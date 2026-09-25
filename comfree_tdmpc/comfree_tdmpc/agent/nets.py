"""Network components, translated from TD-MPC2.

The ensemble is written with explicit batched parameters (K, out, in) and bmm
rather than functorch's `combine_state_for_ensemble`, which is deprecated and
brittle across torch versions.  Behaviour is identical: K independent MLPs
evaluated in one pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
    """Simplicial normalisation (https://arxiv.org/abs/2204.00616)."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear + LayerNorm + Mish."""

    def __init__(self, *args, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(super().forward(x)))


def mlp(in_dim: int, hidden: list[int] | int, out_dim: int, act=None) -> nn.Sequential:
    if isinstance(hidden, int):
        hidden = [hidden]
    dims = [in_dim] + list(hidden) + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1]))
    layers.append(
        NormedLinear(dims[-2], dims[-1], act=act) if act is not None else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*layers)


class EnsembleLinear(nn.Module):
    def __init__(self, k: int, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(k, out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(k, 1, out_features))
        for i in range(k):
            nn.init.trunc_normal_(self.weight[i], std=1.0 / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (K, B, in) -> (K, B, out)
        return torch.baddbmm(self.bias, x, self.weight.transpose(1, 2))


class EnsembleMLP(nn.Module):
    """K independent MLPs with LayerNorm + Mish, evaluated in one pass."""

    def __init__(self, k: int, in_dim: int, hidden: list[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.k = k
        dims = [in_dim] + list(hidden) + [out_dim]
        self.layers = nn.ModuleList(
            [EnsembleLinear(k, dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.norm_weight = nn.ParameterList(
            [nn.Parameter(torch.ones(k, 1, d)) for d in dims[1:-1]]
        )
        self.norm_bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(k, 1, d)) for d in dims[1:-1]]
        )
        self.dropout = dropout
        self.act = nn.Mish(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in) or (K, B, in) -> (K, B, out)."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.k, -1, -1)
        n_hidden = len(self.layers) - 1
        for i in range(n_hidden):
            x = self.layers[i](x)
            if self.dropout and i == 0:
                x = F.dropout(x, self.dropout, self.training)
            x = F.layer_norm(x, (x.shape[-1],))
            x = self.act(x * self.norm_weight[i] + self.norm_bias[i])
        return self.layers[-1](x)

    def zero_last_layer_(self) -> None:
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1 + x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


class TwoHot:
    """Discrete regression on a symlog-spaced bin grid (TD-MPC2 / DreamerV3)."""

    def __init__(self, num_bins: int, vmin: float, vmax: float, device: torch.device):
        self.num_bins = num_bins
        self.vmin, self.vmax = vmin, vmax
        self.bin_size = (vmax - vmin) / (num_bins - 1)
        self.bins = torch.linspace(vmin, vmax, num_bins, device=device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(symlog(x), self.vmin, self.vmax).squeeze(-1)
        idx = torch.floor((x - self.vmin) / self.bin_size).long()
        idx = idx.clamp(0, self.num_bins - 1)
        offset = ((x - self.vmin) / self.bin_size - idx.float()).unsqueeze(-1)
        out = torch.zeros(*x.shape, self.num_bins, device=x.device)
        out.scatter_(-1, idx.unsqueeze(-1), 1 - offset)
        out.scatter_(-1, ((idx + 1) % self.num_bins).unsqueeze(-1), offset)
        return out

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits, dim=-1)
        return symexp((p * self.bins).sum(-1, keepdim=True))

    def soft_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        return -(self.encode(target) * log_p).sum(-1, keepdim=True)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """TD-MPC2's Gaussian log-probability.

    Note the multiplication by the action dimension rather than the textbook
    `- 0.5 * A * log(2 pi)` constant offset.  This is what the reference
    implementation does, and TD-M(PC)^2's `prior_coef * action_dim / 61` was
    tuned against this convention -- writing the textbook version instead makes
    the policy's behaviour-cloning term ~A times too weak, and the policy then
    ignores the planner and follows the value function alone.
    """
    residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
    return (residual - 0.5 * math.log(2 * math.pi)) * eps.size(-1)


def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class RunningScale:
    """Running trimmed scale estimator (TD-MPC2)."""

    def __init__(self, tau: float, device: torch.device):
        self.tau = tau
        self._value = torch.ones(1, dtype=torch.float32, device=device)

    @property
    def value(self) -> float:
        return float(self._value.item())

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().flatten()
        lo = torch.quantile(x, 0.05)
        hi = torch.quantile(x, 0.95)
        self._value.lerp_(torch.clamp(hi - lo, min=1.0).reshape(1), self.tau)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 / self.value)
