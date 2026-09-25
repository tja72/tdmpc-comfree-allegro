# Offline training + real-rollout regularization — design

Design for a dedicated, large-batch
policy-training phase against the SysID-identified sim, decoupled from the
real-env step budget, with an explicit penalty against overfitting to that
sim's known-wrong directions. Design only — no code in this pass.

## 1. Where the synthetic rollouts come from

Reuse, don't rebuild: `self.plan_sim` (`vec_trainer.py:113-122`) is already
a `BatchSim` running at `u_model` (the *identified* params, not
`u_reality`), and `BatchMPPIPlanner._store_elites()`
(`batch_mppi.py:207-259`) already knows how to harvest rollouts from it into
a `ReplayBuffer`-shaped `sim_buffer`. The offline phase is a new loop that
drives the same `self.planner.plan()` / `plan_sim` pair **without** touching
`self.env` (the reality-perturbed env) or the real `self.buffer`, and
without the online loop's update-budget bookkeeping
(`updates_per_env_step`, `max_updates_per_iter`, both exist only to ration
compute against a 640ms real planning call — irrelevant once nothing real
is being collected).

Concretely: `VecTrainer.train_offline(num_updates: int)`, called once a
gating condition trips (§4), that:
1. Runs `self.planner.plan()` against `plan_sim` in a tight loop (no
   `self.env.step()`), same as today's inner loop minus the real-env half,
   feeding `sim_buffer` via the existing `_store_elites` path.
2. Runs `self.agent.update(...)` back-to-back with no per-iteration cap,
   at a **larger batch size** than the online path (see below), until
   `num_updates` gradient steps are done. `update()` is unchanged and still
   trains both heads in one call — `value_loss` (Q) and `pi_loss` (policy)
   — so the offline phase is *not* policy-only; §2 below covers why the two
   heads need different treatment of their real-vs-synthetic batch mix.

This is deliberately *not* a new rollout generator — `plan_sim` +
`BatchMPPIPlanner` + `sim_buffer` already do exactly what's needed; the only
new part is the driving loop and its decoupled budget.

**How large is "large"?** `sim_data_ratio=0.5` caps synthetic data at half
of every *online* batch (`bs=256` → 128 sim samples/update), and a full
150k-step run at the CLI default `updates_per_env_step=1.5` already
accumulates on the order of ~2×10⁵ updates total (consistent with
`max_updates_per_iter=24` being the binding cap at `num_envs=16`,
`1.5 × 16 = 24`). So raw *volume* of sim samples touched isn't the gap —
it's that today's sim data only arrives interleaved with, and capped by,
real-time planning. The offline phase should:
- Use a batch size **4-8x** the online `cfg.agent.batch_size=256` (e.g.
  1024-2048) for the main synthetic-data term, since GPU throughput is now
  the only constraint, not wall-clock parity with a 640ms planning call.
- Default `num_updates` to roughly the online run's own total update count
  (order **50k**, i.e. ~25% of a 150k-step run's ~2×10⁵ online updates) as
  a *consolidation* pass, not a full second training run — this number
  needs an actual sweep (net-size-sweep precedent:
  `outputs/vec/netsize_*`), flagged here as a tuning knob, not a fixed
  constant.

## 2. Real-rollout regularization: exact form and where it plugs in

Both buffers already store `mu`/`std` — the planner's action-distribution
label at collection time (`buffer.py:31-32`, `vec_buffer.py:37-38`) — for
every *real* transition. That's the regularization target: distance
between the policy's current action distribution on real-visited states and
the planner's own recorded mean there, i.e. the same functional form as the
existing `bc_mu` prior term (`tdmpc_agent.py:224-226`), just evaluated on a
real-only cohort instead of the mixed online batch.

Extend `update_pi()` (`tdmpc_agent.py:194-245`) with two optional args,
default `None` so the existing online call site (`update()`, :278-280) is
unaffected:

```python
def update_pi(self, z, action, mu, std, z_real=None, mu_real=None):
    ...
    pi_loss = q_loss + cfg.prior_coef * prior_loss        # unchanged
    if z_real is not None:
        _, pis_real, _, _ = self.pi(z_real)
        real_reg = (pis_real - mu_real).pow(2).sum(-1).mean()
        pi_loss = pi_loss + cfg.real_reg_coef * real_reg
```

`train_offline()` calls this with `z` / `action` / `mu` / `std` from a
large synthetic batch (§1) and `z_real` / `mu_real` from a **separately
sampled**, fixed-size (e.g. 256) real cohort drawn fresh from `self.buffer`
every update — not folded into the big synthetic batch, so its gradient
contribution doesn't get diluted 1-in-2048. `real_reg_coef` is a new
`AgentConfig` field, independent of `prior_coef`.

Why an *explicit distance term* only for the actor, not `value_loss`/Q —
this needs justifying, since `update()` trains both heads every offline
step and both see the same large synthetic batch:

- `value_loss` is TD regression directly against each sample's own
  `reward`/`next_obs`. When a batch sample is real, that loss *is already*
  a real-data anchor for Q on that sample — no separate distance penalty is
  needed, because the primary loss already equals "match real ground
  truth" whenever real data is present in the batch.
- `q_loss = -Q(s, pi(s))` (`tdmpc_agent.py:204`) never touches a real
  action label at all — it only reaches real behavior indirectly, through
  Q. And mixing real+synthetic samples into one `prior_loss` batch (as
  `sample_batch()` does today) is *not* the analogous regularizer for pi
  that it is for Q: each sample pulls the policy toward *its own* label
  (real samples toward real `mu`, synthetic toward the elite/mean target),
  averaged over the batch — that's "match whichever label this batch
  happened to contain," diluted 1-in-N by the mix ratio, not "stay close to
  real behavior." Hence pi needs its own independently-weighted `real_reg`
  term (above); Q's existing TD-on-real-samples mechanism is structurally
  already doing that job, *provided* its batch keeps a healthy real
  fraction.

That proviso is the actual risk: if Q's offline batch reuses the same
mostly-synthetic composition built for the large pi batch (§1's 4-8x-bigger,
sim-dominated batch), nothing stops Q's real fraction from being diluted
too — and an offline-trained Q that drifts on `u_model`-biased synthetic
transitions is exactly the `value/bias`/`value/corr` degradation the
`f_no_q` run already showed Q is capable of
(train-without-Q runs, `outputs/vec/f_no_q*`). So: give Q's batch
composition its own explicit knob, `offline_value_real_ratio` (new
`VecTrainConfig` field, default real-heavy — e.g. **0.7**, i.e. the
opposite bias from the pi/prior batch), separate from whatever ratio feeds
the large synthetic batch used for `q_loss`/`prior_loss`. Concretely,
`train_offline()`'s call to `agent.update()` needs to accept two
differently-mixed batches (one per head) rather than the single
`sample_batch()` output `update()` takes today — a small signature change,
noted in §5 below. After the offline phase, check `value/bias`/
`value/corr` from `diagnose_policy.py`'s existing diagnostics (same fields
already reported for `f_no_q` and the net-size sweep) as the empirical
confirmation that Q didn't drift — reusing an existing diagnostic, not
inventing a new metric.

Note on what this *isn't*: today's `sim_data_ratio` mixing is already a
soft, implicit regularizer, since loss-averaging over a blended batch means
sim-heavy gradients get diluted by whatever real fraction is present. But
its strength is capped by `sim_data_ratio ≤ 1` and tied 1:1 to the online
batch size — it can't be tuned independently of "how much sim data am I
using," which is exactly backwards for an offline phase that *wants* a huge
sim batch. The explicit `real_reg` term above decouples "how much synthetic
data trains the policy" from "how hard is the policy pulled back toward
real behavior."

## 3. Interaction with Workstream A (DAgger) / B (decoupled-Q)

Orthogonal by construction — `train_offline()` doesn't need either to
function, but composes with both:

- **Not dependent on `dagger_beta` (Workstream A).** The offline phase's
  synthetic term runs pure BC/distillation against `plan_sim` rollouts
  regardless of how `self.buffer`'s real data was collected. It *does*
  benefit when `dagger_beta > 0`: the real-rollout diagnostic that matters
  (drift ratio = imitation error on policy-visited vs. planner-visited
  states, measured by `scripts/report/diagnose_policy.py`) is exactly what
  `real_reg` regularizes against, and that regularizer is only as
  informative as the real states it's evaluated on. With `dagger_beta=0`,
  `self.buffer` holds only planner-visited real states, so `real_reg`
  constrains the policy on states it may not itself visit; with DAgger on,
  the real buffer includes policy-visited real states too, which is the
  harder and more relevant target.
- **Not dependent on resolving `bc_no_q`/`bc_mu_no_q` (Workstream B).** The
  synthetic-batch prior term reuses whatever `actor_mode` the run is
  configured with — `train_offline()` takes `actor_mode` as a passthrough,
  same as `update_pi` does today, not a hardcoded choice. Recommend
  defaulting the offline phase to `bc_mu` specifically (best-performing
  target so far — rung F of the ablation ladder)
  until `e_no_q`'s rerun settles the open per-step-target comparison; if
  `bc_no_q`/`bc_mu_no_q` ever wins, dropping `q_loss` from `pi_loss`
  in `train_offline()` is a one-line change to which branch fires, not a
  redesign.
- The net-size sweep (`outputs/vec/netsize_*`) is unaffected either
  way — `mlp_dim`/`latent_dim` are architecture, offline training changes
  nothing about the network shape.

## 4. Gating: when does the offline phase start

Adopt the threshold idea directly: trigger `train_offline()` once SysID's
**validation loss** — `run_sysid()`'s `val_loss`
(`vec_trainer.py:186-224`, already logged to `sysid_log` every
`sysid_every_steps`) — drops below a new `offline_start_val_loss`
threshold, rather than a fixed real-step count. Use `val_loss`
specifically, not `param_err_mean` or `gap_closed`: both of the latter are
computed against `self.u_reality`
(`ref = self.sysid._evaluate(u_reality, ...)`, :199-200), which only exists
because this is a simulated "reality" — a real robot has no `u_reality` to
diff against. `val_loss` alone (SysID's own reconstruction loss on held-out
*real* chunks, `sysid.py`'s `_evaluate(u_model, eval_chunks)`) needs no
privileged ground truth and is the one signal from this list that survives
the move to hardware, matching the real-robot pipeline framing.

Trigger once per training run (offline phase is a one-shot consolidation
stage, not a repeating one, in this v1): after each scheduled `run_sysid()`
call, if `val_loss < offline_start_val_loss` and `train_offline()` hasn't
run yet this training run, run it, then resume the online loop. Requires
`sysid_enabled=True` and at least one completed `run_sysid()` call (i.e.
past `sysid_warmup_steps`) — with SysID off there's no `u_model` update
signal to gate on, and the offline phase already assumes the SysID
machinery is live.

Implementation stays in-process (`VecTrainer` gains `train_offline()` and a
check next to the existing `run_sysid()` call site,
`vec_trainer.py:380-388`), not a separate script: `self.buffer` (real
data), `self.plan_sim`/`self.planner` (at current `u_model`), and
`self.agent` are all already GPU-resident, and none of them are currently
serialized to disk (only `agent.pt`/`params.npy`, `_save()`,
`vec_trainer.py:475-478`) — a standalone "resume from checkpoint, run
offline phase, save" script would need new buffer-checkpointing
infrastructure that doesn't exist yet. That's legitimate future work for
a real-robot pipeline (where the offline phase genuinely must
be a separate process between real-rollout sessions), but out of scope for
this design pass; flagging it here rather than silently assuming it away.

### 4a. Controlled trigger, and periodic consolidation (implemented, v2)

The v1 gate above fires on the *first* `run_sysid()` call whose `val_loss`
dips below `offline_start_val_loss`. In practice this trigger point falls
out of noise, not a controlled choice: the GPU path has no
`torch.use_deterministic_algorithms`, no CUBLAS workspace pin, and atomic
scatter-adds in the contact solver, so `val_loss` traces aren't reproducible
even at fixed seed. Two new knobs on `VecTrainConfig`, both wired through
`scripts/train_vec.py`'s existing `--offline_*` convention, address this —
gating logic lives in `VecTrainer._maybe_offline()`, called from the same
`run_sysid()` call site as the v1 check:

- `offline_confirm_checks: int = 1` — require `val_loss <
  offline_start_val_loss` on this many *consecutive* `run_sysid()` calls,
  not just the first dip. A counter increments on a passing check and resets
  to 0 on any failing one, so a single noisy dip below threshold no longer
  fires the phase on its own.
- `offline_min_step: int | None = None` — a hard floor: the phase never
  fires before this env step, regardless of `val_loss` or the confirm
  counter's state (a qualifying-but-blocked check keeps its progress rather
  than resetting, so the phase fires as soon as `offline_min_step` is
  reached if the gate was already satisfied).
- `offline_force_step: int | None = None` — for controlled experiments that
  need a fixed, non-noisy trigger point instead of a `val_loss` threshold:
  ignores `val_loss`/the confirm counter entirely and fires exactly once as
  soon as `self.step >= offline_force_step`. Takes priority over
  `offline_start_val_loss` when both are set (the `val_loss` path is
  skipped, not merely deprioritized). Still subject to `offline_min_step` if
  both are set.

Separately, `offline_mode: str = "once"` (default, unchanged v1 behavior —
a single one-shot burst gated by `_ran_offline`) now has a `"periodic"`
counterpart: it drops the single-fire guard and instead runs a smaller
`offline_periodic_num_updates` (default `8_000`, vs. `offline_num_updates`'s
`50_000`) burst every time the gate (§4's threshold, sharpened by the three
knobs above) re-qualifies — spreading consolidation across SysID's
improving `u_model` estimate instead of committing it all to one snapshot.
`offline_force_step` still fires only once even under `"periodic"`, since
there is only one forced trigger point to re-qualify against.

## 5. Open question: pretrain in sim, or train from scratch on real data?

**Recommendation: pretrain in sim.**

Reasoning:
- The stated motivation for this phase is to buy the policy many more
  gradient updates per real robot interaction, which is the actual
  bottleneck on hardware. Training from scratch on real data spends
  exactly the resource this phase exists to economize.
- The envisioned real-robot pipeline already has a
  `pretraining-SysID → MPPI` stage before any closed-loop real training,
  specifically to get MPPI to "not too bad" so a dropped cube doesn't cost
  a human reset for nothing. A policy pretrained in sim during that same
  window is close to free — the sim is already running for the MPPI
  shakedown — and gives DAgger-style relabeling (Workstream A) a
  non-random policy to relabel against from the first real episode, instead
  of collecting policy-visited states that are just noise for the first N
  episodes.
- The `real_reg` term (§2) only regularizes against something once real
  `(state, mu)` pairs exist. Before any real data, there's nothing to
  pull the policy back toward, so the natural sequencing is: (a) sim-only
  pretraining with `real_reg_coef=0` (pure prior-loss against
  `plan_sim` rollouts, gated on pretraining-SysID's own `val_loss`
  convergence, not the full `offline_start_val_loss` threshold in §4) →
  (b) closed-loop real training begins from a non-random policy, `buffer`
  starts accumulating real `mu` labels → (c) the §4-gated offline
  consolidation phase now has real data for `real_reg` to bite on.
- Counter-risk, worth stating explicitly since it's the same failure mode
  the real-rollout regularization (§2) is designed to guard against: pretraining too long/too well
  risks locking the policy into the SysID sim's known-wrong directions
  (48 untouched joint/inertial params) before any real correction exists.
  Mitigate by keeping the pretraining stage deliberately short/coarse —
  the pipeline's own "not too bad" bar, not full convergence — and treating
  the first §4-gated offline phase (the one with `real_reg` active) as the
  pass that actually matters for final policy quality.

## Summary of new surface (for a future implementation pass)

- `AgentConfig.real_reg_coef: float` (new field).
- `TDMPCAgent.update_pi(..., z_real=None, mu_real=None)` (extended,
  backward-compatible).
- `TDMPCAgent.update()` needs an offline-mode path (or a sibling
  `update_offline()`) that accepts two batches — a large, synthetic-heavy
  one for `q_loss`/`prior_loss`, and a separate, real-heavy one
  (`offline_value_real_ratio`) for `value_loss` — instead of the single
  `sample_batch()` output the online `update()` takes today.
- `VecTrainConfig.offline_start_val_loss`, `.offline_num_updates`,
  `.offline_batch_size`, `.offline_value_real_ratio` (new fields,
  `train_vec.py` CLI passthrough following its existing `--foo` →
  dataclass-field convention).
- `VecTrainer.train_offline(num_updates)` (new method).
- One new gating check next to `run_sysid()`'s call site
  (`vec_trainer.py:380-388`).
- (v2, §4a) `VecTrainConfig.offline_confirm_checks`, `.offline_min_step`,
  `.offline_force_step`, `.offline_mode`, `.offline_periodic_num_updates`
  (new fields, same CLI passthrough convention);
  `VecTrainer._maybe_offline(s)` (new method, replaces the inline v1 gating
  check at the `run_sysid()` call site).
