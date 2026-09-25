"""Fingertips reward/cost terms, one function per term (IsaacLab mdp/rewards.py style).

Every function takes plain arrays/floats and returns an unweighted float; FingertipsEnv
applies cfg weights and sums (see _cost / _reward_shaped in fingertips.py).
"""

import numpy as np


def pos_error(pos, target_pos):
    return float(np.linalg.norm(pos - target_pos))


def quat_error(quat, target_quat):
    return float(1.0 - np.dot(quat, target_quat) ** 2)


def quat_conj(q):
    """Conjugate of a unit [w, x, y, z] quaternion -- its inverse."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(q1, q2):
    """Hamilton product of two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_to_dcm(q):
    """[w, x, y, z] quaternion to rotation matrix."""
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)],
    ])


def _fingertip_distances(obj_pos, ft_pos):
    return np.linalg.norm(ft_pos - obj_pos[None, :], axis=1)


def grasp_closure(obj_pos, obj_dcm, ft_pos):
    """Antipodal-alignment metric: ~0 when fingertip directions cancel around the object."""
    closure_vecs = [obj_dcm.T @ (ft - obj_pos) for ft in ft_pos]
    closure_sum = sum(v / np.linalg.norm(v) for v in closure_vecs)
    return float(np.dot(closure_sum, closure_sum))


# --- legacy terms ---

def contact_cost(obj_pos, ft_pos):
    return float(np.sum(_fingertip_distances(obj_pos, ft_pos) ** 2))

def position_cost(obj_pos, target_pos):
    err = obj_pos - target_pos
    return float(np.dot(err, err))

def quaternion_cost(obj_quat, target_quat):
    return quat_error(obj_quat, target_quat)

def control_cost(cmd):
    return float(np.dot(cmd, cmd))


# --- shaped terms ---

def finger_obj_distance_tanh(obj_pos, ft_pos, std):
    """1 - tanh(dist / std): bounded (0, 1], longer gradient tail than a Gaussian kernel.
    Uses the farthest fingertip, not the mean -- otherwise two close fingertips can hide one
    idle one, both here and in grasp_distribution_exp's gate below (same function)."""
    max_dist = float(np.max(_fingertip_distances(obj_pos, ft_pos)))
    return 1.0 - np.tanh(max_dist / std)

def grasp_distribution_exp(obj_pos, dcm, ft_pos, std, gate_std):
    # soft-gated by proximity instead of a bool contact check, to avoid reward flicker
    closure = grasp_closure(obj_pos, dcm, ft_pos)
    gate = finger_obj_distance_tanh(obj_pos, ft_pos, gate_std)
    reward = np.exp(-closure / std ** 2)
    return float(gate * reward)

def position_tracking(pos_err, std):
    """1 - tanh(pos_err / std): bounded (0, 1], longer gradient tail than a Gaussian kernel."""
    return 1.0 - np.tanh(pos_err / std)

def track_obj_pos_tanh_gated(pos_err, obj_pos, ft_pos, std, gate_std):
    # continuous engagement gate (not boolean "any contact"): scales the absolute reward down
    # smoothly as the worst-placed fingertip drifts, so a 2-close-1-idle grasp can't collect the
    # same reward as genuine 3-way engagement -- also still zeros out the "idle agent starts
    # lucky-close" case, since the gate -> 0 with no fingertips near the object at all
    gate = finger_obj_distance_tanh(obj_pos, ft_pos, gate_std)
    return position_tracking(pos_err, std) * gate

def position_progress(pos_err, prev_pos_err, std):
    # potential-based: reward reducing error, not just sitting close to the goal
    return position_tracking(pos_err, std) - position_tracking(prev_pos_err, std)

def quaternion_tracking(quat_err, std):
    """1 - tanh(quat_err / std): bounded (0, 1], longer gradient tail than a Gaussian kernel."""
    return 1.0 - np.tanh(quat_err / std)

def track_obj_quad_tanh_gated(quat_err, obj_pos, ft_pos, std, gate_std):
    # same continuous engagement gate as track_obj_pos_tanh_gated above
    gate = finger_obj_distance_tanh(obj_pos, ft_pos, gate_std)
    return quaternion_tracking(quat_err, std) * gate

def quaternion_progress(quat_err, prev_quat_err, std):
    # potential-based: reward reducing error, not just sitting close to the goal
    return quaternion_tracking(quat_err, std) - quaternion_tracking(prev_quat_err, std)
