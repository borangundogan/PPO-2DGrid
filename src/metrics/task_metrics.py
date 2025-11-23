# src/metrics/task_distribution_metrics.py

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


Array = np.ndarray


def compute_mean_std(features: Array) -> Tuple[Array, Array]:
    """
    Compute per-dimension mean and std of a feature matrix.

    Args:
        features: Array of shape (N, D) or (N,) containing samples.

    Returns:
        mean: (D,) array
        std:  (D,) array
    """
    feats = np.asarray(features)
    if feats.ndim == 1:
        feats = feats[:, None]

    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-8  # avoid zero std
    return mean, std


def kl_diag_gaussians(
    mean_p: Array,
    std_p: Array,
    mean_q: Array,
    std_q: Array,
) -> float:
    """
    KL divergence KL(P || Q) between two diagonal Gaussians.

    P ~ N(mean_p, diag(std_p^2)), Q ~ N(mean_q, diag(std_q^2))

    Returns:
        Scalar KL(P || Q)
    """
    var_p = std_p ** 2
    var_q = std_q ** 2

    # KL for diagonal Gaussians, summed over dimensions
    term1 = np.log(std_q / std_p)
    term2 = (var_p + (mean_p - mean_q) ** 2) / (2.0 * var_q)
    kl = np.sum(term1 + term2 - 0.5)
    return float(kl)


def js_diag_gaussians(
    mean_p: Array,
    std_p: Array,
    mean_q: Array,
    std_q: Array,
) -> float:
    """
    Jensen–Shannon divergence between two diagonal Gaussians,
    using the symmetric definition:

        JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        with M = 0.5 * (P + Q)
    """
    # Mixture parameters (simple average of means/stds)
    mean_m = 0.5 * (mean_p + mean_q)
    std_m = 0.5 * (std_p + std_q)

    kl_pm = kl_diag_gaussians(mean_p, std_p, mean_m, std_m)
    kl_qm = kl_diag_gaussians(mean_q, std_q, mean_m, std_m)
    js = 0.5 * (kl_pm + kl_qm)
    return float(js)


def wasserstein_1d(u: Array, v: Array) -> float:
    """
    1D Wasserstein-1 (Earth Mover's Distance) between two
    sets of samples u and v.

    This assumes both arrays are 1D. For multi-dimensional
    data we compute per-dimension distances separately.
    """
    u = np.asarray(u).ravel()
    v = np.asarray(v).ravel()

    if len(u) == 0 or len(v) == 0:
        return 0.0

    u_sorted = np.sort(u)
    v_sorted = np.sort(v)

    n = min(len(u_sorted), len(v_sorted))
    u_sorted = u_sorted[:n]
    v_sorted = v_sorted[:n]

    return float(np.mean(np.abs(u_sorted - v_sorted)))


def wasserstein_mean(features_p: Array, features_q: Array) -> float:
    """
    Average 1D Wasserstein distance over feature dimensions.

    Args:
        features_p: (N_p, D) samples from distribution P
        features_q: (N_q, D) samples from distribution Q

    Returns:
        Scalar average Wasserstein distance across D dims.
    """
    x = np.asarray(features_p)
    y = np.asarray(features_q)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    d = x.shape[1]
    distances = []
    for i in range(d):
        distances.append(wasserstein_1d(x[:, i], y[:, i]))
    return float(np.mean(distances))


def compare_two_feature_sets(
    feats_a: Array,
    feats_b: Array,
) -> Dict[str, float]:
    """
    Convenience function: given two feature sets (states, embeddings, etc.),
    compute mean/std + KL, JS, Wasserstein metrics between them.

    Args:
        feats_a: (N_a, D) samples for task A
        feats_b: (N_b, D) samples for task B

    Returns:
        Dictionary with scalar metrics:
            - mean_norm_diff
            - js_div
            - kl_ab
            - kl_ba
            - wasserstein
    """
    mean_a, std_a = compute_mean_std(feats_a)
    mean_b, std_b = compute_mean_std(feats_b)

    kl_ab = kl_diag_gaussians(mean_a, std_a, mean_b, std_b)
    kl_ba = kl_diag_gaussians(mean_b, std_b, mean_a, std_a)
    js = js_diag_gaussians(mean_a, std_a, mean_b, std_b)
    w = wasserstein_mean(feats_a, feats_b)

    mean_norm_diff = float(np.linalg.norm(mean_a - mean_b))

    return {
        "mean_norm_diff": mean_norm_diff,
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "js_div": js,
        "wasserstein": w,
    }


def compare_task_feature_dict(
    feature_dict: Dict[str, Array]
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Pairwise comparison for multiple tasks.

    Args:
        feature_dict: mapping task_name -> (N, D) feature array

    Returns:
        mapping (task_i, task_j) -> metric_dict
        where metric_dict is the output of compare_two_feature_sets.
    """
    task_names = list(feature_dict.keys())
    results: Dict[Tuple[str, str], Dict[str, float]] = {}

    for i in range(len(task_names)):
        for j in range(i + 1, len(task_names)):
            ti = task_names[i]
            tj = task_names[j]
            feats_i = feature_dict[ti]
            feats_j = feature_dict[tj]

            metrics = compare_two_feature_sets(feats_i, feats_j)
            results[(ti, tj)] = metrics

    return results
