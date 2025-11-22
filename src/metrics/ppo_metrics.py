# src/metrics/ppo_metrics.py

from __future__ import annotations
import torch
from typing import List, Dict

def aggregate_ppo_update_metrics(
    total_pi: float,
    total_v: float,
    total_ent: float,
    total_kl: float,
    total_clip: float,
    total_gnorm: float,
    nbatches: int
):
    """
    Aggregate PPO minibatch statistics into averaged metrics.
    This is clean and future-proof: PPO only needs to pass totals + nbatches.
    """

    if nbatches == 0:
        return {
            "pi_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clipfrac": 0.0,
            "gradnorm": 0.0,
        }
    
    

    return {
        "pi_loss": total_pi / nbatches,
        "v_loss": total_v / nbatches,
        "entropy": total_ent / nbatches,
        "kl": total_kl / nbatches,
        "clipfrac": total_clip / nbatches,
        "gradnorm": total_gnorm / nbatches,
    }


def compute_episode_stats(episode_returns: List[float], episode_lengths: List[int]) -> Dict[str, float]:
    """
    Basic episodic RL statistics to log into TensorBoard.
    """

    if len(episode_returns) == 0:
        return {
            "episode_return_mean": 0.0,
            "episode_length_mean": 0.0,
        }

    return {
        "episode_return_mean": sum(episode_returns) / len(episode_returns),
        "episode_length_mean": sum(episode_lengths) / len(episode_lengths),
    }
