import torch
import numpy as np

from eval_coherence import (
    _cross_spectral_coherence_band_metrics,
)

# Reward function per Batch
def physical_band_energy_reward_batch(metrics, alpha_crossfreq: float = 1.0):
    """
    Convert band-energy coherence metrics into a scalar reward.

    Higher Reward = better physical band-energy agreement
    """
    same_freq = metrics["samefreq_energy_relerr_mean"]
    cross_freq = metrics["crossfreq_energy_relerr_me"]

    loss = same_freq + alpha_crossfreq * cross_freq
    reward = -loss

    return reward

# Reward function per Snapshot
def physical_band_energy_reward_snapshot(
        payload,
        alpha_crossfreq: float = 1.0,
        standarize: bool = True,
        clip: float | None = 3.0,
        device=None,
        dtype=None,
):
    """
    Convert per-snapshot band-energy errors in a scalar reward.

    USED FOR REINFORCE ADJOINY MATCHING.
    """
    same = torch.as_tensor(
        payload["samefreq_relerr_per_snapshot"],
        device=device,
        dtype=dtype,
    )

    cross = torch.as_tensor(
        payload["crossfreq_relerr_per_snapshot"],
        device=device,
        dtype=dtype,
    )

    total_error = same + alpha_crossfreq * cross
    reward = -total_error

    if standarize:
        reward = (reward - reward.mean()) / (reward.std(unbaised=False) + 1e-12)
    
    if clip is not None:
        reward = reward.clamp(-clip, clip)

    return reward
