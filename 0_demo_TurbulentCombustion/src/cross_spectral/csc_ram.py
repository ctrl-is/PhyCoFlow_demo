import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

from eval_coherence import (
    _cross_spectral_coherence_band_metrics_snapshot,
)

@dataclass
class CSCRAMConfig:
    eps: float = 1e-12

    # Weights for RAM
    alpha_samefreq: float = 1.0
    alpha_crossfreq: float = 1.0
    
    # The physical fields to score
    field_pairs: Optional[list[tuple[int, int]]] = None

    # First version of reward function is band-energy only
    use_samefreq_energy: bool = True
    use_crossfreq_energy: bool = True
    
    # Can be a second version where the reward function is the coherence 
    # rather than the energy
    use_samefreq_coherence: bool = False
    use_crossfreq_coupling: bool = False

    # Optional - If false, computes in normalized model units
    use_denorm: bool = False

# ---------------------------------------------------
# Band-energy Reward Functions
# ---------------------------------------------------
def physical_band_energy_reward_snapshot(
        payload,
        alpha_samefreq: float = 1.0,
        alpha_crossfreq: float = 1.0,
        standardize: bool = True,
        clip: float | None = 3.0,
        device=None,
        dtype=None,
):
    """
    Convert per-snapshot band-energy errors in a scalar reward.

    Used for testing.
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

    total_error = alpha_samefreq * same + alpha_crossfreq * cross
    reward = -total_error

    if standardize:
        reward = (reward - reward.mean()) / (reward.std(unbiased=False) + 1e-12)
    
    if clip is not None:
        reward = reward.clamp(-clip, clip)

    return reward

def compute_csc_ram_cost(
        fields_pred,
        fields_true,
        U,
        bands,
        cfg: CSCRAMConfig,
        mean=None,
        std=None,
):
    """
    Compute per-snapshot CSC-RAM cost.

    Lower cost = better physical agreement.
    Trainer should negate the cost for a reward.
    """
    if cfg.use_denorm:
        if mean is None or std is None:
            raise ValueError("use_denorm=True requires mean and std.")
        
        mean = mean.to(device=fields_pred.device, dtype=fields_pred.dtype).view(1, 1, -1)
        std = std.to(device=fields_pred.device, dtype=fields_pred.dtype).view(1, 1, -1)

        fields_pred = fields_pred * std + mean
        fields_true = fields_true * std + mean

    metrics, payload = _cross_spectral_coherence_band_metrics_snapshot(
        fields_true=fields_pred,
        fields_pred=fields_pred,
        U=U,
        bands=bands,
        field_pairs=cfg.field_pairs,
        eps=cfg.eps,
    )

    same = torch.as_tensor(
        payload["samefreq_relerr_per_snapshot"],
        device=fields_pred.device,
        dtype=fields_pred.dtype
    )

    cross = torch.as_tensor(
        payload["crossfreq_relerr_per_snapshot"],
        device=fields_pred.device,
        dtype=fields_pred.dtype
    )

    cost = cfg.alpha_samefreq * same + cfg.alpha_crossfreq * cross

    metrics["csc_ram_cost_mean"] = float(cost.mean().detach().cpu())
    metrics["csc_ram_cost_std"] = float(cost.std(unbiased=False).detach().cpu())

    return cost, metrics