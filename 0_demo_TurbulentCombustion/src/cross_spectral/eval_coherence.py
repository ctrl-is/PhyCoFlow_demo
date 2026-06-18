import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from cross_spectral import (
    gft,
    compute_band_energy,
    compute_all_cross_band_covariances,
)

# -----------------------------------------------------------------------------
# Band Energy Diagnostics
# -----------------------------------------------------------------------------

def _safe_name(name):
    """Make a string safe for filenames."""
    return str(name).replace("/", "_").replace(" ", "_").replace("→", "to")

def _cross_spectral_coherence_band_metrics(
    fields_true,
    fields_pred,
    U,
    bands,
    field_pairs=None,
    eps: float = 1e-12,
):
    """
    Compute band-energy agreement diagnostics for both same-frequency and
    cross-frequency visualizations.

    SAME-FREQUENCY BAND ENERGY:
        For each graph-frequency band m and field c:

            E_same[m, c] = mean_b sum_{k in band m} |x_hat[b, k, c]|^2

        Ratio:

            E_same_pred[m, c] / E_same_true[m, c]

    CROSS-FREQUENCY BAND ENERGY:
        Uses the raw cross-band covariance object S from the cross-frequency
        coherence code:

            S[m, n, c1, c2]

        where S measures batch co-fluctuation between centered band energy of
        field c1 in band m and centered band energy of field c2 in band n.

        Ratio:

            |S_pred[m, n, c1, c2]| / |S_true[m, n, c1, c2]|

    Args:
        fields_true: [B, N, C]
        fields_pred: [B, N, C]
        U: [N, K] graph Fourier basis
        bands: dict mapping band name -> graph-frequency indices
        field_pairs: optional list of (c1, c2). If None, uses all c1 < c2.
        eps: numerical stability

    Returns:
        metrics: flat scalar dict
        payload: arrays for plotting
    """

    # ---------------------------------------------------
    # 1. Graph Fourier transform
    # ---------------------------------------------------
    gft_true = gft(fields_true, U)  # [B, K, C]
    gft_pred = gft(fields_pred, U)  # [B, K, C]

    device = gft_true.device
    C = gft_true.shape[-1]

    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]

    band_names = list(bands.keys())
    num_bands = len(band_names)
    num_pairs = len(field_pairs)

    # ---------------------------------------------------
    # 2. Same-frequency band energy
    # ---------------------------------------------------
    samefreq_energy_true = []
    samefreq_energy_pred = []

    for band_name in band_names:
        band_idx = bands[band_name]

        # [B, C]
        E_true_bc = compute_band_energy(gft_true, band_idx)
        E_pred_bc = compute_band_energy(gft_pred, band_idx)

        # Average over batch -> [C]
        E_true_c = E_true_bc.mean(dim=0)
        E_pred_c = E_pred_bc.mean(dim=0)

        samefreq_energy_true.append(E_true_c)
        samefreq_energy_pred.append(E_pred_c)

    # [M, C]
    samefreq_energy_true = torch.stack(samefreq_energy_true, dim=0)
    samefreq_energy_pred = torch.stack(samefreq_energy_pred, dim=0)

    # [M, C]
    samefreq_energy_ratio = samefreq_energy_pred / (samefreq_energy_true + eps)

    samefreq_energy_relerr = torch.abs(samefreq_energy_pred - samefreq_energy_true) / (samefreq_energy_true + eps)

    # ---------------------------------------------------
    # 3. Cross-frequency band energy
    # ---------------------------------------------------
    # S has shape [M, M, C, C].
    # S[m, n, c1, c2] measures cross-band energy co-fluctuation.
    S_true, band_names_from_S = compute_all_cross_band_covariances(
        gft_true,
        bands,
    )
    S_pred, _ = compute_all_cross_band_covariances(
        gft_pred,
        bands,
    )

    # Use the band ordering returned by the cross-band covariance function.
    band_names = list(band_names_from_S)
    num_bands = len(band_names)

    crossfreq_energy_true = torch.empty(
        (num_bands, num_bands, num_pairs),
        device=device,
        dtype=S_true.real.dtype,
    )

    crossfreq_energy_pred = torch.empty_like(crossfreq_energy_true)

    for p, (c1, c2) in enumerate(field_pairs):
        crossfreq_energy_true[:, :, p] = torch.abs(S_true[:, :, c1, c2])
        crossfreq_energy_pred[:, :, p] = torch.abs(S_pred[:, :, c1, c2])

    # [M, M, P]
    crossfreq_energy_ratio = crossfreq_energy_pred / (crossfreq_energy_true + eps)

    crossfreq_energy_relerr = torch.abs(crossfreq_energy_pred - crossfreq_energy_true) / (crossfreq_energy_true + eps)

    # Cross-frequency means off-diagonal band pairs only.
    off_diag_mask = ~torch.eye(num_bands, dtype=torch.bool, device=device)

    crossfreq_ratio_offdiag = crossfreq_energy_ratio[off_diag_mask, :]
    crossfreq_relerr_offdiag = crossfreq_energy_relerr[off_diag_mask, :]

    # ---------------------------------------------------
    # 4. Scalar metrics
    # ---------------------------------------------------
    metrics = {}

    # Same-frequency summaries.
    samefreq_energy_ratio_mean_by_band = samefreq_energy_ratio.mean(dim=1)
    samefreq_energy_relerr_mean_by_band = samefreq_energy_relerr.mean(dim=1)

    for m, band_name in enumerate(band_names):
        clean = str(band_name).lower()

        metrics[f"samefreq_energy_ratio_{clean}"] = float(
            samefreq_energy_ratio_mean_by_band[m].detach().cpu()
        )

        metrics[f"samefreq_energy_relerr_{clean}"] = float(
            samefreq_energy_relerr_mean_by_band[m].detach().cpu()
        )

    metrics["samefreq_energy_ratio_mean"] = float(
        samefreq_energy_ratio.mean().detach().cpu()
    )

    metrics["samefreq_energy_relerr_mean"] = float(
        samefreq_energy_relerr.mean().detach().cpu()
    )

    # Cross-frequency summaries.
    metrics["crossfreq_energy_ratio_mean"] = float(
        crossfreq_ratio_offdiag.mean().detach().cpu()
    )

    metrics["crossfreq_energy_relerr_mean"] = float(
        crossfreq_relerr_offdiag.mean().detach().cpu()
    )

    for i, band_i in enumerate(band_names):
        for j, band_j in enumerate(band_names):
            if i == j:
                continue

            key = f"{str(band_i).lower()}_to_{str(band_j).lower()}"

            metrics[f"crossfreq_energy_ratio_{key}"] = float(
                crossfreq_energy_ratio[i, j, :].mean().detach().cpu()
            )

            metrics[f"crossfreq_energy_relerr_{key}"] = float(
                crossfreq_energy_relerr[i, j, :].mean().detach().cpu()
            )

    # ---------------------------------------------------
    # 5. Payload for plotting / saving
    # ---------------------------------------------------
    payload = {
        "band_names": np.asarray(band_names),
        "field_pairs": np.asarray(field_pairs),

        # Same-frequency band energy, shape [M, C].
        "samefreq_energy_true": samefreq_energy_true.detach().cpu().numpy(),
        "samefreq_energy_pred": samefreq_energy_pred.detach().cpu().numpy(),
        "samefreq_energy_ratio": samefreq_energy_ratio.detach().cpu().numpy(),
        "samefreq_energy_relerr": samefreq_energy_relerr.detach().cpu().numpy(),

        # Cross-frequency band energy, shape [M, M, P].
        "crossfreq_energy_true": crossfreq_energy_true.detach().cpu().numpy(),
        "crossfreq_energy_pred": crossfreq_energy_pred.detach().cpu().numpy(),
        "crossfreq_energy_ratio": crossfreq_energy_ratio.detach().cpu().numpy(),
        "crossfreq_energy_relerr": crossfreq_energy_relerr.detach().cpu().numpy(),

        # Raw cross-band covariance tensors, shape [M, M, C, C].
        "S_true": S_true.detach().cpu().numpy(),
        "S_pred": S_pred.detach().cpu().numpy(),
    }

    return metrics, payload

def _cross_spectral_coherence_band_metrics_snapshot(
    fields_true,
    fields_pred,
    U,
    bands,
    field_pairs=None,
    eps: float = 1e-12,
):
    """
    Per-snapshot band-energy diagnostics only.

    Computes same-frequency and cross-frequency band-energy relative erros per snapshot.

    Args:
        fields_true: [B, N, C]
        fields_pred: [B, N, C]
        U: [N, K] graph fourier basis
        bands: dict mapping band name -> graph frequency indices
        field_pairs: optional list of (c1, c2). If None, uses all c1 < c2.
        eps: numerical stability
    
    Returns:
        metrics:
    """

    # ---------------------------------------------------
    # 1. Graph Fourier Transform
    # ---------------------------------------------------
    gft_true = gft(fields_true, U)
    gft_pred = gft(fields_pred, U)

    device = gft_true.device
    B, _, C = gft_true.shape

    if field_pairs is None:
        field_pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    
    band_names = list(bands.keys())
    M = len(band_names)
    P = len(field_pairs)

    # ---------------------------------------------------
    # 2. Same-frequency band energy per snapshot
    # ---------------------------------------------------
    samefreq_energy_true = []
    samefreq_energy_pred = []

    for bandName in band_names:
        band_idx = bands[bandName]

        E_true_bc = compute_band_energy(gft_true, band_idx)
        E_pred_bc = compute_band_energy(gft_pred, band_idx)

        # [B, C]
        samefreq_energy_true.append(E_true_bc)
        samefreq_energy_pred.append(E_pred_bc)
    
    # [B, M, C]
    samefreq_energy_true = torch.stack(samefreq_energy_true, dim=1)
    samefreq_energy_pred = torch.stack(samefreq_energy_pred, dim=1)

    samefreq_energy_ratio = samefreq_energy_pred / (samefreq_energy_true + eps)

    samefreq_energy_relerr = torch.abs(samefreq_energy_pred - samefreq_energy_true) / (samefreq_energy_true + eps)

    # [B]
    samefreq_relerr_per_snapshot = samefreq_energy_relerr.mean(dim=(1,2))
    
    # ---------------------------------------------------
    # 3. Cross-frequency band energy per snapshot
    # ---------------------------------------------------
    # Center band energies across batch, matching your cross-frequency
    # covariance construction but keeping per-snapshot products instead
    # of immediately averaging over B.
    z_true = samefreq_energy_true - samefreq_energy_true.mean(dim=0, keepdim=True)
    z_pred = samefreq_energy_pred - samefreq_energy_pred.mean(dim=0, keepdim=True)

    crossfreq_energy_true = torch.empty(
        (B, M, M, P),
        device=device,
        dtype=z_true.dtype,
    )

    crossfreq_energy_pred = torch.empty_like(crossfreq_energy_true)

    for p, (c1, c2) in enumerate(field_pairs):
        # [B, M, M]
        true_pair = torch.einsum(
            "bm,bn->bmn",
            z_true[:, :, c1],
            z_true[:, :, c2],
        )

        pred_pair = torch.einsum(
            "bm,bn->bmn",
            z_pred[:, :, c1],
            z_pred[:, :, c2],
        )

        crossfreq_energy_true[:, :, :, p] = torch.abs(true_pair)
        crossfreq_energy_pred[:, :, :, p] = torch.abs(pred_pair)

    crossfreq_energy_ratio = crossfreq_energy_pred / (crossfreq_energy_true + eps)

    crossfreq_energy_relerr = torch.abs(crossfreq_energy_pred - crossfreq_energy_true) / (crossfreq_energy_true + eps)

    # Cross-frequency = off-diagonal band pairs only.
    off_diag_mask = ~torch.eye(M, dtype=torch.bool, device=device)

    # [B, M*(M-1), P]
    crossfreq_relerr_offdiag = crossfreq_energy_relerr[:, off_diag_mask, :]

    # [B]
    crossfreq_relerr_per_snapshot = crossfreq_relerr_offdiag.mean(dim=(1, 2))

    # ---------------------------------------------------
    # 4. Total error per snapshot, still not a reward
    # ---------------------------------------------------
    total_relerr_per_snapshot = (samefreq_relerr_per_snapshot + crossfreq_relerr_per_snapshot)

    # ---------------------------------------------------
    # 5. Scalar logging metrics
    # ---------------------------------------------------
    metrics = {
        "samefreq_relerr_snapshot_mean": float(
            samefreq_relerr_per_snapshot.mean().detach().cpu()
        ),
        "samefreq_relerr_snapshot_std": float(
            samefreq_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
        "crossfreq_relerr_snapshot_mean": float(
            crossfreq_relerr_per_snapshot.mean().detach().cpu()
        ),
        "crossfreq_relerr_snapshot_std": float(
            crossfreq_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
        "total_relerr_snapshot_mean": float(
            total_relerr_per_snapshot.mean().detach().cpu()
        ),
        "total_relerr_snapshot_std": float(
            total_relerr_per_snapshot.std(unbiased=False).detach().cpu()
        ),
    }

    # Same-frequency per-band means.
    samefreq_relerr_band_mean = samefreq_energy_relerr.mean(dim=(0, 2))  # [M]

    for m, band_name in enumerate(band_names):
        clean = str(band_name).lower()
        metrics[f"samefreq_relerr_{clean}_snapshot_mean"] = float(
            samefreq_relerr_band_mean[m].detach().cpu()
        )

    # Cross-frequency per-band-pair means.
    for i, band_i in enumerate(band_names):
        for j, band_j in enumerate(band_names):
            if i == j:
                continue

            key = f"{str(band_i).lower()}_to_{str(band_j).lower()}"

            metrics[f"crossfreq_relerr_{key}_snapshot_mean"] = float(
                crossfreq_energy_relerr[:, i, j, :].mean().detach().cpu()
            )

    # ---------------------------------------------------
    # 6. Payload
    # ---------------------------------------------------
    payload = {
        "band_names": np.asarray(band_names),
        "field_pairs": np.asarray(field_pairs),

        # Same-frequency per-snapshot energy, [B, M, C].
        "samefreq_energy_true": samefreq_energy_true.detach().cpu().numpy(),
        "samefreq_energy_pred": samefreq_energy_pred.detach().cpu().numpy(),
        "samefreq_energy_ratio": samefreq_energy_ratio.detach().cpu().numpy(),
        "samefreq_energy_relerr": samefreq_energy_relerr.detach().cpu().numpy(),

        # Cross-frequency per-snapshot energy contribution, [B, M, M, P].
        "crossfreq_energy_true": crossfreq_energy_true.detach().cpu().numpy(),
        "crossfreq_energy_pred": crossfreq_energy_pred.detach().cpu().numpy(),
        "crossfreq_energy_ratio": crossfreq_energy_ratio.detach().cpu().numpy(),
        "crossfreq_energy_relerr": crossfreq_energy_relerr.detach().cpu().numpy(),

        # Per-snapshot error pieces, [B].
        "samefreq_relerr_per_snapshot": (
            samefreq_relerr_per_snapshot.detach().cpu().numpy()
        ),
        "crossfreq_relerr_per_snapshot": (
            crossfreq_relerr_per_snapshot.detach().cpu().numpy()
        ),
        "total_relerr_per_snapshot": (
            total_relerr_per_snapshot.detach().cpu().numpy()
        ),
    }

    return metrics, payload

# Same-Frequency Band Energy Plot
def _save_samefreq_band_energy_ratio_plot(
    band_names,
    samefreq_energy_ratio,
    save_path,
    title="Same-Frequency Band Energy Ratio",
    field_names=None,
):
    """
    Save a band energy ratio plot for same-frequency bands.

    Plots:
        E_pred[m, c] / E_GT[m, c]

    Args:
        band_names: [M]
        samefreq_energy_ratio: [M, C]
        save_path: output path
        title: plot title
        field_names: optional list of physical field names
    """
    band_names = np.asarray(band_names)
    ratio = np.asarray(samefreq_energy_ratio)

    M, C = ratio.shape

    if field_names is None:
        field_names = [f"field_{c}" for c in range(C)]

    x = np.arange(M)
    width = 0.8 / max(C, 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    for c in range(C):
        offset = (c - (C - 1) / 2.0) * width
        ax.bar(
            x + offset,
            ratio[:, c],
            width=width,
            label=str(field_names[c]),
        )

    ax.axhline(1.0, linestyle=":", linewidth=1.8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v).capitalize() for v in band_names])
    ax.set_ylabel(r"$E_{\mathrm{pred}} / E_{\mathrm{GT}}$")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

# Cross-Frequency Band Energy Plot
def _save_crossfreq_band_energy_ratio_plot(
    band_names,
    crossfreq_energy_ratio,
    save_path,
    title="Cross-Frequency Band Energy Ratio",
    field_pairs=None,
    field_names=None,
    pair_idx=None,
    unordered: bool = False,
):
    """
    Save a band energy ratio plot for cross-frequency bands.

    Plots:
        |S_pred[m, n, c1, c2]| / |S_GT[m, n, c1, c2]|

    Args:
        band_names: [M]
        crossfreq_energy_ratio: [M, M, P]
        save_path: output path
        title: plot title
        field_pairs: optional list of physical field pairs
        field_names: optional list of physical field names
        pair_idx: if None, average over all physical field pairs.
                  Otherwise, plot one field pair.
        unordered: if True, plot only i < j band pairs.
                   If False, plot all directed off-diagonal pairs.
    """
    band_names = np.asarray(band_names)
    ratio = np.asarray(crossfreq_energy_ratio)

    M = len(band_names)

    labels = []
    values = []

    for i in range(M):
        for j in range(M):
            if i == j:
                continue

            if unordered and not (i < j):
                continue

            labels.append(
                f"{str(band_names[i]).capitalize()}→{str(band_names[j]).capitalize()}"
            )

            if pair_idx is None:
                values.append(float(np.mean(ratio[i, j, :])))
            else:
                values.append(float(ratio[i, j, pair_idx]))

    values = np.asarray(values, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(labels))

    ax.bar(x, values, width=0.65)
    ax.axhline(1.0, linestyle=":", linewidth=1.8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"$|S_{\mathrm{pred}}| / |S_{\mathrm{GT}}|$")

    if pair_idx is not None and field_pairs is not None and field_names is not None:
        c1, c2 = field_pairs[pair_idx]
        pair_name = f"{field_names[c1]}-{field_names[c2]}"
        ax.set_title(f"{title}: {pair_name}")
    else:
        ax.set_title(title)

    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

# Convenience Saver
def save_band_energy_diagnostic_plots(
    payload,
    save_dir,
    field_names=None,
    save_per_pair: bool = True,
    unordered_crossfreq: bool = False,
):
    """
    Save band-energy diagnostic plots for same-frequency and cross-frequency.

    Outputs:
        samefreq_band_energy_ratio.png
        crossfreq_band_energy_ratio_all_pairs.png
        crossfreq_band_energy_ratio_<fieldpair>.png, optional
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    band_names = payload["band_names"]
    field_pairs = payload["field_pairs"]

    _save_samefreq_band_energy_ratio_plot(
        band_names=band_names,
        samefreq_energy_ratio=payload["samefreq_energy_ratio"],
        save_path=save_dir / "samefreq_band_energy_ratio.png",
        title="Same-Frequency Band Energy Ratio",
        field_names=field_names,
    )

    _save_crossfreq_band_energy_ratio_plot(
        band_names=band_names,
        crossfreq_energy_ratio=payload["crossfreq_energy_ratio"],
        save_path=save_dir / "crossfreq_band_energy_ratio_all_pairs.png",
        title="Cross-Frequency Band Energy Ratio",
        field_pairs=field_pairs,
        field_names=field_names,
        pair_idx=None,
        unordered=unordered_crossfreq,
    )

    if save_per_pair:
        for p, (c1, c2) in enumerate(field_pairs):
            if field_names is not None:
                pair_name = f"{field_names[c1]}-{field_names[c2]}"
            else:
                pair_name = f"field{c1}-field{c2}"

            _save_crossfreq_band_energy_ratio_plot(
                band_names=band_names,
                crossfreq_energy_ratio=payload["crossfreq_energy_ratio"],
                save_path=save_dir / f"crossfreq_band_energy_ratio_{_safe_name(pair_name)}.png",
                title="Cross-Frequency Band Energy Ratio",
                field_pairs=field_pairs,
                field_names=field_names,
                pair_idx=p,
                unordered=unordered_crossfreq,
            )