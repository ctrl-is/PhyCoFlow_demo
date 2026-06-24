import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from helpers import (
    TurbulentCombustionH5Dataset,
    reconstruct_snapshot,
)
from model_finetune import (
    load_pretrained_ffm,
    load_source_config,
)

from graph import make_graph_frequency_bands

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

def main():
    # eval_coherence.py is assumed to be inside src/
    project_root = Path(__file__).resolve().parents[2]

    # ---------------------------------------------------
    # 1. Paths
    # ---------------------------------------------------
    run_dir = (
        project_root
        / "Save_TrainedModel"
        / "ffm_tc_pointcloud_DemoN30_20260610_173403"
    )

    # This must be the saved tensor file, not the Python script.
    graph_basis_path = (
        project_root
        / "Save_Graph"
        / "graph_basis_k16_modes384.pt"
    )

    output_dir = run_dir / "Evaluation" / "BandEnergy"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = "best"
    split = "test"

    # Cross-frequency covariance requires multiple snapshots.
    num_snapshots = 16

    # Set True to calculate energy after returning to physical units.
    use_denorm = False

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    torch.manual_seed(42)
    np.random.seed(42)

    # ---------------------------------------------------
    # 2. Load the saved run configuration
    # ---------------------------------------------------
    source_cfg = load_source_config(run_dir)

    data_path = Path(
        source_cfg.get(
            "data",
            "Dataset/Merged_CH4COTU1P.h5",
        )
    )

    if not data_path.is_absolute():
        data_path = project_root / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    # ---------------------------------------------------
    # 3. Load dataset using this model's saved statistics
    # ---------------------------------------------------
    dataset = TurbulentCombustionH5Dataset(
        h5_path=str(data_path),
        split=split,
        train_ratio=float(source_cfg.get("train_ratio", 0.9)),
        seed=int(source_cfg.get("seed", 42)),
        time_stride=int(source_cfg.get("time_stride", 1)),
        stats_path=str(run_dir / "dataset_stats.pt"),
    )

    field_names = list(dataset.field_names)


    # ---------------------------------------------------
    # Conditioning configuration
    # ---------------------------------------------------
    cond_fields = (
        source_cfg.get("vis_cond_fields")
        or source_cfg.get("cond_fields")
    )

    if cond_fields is None:
        cond_fields = [int(source_cfg.get("cond_field") or 2)]
    elif isinstance(cond_fields, (int, np.integer)):
        cond_fields = [int(cond_fields)]
    else:
        cond_fields = [int(v) for v in cond_fields]


    n_obs_list = (
        source_cfg.get("vis_n_obs_list")
        or source_cfg.get("n_obs_max_list")
    )

    if n_obs_list is None:
        n_obs_list = [int(source_cfg.get("n_obs_max") or 256)]
    elif isinstance(n_obs_list, (int, np.integer)):
        n_obs_list = [int(n_obs_list)]
    else:
        n_obs_list = [int(v) for v in n_obs_list]


    # Broadcast one observation count across all conditioned fields.
    if len(n_obs_list) == 1 and len(cond_fields) > 1:
        n_obs_list = n_obs_list * len(cond_fields)

    if len(n_obs_list) != len(cond_fields):
        raise ValueError(
            "n_obs_list must have length 1 or match cond_fields. "
            f"Got cond_fields={cond_fields}, n_obs_list={n_obs_list}."
        )


    n_steps_generation = int(
        source_cfg.get("n_steps_generation") or 100
    )

    ode_solver = str(
        source_cfg.get("ode_solver") or "euler"
    ).lower()

    if ode_solver not in {"euler", "heun"}:
        raise ValueError(
            f"Unsupported ode_solver={ode_solver!r}; "
            "expected 'euler' or 'heun'."
        )


    print(f"Conditioned fields: {cond_fields}")
    print(f"Observation counts: {n_obs_list}")
    print(f"Generation steps: {n_steps_generation}")
    print(f"ODE solver: {ode_solver}")

    # ---------------------------------------------------
    # 4. Load best.pt
    # ---------------------------------------------------
    model, source_cfg_loaded, checkpoint = load_pretrained_ffm(
    source_run_dir=run_dir,
    checkpoint=checkpoint_name,
    dataset=dataset,
    device=device,
    )

    model.eval()

    # ---------------------------------------------------
    # Force PyTorch KNN instead of KeOps
    # ---------------------------------------------------
    outer_model = model.module if hasattr(model, "module") else model
    backbone = outer_model.model

    if not hasattr(backbone, "neighbor_backend"):
        raise AttributeError(
            f"Backbone {type(backbone).__name__} does not expose "
            "neighbor_backend."
        )

    backbone.neighbor_backend = "torch"
    backbone.gather_query_chunk_size = 4096

    print(f"Neighbor backend: {backbone.neighbor_backend}")
    print(
        "Gather query chunk size: "
        f"{backbone.gather_query_chunk_size}"
    )

    print(f"Loaded checkpoint: {run_dir / 'best.pt'}")
    print(f"Evaluation split: {split}")
    print(f"Conditioned fields: {cond_fields}")
    print(f"Observation counts: {n_obs_list}")

    # load_pretrained_ffm rebuilds the model from the saved run
    # configuration and loads checkpoint['model'] when present.
    # ---------------------------------------------------
    # 5. Load graph basis
    # ---------------------------------------------------
    if not graph_basis_path.exists():
        raise FileNotFoundError(
            f"Graph basis not found: {graph_basis_path}\n"
            "The graph basis must be a saved .pt file, not a .py script."
        )

    graph_obj = torch.load(
        graph_basis_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(graph_obj, dict):
        raise TypeError(
            "Expected the graph-basis .pt file to contain a dictionary."
        )

    print(f"Graph basis keys: {list(graph_obj.keys())}")

    U = graph_obj.get("U")
    if U is None:
        U = graph_obj.get("eigenvectors")
    if U is None:
        U = graph_obj.get("evecs")

    eigenvalues = graph_obj.get("eigenvalues")
    if eigenvalues is None:
        eigenvalues = graph_obj.get("evals")

    if U is None:
        raise KeyError(
            "Graph basis does not contain U, eigenvectors, or evecs."
        )

    U = torch.as_tensor(
        U,
        dtype=torch.float32,
        device=device,
    )

    if U.ndim != 2:
        raise ValueError(
            f"Expected U to have shape [N,K], got {tuple(U.shape)}"
        )

    if U.shape[0] != dataset.num_points:
        raise ValueError(
            "Graph basis spatial size does not match the dataset: "
            f"U has N={U.shape[0]}, while the dataset has "
            f"N={dataset.num_points}."
        )

    # Use saved bands when the graph-basis file contains them.
    saved_bands = graph_obj.get("bands")

    if saved_bands is not None:
        bands = {
            str(name): torch.as_tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            for name, indices in saved_bands.items()
        }
    else:
        if eigenvalues is None:
            raise KeyError(
                "Graph basis has neither saved bands nor eigenvalues."
            )

        eigenvalues = torch.as_tensor(
            eigenvalues,
            dtype=torch.float32,
            device=device,
        )

        bands = make_graph_frequency_bands(
            eigenvalues=eigenvalues,
            exclude_zero=True,
            split="thirds",
        )

        bands = {
            str(name): torch.as_tensor(
                indices,
                dtype=torch.long,
                device=device,
            )
            for name, indices in bands.items()
        }

    print(f"Graph basis shape: {tuple(U.shape)}")
    print(
        "Band sizes:",
        {
            name: int(indices.numel())
            for name, indices in bands.items()
        },
    )

    # ---------------------------------------------------
    # 6. Reconstruct several test snapshots
    # ---------------------------------------------------
    fields_true_list = []
    fields_pred_list = []

    snapshot_count = min(num_snapshots, len(dataset))

    if snapshot_count < 2:
        raise ValueError(
            "Cross-frequency band covariance requires at least "
            "two snapshots."
        )

    with torch.no_grad():
        for snapshot_index in range(snapshot_count):
            print(
                f"Reconstructing snapshot "
                f"{snapshot_index + 1}/{snapshot_count}"
            )

            result = reconstruct_snapshot(
                model=model,
                dataset=dataset,
                device=device,
                snapshot_index=snapshot_index,
                cond_fields=cond_fields,
                n_obs_list=n_obs_list,
                n_steps=n_steps_generation,
                ode_solver=ode_solver,
            )

            # Each one is [1,N,C].
            fields_true_list.append(
                result["truth"].detach().cpu()
            )
            fields_pred_list.append(
                result["recon"].detach().cpu()
            )

    # Final shape: [B,N,C].
    fields_true = torch.cat(
        fields_true_list,
        dim=0,
    ).to(device)

    fields_pred = torch.cat(
        fields_pred_list,
        dim=0,
    ).to(device)

    print(f"fields_true: {tuple(fields_true.shape)}")
    print(f"fields_pred: {tuple(fields_pred.shape)}")

    # reconstruct_snapshot returns normalized full-field tensors.
    # ---------------------------------------------------
    # 7. Optional denormalization
    # ---------------------------------------------------
    if use_denorm:
        mean = dataset.mean.to(
            device=device,
            dtype=fields_true.dtype,
        ).view(1, 1, -1)

        std = dataset.std.to(
            device=device,
            dtype=fields_true.dtype,
        ).view(1, 1, -1)

        fields_true = fields_true * std + mean
        fields_pred = fields_pred * std + mean

    # ---------------------------------------------------
    # 8. Compute same- and cross-frequency band energy
    # ---------------------------------------------------
    with torch.no_grad():
        metrics, payload = _cross_spectral_coherence_band_metrics(
            fields_true=fields_true,
            fields_pred=fields_pred,
            U=U,
            bands=bands,
            field_pairs=None,  # all c1 < c2 physical-field pairs
            eps=1e-12,
        )

    # ---------------------------------------------------
    # 9. Save both categories of plots
    # ---------------------------------------------------
    save_band_energy_diagnostic_plots(
        payload=payload,
        save_dir=output_dir,
        field_names=field_names,
        save_per_pair=True,
        unordered_crossfreq=False,
    )

    # ---------------------------------------------------
    # 10. Save numerical results
    # ---------------------------------------------------
    with open(
        output_dir / "band_energy_metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2)

    np.savez_compressed(
        output_dir / "band_energy_payload.npz",
        **payload,
    )

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "graph_basis_path": str(graph_basis_path),
        "split": split,
        "num_snapshots": snapshot_count,
        "snapshot_indices": list(range(snapshot_count)),
        "cond_fields": (
            list(cond_fields)
            if isinstance(cond_fields, (list, tuple))
            else [int(cond_fields)]
        ),
        "n_obs_list": (
            list(n_obs_list)
            if isinstance(n_obs_list, (list, tuple))
            else [int(n_obs_list)]
        ),
        "n_steps_generation": n_steps_generation,
        "ode_solver": ode_solver,
        "use_denorm": use_denorm,
        "field_names": field_names,
    }

    with open(
        output_dir / "band_energy_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2)

    print("\nBand-energy plots complete.")
    print(f"Saved to: {output_dir}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6e}")


if __name__ == "__main__":
    main()