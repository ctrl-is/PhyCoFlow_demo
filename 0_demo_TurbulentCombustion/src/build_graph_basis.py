import argparse
from pathlib import Path

import torch

from helpers import TurbulentCombustionH5Dataset
from cross_spectral.graph import (
    build_weighted_matrix,
    get_graph_laplacian,
    spectral_decomposition,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-stride", type=int, default=1)
    args = parser.parse_args()

    dataset = TurbulentCombustionH5Dataset(
        args.data,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        time_stride=args.time_stride,
        stats_path=None,
    )

    sample = dataset[0]
    coords = sample["coords"]

    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    print(f"coords shape: {coords_np.shape}")
    print("[*] Building weighted graph...")
    W, sigma = build_weighted_matrix(coords_np, k=args.k_neighbors)

    print(f"[*] sigma = {sigma}")
    print("[*] Building graph Laplacian...")
    L = get_graph_laplacian(W)

    print("[*] Computing spectral decomposition...")
    eigenvalues, U = spectral_decomposition(L, num_modes=args.num_modes)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "eigenvalues": torch.as_tensor(eigenvalues, dtype=torch.float32),
            "U": torch.as_tensor(U, dtype=torch.float32),
            "coords": torch.as_tensor(coords_np, dtype=torch.float32),
            "k_neighbors": args.k_neighbors,
            "num_modes": args.num_modes,
            "sigma": sigma,
        },
        out_path,
    )

    print(f"[*] Saved graph basis to: {out_path}")
    print(f"[*] U shape: {U.shape}")
    print(f"[*] eigenvalues shape: {eigenvalues.shape}")


if __name__ == "__main__":
    main()