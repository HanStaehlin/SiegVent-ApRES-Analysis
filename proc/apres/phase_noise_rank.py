#!/usr/bin/env python3
"""
Rank echo-less phase histograms by Gaussianity.

Scans phase-difference CSVs and computes normality statistics, then
reports the least-Gaussian depths and optionally saves a summary plot.
"""

import argparse
from pathlib import Path
import re
import numpy as np
from scipy import stats


def parse_depth_from_name(name: str) -> float | None:
    match = re.search(r"_(\d+\.\d)m\.csv$", name)
    if match:
        return float(match.group(1))
    return None


def load_values(csv_path: Path) -> np.ndarray:
    values = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if values.ndim == 0:
        values = np.array([values])
    values = values[np.isfinite(values)]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank phase-noise histograms by Gaussianity.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/apres/layer_analysis",
        help="Directory containing phase_noise_*.csv files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of least-Gaussian depths to report.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="wrapped",
        choices=["wrapped", "unwrapped"],
        help="Phase mode to rank.",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a summary plot of the least-Gaussian histograms.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="phase_noise",
        help="Filename prefix for input files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    pattern = f"{args.prefix}_{args.mode}_*.csv"
    csv_files = [p for p in sorted(output_dir.glob(pattern)) if "least_gaussian" not in p.name]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found matching {pattern} in {output_dir}")

    rows = []
    for csv_path in csv_files:
        values = load_values(csv_path)
        if values.size < 20:
            continue

        mu = float(np.mean(values))
        sigma = float(np.std(values))
        skew = float(stats.skew(values))
        kurt = float(stats.kurtosis(values, fisher=True))
        ks_p = float(stats.kstest(values, 'norm', args=(mu, sigma)).pvalue) if sigma > 0 else np.nan
        normal_p = float(stats.normaltest(values).pvalue)

        depth = parse_depth_from_name(csv_path.name)
        rows.append({
            "depth": depth,
            "file": csv_path.name,
            "n": int(values.size),
            "mean": mu,
            "std": sigma,
            "skew": skew,
            "kurtosis": kurt,
            "ks_p": ks_p,
            "normal_p": normal_p,
        })

    if not rows:
        raise ValueError("No usable CSV files with enough samples.")

    # Sort by normality p-value ascending (least Gaussian first)
    rows.sort(key=lambda r: (r["normal_p"], r["ks_p"]))

    top_n = rows[: args.top]

    summary_csv = output_dir / f"{args.prefix}_{args.mode}_least_gaussian.csv"
    header = "depth,file,n,mean,std,skew,kurtosis,ks_p,normal_p"
    with open(summary_csv, "w") as f:
        f.write(header + "\n")
        for r in top_n:
            f.write(
                f"{r['depth']:.1f},{r['file']},{r['n']},{r['mean']:.6f},{r['std']:.6f},"
                f"{r['skew']:.6f},{r['kurtosis']:.6f},{r['ks_p']:.6g},{r['normal_p']:.6g}\n"
            )

    print("Least-Gaussian depths:")
    for r in top_n:
        print(
            f"  {r['depth']:.1f} m | normal p={r['normal_p']:.3g} | KS p={r['ks_p']:.3g} | std={r['std']:.4f}"
        )
    print(f"Saved: {summary_csv}")

    if args.save_plot:
        import matplotlib.pyplot as plt

        cols = 2
        rows_count = int(np.ceil(len(top_n) / cols))
        fig, axes = plt.subplots(rows_count, cols, figsize=(10, 4 * rows_count))
        axes = np.atleast_2d(axes)

        for i, r in enumerate(top_n):
            ax = axes[i // cols, i % cols]
            values = load_values(output_dir / r["file"])
            ax.hist(values, bins=40, density=True, alpha=0.7, color="#1f77b4")

            if r["std"] > 0:
                x = np.linspace(values.min(), values.max(), 300)
                pdf = stats.norm.pdf(x, loc=r["mean"], scale=r["std"])
                ax.plot(x, pdf, "r--", linewidth=1.5)

            ax.set_title(
                f"{r['depth']:.1f} m | normal p={r['normal_p']:.2g}")
            ax.set_xlabel("Phase Δ (rad)")
            ax.set_ylabel("Density")

        # Hide unused axes
        for j in range(len(top_n), rows_count * cols):
            axes[j // cols, j % cols].axis("off")

        fig.tight_layout()
        out_plot = output_dir / f"{args.prefix}_{args.mode}_least_gaussian.png"
        fig.savefig(out_plot, dpi=150)
        print(f"Saved: {out_plot}")


if __name__ == "__main__":
    main()
