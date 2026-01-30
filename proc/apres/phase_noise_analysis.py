#!/usr/bin/env python3
"""
Echo-less region phase-change analysis.

Computes phase differences between consecutive time samples at a fixed depth,
then evaluates whether the distribution resembles Gaussian noise.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy import stats


def wrap_phase(phase: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * phase))


def fit_gmm_1d(
    values: np.ndarray,
    n_components: int = 2,
    random_state: int = 0,
    reg_covar: float = 1e-4,
    init_strategy: str = "percentile",
    zero_mean: bool = False,
) -> dict:
    values = np.asarray(values).reshape(-1, 1)
    if values.size < max(5, n_components * 2):
        raise ValueError("Not enough samples for GMM.")

    x = values.flatten()
    x_std = float(np.std(x))
    if init_strategy == "percentile":
        percentiles = np.linspace(20, 80, n_components)
        init_means = np.percentile(x, percentiles)
    else:
        init_means = np.linspace(np.min(x), np.max(x), n_components)
    init_stds = np.linspace(max(x_std * 0.4, 1e-6), max(x_std * 1.6, 2e-6), n_components)
    init_weights = np.full(n_components, 1.0 / n_components)
    init_precisions = 1.0 / (init_stds ** 2 + reg_covar)

    def norm_pdf(xv, m, s):
        s = max(s, np.sqrt(reg_covar))
        return np.exp(-0.5 * ((xv - m) / s) ** 2) / (np.sqrt(2 * np.pi) * s)

    def run_em(mu, sigma, w, fix_zero_mean: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(200):
            probs = np.stack([w[k] * norm_pdf(x, mu[k], sigma[k]) for k in range(n_components)], axis=0)
            denom = np.sum(probs, axis=0) + 1e-12
            resp = probs / denom
            w = resp.mean(axis=1)
            mu = np.array([np.sum(resp[k] * x) / np.sum(resp[k]) for k in range(n_components)])
            if fix_zero_mean:
                mu[0] = 0.0
            sigma = np.array([
                np.sqrt(np.sum(resp[k] * (x - mu[k]) ** 2) / np.sum(resp[k]))
                for k in range(n_components)
            ])
        return mu, sigma, w

    if zero_mean:
        mu = init_means.copy()
        mu[0] = 0.0
        sigma = init_stds
        w = init_weights
        mu, sigma, w = run_em(mu, sigma, w, fix_zero_mean=True)
        means = mu
        stds = sigma
        weights = w
        converged = True
        n_iter = 200
        method = "fixed_mean_em"
    else:
        try:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=random_state,
                reg_covar=reg_covar,
                init_params="random",
                n_init=1,
            )
            gmm.means_init = init_means.reshape(-1, 1)
            gmm.weights_init = init_weights
            gmm.precisions_init = init_precisions.reshape(-1, 1, 1)
            gmm.fit(values)
            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()
            converged = bool(gmm.converged_)
            n_iter = int(gmm.n_iter_)
            method = "sklearn.GaussianMixture"
        except Exception as exc:
            mu = init_means
            sigma = init_stds
            w = init_weights
            mu, sigma, w = run_em(mu, sigma, w, fix_zero_mean=False)
            means = mu
            stds = sigma
            weights = w
            converged = True
            n_iter = 200
            method = f"fallback_em ({type(exc).__name__})"

    order = np.argsort(stds)
    return {
        "means": means[order].tolist(),
        "stds": stds[order].tolist(),
        "weights": weights[order].tolist(),
        "converged": converged,
        "n_iter": n_iter,
        "method": method,
        "reg_covar": reg_covar,
        "init_strategy": init_strategy,
        "zero_mean": zero_mean,
    }


def select_depth_index(depths: np.ndarray, mean_db: np.ndarray, depth_m: float | None,
                       min_depth: float | None, max_depth: float | None,
                       amp_threshold_db: float) -> int:
    if depth_m is not None:
        return int(np.argmin(np.abs(depths - depth_m)))

    mask = np.ones_like(depths, dtype=bool)
    if min_depth is not None:
        mask &= depths >= min_depth
    if max_depth is not None:
        mask &= depths <= max_depth

    mask &= mean_db < amp_threshold_db
    if not np.any(mask):
        raise ValueError("No depths found below amplitude threshold in the selected range.")

    eligible = np.where(mask)[0]
    # Choose the lowest-amplitude depth in the echo-less region
    idx = eligible[np.argmin(mean_db[eligible])]
    return int(idx)


def select_depth_indices(depths: np.ndarray, mean_db: np.ndarray,
                         min_depth: float, max_depth: float,
                         depth_step: float, amp_threshold_db: float) -> list[int]:
    mask = (depths >= min_depth) & (depths <= max_depth)
    mask &= mean_db < amp_threshold_db
    candidates = depths[mask]
    if candidates.size == 0:
        raise ValueError("No depths found below amplitude threshold in the selected range.")

    target_depths = np.arange(min_depth, max_depth + 1e-6, depth_step)
    indices = []
    for d in target_depths:
        idx = int(np.argmin(np.abs(depths - d)))
        if mask[idx]:
            indices.append(idx)

    # Deduplicate while preserving order
    seen = set()
    unique_indices = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique_indices.append(idx)

    if not unique_indices:
        raise ValueError("No matching depths found after applying step and threshold.")
    return unique_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze phase differences in low-amplitude (echo-less) regions.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/apres/ImageP2_python.mat",
        help="Path to ImageP2_python.mat",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="data/apres/layer_analysis/phase_tracking.mat",
        help="Path to phase_tracking.mat (for lambdac)",
    )
    parser.add_argument(
        "--depth-m",
        type=float,
        default=None,
        help="Depth (m) to analyze. If omitted, auto-select from echo-less region.",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="Minimum depth (m) for auto-selection.",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum depth (m) for auto-selection.",
    )
    parser.add_argument(
        "--amp-threshold-db",
        type=float,
        default=-85.0,
        help="Mean amplitude threshold (dB) defining echo-less region.",
    )
    parser.add_argument(
        "--depth-step",
        type=float,
        default=None,
        help="If set, sweep depths from min-depth to max-depth with this step.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="When sweeping, plot every Nth depth (default 1 = plot all).",
    )
    parser.add_argument(
        "--unwrap",
        action="store_true",
        help="Use unwrapped phase before differencing.",
    )
    parser.add_argument(
        "--gmm",
        action="store_true",
        help="Fit a Gaussian mixture (EM) to the phase-difference samples.",
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=2,
        help="Number of Gaussian mixture components (default 2).",
    )
    parser.add_argument(
        "--gmm-reg-covar",
        type=float,
        default=1e-4,
        help="Regularization added to covariance (default 1e-4).",
    )
    parser.add_argument(
        "--gmm-init",
        type=str,
        default="percentile",
        choices=["percentile", "linear"],
        help="Initialization strategy for GMM means (default percentile).",
    )
    parser.add_argument(
        "--gmm-zero-mean",
        action="store_true",
        help="Fix one component mean at zero (white-noise assumption).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/apres/layer_analysis",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="phase_noise",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    mat = loadmat(data_path)
    if "RawImage" not in mat or "Rcoarse" not in mat or "TimeInDays" not in mat:
        raise KeyError("ImageP2_python.mat missing required fields.")

    if "RfineBarTime" not in mat:
        raise KeyError("ImageP2_python.mat missing RfineBarTime (needed for phase).")

    range_img = np.array(mat["RawImage"])
    rfine = np.array(mat["RfineBarTime"])
    depths = np.array(mat["Rcoarse"]).flatten()
    time_days = np.array(mat["TimeInDays"]).flatten()

    # Mean amplitude in dB for echo-less selection
    mean_db = 10 * np.log10(np.mean(range_img**2, axis=1) + 1e-30)

    # Get lambdac from phase_tracking.mat if available
    lambdac = 0.5608
    phase_path = Path(args.phase)
    if phase_path.exists():
        d = loadmat(phase_path)
        if "lambdac" in d:
            lambdac = float(np.array(d["lambdac"]).squeeze())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_mode = "unwrapped" if args.unwrap else "wrapped"

    if args.depth_step is not None:
        if args.min_depth is None or args.max_depth is None:
            raise ValueError("--min-depth and --max-depth are required when using --depth-step.")
        indices = select_depth_indices(
            depths,
            mean_db,
            args.min_depth,
            args.max_depth,
            args.depth_step,
            args.amp_threshold_db,
        )
    else:
        idx = select_depth_index(
            depths,
            mean_db,
            args.depth_m,
            args.min_depth,
            args.max_depth,
            args.amp_threshold_db,
        )
        indices = [idx]

    def analyze_depth(idx: int, do_plot: bool) -> None:
        depth_sel = depths[idx]
        raw_phase = (4 * np.pi / lambdac) * rfine[idx, :]
        if args.unwrap:
            phase_series = np.unwrap(raw_phase)
        else:
            phase_series = wrap_phase(raw_phase)

        phase_diff = np.diff(phase_series)
        if not args.unwrap:
            phase_diff = wrap_phase(phase_diff)

        valid = np.isfinite(phase_diff)
        phase_diff = phase_diff[valid]

        if phase_diff.size < 5:
            print(f"Skipping depth {depth_sel:.1f} m (insufficient samples)")
            return

        mu = float(np.mean(phase_diff))
        sigma = float(np.std(phase_diff))
        skew = float(stats.skew(phase_diff))
        kurt = float(stats.kurtosis(phase_diff, fisher=True))

        ks_p = np.nan
        if sigma > 0:
            ks_p = float(stats.kstest(phase_diff, 'norm', args=(mu, sigma)).pvalue)

        normaltest_p = np.nan
        if phase_diff.size >= 20:
            normaltest_p = float(stats.normaltest(phase_diff).pvalue)

        csv_path = output_dir / f"{args.prefix}_{phase_mode}_{depth_sel:.1f}m.csv"
        np.savetxt(csv_path, phase_diff, delimiter=",", header="phase_diff_rad", comments="")

        gmm_result = None
        if args.gmm:
            try:
                gmm_result = fit_gmm_1d(
                    phase_diff,
                    n_components=args.gmm_components,
                    reg_covar=args.gmm_reg_covar,
                    init_strategy=args.gmm_init,
                    zero_mean=args.gmm_zero_mean,
                )
                gmm_path = output_dir / f"{args.prefix}_{phase_mode}_{depth_sel:.1f}m_gmm.json"
                with gmm_path.open("w") as f:
                    json.dump({
                        "depth_m": float(depth_sel),
                        "n": int(phase_diff.size),
                        "phase_mode": phase_mode,
                        "components": gmm_result,
                    }, f, indent=2)
            except Exception as exc:
                print(f"GMM fit failed at {depth_sel:.1f} m: {exc}")

        if do_plot:
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 4.2))
                ax.hist(phase_diff, bins=40, density=True, alpha=0.7, color="#1f77b4")

                x = np.linspace(phase_diff.min(), phase_diff.max(), 300)
                if gmm_result is not None:
                    means = np.array(gmm_result["means"])
                    stds = np.array(gmm_result["stds"])
                    weights = np.array(gmm_result["weights"])
                    mix_pdf = np.zeros_like(x)
                    for i, (m, s, w) in enumerate(zip(means, stds, weights)):
                        if s <= 0:
                            continue
                        comp_pdf = w * stats.norm.pdf(x, loc=m, scale=s)
                        mix_pdf += comp_pdf
                        ax.plot(x, comp_pdf, linewidth=1.5, linestyle="--", label=f"GMM comp {i+1}")
                    ax.plot(x, mix_pdf, "k-", linewidth=2, label="GMM mixture")
                    ax.legend(frameon=False)
                elif sigma > 0:
                    pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
                    ax.plot(x, pdf, "r--", linewidth=2, label="Gaussian fit")
                    ax.legend(frameon=False)

                ax.set_title(f"Phase Δ histogram ({phase_mode}) at {depth_sel:.1f} m")
                ax.set_xlabel("Phase Δ (rad)")
                ax.set_ylabel("Density")
                fig.tight_layout()

                fig_path = output_dir / f"{args.prefix}_{phase_mode}_{depth_sel:.1f}m.png"
                fig.savefig(fig_path, dpi=150)
            except Exception as exc:
                print(f"Plotting skipped for {depth_sel:.1f} m: {exc}")

        print(
            f"Depth {depth_sel:.1f} m | n={phase_diff.size} | mean={mu:.4f} | std={sigma:.4f} | "
            f"skew={skew:.4f} | kurt={kurt:.4f} | KS p={ks_p:.4g} | normal p={normaltest_p:.4g}"
        )

    for i, idx in enumerate(indices):
        do_plot = (i % max(args.plot_every, 1) == 0)
        analyze_depth(idx, do_plot)


if __name__ == "__main__":
    main()
