# ApRES processing

This folder contains the ApRES processing pipeline used in *Siegfried*, Venturelli*, et al. (2023), **Geology**. The ApRES site is Mercer Subglacial Lake, West Antarctica (drill site: **84.64029°S, 149.50134°W**).

## Inputs

- Raw ApRES data: `apres/raw` inside the Zenodo `data.zip` archive
- Processed ApRES image: `data/apres/ImageP2_python.mat`
- Layer analysis outputs: `data/apres/layer_analysis/`

## Environment

Follow the top-level `README.md` to create and activate the `siegvent2023` conda environment. All commands below assume you run them from the repo root.

## MATLAB preprocessing (raw to range)

Written by Paul Summers. Run the MATLAB steps below to generate the fine range estimate from raw data:

```matlab
% From within MATLAB, with the repo as your working directory
addpath('proc/apres');
cd('proc/apres');

% Run on the raw ApRES folder inside the data archive
mainCode_simple('apres/raw');

% Save fine range estimate
RangeEstFine;
```

This should produce the ApRES image file used by the Python workflow: `data/apres/ImageP2_python.mat`.

## Python preprocessing (raw to ImageP2_python.mat)

Use the standalone script to process `.DAT` files and preserve complex data:

```bash
python proc/apres/process_apres_raw.py \
	--data-folder data/apres/raw \
	--output data/apres/ImageP2_python.mat \
	--pad-factor 8 \
	--max-range 2000
```

This writes:
- `RawImage` (magnitude)
- `RawImageComplex` (complex spectrum, if not disabled)
- `RfineBarTime`, `Rcoarse`, `TimeInDays`

Use `--no-complex` to skip the complex field if needed.

## Python workflow (layers → tracking → velocity)

1) **Detect layers**

```bash
python proc/apres/layer_detection.py \
	--data data/apres/ImageP2_python.mat \
	--output data/apres/layer_analysis/detected_layers
```

Common optional arguments:

- `--max-depth 1050` (meters)
- `--min-snr 3.0`
- `--min-persistence 30`

2) **Track layers (phase tracking)**

```bash
python proc/apres/phase_tracking.py \
	--layers data/apres/layer_analysis/detected_layers \
	--data data/apres/ImageP2_python.mat \
	--output data/apres/layer_analysis/phase_tracking
```

Useful optional arguments:

- `--tracking-mode fixed`
- `--search-window-m 1.5`
- `--unwrap-mode robust_derivative`

3) **Velocity profile**

```bash
python proc/apres/velocity_profile.py \
	--phase data/apres/layer_analysis/phase_tracking \
	--output data/apres/layer_analysis/velocity_profile \
	--no-plot
```

## Visualization (Dash app)

Start the interactive viewer (3D echogram, tracked layers, 2D views):

```bash
python proc/apres/visualization_app.py \
	--output-dir data/apres/layer_analysis \
	--data data/apres/ImageP2_python.mat \
	--port 8050
```

Then open `http://127.0.0.1:8050/` in your browser.

## Echo-less region phase analysis

To test whether low-amplitude regions contain signal beyond Gaussian noise, use:

```bash
python proc/apres/phase_noise_analysis.py \
	--data data/apres/ImageP2_python.mat \
	--phase data/apres/layer_analysis/phase_tracking.mat \
	--amp-threshold-db -85 \
	--output-dir data/apres/layer_analysis
```

Options:

- `--depth-m 900` to analyze a specific depth
- `--min-depth 800 --max-depth 1200` to constrain auto-selection
- `--unwrap` to compute unwrapped phase differences
- `--gmm` to fit a Gaussian mixture (EM) and save component stats
- `--gmm-components 2` to set the number of mixture components
- `--gmm-reg-covar 1e-4` to add covariance regularization
- `--gmm-init percentile` to choose initialization strategy (`percentile` or `linear`)
- `--gmm-zero-mean` to fix one component mean at zero (white-noise assumption)

Outputs:

- Histogram PNG and CSV in `data/apres/layer_analysis/`
- Optional GMM summary JSON (`*_gmm.json`) when `--gmm` is enabled
- Summary stats and Gaussianity tests in the console

### Phase Coherence Factor (PCF)

The interactive UI now includes a **PCF / mean phase vector** panel. It stacks complex phasors
over a small vertical window and reports the mean resultant length (PCF). Set the window size
and optionally weight by amplitude to improve SNR in weak-signal regions.

### Find least-Gaussian points

After generating a set of histograms/CSVs, rank the least-Gaussian depths and plot them:

```bash
python proc/apres/phase_noise_rank.py \
	--output-dir data/apres/layer_analysis \
	--mode wrapped \
	--top 10 \
	--save-plot
```

Outputs:

- `phase_noise_wrapped_least_gaussian.csv`
- `phase_noise_wrapped_least_gaussian.png`

