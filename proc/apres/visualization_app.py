#!/usr/bin/env python3
"""
ApRES Layer Analysis - Interactive Browser Visualization

Creates a Dash-based web application for exploring internal ice layer
velocity analysis results.

Features:
- Interactive echogram with layer overlay
- Clickable layer selection
- Phase/range time series for selected layers
- Velocity profile with depth
- Export capabilities

Usage:
    python visualization_app.py --output-dir /path/to/results
    
Then open http://localhost:8050 in your browser.

Author: SiegVent2023 project
"""

import numpy as np
from scipy.io import loadmat
from scipy import stats
from scipy.ndimage import median_filter, uniform_filter
from pathlib import Path
import json
import base64
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check if Dash is available
try:
    from dash import Dash, dcc, html, callback, Input, Output, State
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Dash not installed. Install with: pip install dash")

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from phase_noise_analysis import fit_gmm_1d, wrap_phase
except Exception:
    fit_gmm_1d = None
    wrap_phase = None


# Cache for denoised echogram
_denoised_cache = {}


def apply_fast_denoising(range_img: np.ndarray) -> np.ndarray:
    """
    Apply fast denoising to the echogram using median + smoothing filter.
    
    This is much faster than SVD and works well for horizontal layer enhancement.
    Uses a horizontally-oriented filter to preserve layer structure.
    
    Parameters
    ----------
    range_img : np.ndarray
        Original echogram (depth x time)
    
    Returns
    -------
    np.ndarray
        Denoised echogram (same shape as input)
    """
    # Convert to dB for filtering (log-domain filtering works better)
    img_db = 10 * np.log10(range_img**2 + 1e-30)
    
    # Apply median filter with horizontal emphasis (preserves layers)
    # Size is (depth, time) - larger in time dimension to smooth along layers
    filtered = median_filter(img_db, size=(3, 7))
    
    # Additional smoothing along time axis only
    filtered = uniform_filter(filtered, size=(1, 5))
    
    # Convert back from dB
    denoised = np.sqrt(10 ** (filtered / 10))
    
    return denoised


def load_denoised_echogram(output_dir: str) -> np.ndarray | None:
    """Load pre-computed denoised echogram if available."""
    denoised_path = Path(output_dir) / 'echogram_denoised.mat'
    if denoised_path.exists():
        try:
            data = loadmat(str(denoised_path))
            return np.array(data['range_img_denoised'])
        except Exception as e:
            print(f"Could not load denoised echogram: {e}")
    return None


def get_denoised_echogram(apres_data: dict, output_dir: str = None) -> np.ndarray:
    """Get denoised echogram from pre-computed file or compute on the fly."""
    global _denoised_cache
    
    cache_key = id(apres_data['range_img'])
    if cache_key in _denoised_cache:
        return _denoised_cache[cache_key]
    
    # Try to load pre-computed
    if output_dir:
        denoised = load_denoised_echogram(output_dir)
        if denoised is not None:
            print("Loaded pre-computed denoised echogram")
            _denoised_cache[cache_key] = denoised
            return denoised
    
    # Fall back to computing
    print("Computing denoised echogram (run precompute_denoised.py for faster startup)...")
    _denoised_cache[cache_key] = apply_fast_denoising(apres_data['range_img'])
    print("Denoising complete.")
    
    return _denoised_cache[cache_key]


def load_all_results(output_dir: str) -> dict:
    """Load all analysis results."""
    output_path = Path(output_dir)
    
    results = {}
    
    # Load layer detection
    layer_path = output_path / 'detected_layers.mat'
    if layer_path.exists():
        results['layers'] = loadmat(str(layer_path))
    
    # Load phase tracking
    phase_path = output_path / 'phase_tracking.mat'
    if phase_path.exists():
        results['phase'] = loadmat(str(phase_path))
    
    # Load velocity profile
    velocity_path = output_path / 'velocity_profile.mat'
    if velocity_path.exists():
        results['velocity'] = loadmat(str(velocity_path))
    
    # Load velocity JSON for summary
    velocity_json = output_path / 'velocity_profile.json'
    if velocity_json.exists():
        with open(velocity_json, 'r') as f:
            results['velocity_summary'] = json.load(f)
    
    return results


def load_image_base64(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    data = image_path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode('utf-8')}"


def load_gmm_summary(output_dir: Path, image_name: str) -> dict | None:
    gmm_name = image_name.replace(".png", "_gmm.json")
    gmm_path = output_dir / gmm_name
    if not gmm_path.exists():
        return None
    try:
        return json.loads(gmm_path.read_text())
    except Exception:
        return None


def load_gmm_sweep_rows(output_dir: Path, prefix: str = "phase_noise") -> list[dict]:
    rows = []
    for path in sorted(output_dir.glob(f"{prefix}_*_gmm.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        comps = data.get("components", {})
        means = comps.get("means", [])
        stds = comps.get("stds", [])
        weights = comps.get("weights", [])
        if not means or not stds or not weights:
            continue
        order = np.argsort(stds)
        signal_idx = int(order[0])
        rows.append({
            "depth": float(data.get("depth_m", np.nan)),
            "signal_mean": float(means[signal_idx]),
            "signal_std": float(stds[signal_idx]),
            "signal_weight": float(weights[signal_idx]),
        })
    rows = [r for r in rows if np.isfinite(r["depth"])]
    rows.sort(key=lambda r: r["depth"])
    return rows


def create_gmm_sweep_figure(rows: list[dict]) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text='Run phase_noise_analysis.py with --gmm to populate sweep summary.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=360, template='plotly_white')
        return fig

    depths = [r["depth"] for r in rows]
    signal_std = [r["signal_std"] for r in rows]
    signal_mean = [r["signal_mean"] for r in rows]
    signal_weight = [r["signal_weight"] for r in rows]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=signal_std,
            mode='lines+markers',
            name='Signal std',
            line=dict(color='#2563eb', width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=signal_weight,
            mode='lines+markers',
            name='Signal weight',
            line=dict(color='#16a34a', width=2),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=signal_mean,
            mode='markers',
            name='Signal mean',
            marker=dict(color='#ef4444', size=6),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title='GMM Sweep Summary (narrowest component = signal)',
        xaxis_title='Depth (m)',
        yaxis_title='Signal std / mean (rad)',
        height=380,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    fig.update_yaxes(title_text='Signal weight', secondary_y=True)
    return fig


def list_histogram_images(output_dir: Path) -> list[dict]:
    images = sorted(output_dir.glob("phase_noise_*.png"))
    options = []
    for img in images:
        if "least_gaussian" in img.name:
            continue
        if not img.name.endswith("m.png"):
            continue
        label = img.name.replace("phase_noise_", "").replace(".png", "")
        options.append({"label": label, "value": img.name})
    return options


def load_least_gaussian_rows(output_dir: Path, mode: str = "wrapped", prefix: str = "phase_noise") -> list[dict]:
    summary_path = output_dir / f"{prefix}_{mode}_least_gaussian.csv"
    if not summary_path.exists():
        return []
    import csv

    rows = []
    with open(summary_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "depth": float(row["depth"]),
                "file": str(row.get("file", "")),
                "n": int(row["n"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "skew": float(row["skew"]),
                "kurtosis": float(row["kurtosis"]),
                "ks_p": float(row["ks_p"]),
                "normal_p": float(row["normal_p"]),
            })
    return rows


def create_least_gaussian_figure(rows: list[dict]) -> go.Figure:
    if not rows:
        fig = go.Figure()
        fig.add_annotation(
            text='Run phase_noise_rank.py to generate least-Gaussian summary.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=320, template='plotly_white')
        return fig

    depths = [r["depth"] for r in rows]
    normal_p = [r["normal_p"] for r in rows]
    ks_p = [r["ks_p"] for r in rows]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=depths,
            y=normal_p,
            name='Normality p-value',
            marker=dict(color=normal_p, colorscale='Viridis', showscale=True),
            hovertemplate='Depth: %{x:.1f} m<br>Normal p: %{y:.3g}<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=depths,
            y=ks_p,
            mode='markers+lines',
            name='KS p-value',
            marker=dict(color='#ef4444', size=8),
            hovertemplate='Depth: %{x:.1f} m<br>KS p: %{y:.3g}<extra></extra>'
        )
    )
    fig.update_layout(
        title='Least-Gaussian Depths (lower p = less Gaussian)',
        xaxis_title='Depth (m)',
        yaxis_title='p-value',
        height=360,
        template='plotly_white',
    )
    return fig


def build_least_gaussian_table(rows: list[dict]) -> html.Table:
    if not rows:
        return html.Table([html.Tr([html.Td('No summary found.')])])

    header = html.Tr([
        html.Th('Depth (m)'), html.Th('Normal p'), html.Th('KS p'), html.Th('Std'), html.Th('Skew'), html.Th('Kurt')
    ])
    body = [
        html.Tr([
            html.Td(f"{r['depth']:.1f}"),
            html.Td(f"{r['normal_p']:.3g}"),
            html.Td(f"{r['ks_p']:.3g}"),
            html.Td(f"{r['std']:.4f}"),
            html.Td(f"{r['skew']:.4f}"),
            html.Td(f"{r['kurtosis']:.4f}"),
        ])
        for r in rows
    ]
    return html.Table([header] + body, style={'width': '100%', 'borderCollapse': 'collapse'})


def create_3d_echogram_figure(apres_data: dict, layer_depths: np.ndarray, 
                               highlighted_layers: list = None,
                               depth_range: tuple = (50, 600),
                               time_subsample: int = 10,
                               depth_subsample: int = 5,
                               denoise: bool = False,
                               output_dir: str = None,
                               range_timeseries: np.ndarray = None,
                               phase_time: np.ndarray = None) -> go.Figure:
    """
    Create an interactive 3D surface plot of the echogram.
    
    Parameters
    ----------
    apres_data : dict
        Dictionary with 'range_img', 'Rcoarse', 'time_days'
    layer_depths : np.ndarray
        Depths of detected layers
    highlighted_layers : list, optional
        Indices of layers to highlight
    depth_range : tuple
        (min_depth, max_depth) to display
    time_subsample : int
        Subsample factor for time dimension (for performance)
    depth_subsample : int
        Subsample factor for depth dimension (for performance)
    denoise : bool
        If True, apply SVD denoising to the echogram
    range_timeseries : np.ndarray, optional
        Layer tracking data [n_layers, n_times] - range change in meters
    phase_time : np.ndarray, optional
        Time array for range_timeseries
    
    Returns
    -------
    go.Figure
        Plotly figure with 3D surface
    """
    # Use denoised data if requested
    if denoise:
        range_img = get_denoised_echogram(apres_data, output_dir)
    else:
        range_img = apres_data['range_img']
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']
    
    # Mask to selected depth range
    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    depths_sel = Rcoarse[depth_mask][::depth_subsample]
    echogram_sel = range_img[depth_mask, :][::depth_subsample, ::time_subsample]
    time_sel = time_days[::time_subsample]
    
    # Convert to dB
    echogram_db = 10 * np.log10(echogram_sel**2 + 1e-30)
    
    # Clip for better visualization
    echogram_db = np.clip(echogram_db, -20, 50)
    
    # Create meshgrid for surface
    T, D = np.meshgrid(time_sel, depths_sel)
    
    # Create figure
    fig = go.Figure()
    
    # Add title indicating denoised state
    title_suffix = " (Denoised)" if denoise else ""
    
    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=T,
            y=D,
            z=echogram_db,
            colorscale='Viridis',
            cmin=-20,
            cmax=50,
            colorbar=dict(
                title=dict(text='Amplitude (dB)', side='right'),
                x=1.02,
                len=0.7,
            ),
            opacity=0.9,
            name='Echogram',
            showlegend=False,
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Amp: %{z:.1f} dB<extra></extra>',
        )
    )
    
    # Add layer highlighting as 3D lines on the surface
    # If range_timeseries is provided, show tracked layer positions over time
    if layer_depths is not None:
        # All layers - subtle lines
        for i, depth in enumerate(layer_depths):
            if depth < depth_range[0] or depth > depth_range[1]:
                continue
            
            is_highlighted = highlighted_layers is not None and i in highlighted_layers
            
            # Use tracked depth positions if available
            if range_timeseries is not None and phase_time is not None and i < range_timeseries.shape[0]:
                # range_timeseries contains range CHANGE in meters
                # Actual depth = initial depth + range change
                tracked_depths = depth + range_timeseries[i, :]
                
                # Subsample to match the surface time sampling
                time_for_layer = phase_time[::time_subsample]
                depth_for_layer = tracked_depths[::time_subsample]
                
                # Make sure lengths match
                n_points = min(len(time_for_layer), len(time_sel))
                time_for_layer = time_for_layer[:n_points]
                depth_for_layer = depth_for_layer[:n_points]
                
                # Get z values by finding nearest depth indices for each time point
                z_values = np.zeros(n_points)
                for j in range(n_points):
                    depth_idx = np.argmin(np.abs(depths_sel - depth_for_layer[j]))
                    z_values[j] = echogram_db[depth_idx, j] if j < echogram_db.shape[1] else echogram_db[depth_idx, -1]
            else:
                # Fallback to constant depth (original behavior)
                time_for_layer = time_sel
                depth_idx = np.argmin(np.abs(depths_sel - depth))
                depth_for_layer = np.full_like(time_sel, depths_sel[depth_idx])
                z_values = echogram_db[depth_idx, :]
            
            if is_highlighted:
                # Highlighted layers - bright and thick
                fig.add_trace(
                    go.Scatter3d(
                        x=time_for_layer,
                        y=depth_for_layer,
                        z=z_values + 3,  # Slightly above surface
                        mode='lines',
                        line=dict(color='#ff4444', width=8),
                        name=f'Layer {depth:.0f}m (tracked)',
                        hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.2f} m<extra></extra>',
                    )
                )
            else:
                # Regular layers - subtle
                fig.add_trace(
                    go.Scatter3d(
                        x=time_for_layer,
                        y=depth_for_layer,
                        z=z_values + 1,  # Slightly above surface
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.4)', width=2),
                        name=f'Layer {depth:.0f}m',
                        visible='legendonly',  # Hidden by default
                        hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.2f} m<extra></extra>',
                    )
                )
    
    # Update layout for 3D
    fig.update_layout(
        title=dict(
            text=f'3D Echogram Visualization{title_suffix}',
            font=dict(size=18),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title='Time (days)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
            ),
            yaxis=dict(
                title='Depth (m)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
                autorange='reversed',  # Depth increases downward
            ),
            zaxis=dict(
                title='Amplitude (dB)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),  # Nice viewing angle
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.5),
        ),
        height=700,
        margin=dict(l=0, r=0, t=50, b=0),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
    )
    
    return fig


def create_3d_phase_echogram_figure(apres_data: dict, layer_depths: np.ndarray,
                                    lambdac: float,
                                    highlighted_layers: list = None,
                                    depth_range: tuple = (50, 600),
                                    time_subsample: int = 10,
                                    depth_subsample: int = 5,
                                    phase_mode: str = 'wrapped') -> go.Figure:
    """
    Create an interactive 3D surface plot of the phase echogram.

    Phase is computed directly from np.angle(RawImageComplex).
    """
    raw_complex = apres_data.get('raw_complex')
    if raw_complex is None:
        fig = go.Figure()
        fig.add_annotation(
            text='Phase echogram not available (RawImageComplex missing).',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            height=650,
            template='plotly_white',
            font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    # Mask to selected depth range
    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    depths_sel = Rcoarse[depth_mask][::depth_subsample]
    time_sel = time_days[::time_subsample]

    # Get phase directly from complex data
    complex_sel = raw_complex[depth_mask, :][::depth_subsample, ::time_subsample]
    raw_phase = np.angle(complex_sel)
    
    if phase_mode == 'unwrapped':
        phase_values = np.unwrap(raw_phase, axis=1)
        phase_title = 'Phase (rad, unwrapped)'
        plot_title = '3D Phase Echogram (unwrapped)'
    else:
        phase_values = raw_phase  # Already wrapped from np.angle
        phase_title = 'Phase (rad, wrapped)'
        plot_title = '3D Phase Echogram (wrapped)'

    # Create meshgrid for surface
    T, D = np.meshgrid(time_sel, depths_sel)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=T,
            y=D,
            z=phase_values,
            colorscale='Twilight',
            cmin=-np.pi if phase_mode != 'unwrapped' else None,
            cmax=np.pi if phase_mode != 'unwrapped' else None,
            colorbar=dict(
                title=dict(text=phase_title, side='right'),
                x=1.02,
                len=0.7,
            ),
            opacity=0.9,
            name='Phase Echogram',
            showlegend=False,
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Phase: %{z:.2f} rad<extra></extra>',
        )
    )

    # Add layer highlighting as 3D lines on the surface
    if layer_depths is not None:
        for i, depth in enumerate(layer_depths):
            if depth < depth_range[0] or depth > depth_range[1]:
                continue

            depth_idx = np.argmin(np.abs(depths_sel - depth))
            actual_depth = depths_sel[depth_idx]
            z_values = phase_values[depth_idx, :]

            is_highlighted = highlighted_layers is not None and i in highlighted_layers

            if is_highlighted:
                fig.add_trace(
                    go.Scatter3d(
                        x=time_sel,
                        y=np.full_like(time_sel, actual_depth),
                        z=z_values + 0.1,
                        mode='lines',
                        line=dict(color='#ff4444', width=7),
                        name=f'Layer {depth:.0f}m (highlighted)',
                        hovertemplate=f'Layer at {depth:.0f}m<br>Time: %{{x:.1f}} days<extra></extra>',
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=time_sel,
                        y=np.full_like(time_sel, actual_depth),
                        z=z_values + 0.05,
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.4)', width=2),
                        name=f'Layer {depth:.0f}m',
                        visible='legendonly',
                        hovertemplate=f'Layer at {depth:.0f}m<br>Time: %{{x:.1f}} days<extra></extra>',
                    )
                )

    fig.update_layout(
        title=dict(
            text=plot_title,
            font=dict(size=18),
            x=0.5,
        ),
        scene=dict(
            xaxis=dict(
                title='Time (days)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
            ),
            yaxis=dict(
                title='Depth (m)',
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
                autorange='reversed',
            ),
            zaxis=dict(
                title=phase_title,
                backgroundcolor='rgb(230, 230, 230)',
                gridcolor='white',
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.5, y=1, z=0.4),
        ),
        height=650,
        margin=dict(l=0, r=0, t=50, b=0),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
    )

    return fig


def create_2d_echogram_figure(apres_data: dict, depth_range: tuple = (50, 1200)) -> go.Figure:
    """Create a 2D echogram heatmap view."""
    range_img = apres_data['range_img']
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    echogram_db = 10 * np.log10(range_img[depth_mask, :]**2 + 1e-30)
    echogram_db = np.clip(echogram_db, -20, 50)

    step = max(1, len(time_days) // 400)

    fig = go.Figure(
        data=go.Heatmap(
            x=time_days[::step],
            y=Rcoarse[depth_mask],
            z=echogram_db[:, ::step],
            colorscale='Viridis',
            zmin=-20,
            zmax=50,
            colorbar=dict(title='dB'),
        )
    )

    fig.update_layout(
        title='Echogram (2D)',
        height=500,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
        margin=dict(l=40, r=10, t=40, b=40),
    )
    fig.update_yaxes(title='Depth (m)', autorange='reversed')
    fig.update_xaxes(title='Time (days)')
    return fig


def create_summary_figure(results: dict, apres_data: dict) -> go.Figure:
    """Create a comprehensive summary figure."""
    
    # Extract data
    depths = results['velocity']['depths'].flatten()
    velocities = results['velocity']['velocities'].flatten()
    velocities_smooth = results['velocity']['velocities_smooth'].flatten()
    r_squared = results['velocity']['r_squared'].flatten()
    reliable = results['velocity']['reliable'].flatten().astype(bool)
    
    range_img = apres_data['range_img']
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Echogram with Detected Layers',
            'Velocity Profile',
            'Phase Time Series (Selected Layers)',
            'Velocity Quality Assessment'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}],
        ],
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )
    
    # Subsample echogram
    depth_mask = (Rcoarse >= 50) & (Rcoarse <= 1000)
    echogram_db = 10 * np.log10(range_img[depth_mask, :]**2 + 1e-30)
    
    step = max(1, len(time_days) // 300)
    
    # 1. Echogram
    fig.add_trace(
        go.Heatmap(
            x=time_days[::step],
            y=Rcoarse[depth_mask],
            z=echogram_db[:, ::step],
            colorscale='Viridis',
            colorbar=dict(title='dB', x=0.45, len=0.4, y=0.8),
        ),
        row=1, col=1
    )
    
    # Add layer markers
    for depth in depths:
        fig.add_hline(y=depth, line=dict(color='red', width=0.5, dash='dot'), row=1, col=1)
    
    # 2. Velocity profile
    # Unreliable
    fig.add_trace(
        go.Scatter(
            x=velocities[~reliable],
            y=depths[~reliable],
            mode='markers',
            marker=dict(color='lightgray', size=5),
            name='Unreliable',
            showlegend=True,
        ),
        row=1, col=2
    )
    
    # Reliable
    fig.add_trace(
        go.Scatter(
            x=velocities[reliable],
            y=depths[reliable],
            mode='markers',
            marker=dict(
                color=r_squared[reliable],
                colorscale='Viridis',
                size=8,
                cmin=0.3, cmax=1.0,
                colorbar=dict(title='R²', x=1.02, len=0.4, y=0.8),
            ),
            name='Reliable',
            showlegend=True,
        ),
        row=1, col=2
    )
    
    # Smoothed
    fig.add_trace(
        go.Scatter(
            x=velocities_smooth,
            y=depths,
            mode='lines',
            line=dict(color='red', width=3),
            name='Smoothed',
        ),
        row=1, col=2
    )
    
    fig.add_vline(x=0, line=dict(color='black', dash='dash'), row=1, col=2)
    
    # 3. Phase time series for a few layers
    phase_data = results['phase']
    range_ts = phase_data['range_timeseries']
    phase_time = phase_data['time_days'].flatten()
    
    # Select 4 layers evenly spaced
    n_layers = len(depths)
    show_indices = np.linspace(0, n_layers-1, min(4, n_layers), dtype=int)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, idx in enumerate(show_indices):
        range_cm = range_ts[idx, :] * 100
        fig.add_trace(
            go.Scatter(
                x=phase_time,
                y=range_cm,
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=1),
                name=f'{depths[idx]:.0f}m',
            ),
            row=2, col=1
        )
    
    # 4. Quality scatter
    fig.add_trace(
        go.Scatter(
            x=r_squared,
            y=velocities,
            mode='markers',
            marker=dict(
                color=depths,
                colorscale='Viridis',
                size=6,
                colorbar=dict(title='Depth (m)', x=1.02, len=0.4, y=0.2),
            ),
            showlegend=False,
        ),
        row=2, col=2
    )
    
    fig.add_vline(x=0.3, line=dict(color='red', dash='dot'), row=2, col=2)
    
    # Update axes
    fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title='Time (days)', row=1, col=1)
    
    fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=2)
    fig.update_xaxes(title='Velocity (m/yr)', row=1, col=2)
    
    fig.update_yaxes(title='ΔRange (cm)', row=2, col=1)
    fig.update_xaxes(title='Time (days)', row=2, col=1)
    
    fig.update_yaxes(title='Velocity (m/yr)', row=2, col=2)
    fig.update_xaxes(title='R²', range=[0, 1], row=2, col=2)
    
    # Layout
    n_reliable = int(np.sum(reliable))
    fig.update_layout(
        title=dict(
            text=f'ApRES Internal Ice Layer Velocity Analysis<br><sup>{n_reliable}/{n_layers} reliable layers</sup>',
            font=dict(size=18),
        ),
        height=900,
        width=1400,
        showlegend=True,
        legend=dict(x=0.52, y=1.0, orientation='h'),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
    )
    
    return fig


def create_time_averaged_echogram_figure(apres_data: dict) -> go.Figure:
    range_img = apres_data['range_img']
    depths = apres_data['Rcoarse']

    mean_db = 10 * np.log10(np.mean(range_img**2, axis=1) + 1e-30)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mean_db,
            y=depths,
            mode='lines',
            line=dict(color='#2563eb', width=2),
            name='Mean amplitude (dB)'
        )
    )
    fig.update_layout(
        title='Time-averaged Echogram (Mean amplitude vs depth)',
        xaxis_title='Mean amplitude (dB)',
        yaxis_title='Depth (m)',
        height=420,
        template='plotly_white',
        margin=dict(l=60, r=20, t=50, b=40),
    )
    fig.update_yaxes(autorange='reversed')
    return fig


def create_velocity_profile_with_sea_surface(
    results: dict, 
    sea_surface_depth: float, 
    sea_surface_velocity: float,
    sea_surface_r_squared: float,
    ice_thickness: float = 1094.0,
) -> go.Figure:
    """Create velocity profile with interpolation including lake/sea surface.
    
    Shows observed layer velocities and an interpolated velocity trend
    from the ice surface through internal layers to the lake surface.
    
    Args:
        results: Analysis results dict with velocity data
        sea_surface_depth: Depth of sea/lake surface (m)
        sea_surface_velocity: Velocity of sea surface (m/yr)
        sea_surface_r_squared: R² of sea surface fit
        ice_thickness: Total ice thickness (m), unused but kept for API compatibility
    """
    from scipy.interpolate import UnivariateSpline
    
    velocity_data = results['velocity']
    depths = velocity_data['depths'].flatten()
    velocities = velocity_data['velocities'].flatten()
    velocities_smooth = velocity_data['velocities_smooth'].flatten()
    r_squared = velocity_data['r_squared'].flatten()
    reliable = velocity_data['reliable'].flatten().astype(bool)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Observed Layer Velocities",
            "Interpolated Velocity Profile"
        ),
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.12,
    )
    
    # Panel 1: Observed Velocity Profile
    # Unreliable layers (gray)
    fig.add_trace(
        go.Scatter(
            x=velocities[~reliable],
            y=depths[~reliable],
            mode='markers',
            marker=dict(color='#cbd5e1', size=7, symbol='circle'),
            name='Unreliable (R² < 0.3)',
            showlegend=True,
            hovertemplate='Depth: %{y:.0f}m<br>Velocity: %{x:.3f} m/yr<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Reliable layers (colored by R²)
    fig.add_trace(
        go.Scatter(
            x=velocities[reliable],
            y=depths[reliable],
            mode='markers',
            marker=dict(
                color=r_squared[reliable],
                colorscale='Viridis',
                size=10,
                cmin=0.3, cmax=1.0,
                colorbar=dict(title='R²', x=0.45, len=0.7, y=0.5),
                line=dict(color='white', width=1),
            ),
            name='Reliable (R² ≥ 0.3)',
            showlegend=True,
            hovertemplate='Depth: %{y:.0f}m<br>Velocity: %{x:.3f} m/yr<br>R²: %{marker.color:.3f}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Lake/sea surface point
    if not np.isnan(sea_surface_velocity):
        fig.add_trace(
            go.Scatter(
                x=[sea_surface_velocity],
                y=[sea_surface_depth],
                mode='markers',
                marker=dict(
                    color='#ef4444',
                    size=16,
                    symbol='star',
                    line=dict(color='white', width=2),
                ),
                name=f'Lake surface ({sea_surface_velocity:.3f} m/yr)',
                showlegend=True,
                hovertemplate=f'Lake Surface<br>Depth: {sea_surface_depth:.0f}m<br>Velocity: {sea_surface_velocity:.3f} m/yr<extra></extra>',
            ),
            row=1, col=1
        )
    
    # Smoothed velocity line
    fig.add_trace(
        go.Scatter(
            x=velocities_smooth,
            y=depths,
            mode='lines',
            line=dict(color='#3b82f6', width=2),
            name='Smoothed profile',
        ),
        row=1, col=1
    )
    
    # Zero velocity reference
    fig.add_vline(x=0, line=dict(color='#64748b', dash='dash', width=1), row=1, col=1)
    
    # Panel 2: Interpolated profile including lake surface
    if np.sum(reliable) > 3 and not np.isnan(sea_surface_velocity):
        # Combine reliable layer velocities with lake surface
        interp_depths = np.concatenate([depths[reliable], [sea_surface_depth]])
        interp_velocities = np.concatenate([velocities[reliable], [sea_surface_velocity]])
        
        # Sort by depth
        sort_idx = np.argsort(interp_depths)
        interp_depths = interp_depths[sort_idx]
        interp_velocities = interp_velocities[sort_idx]
        
        # Remove NaN values
        valid = ~np.isnan(interp_velocities)
        interp_depths = interp_depths[valid]
        interp_velocities = interp_velocities[valid]
        
        # Create smooth interpolation
        depth_grid = np.linspace(interp_depths.min(), interp_depths.max(), 200)
        
        try:
            # Use spline interpolation with moderate smoothing
            spline = UnivariateSpline(interp_depths, interp_velocities, 
                                       s=len(interp_depths) * 0.05, k=3)
            velocity_interp = spline(depth_grid)
        except Exception:
            # Fallback to linear interpolation
            velocity_interp = np.interp(depth_grid, interp_depths, interp_velocities)
        
        # Compute fit statistics
        predicted_at_obs = np.interp(interp_depths, depth_grid, velocity_interp)
        ss_res = np.sum((interp_velocities - predicted_at_obs) ** 2)
        ss_tot = np.sum((interp_velocities - np.mean(interp_velocities)) ** 2)
        interp_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Plot interpolated trend
        fig.add_trace(
            go.Scatter(
                x=velocity_interp,
                y=depth_grid,
                mode='lines',
                line=dict(color='#8b5cf6', width=3),
                name='Spline interpolation',
                fill='tozerox',
                fillcolor='rgba(139, 92, 246, 0.1)',
            ),
            row=1, col=2
        )
        
        # Add reliable layer data points
        fig.add_trace(
            go.Scatter(
                x=velocities[reliable],
                y=depths[reliable],
                mode='markers',
                marker=dict(color='#3b82f6', size=8, line=dict(color='white', width=1)),
                name='Layer velocities',
                showlegend=False,
                hovertemplate='Depth: %{y:.0f}m<br>Velocity: %{x:.3f} m/yr<extra></extra>',
            ),
            row=1, col=2
        )
        
        # Lake surface marker (important constraint)
        fig.add_trace(
            go.Scatter(
                x=[sea_surface_velocity],
                y=[sea_surface_depth],
                mode='markers',
                marker=dict(color='#ef4444', size=16, symbol='star', line=dict(color='white', width=2)),
                name='Lake surface',
                showlegend=False,
                hovertemplate=f'Lake Surface<br>Depth: {sea_surface_depth:.0f}m<br>Velocity: {sea_surface_velocity:.3f} m/yr<extra></extra>',
            ),
            row=1, col=2
        )
        
        # Add horizontal line at lake surface depth
        fig.add_hline(
            y=sea_surface_depth, 
            line=dict(color='#ef4444', dash='dot', width=1.5),
            row=1, col=2
        )
        
        # Add annotation with statistics
        fig.add_annotation(
            x=0.98, y=0.02,
            xref='x2 domain', yref='y2 domain',
            text=f"Lake surface: {sea_surface_velocity:.4f} m/yr (R²={sea_surface_r_squared:.2f})<br>"
                 f"Interpolation includes {np.sum(reliable)} layers + lake",
            showarrow=False,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=4,
            align='right',
        )
    
    # Zero velocity reference
    fig.add_vline(x=0, line=dict(color='#64748b', dash='dash', width=1), row=1, col=2)
    
    # Update axes
    fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title='Velocity (m/yr)', row=1, col=1)
    fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=2)
    fig.update_xaxes(title='Velocity (m/yr)', row=1, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e2e8f0', borderwidth=1),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
        margin=dict(l=60, r=60, t=60, b=50),
    )
    
    return fig


def estimate_lake_surface(apres_data: dict, target_depth: float = 1094.0, window_m: float = 20.0) -> tuple:
    """Estimate lake surface depth over time by peak amplitude near target depth."""
    range_img = apres_data['range_img']
    depths = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    depth_mask = np.abs(depths - target_depth) <= window_m
    if not np.any(depth_mask):
        return time_days, np.full_like(time_days, np.nan, dtype=float), np.full_like(time_days, np.nan, dtype=float)

    depth_candidates = depths[depth_mask]
    amp_window = range_img[depth_mask, :]

    lake_depths = np.full(amp_window.shape[1], np.nan, dtype=float)
    lake_amp = np.full(amp_window.shape[1], np.nan, dtype=float)

    for i in range(amp_window.shape[1]):
        column = np.abs(amp_window[:, i])
        if column.size == 0:
            continue
        idx = int(np.argmax(column))
        lake_depths[i] = depth_candidates[idx]
        lake_amp[i] = column[idx]

    return time_days, lake_depths, lake_amp


def track_sea_surface_phase(apres_data: dict, target_depth: float = 1094.0, 
                            window_m: float = 10.0, lambdac: float = 0.5608) -> tuple:
    """
    Track sea surface using phase-based method (same as layer tracking).
    
    This is more robust than peak detection as it uses phase unwrapping
    to track sub-wavelength motion.
    
    Args:
        apres_data: Dictionary with range_img, Rcoarse, time_days, raw_complex
        target_depth: Approximate sea surface depth (m)
        window_m: Search window around target depth (m)
        lambdac: Center wavelength in ice (m)
        
    Returns:
        time_days, tracked_depths, range_change, amplitude
    """
    raw_complex = apres_data.get('raw_complex')
    if raw_complex is None:
        # Fallback to amplitude-only tracking
        time_days, lake_depths, lake_amp = estimate_lake_surface(apres_data, target_depth, window_m)
        range_change = lake_depths - np.nanmedian(lake_depths[:50])  # Change from initial median
        return time_days, lake_depths, range_change, lake_amp
    
    range_img = apres_data['range_img']
    depths = apres_data['Rcoarse']
    time_days = apres_data['time_days']
    n_times = len(time_days)
    
    # Find the peak in the first time step to initialize
    depth_mask = np.abs(depths - target_depth) <= window_m
    if not np.any(depth_mask):
        return time_days, np.full(n_times, np.nan), np.full(n_times, np.nan), np.full(n_times, np.nan)
    
    depth_indices = np.where(depth_mask)[0]
    
    # Find initial peak location
    amp_init = np.abs(range_img[depth_mask, 0])
    peak_idx_local = np.argmax(amp_init)
    peak_idx = depth_indices[peak_idx_local]
    initial_depth = depths[peak_idx]
    
    # Compute rfine from complex phase
    phase = np.angle(raw_complex)
    rfine = phase * lambdac / (4 * np.pi)
    
    # Track the peak over time using local peak following
    tracked_indices = np.zeros(n_times, dtype=int)
    tracked_indices[0] = peak_idx
    
    search_bins = int(window_m / (depths[1] - depths[0])) if len(depths) > 1 else 5
    
    for t in range(1, n_times):
        prev_idx = tracked_indices[t - 1]
        search_start = max(0, prev_idx - search_bins)
        search_end = min(len(depths), prev_idx + search_bins + 1)
        
        local_amp = np.abs(range_img[search_start:search_end, t])
        if local_amp.size > 0:
            local_peak = np.argmax(local_amp)
            tracked_indices[t] = search_start + local_peak
        else:
            tracked_indices[t] = prev_idx
    
    # Extract phase-based fine range at tracked positions
    time_idx = np.arange(n_times)
    fine_range = rfine[tracked_indices, time_idx]
    amplitude = range_img[tracked_indices, time_idx]
    coarse_range = depths[tracked_indices]
    total_range = coarse_range + fine_range
    
    # Apply phase unwrapping
    wrap_period = lambdac / 2
    
    # Simple unwrapping: correct jumps > λ/4
    unwrapped = np.zeros_like(total_range)
    unwrapped[0] = total_range[0]
    threshold = wrap_period / 2
    
    for i in range(1, n_times):
        candidates = [total_range[i] + k * wrap_period for k in (-2, -1, 0, 1, 2)]
        deltas = [abs(c - unwrapped[i - 1]) for c in candidates]
        unwrapped[i] = candidates[int(np.argmin(deltas))]
    
    # Apply jump correction (same as layer tracking)
    range_change = unwrapped - unwrapped[0]
    
    # Correct large jumps
    for _ in range(50):
        diffs = np.diff(range_change)
        abs_diffs = np.abs(diffs)
        if np.max(abs_diffs) <= 0.20:
            break
        idx = np.argmax(abs_diffs)
        delta = diffs[idx]
        
        best_shift = 0
        best_residual = abs(delta)
        for k in [-3, -2, -1, 1, 2, 3]:
            test_shift = k * wrap_period
            residual = abs(delta + test_shift)
            if residual < best_residual:
                best_residual = residual
                best_shift = test_shift
        
        if best_shift != 0:
            range_change[idx + 1:] = range_change[idx + 1:] + best_shift
        else:
            break
    
    # Compute tracked depths
    tracked_depths = initial_depth + range_change
    
    return time_days, tracked_depths, range_change, amplitude


def smooth_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple median smoother that preserves NaNs."""
    if window < 3:
        return values.copy()
    half = window // 2
    smoothed = np.full_like(values, np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - half)
        end = min(len(values), i + half + 1)
        smoothed[i] = np.nanmedian(values[start:end])
    return smoothed


def create_lake_surface_figure(time_days: np.ndarray, tracked_depths: np.ndarray, 
                                range_change: np.ndarray, amplitude: np.ndarray) -> go.Figure:
    """Create a time series figure for lake/sea surface tracking with velocity.
    
    Args:
        time_days: Time axis in days
        tracked_depths: Phase-tracked surface depths (m)
        range_change: Range change from initial (m), positive = moving away
        amplitude: Amplitude at tracked positions
    """
    # Compute velocity from range change using linear fit
    valid = ~np.isnan(range_change)
    velocity_m_day = np.nan
    intercept = 0.0
    r_squared = np.nan
    velocity_m_yr = np.nan
    if np.sum(valid) > 10:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(time_days[valid], range_change[valid])
        velocity_m_day = slope
        velocity_m_yr = velocity_m_day * 365.25
        r_squared = r_value ** 2
    
    # Smooth for display
    tracked_smooth = smooth_series(tracked_depths, window=9)
    amp_db = 20 * np.log10(np.abs(amplitude) + 1e-30)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Sea Surface Depth (Phase-Tracked)', 
            f'Range Change (v = {velocity_m_yr:.3f} m/yr, R² = {r_squared:.3f})',
            'Amplitude'
        ),
        column_widths=[0.35, 0.35, 0.30],
    )

    # Panel 1: Tracked depth
    fig.add_trace(
        go.Scatter(
            x=time_days,
            y=tracked_depths,
            mode='lines',
            name='Phase-tracked',
            line=dict(color='#0ea5e9', width=1),
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=time_days,
            y=tracked_smooth,
            mode='lines',
            name='Smoothed',
            line=dict(color='#1e293b', width=2),
        ),
        row=1, col=1
    )

    # Panel 2: Range change with linear fit
    fig.add_trace(
        go.Scatter(
            x=time_days,
            y=range_change,
            mode='lines',
            name='Range change',
            line=dict(color='#10b981', width=1.5),
        ),
        row=1, col=2
    )
    # Add linear fit line
    if not np.isnan(velocity_m_yr):
        fit_line = velocity_m_day * time_days + intercept
        fig.add_trace(
            go.Scatter(
                x=time_days,
                y=fit_line,
                mode='lines',
                name='Linear fit',
                line=dict(color='#ef4444', width=2, dash='dash'),
            ),
            row=1, col=2
        )

    # Panel 3: Amplitude
    fig.add_trace(
        go.Scatter(
            x=time_days,
            y=amp_db,
            mode='lines',
            name='Amplitude (dB)',
            line=dict(color='#6366f1', width=1),
        ),
        row=1, col=3
    )

    fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)
    fig.update_xaxes(title='Time (days)', row=1, col=1)
    fig.update_yaxes(title='Range change (m)', row=1, col=2)
    fig.update_xaxes(title='Time (days)', row=1, col=2)
    fig.update_yaxes(title='Amplitude (dB)', row=1, col=3)
    fig.update_xaxes(title='Time (days)', row=1, col=3)

    fig.update_layout(
        height=360,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
    )

    return fig


def create_dash_app(output_dir: str, apres_data_path: str) -> 'Dash':
    """Create interactive Dash application."""
    
    if not DASH_AVAILABLE:
        raise ImportError("Dash is required for the interactive app. Install with: pip install dash")
    
    # Load data
    results = load_all_results(output_dir)
    apres_mat = loadmat(apres_data_path)
    
    # Only load RawImageComplex - skip RawImage and RfineBarTime to save RAM
    # RawImageComplex is sufficient: amplitude = abs(), phase = angle()
    raw_complex = np.array(apres_mat['RawImageComplex']) if 'RawImageComplex' in apres_mat else None
    
    # Compute amplitude from complex (more accurate than RawImage)
    if raw_complex is not None:
        range_img = np.abs(raw_complex)
    else:
        range_img = np.array(apres_mat['RawImage'])

    apres_data = {
        'range_img': range_img,
        'raw_complex': raw_complex,
        'Rcoarse': apres_mat['Rcoarse'].flatten(),
        'time_days': apres_mat['TimeInDays'].flatten(),
    }
    
    # Free memory from loaded mat file
    del apres_mat
    
    # Try to load pre-computed denoised echogram (for faster toggle)
    denoised = load_denoised_echogram(output_dir)
    if denoised is not None:
        print("Loaded pre-computed denoised echogram")
        _denoised_cache[id(apres_data['range_img'])] = denoised
    else:
        print("No pre-computed denoised echogram found. Run precompute_denoised.py for faster startup.")
    
    # Create summary figure
    summary_fig = create_summary_figure(results, apres_data)
    mean_echogram_fig = create_time_averaged_echogram_figure(apres_data)
    
    # Create Dash app
    app = Dash(__name__, title='ApRES Layer Analysis')

    theme = {
        'bg': '#f8fafc',
        'panel': '#ffffff',
        'card': '#ffffff',
        'text': '#0f172a',
        'muted': '#64748b',
        'accent': '#0ea5e9',
        'border': '#e2e8f0',
        'shadow': '0 18px 45px rgba(15, 23, 42, 0.08)',
    }
    
    # Get layer options for dropdown
    velocity_data = results['velocity']
    depths = velocity_data['depths'].flatten()
    velocities = velocity_data['velocities'].flatten()
    r_squared = velocity_data['r_squared'].flatten()
    reliable = velocity_data['reliable'].flatten().astype(bool)
    
    layer_options = [
        {
            'label': f"{d:.0f} m (v={v:.2f} m/yr, R²={r:.2f})" if not np.isnan(v) else f"{d:.0f} m (no valid velocity)",
            'value': i
        }
        for i, (d, v, r) in enumerate(zip(depths, velocities, r_squared))
    ]
    
    # Options for 3D highlighting (only reliable layers)
    highlight_options = [
        {
            'label': f"{depths[i]:.0f} m (v={velocities[i]:.2f} m/yr, R²={r_squared[i]:.2f})",
            'value': i
        }
        for i in range(len(depths)) if reliable[i]
    ]
    
    # Create initial 3D echograms
    lambdac = 0.5608
    if 'phase' in results and 'lambdac' in results['phase']:
        lambdac = float(np.array(results['phase']['lambdac']).squeeze())
    rcoarse = apres_data['Rcoarse']

    # Initial figures (using default depth range)
    initial_3d_fig = create_3d_echogram_figure(apres_data, depths, highlighted_layers=None, depth_range=(50, 250))
    initial_3d_phase = create_3d_phase_echogram_figure(
        apres_data,
        depths,
        lambdac,
        highlighted_layers=None,
        phase_mode='wrapped',
        depth_range=(50, 250),
    )

    least_rows = load_least_gaussian_rows(Path(output_dir))
    least_fig = create_least_gaussian_figure(least_rows)
    least_table = build_least_gaussian_table(least_rows)
    gmm_rows = load_gmm_sweep_rows(Path(output_dir))
    gmm_sweep_fig = create_gmm_sweep_figure(gmm_rows)

    initial_2d_echogram = create_2d_echogram_figure(apres_data)

    # Sea surface tracking using phase-based method
    SEA_SURFACE_DEPTH = 1094.0  # Target depth for sea surface
    lake_time, lake_depths, lake_range_change, lake_amp = track_sea_surface_phase(apres_data, target_depth=SEA_SURFACE_DEPTH)
    lake_surface_fig = create_lake_surface_figure(lake_time, lake_depths, lake_range_change, lake_amp)
    
    # Calculate sea surface velocity for the velocity profile
    from scipy.stats import linregress as lr_stats
    valid_sea = ~np.isnan(lake_range_change)
    if np.sum(valid_sea) > 10:
        sea_slope, sea_intercept, sea_r, _, _ = lr_stats(lake_time[valid_sea], lake_range_change[valid_sea])
        sea_surface_velocity = sea_slope * 365.25  # m/yr
        sea_surface_r_squared = sea_r ** 2
    else:
        sea_surface_velocity = np.nan
        sea_surface_r_squared = np.nan
    
    # Create velocity profile with Glen's law comparison
    velocity_profile_fig = create_velocity_profile_with_sea_surface(
        results, 
        sea_surface_depth=SEA_SURFACE_DEPTH,
        sea_surface_velocity=sea_surface_velocity,
        sea_surface_r_squared=sea_surface_r_squared,
        ice_thickness=SEA_SURFACE_DEPTH,
    )
    
    app.layout = html.Div([
        html.Div([
            html.H1(
                'ApRES Internal Ice Layer Velocity Analysis',
                style={
                    'textAlign': 'center',
                    'marginBottom': '8px',
                    'color': theme['text'],
                    'fontWeight': '600',
                    'letterSpacing': '0.3px',
                },
            ),
            html.P(
                'Interactive exploration of internal layer velocities',
                style={
                    'textAlign': 'center',
                    'marginTop': '0',
                    'color': theme['muted'],
                },
            ),
        ], style={'marginBottom': '24px'}),
        dcc.Tabs([
            dcc.Tab(label='Summary', children=[
                # Velocity Profile with Interpolation
                html.Div([
                    html.H3("Velocity Profile with Lake Surface", 
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P("Observed layer velocities with spline interpolation from surface to lake",
                          style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    dcc.Graph(id='velocity-profile-sea-surface', figure=velocity_profile_fig, style={'height': '600px'}),
                ], style={
                    'margin': '20px 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                
                # Sea Surface Tracking Detail
                html.Div([
                    html.H3('Sea Surface Phase Tracking', 
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P('Sub-wavelength tracking of sea surface using phase unwrapping',
                          style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    dcc.Graph(id='lake-surface-figure', figure=lake_surface_fig, style={'height': '360px'}),
                ], style={
                    'margin': '0 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                
                # Analysis Overview
                html.Div([
                    html.H3('Detailed Analysis Overview', 
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P('Echogram with detected layers, velocity profile, phase time series, and quality assessment',
                          style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    dcc.Graph(id='summary-figure', figure=summary_fig, style={'height': '900px'}),
                ], style={
                    'margin': '0 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                
                # Time-averaged Echogram
                html.Div([
                    html.H3('Time-Averaged Echogram', 
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P('Mean amplitude profile showing reflector strengths across depth',
                          style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    dcc.Graph(id='mean-echogram-fig', figure=mean_echogram_fig, style={'height': '420px'}),
                ], style={
                    'margin': '0 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
            dcc.Tab(label='3D Views', children=[
                html.Div([
                    html.H3('3D Echogram Visualization', style={'color': theme['text'], 'marginBottom': '10px'}),
                    html.P(
                        'Rotate: drag · Zoom: scroll · Pan: shift + drag',
                        style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '16px'},
                    ),
                    html.Div([
                        html.Div([
                            html.Label('Start depth (m):', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='depth-start-slider',
                                type='number',
                                min=0,
                                max=2000,
                                step=1,
                                value=50,
                                style={
                                    'width': '100px',
                                    'padding': '8px',
                                    'borderRadius': '4px',
                                    'border': f"1px solid {theme['border']}",
                                    'marginTop': '6px',
                                },
                                debounce=True,
                            ),
                        ], style={'width': '150px', 'marginRight': '20px'}),
                        html.Div([
                            html.Label('Depth interval (m):', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='depth-interval-input',
                                type='number',
                                min=0.5,
                                max=2000,
                                step=0.5,
                                value=200,
                                style={
                                    'width': '100px',
                                    'padding': '8px',
                                    'borderRadius': '4px',
                                    'border': f"1px solid {theme['border']}",
                                    'marginTop': '6px',
                                },
                                debounce=True,
                            ),
                        ], style={'width': '150px', 'marginRight': '20px'}),
                        html.Div([
                            html.Label('View:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.RadioItems(
                                id='denoise-toggle',
                                options=[
                                    {'label': 'Original', 'value': 'original'},
                                    {'label': 'Denoised', 'value': 'denoised'},
                                ],
                                value='original',
                                inline=True,
                                style={'marginTop': '6px'},
                                inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                            ),
                        ], style={'marginRight': '20px'}),
                        html.Div([
                            html.Label('Highlight layers:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='highlight-layers',
                                options=highlight_options,
                                value=[],
                                multi=True,
                                placeholder='Select layers...',
                                style={'width': '100%', 'marginTop': '5px'},
                            ),
                        ], style={'flex': '1', 'maxWidth': '300px'}),
                    ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '20px', 'flexWrap': 'wrap', 'gap': '15px'}),
                    dcc.Graph(id='echogram-3d', figure=initial_3d_fig, style={'height': '700px'}),
                ], style={
                    'padding': '22px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'margin': '18px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                html.Div([
                    html.H3('3D Phase Echogram', style={'color': theme['text'], 'marginBottom': '10px'}),
                    html.P(
                        'Phase from complex image. Wrapped shows values in [-π, π].',
                        style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '12px'},
                    ),
                    html.Div([
                        html.Label('Phase view:', style={'fontWeight': '600', 'color': theme['text']}),
                        dcc.RadioItems(
                            id='phase-wrap-toggle',
                            options=[
                                {'label': 'Wrapped', 'value': 'wrapped'},
                                {'label': 'Unwrapped', 'value': 'unwrapped'},
                            ],
                            value='wrapped',
                            inline=True,
                            style={'marginTop': '6px'},
                            inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                        ),
                    ], style={'marginBottom': '10px'}),
                    dcc.Graph(id='echogram-3d-phase', figure=initial_3d_phase, style={'height': '650px'}),
                ], style={
                    'padding': '22px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'margin': '18px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
            dcc.Tab(label='2D Views', children=[
                html.Div([
                    html.Div([
                        html.H3('2D Echogram', style={'color': theme['text'], 'marginBottom': '10px'}),
                    ]),
                    dcc.Graph(id='echogram-2d', figure=initial_2d_echogram, style={'height': '500px'}),
                ], style={
                    'padding': '22px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'margin': '18px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
            dcc.Tab(label='Layer Detail', children=[
                html.Div([
                    html.H3('Explore Individual Layers', style={'color': theme['text']}),
                    html.Div([
                        html.Label('Select layer depth to examine:', style={'fontWeight': '600', 'color': theme['text']}),
                        dcc.Dropdown(
                            id='layer-selector',
                            options=layer_options,
                            value=0,
                            style={'width': '500px'},
                        ),
                    ], style={'marginTop': '10px', 'marginBottom': '20px'}),
                    dcc.Graph(id='layer-detail', style={'height': '400px'}),
                ], style={
                    'padding': '20px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'margin': '18px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
            dcc.Tab(label='Lake Surface', children=[
                html.Div([
                    html.H3('Lake Surface Tracking', style={'color': theme['text'], 'marginBottom': '12px'}),
                    dcc.Graph(id='lake-surface-figure', figure=lake_surface_fig, style={'height': '360px'}),
                ], style={
                    'margin': '18px',
                    'padding': '18px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
            dcc.Tab(label='Echo-less Analysis', children=[
                html.Div([
                    html.H3('Echo-less Phase Histogram', style={'color': theme['text'], 'marginBottom': '10px'}),
                    html.P(
                        'Loads histogram images created by `phase_noise_analysis.py`.',
                        style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '12px'},
                    ),
                    html.Div([
                        html.Div([
                            html.Label('Available histograms:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='echo-file',
                                options=list_histogram_images(Path(output_dir)),
                                value=None,
                                placeholder='Select a histogram…',
                                style={'width': '320px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Button('Refresh', id='echo-refresh', n_clicks=0, style={
                            'marginTop': '24px',
                            'height': '36px',
                            'padding': '0 16px',
                            'backgroundColor': '#e2e8f0',
                            'color': theme['text'],
                            'border': 'none',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                        }),
                        html.Button('Load', id='echo-load', n_clicks=0, style={
                            'marginTop': '24px',
                            'height': '36px',
                            'padding': '0 16px',
                            'backgroundColor': theme['accent'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                        }),
                    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'}),
                    html.P(id='echo-status', style={'marginTop': '10px', 'color': theme['muted']}),
                    html.Div(id='echo-summary', style={'marginTop': '8px', 'color': theme['text']}),
                    html.Div(id='echo-gmm-summary', style={'marginTop': '8px', 'color': theme['text']}),
                    html.Div([
                        html.Div([
                            html.H4('Histogram', style={'color': theme['text'], 'marginBottom': '8px'}),
                            dcc.Graph(id='echo-histogram-fig', style={'height': '340px'}),
                        ], style={
                            'flex': '1 1 380px',
                            'backgroundColor': theme['bg'],
                            'borderRadius': '12px',
                            'padding': '12px',
                            'border': f"1px solid {theme['border']}",
                        }),
                        html.Div([
                            html.H4('GMM Split', style={'color': theme['text'], 'marginBottom': '8px'}),
                            dcc.Graph(id='echo-gmm-fig', style={'height': '340px'}),
                        ], style={
                            'flex': '1 1 380px',
                            'backgroundColor': theme['bg'],
                            'borderRadius': '12px',
                            'padding': '12px',
                            'border': f"1px solid {theme['border']}",
                        }),
                    ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '12px'}),
                ], style={
                    'margin': '18px',
                    'padding': '18px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                html.Div([
                    html.H3('Interactive Histogram + EM/GMM', style={'color': theme['text'], 'marginBottom': '10px'}),
                    html.P(
                        'Select a depth, create a histogram, then tune EM/GMM parameters and run the fit.',
                        style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '12px'},
                    ),
                    html.P(
                        'Uses the nearest available depth in the Rcoarse grid.',
                        style={'color': theme['muted'], 'fontSize': '11px', 'marginBottom': '12px'},
                    ),
                    html.Div([
                        html.Div([
                            html.Label('Depth (m)', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='interactive-depth',
                                type='number',
                                value=112.0,
                                step=1,
                                style={'width': '140px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Unwrap phase', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Checklist(
                                id='interactive-unwrap',
                                options=[{'label': 'Use unwrapped', 'value': 'unwrap'}],
                                value=[],
                                style={'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Window (m)', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='interactive-window',
                                type='number',
                                value=5.0,
                                step=0.5,
                                min=0.0,
                                style={'width': '120px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Amplitude weight', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Checklist(
                                id='interactive-weight',
                                options=[{'label': 'Weight by amplitude', 'value': 'weight'}],
                                value=[],
                                style={'marginTop': '6px'},
                            ),
                        ]),
                        html.Button('Create Histogram', id='interactive-histogram', n_clicks=0, style={
                            'height': '36px',
                            'padding': '0 16px',
                            'backgroundColor': theme['accent'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'marginTop': '22px',
                        }),
                    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap', 'alignItems': 'flex-end'}),
                    html.Div([
                        html.Div([
                            html.Label('GMM components', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='interactive-gmm-components',
                                type='number',
                                value=2,
                                min=1,
                                step=1,
                                style={'width': '120px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Reg covar', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='interactive-gmm-reg',
                                type='number',
                                value=1e-4,
                                step=1e-5,
                                style={'width': '140px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Init strategy', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='interactive-gmm-init',
                                options=[
                                    {'label': 'Percentile', 'value': 'percentile'},
                                    {'label': 'Linear', 'value': 'linear'},
                                ],
                                value='percentile',
                                style={'width': '160px', 'marginTop': '6px'},
                            ),
                        ]),
                        html.Div([
                            html.Label('Zero-mean comp', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Checklist(
                                id='interactive-gmm-zero',
                                options=[{'label': 'Fix one mean at 0', 'value': 'zero'}],
                                value=[],
                                style={'marginTop': '6px'},
                            ),
                        ]),
                        html.Button('Run EM/GMM', id='interactive-gmm-run', n_clicks=0, style={
                            'height': '36px',
                            'padding': '0 16px',
                            'backgroundColor': '#0f172a',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '8px',
                            'cursor': 'pointer',
                            'marginTop': '22px',
                        }),
                    ], style={'display': 'flex', 'gap': '18px', 'flexWrap': 'wrap', 'alignItems': 'flex-end', 'marginTop': '12px'}),
                    html.P(id='interactive-status', style={'marginTop': '10px', 'color': theme['muted']}),
                    dcc.Store(id='interactive-phase-diff'),
                    dcc.Store(id='interactive-pcf'),
                    html.Div([
                        html.Div([
                            html.H4('Histogram', style={'color': theme['text'], 'marginBottom': '8px'}),
                            dcc.Graph(id='interactive-hist-fig', style={'height': '340px'}),
                        ], style={
                            'flex': '1 1 380px',
                            'backgroundColor': theme['bg'],
                            'borderRadius': '12px',
                            'padding': '12px',
                            'border': f"1px solid {theme['border']}",
                        }),
                        html.Div([
                            html.H4('GMM Split', style={'color': theme['text'], 'marginBottom': '8px'}),
                            dcc.Graph(id='interactive-gmm-fig', style={'height': '340px'}),
                        ], style={
                            'flex': '1 1 380px',
                            'backgroundColor': theme['bg'],
                            'borderRadius': '12px',
                            'padding': '12px',
                            'border': f"1px solid {theme['border']}",
                        }),
                    ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '12px'}),
                    html.Div([
                        html.Div([
                            html.H4('Phase Coherence (PCF)', style={'color': theme['text'], 'marginBottom': '8px'}),
                            dcc.Graph(id='interactive-pcf-fig', figure=go.Figure(), style={'height': '320px'}),
                        ], style={
                            'flex': '1 1 640px',
                            'backgroundColor': theme['bg'],
                            'borderRadius': '12px',
                            'padding': '12px',
                            'border': f"1px solid {theme['border']}",
                        }),
                    ], style={'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap', 'marginTop': '12px'}),
                ], style={
                    'margin': '18px',
                    'padding': '18px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                html.Div([
                    html.Div([
                        html.H3('Least-Gaussian Summary', style={'color': theme['text'], 'marginBottom': '10px'}),
                        html.P(
                            'Based on phase_noise_rank.py output.',
                            style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '12px'},
                        ),
                    ]),
                    html.Button('Refresh Summary', id='least-gaussian-refresh', n_clicks=0, style={
                        'marginBottom': '12px',
                        'height': '36px',
                        'padding': '0 16px',
                        'backgroundColor': '#e2e8f0',
                        'color': theme['text'],
                        'border': 'none',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                    }),
                    dcc.Graph(id='least-gaussian-fig', figure=least_fig, style={'height': '360px'}),
                    html.Div(id='least-gaussian-table', children=least_table, style={'marginTop': '10px'}),
                ], style={
                    'margin': '18px',
                    'padding': '18px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                html.Div([
                    html.Div([
                        html.H3('GMM Sweep Summary', style={'color': theme['text'], 'marginBottom': '10px'}),
                        html.P(
                            'Shows the narrowest (signal-like) component across depths.',
                            style={'color': theme['muted'], 'fontSize': '12px', 'marginBottom': '12px'},
                        ),
                    ]),
                    html.Button('Refresh GMM Sweep', id='gmm-sweep-refresh', n_clicks=0, style={
                        'marginBottom': '12px',
                        'height': '36px',
                        'padding': '0 16px',
                        'backgroundColor': '#e2e8f0',
                        'color': theme['text'],
                        'border': 'none',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                    }),
                    dcc.Graph(id='gmm-sweep-fig', figure=gmm_sweep_fig, style={'height': '380px'}),
                ], style={
                    'margin': '18px',
                    'padding': '18px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
            ]),
        ]),
        html.Div([
            html.Hr(style={'borderColor': theme['border']}),
            html.P('Based on methodology from Summers et al. (2021) - IGARSS', style={'fontStyle': 'italic'}),
            html.P('SiegVent2023 Project - ApRES Internal Layer Velocity Analysis'),
        ], style={'textAlign': 'center', 'marginTop': '30px', 'color': theme['muted']}),
    ], style={
        'padding': '32px',
        'maxWidth': '1500px',
        'margin': 'auto',
        'fontFamily': 'Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif',
        'backgroundColor': theme['bg'],
    })

    @app.callback(
        Output('layer-detail', 'figure'),
        Input('layer-selector', 'value'),
    )
    def update_layer_detail(layer_idx):
        if layer_idx is None:
            layer_idx = 0

        phase_data = results['phase']
        velocity_data_local = results['velocity']

        depths_local = velocity_data_local['depths'].flatten()
        range_ts = phase_data['range_timeseries'][layer_idx, :] * 100  # cm
        amp_ts = phase_data['amplitude_timeseries'][layer_idx, :]
        amp_db = 10 * np.log10(amp_ts**2 + 1e-30)
        time_days = phase_data['time_days'].flatten()

        depth = depths_local[layer_idx]
        velocity = velocity_data_local['velocities'].flatten()[layer_idx]
        r_sq = velocity_data_local['r_squared'].flatten()[layer_idx]

        valid = ~np.isnan(range_ts)
        if np.sum(valid) > 10:
            slope, intercept, _, _, _ = stats.linregress(time_days[valid], range_ts[valid])
            fit_line = slope * time_days + intercept
        else:
            fit_line = np.full_like(time_days, np.nan, dtype=float)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'Range Change at {depth:.0f}m',
                f'Amplitude Stability at {depth:.0f}m'
            ),
        )

        fig.add_trace(
            go.Scatter(x=time_days, y=range_ts, mode='lines', name='Data',
                       line=dict(color='#3498db', width=1)),
            row=1, col=1
        )

        if not np.all(np.isnan(fit_line)):
            fig.add_trace(
                go.Scatter(x=time_days, y=fit_line, mode='lines',
                           name=f'Fit: {velocity:.2f} m/yr',
                           line=dict(color='#e74c3c', width=2, dash='dash')),
                row=1, col=1
            )

        fig.add_trace(
            go.Scatter(x=time_days, y=amp_db, mode='lines', name='Amplitude',
                       line=dict(color='#27ae60', width=1)),
            row=1, col=2
        )

        fig.update_xaxes(title='Time (days)', row=1, col=1)
        fig.update_yaxes(title='ΔRange (cm)', row=1, col=1)
        fig.update_xaxes(title='Time (days)', row=1, col=2)
        fig.update_yaxes(title='Amplitude (dB)', row=1, col=2)

        vel_str = f'{velocity:.2f}' if not np.isnan(velocity) else 'N/A'
        r_sq_str = f'{r_sq:.3f}' if not np.isnan(r_sq) else 'N/A'

        fig.update_layout(
            title=dict(
                text=f'Layer at {depth:.0f}m: v = {vel_str} m/yr, R² = {r_sq_str}',
                font=dict(size=14),
            ),
            height=350,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
        )

        return fig
    
    # Callback for 3D echogram layer highlighting, depth range, and denoising
    @app.callback(
        Output('echogram-3d', 'figure'),
        Input('highlight-layers', 'value'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
        Input('denoise-toggle', 'value'),
    )
    def update_3d_echogram(highlighted_indices, depth_start, depth_interval, denoise_mode):
        if highlighted_indices is None:
            highlighted_indices = []
        # Parse start depth from text input
        try:
            depth_start = float(depth_start) if depth_start else 50
            depth_start = max(0, min(2000, depth_start))
        except (ValueError, TypeError):
            depth_start = 50
        # Parse interval from text input
        try:
            depth_interval = float(depth_interval) if depth_interval else 200
            depth_interval = max(0.5, min(2000, depth_interval))
        except (ValueError, TypeError):
            depth_interval = 200
        depth_range = (depth_start, depth_start + depth_interval)
        denoise = (denoise_mode == 'denoised') if denoise_mode else False
        
        depths = velocity_data['depths'].flatten()
        
        # Get range_timeseries for layer tracking
        phase_data = results.get('phase', {})
        range_ts = phase_data.get('range_timeseries', None)
        phase_time = phase_data.get('time_days', None)
        if phase_time is not None:
            phase_time = phase_time.flatten()
        
        fig = create_3d_echogram_figure(
            apres_data, 
            depths, 
            highlighted_layers=highlighted_indices,
            depth_range=tuple(depth_range),
            denoise=denoise,
            output_dir=output_dir,
            range_timeseries=range_ts,
            phase_time=phase_time,
        )
        return fig

    @app.callback(
        Output('echogram-3d-phase', 'figure'),
        Input('highlight-layers', 'value'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
        Input('phase-wrap-toggle', 'value'),
    )
    def update_3d_phase_echogram(highlighted_indices, depth_start, depth_interval, phase_mode):
        if highlighted_indices is None:
            highlighted_indices = []
        try:
            depth_start = float(depth_start) if depth_start else 50
            depth_start = max(0, min(2000, depth_start))
        except (ValueError, TypeError):
            depth_start = 50
        try:
            depth_interval = float(depth_interval) if depth_interval else 200
            depth_interval = max(0.5, min(2000, depth_interval))
        except (ValueError, TypeError):
            depth_interval = 200
        depth_range = (depth_start, depth_start + depth_interval)
        phase_mode = phase_mode or 'wrapped'

        fig = create_3d_phase_echogram_figure(
            apres_data,
            depths,
            lambdac,
            highlighted_layers=highlighted_indices,
            depth_range=depth_range,
            phase_mode=phase_mode,
        )
        return fig

    @app.callback(
        Output('echogram-2d', 'figure'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
    )
    def update_2d_views(depth_start, depth_interval):
        try:
            depth_start = float(depth_start) if depth_start else 50
            depth_start = max(0, min(2000, depth_start))
        except (ValueError, TypeError):
            depth_start = 50
        try:
            depth_interval = float(depth_interval) if depth_interval else 200
            depth_interval = max(0.5, min(2000, depth_interval))
        except (ValueError, TypeError):
            depth_interval = 200
        depth_range = (depth_start, depth_start + depth_interval)
        return create_2d_echogram_figure(apres_data, depth_range=depth_range)

    @app.callback(
        Output('echo-file', 'options'),
        Input('echo-refresh', 'n_clicks'),
    )
    def refresh_echo_files(n_clicks):
        return list_histogram_images(Path(output_dir))

    @app.callback(
        Output('echo-histogram-fig', 'figure'),
        Output('echo-gmm-fig', 'figure'),
        Output('echo-status', 'children'),
        Output('echo-summary', 'children'),
        Output('echo-gmm-summary', 'children'),
        Input('echo-load', 'n_clicks'),
        State('echo-file', 'value'),
    )
    def update_echo_histogram(n_clicks, selected_file):
        empty_hist = go.Figure()
        empty_hist.add_annotation(
            text='Click Load to display the histogram.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        empty_hist.update_layout(height=320, template='plotly_white')

        empty_gmm = go.Figure()
        empty_gmm.add_annotation(
            text='Run with --gmm to display GMM split.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        empty_gmm.update_layout(height=320, template='plotly_white')

        if not n_clicks:
            return empty_hist, empty_gmm, 'Click Load to display the histogram.', None, None

        if not selected_file:
            return empty_hist, empty_gmm, 'Select a histogram from the dropdown.', None, None

        image_name = selected_file
        image_path = Path(output_dir) / image_name
        if not image_path.exists():
            return empty_hist, empty_gmm, f"Image not found: {image_path}", None, None

        summary = None
        gmm_summary = None
        hist_fig = empty_hist
        gmm_fig = empty_gmm
        csv_name = image_name.replace('.png', '.csv')
        csv_path = Path(output_dir) / csv_name
        if csv_path.exists():
            try:
                values = np.loadtxt(csv_path, delimiter=',', skiprows=1)
                if values.ndim == 0:
                    values = np.array([values])
                values = values[np.isfinite(values)]
                if values.size > 0:
                    mu = float(np.mean(values))
                    sigma = float(np.std(values))
                    skew = float(stats.skew(values))
                    kurt = float(stats.kurtosis(values, fisher=True))
                    ks_p = float(stats.kstest(values, 'norm', args=(mu, sigma)).pvalue) if sigma > 0 else np.nan
                    normal_p = float(stats.normaltest(values).pvalue) if values.size >= 20 else np.nan
                    summary = html.Ul([
                        html.Li(f"Mean: {mu:.4f} rad"),
                        html.Li(f"Std: {sigma:.4f} rad"),
                        html.Li(f"Skew: {skew:.4f}"),
                        html.Li(f"Kurtosis: {kurt:.4f}"),
                        html.Li(f"KS p-value: {ks_p:.4g}"),
                        html.Li(f"Normaltest p-value: {normal_p:.4g}"),
                    ])
                    hist_fig = go.Figure()
                    hist_fig.add_trace(
                        go.Histogram(
                            x=values,
                            nbinsx=40,
                            name='Phase Δ',
                            marker=dict(color='#2563eb'),
                            opacity=0.8,
                        )
                    )
                    if sigma > 0:
                        x_line = np.linspace(values.min(), values.max(), 300)
                        pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
                        pdf_scaled = pdf * (values.size * (values.max() - values.min()) / 40)
                        hist_fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=pdf_scaled,
                                mode='lines',
                                name='Gaussian fit',
                                line=dict(color='#ef4444', width=2, dash='dash'),
                            )
                        )
                    hist_fig.update_layout(
                        template='plotly_white',
                        height=320,
                        margin=dict(l=40, r=20, t=30, b=40),
                        xaxis_title='Phase Δ (rad)',
                        yaxis_title='Count',
                        showlegend=True,
                    )
            except Exception as exc:
                summary = html.Div(f"Failed to read CSV: {exc}")

        gmm_data = load_gmm_summary(Path(output_dir), image_name)
        if gmm_data and "components" in gmm_data:
            comps = gmm_data["components"]
            means = comps.get("means", [])
            stds = comps.get("stds", [])
            weights = comps.get("weights", [])
            items = []
            for i, (m, s, w) in enumerate(zip(means, stds, weights), start=1):
                items.append(html.Li(f"Comp {i}: mean={m:.4f}, std={s:.4f}, weight={w:.3f}"))
            gmm_summary = html.Div([
                html.Strong("GMM (EM) components:"),
                html.Ul(items),
                html.Div(
                    f"Method: {comps.get('method', 'unknown')} | "
                    f"Converged: {comps.get('converged', 'n/a')} | "
                    f"Iters: {comps.get('n_iter', 'n/a')} | "
                    f"Zero-mean: {comps.get('zero_mean', False)}",
                    style={"color": theme["muted"], "fontSize": "12px"},
                ),
            ])
            if csv_path.exists():
                try:
                    values = np.loadtxt(csv_path, delimiter=',', skiprows=1)
                    if values.ndim == 0:
                        values = np.array([values])
                    values = values[np.isfinite(values)]
                    if values.size > 0:
                        x_line = np.linspace(values.min(), values.max(), 300)
                        gmm_fig = go.Figure()
                        mix_pdf = np.zeros_like(x_line, dtype=float)
                        for i, (m, s, w) in enumerate(zip(means, stds, weights), start=1):
                            if s <= 0:
                                continue
                            comp_pdf = w * stats.norm.pdf(x_line, loc=m, scale=s)
                            mix_pdf += comp_pdf
                            gmm_fig.add_trace(
                                go.Scatter(
                                    x=x_line,
                                    y=comp_pdf,
                                    mode='lines',
                                    name=f'Comp {i}',
                                    line=dict(width=2),
                                )
                            )
                        gmm_fig.add_trace(
                            go.Scatter(
                                x=x_line,
                                y=mix_pdf,
                                mode='lines',
                                name='Mixture',
                                line=dict(color='#111827', width=2),
                            )
                        )
                        gmm_fig.update_layout(
                            template='plotly_white',
                            height=320,
                            margin=dict(l=40, r=20, t=30, b=40),
                            xaxis_title='Phase Δ (rad)',
                            yaxis_title='Density',
                            showlegend=True,
                        )
                except Exception:
                    gmm_fig = empty_gmm
        else:
            gmm_summary = html.Div(
                "No GMM summary found. Run phase_noise_analysis.py with --gmm.",
                style={"color": theme["muted"], "fontSize": "12px"},
            )

        return hist_fig, gmm_fig, f"Loaded: {image_name}", summary, gmm_summary

    @app.callback(
        Output('least-gaussian-fig', 'figure'),
        Output('least-gaussian-table', 'children'),
        Input('least-gaussian-refresh', 'n_clicks'),
    )
    def update_least_gaussian_summary(n_clicks):
        rows = load_least_gaussian_rows(Path(output_dir))
        fig = create_least_gaussian_figure(rows)
        table = build_least_gaussian_table(rows)
        return fig, table

    @app.callback(
        Output('gmm-sweep-fig', 'figure'),
        Input('gmm-sweep-refresh', 'n_clicks'),
    )
    def update_gmm_sweep_summary(n_clicks):
        rows = load_gmm_sweep_rows(Path(output_dir))
        fig = create_gmm_sweep_figure(rows)
        return fig

    @app.callback(
        Output('interactive-phase-diff', 'data'),
        Output('interactive-pcf', 'data'),
        Output('interactive-status', 'children'),
        Output('interactive-hist-fig', 'figure'),
        Input('interactive-histogram', 'n_clicks'),
        State('interactive-depth', 'value'),
        State('interactive-unwrap', 'value'),
        State('interactive-window', 'value'),
        State('interactive-weight', 'value'),
    )
    def build_interactive_histogram(n_clicks, depth_value, unwrap_value, window_m, weight_value):
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text='Click “Create Histogram” to compute.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        empty_fig.update_layout(height=320, template='plotly_white')

        if not n_clicks:
            return None, None, 'Ready to compute a histogram.', empty_fig

        if depth_value is None:
            return None, None, 'Enter a depth to continue.', empty_fig

        if wrap_phase is None:
            return None, None, 'Phase helpers not available in this session.', empty_fig

        depth_value = float(depth_value)
        idx = int(np.argmin(np.abs(rcoarse - depth_value)))
        depth_sel = float(rcoarse[idx])
        delta_depth = abs(depth_sel - depth_value)
        
        raw_complex = apres_data.get('raw_complex')
        if raw_complex is None:
            return None, None, 'Complex image not available in data file.', empty_fig

        # Get phase directly from complex data
        raw_phase = np.angle(raw_complex[idx, :])
        use_unwrap = 'unwrap' in (unwrap_value or [])
        phase_series = np.unwrap(raw_phase) if use_unwrap else wrap_phase(raw_phase)
        phase_diff = np.diff(phase_series)
        if not use_unwrap:
            phase_diff = wrap_phase(phase_diff)
        phase_diff = phase_diff[np.isfinite(phase_diff)]

        if phase_diff.size < 5:
            return None, None, (
                f'Not enough samples at {depth_sel:.1f} m '
                f'(requested {depth_value:.1f} m, Δ={delta_depth:.3f} m).'
            ), empty_fig

        window_m = float(window_m or 0.0)
        use_weight = 'weight' in (weight_value or [])
        if window_m <= 0:
            window_mask = np.zeros_like(rcoarse, dtype=bool)
            window_mask[idx] = True
        else:
            half = window_m / 2.0
            window_mask = (rcoarse >= depth_sel - half) & (rcoarse <= depth_sel + half)

        complex_window = raw_complex[window_mask, :]
        if complex_window.ndim == 1:
            complex_window = complex_window.reshape(1, -1)
        
        if use_weight:
            amp_window = np.abs(complex_window)
            weights = amp_window
            phasor_sum = np.sum(complex_window * weights, axis=0)
            weight_sum = np.sum(weights, axis=0) + 1e-12
            phasor_mean = phasor_sum / weight_sum
        else:
            phasor_mean = np.mean(complex_window, axis=0)

        phase_series = np.angle(phasor_mean)
        phase_deltas = np.diff(phase_series)
        phase_deltas = wrap_phase(phase_deltas)
        if phase_deltas.size == 0:
            pcf = float('nan')
            mean_phase = float('nan')
        else:
            mean_vec = np.mean(np.exp(1j * phase_deltas))
            pcf = float(np.abs(mean_vec))
            mean_phase = float(np.angle(mean_vec))
        pcf_data = {
            "pcf": float(pcf),
            "mean_phase": mean_phase,
            "window_m": float(window_m),
            "use_weight": bool(use_weight),
            "depth_sel": float(depth_sel),
        }

        mu = float(np.mean(phase_diff))
        sigma = float(np.std(phase_diff))

        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=phase_diff,
                nbinsx=40,
                name='Phase Δ',
                marker=dict(color='#2563eb'),
                opacity=0.8,
            )
        )
        if sigma > 0:
            x_line = np.linspace(phase_diff.min(), phase_diff.max(), 300)
            pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
            pdf_scaled = pdf * (phase_diff.size * (phase_diff.max() - phase_diff.min()) / 40)
            hist_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=pdf_scaled,
                    mode='lines',
                    name='Gaussian fit',
                    line=dict(color='#ef4444', width=2, dash='dash'),
                )
            )
        hist_fig.update_layout(
            template='plotly_white',
            height=320,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title='Phase Δ (rad)',
            yaxis_title='Count',
        )

        status = (
            f'Histogram created at {depth_sel:.1f} m '
            f'(requested {depth_value:.1f} m, Δ={delta_depth:.3f} m).'
        )
        return phase_diff.tolist(), pcf_data, status, hist_fig

    @app.callback(
        Output('interactive-gmm-fig', 'figure'),
        Output('interactive-status', 'children', allow_duplicate=True),
        Input('interactive-gmm-run', 'n_clicks'),
        State('interactive-phase-diff', 'data'),
        State('interactive-gmm-components', 'value'),
        State('interactive-gmm-reg', 'value'),
        State('interactive-gmm-init', 'value'),
        State('interactive-gmm-zero', 'value'),
        prevent_initial_call=True,
    )
    def run_interactive_gmm(n_clicks, phase_diff_data, n_components, reg_covar, init_strategy, zero_mean_value):
        empty_fig = go.Figure()
        empty_fig.add_annotation(
            text='Create a histogram first.',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        empty_fig.update_layout(height=320, template='plotly_white')

        if phase_diff_data is None:
            return empty_fig, 'Create a histogram before running EM/GMM.'

        if fit_gmm_1d is None:
            return empty_fig, 'GMM helpers not available in this session.'

        values = np.array(phase_diff_data, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 5:
            return empty_fig, 'Not enough samples to fit GMM.'

        try:
            n_components = int(n_components or 2)
            reg_covar = float(reg_covar or 1e-4)
            init_strategy = init_strategy or 'percentile'
            gmm_result = fit_gmm_1d(
                values,
                n_components=n_components,
                reg_covar=reg_covar,
                init_strategy=init_strategy,
                zero_mean=('zero' in (zero_mean_value or [])),
            )
        except Exception as exc:
            return empty_fig, f'GMM fit failed: {exc}'

        means = gmm_result["means"]
        stds = gmm_result["stds"]
        weights = gmm_result["weights"]

        x_line = np.linspace(values.min(), values.max(), 300)
        gmm_fig = go.Figure()
        mix_pdf = np.zeros_like(x_line, dtype=float)
        for i, (m, s, w) in enumerate(zip(means, stds, weights), start=1):
            if s <= 0:
                continue
            comp_pdf = w * stats.norm.pdf(x_line, loc=m, scale=s)
            mix_pdf += comp_pdf
            gmm_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=comp_pdf,
                    mode='lines',
                    name=f'Comp {i}',
                    line=dict(width=2),
                )
            )
        gmm_fig.add_trace(
            go.Scatter(
                x=x_line,
                y=mix_pdf,
                mode='lines',
                name='Mixture',
                line=dict(color='#111827', width=2),
            )
        )
        gmm_fig.update_layout(
            template='plotly_white',
            height=320,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title='Phase Δ (rad)',
            yaxis_title='Density',
            showlegend=True,
        )

        status = (
            f"GMM fit: means={np.round(means, 4)}, stds={np.round(stds, 4)}, "
            f"weights={np.round(weights, 3)}"
        )
        return gmm_fig, status

    @app.callback(
        Output('interactive-pcf-fig', 'figure'),
        Input('interactive-pcf', 'data'),
    )
    def update_interactive_pcf(pcf_data):
        fig = go.Figure()
        if not pcf_data:
            fig.add_annotation(
                text='Create a histogram to compute PCF.',
                xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
            )
            fig.update_layout(height=300, template='plotly_white')
            return fig

        pcf = pcf_data.get('pcf', np.nan)
        mean_phase = pcf_data.get('mean_phase', np.nan)
        fig.add_trace(
            go.Bar(
                x=['PCF'],
                y=[pcf],
                marker=dict(color='#2563eb'),
                name='PCF',
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=[pcf],
                theta=[np.degrees(mean_phase)],
                mode='markers',
                marker=dict(size=12, color='#ef4444'),
                name='Mean phase',
            )
        )
        fig.update_layout(
            height=300,
            template='plotly_white',
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=True),
                angularaxis=dict(direction='counterclockwise'),
            ),
            showlegend=True,
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig
    
    return app


def generate_static_report(output_dir: str, apres_data_path: str) -> str:
    """Generate a static HTML report (no Dash required)."""
    
    results = load_all_results(output_dir)
    apres_mat = loadmat(apres_data_path)
    raw_img = np.array(apres_mat['RawImage'])

    apres_data = {
        'range_img': raw_img,
        'Rcoarse': apres_mat['Rcoarse'].flatten(),
        'time_days': apres_mat['TimeInDays'].flatten(),
    }
    
    fig = create_summary_figure(results, apres_data)
    
    output_file = Path(output_dir) / 'analysis_summary.html'
    
    # Create a more complete HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ApRES Internal Layer Velocity Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1500px;
            margin: auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ApRES Internal Ice Layer Velocity Analysis</h1>
        <p>SiegVent2023 Project</p>
    </div>
    
    <div class="summary">
        <h3>Analysis Summary</h3>
        <p>Based on methodology from Summers et al. (2021) - IGARSS</p>
    </div>
    
    {fig.to_html(include_plotlyjs=True, full_html=False)}
    
    <div class="footer">
        <p>Generated by ApRES Layer Analysis Pipeline</p>
    </div>
</body>
</html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Static report saved: {output_file}")
    return str(output_file)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ApRES Layer Analysis Visualization')
    parser.add_argument('--output-dir', type=str, 
                        default='../../data/apres/layer_analysis',
                        help='Directory with analysis results')
    parser.add_argument('--data', type=str,
                        default='../../data/apres/ImageP2_python.mat',
                        help='Path to ApRES data file')
    parser.add_argument('--port', type=int, default=8050,
                        help='Port for Dash server')
    parser.add_argument('--static', action='store_true',
                        help='Generate static HTML instead of running server')
    
    args = parser.parse_args()
    
    if args.static or not DASH_AVAILABLE:
        # Generate static report
        report_path = generate_static_report(args.output_dir, args.data)
        print(f"\nOpen in browser: file://{Path(report_path).absolute()}")
    else:
        # Run Dash app
        app = create_dash_app(args.output_dir, args.data)
        print("\nStarting visualization server...")
        print(f"Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop the server\n")
        app.run(debug=True, port=args.port)


if __name__ == '__main__':
    main()
