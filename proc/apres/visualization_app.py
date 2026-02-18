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
from scipy.optimize import curve_fit
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
    # Stronger denoising: (5, 11) instead of (3, 7)
    filtered = median_filter(img_db, size=(5, 11))
    
    # Additional smoothing along time axis only
    # Stronger: (1, 9) instead of (1, 5)
    filtered = uniform_filter(filtered, size=(1, 9))
    
    # Convert back from dB
    denoised = np.sqrt(10 ** (filtered / 10))
    
    return denoised


def load_denoised_echogram(output_dir: str, method: str = 'median') -> np.ndarray | None:
    """Load pre-computed denoised echogram if available.
    
    Parameters
    ----------
    output_dir : str
        Directory containing pre-computed files
    method : str
        Denoising method: 'median' or 'svd'
    """
    denoised_path = Path(output_dir) / 'echogram_denoised.mat'
    
    if denoised_path.exists():
        try:
            data = loadmat(str(denoised_path))
            # Try method-specific key first
            key = f'range_img_denoised_{method}'
            if key in data:
                return np.array(data[key])
            # Fall back to generic key
            if 'range_img_denoised' in data:
                return np.array(data['range_img_denoised'])
        except Exception as e:
            print(f"Could not load denoised echogram: {e}")
    return None


def get_denoised_echogram(apres_data: dict, output_dir: str = None, method: str = 'median') -> np.ndarray:
    """Get denoised echogram from pre-computed file or compute on the fly.
    
    Parameters
    ----------
    apres_data : dict
        ApRES data dictionary
    output_dir : str, optional
        Directory containing pre-computed files
    method : str
        Denoising method: 'median' or 'svd' (only median supported for on-the-fly)
    """
    global _denoised_cache
    
    cache_key = (id(apres_data['range_img']), method)
    if cache_key in _denoised_cache:
        return _denoised_cache[cache_key]
    
    # Try to load pre-computed
    if output_dir:
        denoised = load_denoised_echogram(output_dir, method)
        if denoised is not None:
            print(f"Loaded pre-computed denoised echogram ({method})")
            _denoised_cache[cache_key] = denoised
            return denoised
    
    # Fall back to computing (only median supported on-the-fly, SVD is too slow)
    if method == 'svd':
        print("SVD denoising not pre-computed. Run precompute_denoised.py first.")
        print("Falling back to median filter...")
        # Check if we have median cached
        median_key = (id(apres_data['range_img']), 'median')
        if median_key in _denoised_cache:
            return _denoised_cache[median_key]
    
    print("Computing denoised echogram (run precompute_denoised.py for faster startup)...")
    denoised = apply_fast_denoising(apres_data['range_img'])
    # Cache as median
    _denoised_cache[(id(apres_data['range_img']), 'median')] = denoised
    print("Denoising complete.")
    
    return denoised


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
    
    # Extract deep layers from the velocity summary JSON
    # These have tracking_mode == 'deep_segment_stitched'
    results['deep_layers'] = _extract_deep_layers(results.get('velocity_summary'))
    
    return results


def _extract_deep_layers(velocity_summary: dict | None) -> dict:
    """Extract deep layer data from velocity_profile.json into arrays for plotting.
    
    Returns dict with arrays: depths, velocities, r_squared, nye_velocities,
    quality_tier, n_tracked_pts, total_elevated_frac, and metadata.
    """
    empty = {
        'depths': np.array([]), 'velocities': np.array([]),
        'r_squared': np.array([]), 'nye_velocities': np.array([]),
        'quality_tier': np.array([], dtype=int),
        'n_tracked_pts': np.array([], dtype=int),
        'n_layers': 0, 'available': False,
    }
    if velocity_summary is None:
        return empty
    
    layers = velocity_summary.get('layers', [])
    deep = [l for l in layers if l.get('tracking_mode') == 'deep_segment_stitched']
    if not deep:
        return empty
    
    return {
        'depths': np.array([l['depth_m'] for l in deep]),
        'velocities': np.array([l['velocity_m_yr'] for l in deep]),
        'r_squared': np.array([l['r_squared'] for l in deep]),
        'nye_velocities': np.array([l.get('nye_velocity_m_yr', np.nan) for l in deep]),
        'quality_tier': np.array([l.get('quality_tier', 3) for l in deep], dtype=int),
        'n_tracked_pts': np.array([l.get('n_tracked_pts', 0) for l in deep], dtype=int),
        'n_layers': len(deep),
        'available': True,
        'method': velocity_summary.get('deep_layer_method', ''),
        'description': velocity_summary.get('deep_layer_description', ''),
        'nye_model': velocity_summary.get('deep_layer_nye_model', {}),
        'summary': velocity_summary.get('deep_layer_summary', {}),
    }


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
                               denoise_method: str = 'median',
                               output_dir: str = None,
                               range_timeseries: np.ndarray = None,
                               phase_time: np.ndarray = None,
                               color_mode: str = 'amplitude',
                               initial_depths: np.ndarray = None) -> go.Figure:
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
        If True, apply denoising to the echogram
    denoise_method : str
        Denoising method: 'median' or 'svd'
    range_timeseries : np.ndarray, optional
        Layer tracking data [n_layers, n_times] - range change in meters
    phase_time : np.ndarray, optional
        Time array for range_timeseries
    color_mode : str, optional
        'amplitude' (default) or 'phase' to color surface by phase
    initial_depths : np.ndarray, optional
        Actual depths at t=0 for each layer (used for tracking overlay)
    
    Returns
    -------
    go.Figure
        Plotly figure with 3D surface
    """
    # Use denoised data if requested
    if denoise:
        range_img = get_denoised_echogram(apres_data, output_dir, method=denoise_method)
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
    echogram_db = np.clip(echogram_db, -25, 50)
    
    # Create meshgrid for surface
    T, D = np.meshgrid(time_sel, depths_sel)
    
    # Create figure
    fig = go.Figure()
    
    # Add title indicating denoised state and method
    if denoise:
        method_name = denoise_method.upper()
        title_suffix = f" ({method_name} Denoised)"
    else:
        title_suffix = ""
    if color_mode == 'phase' and apres_data.get('raw_complex') is not None:
        title_suffix += " (Phase Colored)"
    
    raw_complex = apres_data.get('raw_complex')
    phase_color = color_mode == 'phase' and raw_complex is not None

    surfacecolor = None
    colorscale = 'Turbo'
    cmin = -25
    cmax = 50
    colorbar_title = 'Amplitude (dB)'
    hovertemplate = 'Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Amp: %{z:.1f} dB<extra></extra>'

    if phase_color:
        complex_sel = raw_complex[depth_mask, :][::depth_subsample, ::time_subsample]
        phase_values = np.angle(complex_sel)
        surfacecolor = phase_values
        colorscale = 'Twilight'
        cmin = -np.pi
        cmax = np.pi
        colorbar_title = 'Phase (rad)'
        hovertemplate = 'Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Amp: %{z:.1f} dB<br>Phase: %{surfacecolor:.2f} rad<extra></extra>'

    # Add 3D surface
    fig.add_trace(
        go.Surface(
            x=T,
            y=D,
            z=echogram_db,
            surfacecolor=surfacecolor,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=dict(text=colorbar_title, side='right'),
                x=1.02,
                len=0.7,
            ),
            opacity=0.9,
            name='Echogram',
            showlegend=False,
            hovertemplate=hovertemplate,
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
                # Use initial depth (t=0) if available, otherwise fall back to mean depth
                if initial_depths is not None and i < len(initial_depths):
                    base_depth = initial_depths[i]
                else:
                    base_depth = depth
                tracked_depths = base_depth + range_timeseries[i, :]
                
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
                                    phase_mode: str = 'wrapped',
                                    range_timeseries: np.ndarray = None,
                                    phase_time: np.ndarray = None,
                                    initial_depths: np.ndarray = None) -> go.Figure:
    """
    Create an interactive 3D surface plot of the phase echogram.

    Phase is computed directly from np.angle(RawImageComplex).
    
    Parameters
    ----------
    apres_data : dict
        ApRES data with 'raw_complex', 'Rcoarse', 'time_days'
    layer_depths : np.ndarray
        Mean depths of detected layers
    lambdac : float
        Center wavelength in ice
    highlighted_layers : list, optional
        Indices of layers to highlight
    depth_range : tuple
        (min_depth, max_depth) to display
    time_subsample : int
        Subsample factor for time dimension
    depth_subsample : int
        Subsample factor for depth dimension
    phase_mode : str
        'wrapped' or 'unwrapped'
    range_timeseries : np.ndarray, optional
        Layer tracking data [n_layers, n_times] - range change in meters
    phase_time : np.ndarray, optional
        Time array for range_timeseries
    initial_depths : np.ndarray, optional
        Actual depths at t=0 for each layer
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
    # If range_timeseries is provided, show tracked layer positions over time
    if layer_depths is not None:
        for i, depth in enumerate(layer_depths):
            if depth < depth_range[0] or depth > depth_range[1]:
                continue

            is_highlighted = highlighted_layers is not None and i in highlighted_layers
            
            # Use tracked depth positions if available
            if range_timeseries is not None and phase_time is not None and i < range_timeseries.shape[0]:
                # Use initial depth (t=0) if available, otherwise fall back to mean depth
                if initial_depths is not None and i < len(initial_depths):
                    base_depth = initial_depths[i]
                else:
                    base_depth = depth
                tracked_depths = base_depth + range_timeseries[i, :]
                
                # Subsample to match the surface time sampling
                time_for_layer = phase_time[::time_subsample]
                depth_for_layer = tracked_depths[::time_subsample]
                
                # Make sure lengths match
                n_points = min(len(time_for_layer), len(time_sel))
                time_for_layer = time_for_layer[:n_points]
                depth_for_layer = depth_for_layer[:n_points]
                
                # Get z values (phase) by finding nearest depth indices for each time point
                z_values = np.zeros(n_points)
                for j in range(n_points):
                    depth_idx = np.argmin(np.abs(depths_sel - depth_for_layer[j]))
                    z_values[j] = phase_values[depth_idx, j] if j < phase_values.shape[1] else phase_values[depth_idx, -1]
            else:
                # Fallback to constant depth (original behavior)
                time_for_layer = time_sel
                depth_idx = np.argmin(np.abs(depths_sel - depth))
                depth_for_layer = np.full_like(time_sel, depths_sel[depth_idx])
                z_values = phase_values[depth_idx, :]

            if is_highlighted:
                fig.add_trace(
                    go.Scatter3d(
                        x=time_for_layer,
                        y=depth_for_layer,
                        z=z_values + 0.1,
                        mode='lines',
                        line=dict(color='#ff4444', width=7),
                        name=f'Layer {depth:.0f}m (tracked)',
                        hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.2f} m<extra></extra>',
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter3d(
                        x=time_for_layer,
                        y=depth_for_layer,
                        z=z_values + 0.05,
                        mode='lines',
                        line=dict(color='rgba(255,255,255,0.4)', width=2),
                        name=f'Layer {depth:.0f}m',
                        visible='legendonly',
                        hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.2f} m<extra></extra>',
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


def create_2d_echogram_figure(
    apres_data: dict, 
    depth_range: tuple = (50, 1200),
    denoise: bool = False,
    denoise_method: str = 'median',
    output_dir: str = None,
    highlighted_layers: list = None,
    layer_depths: np.ndarray = None,
    range_timeseries: np.ndarray = None,
    phase_time: np.ndarray = None,
    initial_depths: np.ndarray = None,
    time_step: int = None,
    depth_step: int = 1,
) -> go.Figure:
    """
    Create a 2D echogram heatmap view with optional denoising and layer overlay.
    
    Parameters
    ----------
    apres_data : dict
        ApRES data dictionary
    depth_range : tuple
        (min_depth, max_depth) to display
    denoise : bool
        If True, apply denoising
    denoise_method : str
        'median' or 'svd'
    output_dir : str
        Output directory for loading precomputed denoised data
    highlighted_layers : list
        Indices of layers to highlight
    layer_depths : np.ndarray
        Mean depths of detected layers (for display labels)
    range_timeseries : np.ndarray
        Layer tracking data [n_layers, n_times]
    phase_time : np.ndarray
        Time array for range_timeseries
    initial_depths : np.ndarray
        Actual depths at t=0 for each layer (used for tracking overlay)
    """
    # Use denoised data if requested
    if denoise:
        range_img = get_denoised_echogram(apres_data, output_dir, method=denoise_method)
    else:
        range_img = apres_data['range_img']
    
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    echogram_db = 10 * np.log10(range_img[depth_mask, :]**2 + 1e-30)
    echogram_db = np.clip(echogram_db, -25, 50)

    step = time_step if time_step is not None else max(1, len(time_days) // 400)
    step = max(1, step)
    depth_step = max(1, depth_step)

    # Build title
    if denoise:
        title_suffix = f" ({denoise_method.upper()} Denoised)"
    else:
        title_suffix = ""

    fig = go.Figure(
        data=go.Heatmap(
            x=time_days[::step],
            y=Rcoarse[depth_mask][::depth_step],
            z=echogram_db[::depth_step, ::step],
            colorscale='Turbo',
            zmin=-25,
            zmax=50,
            colorbar=dict(title='dB'),
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Amp: %{z:.1f} dB<extra></extra>',
        )
    )
    
    # Add layer overlay if requested
    if highlighted_layers and layer_depths is not None:
        for layer_idx in highlighted_layers:
            if layer_idx >= len(layer_depths):
                continue
            
            # Use mean depth for label
            mean_depth = layer_depths[layer_idx]
            
            # Use initial depth (t=0) for tracking if available, otherwise fall back to mean
            if initial_depths is not None and layer_idx < len(initial_depths):
                base_depth = initial_depths[layer_idx]
            else:
                base_depth = mean_depth
            
            # Use tracked position if available
            if range_timeseries is not None and phase_time is not None:
                # Get range change for this layer
                range_change = range_timeseries[layer_idx, :]  # in meters
                tracked_depths = base_depth + range_change
                
                # Plot tracked layer path
                fig.add_trace(
                    go.Scatter(
                        x=phase_time,
                        y=tracked_depths,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Layer {layer_idx} ({mean_depth:.0f}m)',
                        hovertemplate='Time: %{x:.1f} days<br>Tracked depth: %{y:.2f} m<extra></extra>',
                    )
                )
            else:
                # Just show horizontal line at initial depth
                fig.add_hline(
                    y=base_depth,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=f'Layer {layer_idx}',
                    annotation_position='right',
                )

    fig.update_layout(
        title=f'Echogram (2D){title_suffix}',
        height=500,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
        margin=dict(l=40, r=10, t=40, b=40),
        showlegend=len(highlighted_layers) > 0 if highlighted_layers else False,
        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
    )
    fig.update_yaxes(title='Depth (m)', autorange='reversed')
    fig.update_xaxes(title='Time (days)')
    return fig


def create_2d_phase_figure(
    apres_data: dict, 
    depth_range: tuple = (50, 1200),
    phase_mode: str = 'wrapped',
    highlighted_layers: list = None,
    layer_depths: np.ndarray = None,
    range_timeseries: np.ndarray = None,
    phase_time: np.ndarray = None,
    initial_depths: np.ndarray = None,
    time_step: int = None,
    depth_step: int = 1,
) -> go.Figure:
    """
    Create a 2D phase heatmap view with optional layer overlay.
    
    Parameters
    ----------
    apres_data : dict
        ApRES data dictionary (must contain 'raw_complex')
    depth_range : tuple
        (min_depth, max_depth) to display
    phase_mode : str
        'wrapped' for [-π, π] or 'unwrapped' for cumulative phase
    highlighted_layers : list
        Indices of layers to highlight
    layer_depths : np.ndarray
        Mean depths of detected layers
    range_timeseries : np.ndarray
        Layer tracking data [n_layers, n_times]
    phase_time : np.ndarray
        Time array for range_timeseries
    initial_depths : np.ndarray
        Actual depths at t=0 for each layer
    
    Returns
    -------
    go.Figure
        Plotly figure with 2D phase heatmap
    """
    raw_complex = apres_data.get('raw_complex')
    if raw_complex is None:
        # No complex data available
        fig = go.Figure()
        fig.add_annotation(
            text='No complex data available for phase visualization',
            xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=500, template='plotly_white')
        return fig
    
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    complex_sel = raw_complex[depth_mask, :]
    
    # Compute phase
    phase = np.angle(complex_sel)
    
    # Optionally unwrap along time axis
    if phase_mode == 'unwrapped':
        phase = np.unwrap(phase, axis=1)
        colorscale = 'RdBu'
        cmin = None
        cmax = None
        colorbar_title = 'Phase (rad, unwrapped)'
    else:
        colorscale = 'Twilight'
        cmin = -np.pi
        cmax = np.pi
        colorbar_title = 'Phase (rad)'
    
    step = time_step if time_step is not None else max(1, len(time_days) // 400)
    step = max(1, step)
    depth_step = max(1, depth_step)

    fig = go.Figure(
        data=go.Heatmap(
            x=time_days[::step],
            y=Rcoarse[depth_mask][::depth_step],
            z=phase[::depth_step, ::step],
            colorscale=colorscale,
            zmin=cmin,
            zmax=cmax,
            colorbar=dict(title=colorbar_title),
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Phase: %{z:.2f} rad<extra></extra>',
        )
    )
    
    # Add layer overlay if requested
    if highlighted_layers and layer_depths is not None:
        for layer_idx in highlighted_layers:
            if layer_idx >= len(layer_depths):
                continue
            
            mean_depth = layer_depths[layer_idx]
            
            if initial_depths is not None and layer_idx < len(initial_depths):
                base_depth = initial_depths[layer_idx]
            else:
                base_depth = mean_depth
            
            if range_timeseries is not None and phase_time is not None:
                range_change = range_timeseries[layer_idx, :]
                tracked_depths = base_depth + range_change
                
                fig.add_trace(
                    go.Scatter(
                        x=phase_time,
                        y=tracked_depths,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Layer {layer_idx} ({mean_depth:.0f}m)',
                        hovertemplate='Time: %{x:.1f} days<br>Tracked depth: %{y:.2f} m<extra></extra>',
                    )
                )
            else:
                fig.add_hline(
                    y=base_depth,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=f'Layer {layer_idx}',
                    annotation_position='right',
                )

    title_suffix = ' (Wrapped)' if phase_mode == 'wrapped' else ' (Unwrapped)'
    fig.update_layout(
        title=f'Phase (2D){title_suffix}',
        height=500,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
        margin=dict(l=40, r=10, t=40, b=40),
        showlegend=len(highlighted_layers) > 0 if highlighted_layers else False,
        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
    )
    fig.update_yaxes(title='Depth (m)', autorange='reversed')
    fig.update_xaxes(title='Time (days)')
    return fig


def compute_layer_detection_overlay(range_img: np.ndarray, method: str = 'gradient') -> np.ndarray:
    """
    Compute layer detection overlay using various methods.
    
    These methods highlight horizontal layer-like structures in the echogram,
    which can help identify layers for tracking.
    
    Parameters
    ----------
    range_img : np.ndarray
        Amplitude echogram (depth x time)
    method : str
        Detection method:
        - 'gradient': Vertical gradient magnitude (highlights layer boundaries)
        - 'variance': Local variance (highlights coherent structures)
        - 'coherence': Horizontal coherence (high where signal is consistent along layers)
    
    Returns
    -------
    np.ndarray
        Detection overlay (same shape as input), higher values = stronger layer signal
    """
    # Convert to dB for processing
    img_db = 10 * np.log10(range_img**2 + 1e-30)
    
    if method == 'gradient':
        # Vertical gradient magnitude - layer boundaries have high gradient
        from scipy.ndimage import sobel
        gradient = sobel(img_db, axis=0)  # Vertical gradient
        return np.abs(gradient)
    
    elif method == 'variance':
        # Local variance in depth direction - layers have low vertical variance
        # We actually want LOW variance (coherent), so invert it
        from scipy.ndimage import generic_filter
        
        def local_variance(x):
            return np.var(x)
        
        # Compute variance over small vertical window
        variance = generic_filter(img_db, local_variance, size=(5, 1))
        # Invert: low variance = high layer signal
        max_var = np.percentile(variance, 99)
        return max_var - variance
    
    elif method == 'coherence':
        # Horizontal coherence - layers should be coherent along time
        # Use correlation with horizontal neighbors
        from scipy.ndimage import uniform_filter
        
        # Compute local mean and std
        local_mean = uniform_filter(img_db, size=(1, 11))
        local_sq_mean = uniform_filter(img_db**2, size=(1, 11))
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        
        # Coherence: how similar is each point to its horizontal neighborhood
        # Low local std relative to signal = high coherence
        coherence = np.abs(img_db - local_mean) / (local_std + 1e-6)
        # Invert: low deviation = high coherence
        return 1 / (1 + coherence)
    
    else:
        raise ValueError(f"Unknown detection method: {method}")


def create_2d_detection_overlay_figure(
    apres_data: dict, 
    depth_range: tuple = (50, 1200),
    method: str = 'gradient',
    denoise: bool = False,
    denoise_method: str = 'median',
    output_dir: str = None,
    highlighted_layers: list = None,
    layer_depths: np.ndarray = None,
    range_timeseries: np.ndarray = None,
    phase_time: np.ndarray = None,
    initial_depths: np.ndarray = None,
    time_step: int = None,
    depth_step: int = 1,
) -> go.Figure:
    """
    Create a 2D layer detection overlay visualization.
    
    Parameters
    ----------
    apres_data : dict
        ApRES data dictionary
    depth_range : tuple
        (min_depth, max_depth) to display
    method : str
        Detection method: 'gradient', 'variance', or 'coherence'
    denoise : bool
        If True, apply denoising before detection
    denoise_method : str
        'median' or 'svd'
    output_dir : str
        Output directory for loading precomputed denoised data
    highlighted_layers : list
        Indices of layers to highlight
    layer_depths : np.ndarray
        Mean depths of detected layers
    range_timeseries : np.ndarray
        Layer tracking data [n_layers, n_times]
    phase_time : np.ndarray
        Time array for range_timeseries
    initial_depths : np.ndarray
        Actual depths at t=0 for each layer
    
    Returns
    -------
    go.Figure
        Plotly figure with detection overlay
    """
    # Use denoised data if requested
    if denoise:
        range_img = get_denoised_echogram(apres_data, output_dir, method=denoise_method)
    else:
        range_img = apres_data['range_img']
    
    Rcoarse = apres_data['Rcoarse']
    time_days = apres_data['time_days']

    depth_mask = (Rcoarse >= depth_range[0]) & (Rcoarse <= depth_range[1])
    img_sel = range_img[depth_mask, :]
    
    # Compute detection overlay
    detection = compute_layer_detection_overlay(img_sel, method=method)
    
    # Normalize for display
    p1, p99 = np.percentile(detection, [1, 99])
    detection_norm = np.clip((detection - p1) / (p99 - p1 + 1e-6), 0, 1)
    
    step = time_step if time_step is not None else max(1, len(time_days) // 400)
    step = max(1, step)
    depth_step = max(1, depth_step)
    
    # Method-specific settings
    method_names = {
        'gradient': 'Gradient Magnitude',
        'variance': 'Local Coherence (inv. variance)',
        'coherence': 'Horizontal Coherence',
    }
    method_name = method_names.get(method, method.capitalize())

    fig = go.Figure(
        data=go.Heatmap(
            x=time_days[::step],
            y=Rcoarse[depth_mask][::depth_step],
            z=detection_norm[::depth_step, ::step],
            colorscale='Viridis',
            zmin=0,
            zmax=1,
            colorbar=dict(title='Detection strength'),
            hovertemplate='Time: %{x:.1f} days<br>Depth: %{y:.1f} m<br>Detection: %{z:.2f}<extra></extra>',
        )
    )
    
    # Add layer overlay if requested
    if highlighted_layers and layer_depths is not None:
        for layer_idx in highlighted_layers:
            if layer_idx >= len(layer_depths):
                continue
            
            mean_depth = layer_depths[layer_idx]
            
            if initial_depths is not None and layer_idx < len(initial_depths):
                base_depth = initial_depths[layer_idx]
            else:
                base_depth = mean_depth
            
            if range_timeseries is not None and phase_time is not None:
                range_change = range_timeseries[layer_idx, :]
                tracked_depths = base_depth + range_change
                
                fig.add_trace(
                    go.Scatter(
                        x=phase_time,
                        y=tracked_depths,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'Layer {layer_idx} ({mean_depth:.0f}m)',
                        hovertemplate='Time: %{x:.1f} days<br>Tracked depth: %{y:.2f} m<extra></extra>',
                    )
                )
            else:
                fig.add_hline(
                    y=base_depth,
                    line=dict(color='red', width=2, dash='dash'),
                    annotation_text=f'Layer {layer_idx}',
                    annotation_position='right',
                )

    title = f'Layer Detection: {method_name}'
    if denoise:
        title += f' ({denoise_method.upper()} Denoised)'
    
    fig.update_layout(
        title=title,
        height=500,
        template='plotly_white',
        font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
        margin=dict(l=40, r=10, t=40, b=40),
        showlegend=len(highlighted_layers) > 0 if highlighted_layers else False,
        legend=dict(x=1.02, y=1, bgcolor='rgba(255,255,255,0.8)'),
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
            colorscale='Turbo',
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
    deep_layers: dict | None = None,
) -> go.Figure:
    """Create velocity profile with interpolation including lake/sea surface.
    
    Shows observed layer velocities and an interpolated velocity trend
    from the ice surface through internal layers to the lake surface.
    Deep layers (if provided) are shown as distinct diamond/square/triangle
    markers colored by quality tier.
    
    Args:
        results: Analysis results dict with velocity data
        sea_surface_depth: Depth of sea/lake surface (m)
        sea_surface_velocity: Velocity of sea surface (m/yr)
        sea_surface_r_squared: R² of sea surface fit
        ice_thickness: Total ice thickness (m), unused but kept for API compatibility
        deep_layers: Deep layer detection results dict (from _extract_deep_layers)
    """
    from scipy.interpolate import UnivariateSpline
    
    velocity_data = results['velocity']
    depths = velocity_data['depths'].flatten()
    velocities = velocity_data['velocities'].flatten()
    velocities_smooth = velocity_data['velocities_smooth'].flatten()
    r_squared = velocity_data['r_squared'].flatten()
    reliable = velocity_data['reliable'].flatten().astype(bool)
    
    # Load Kingslake uncertainty if available
    has_uncertainty = 'uncertainty_wls' in velocity_data
    if has_uncertainty:
        uncertainty_wls = velocity_data['uncertainty_wls'].flatten()
        uncertainty_kingslake = velocity_data['uncertainty_kingslake'].flatten()
    else:
        uncertainty_wls = None
        uncertainty_kingslake = None
    
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
    
    # Reliable layers (colored by R²) with uncertainty bars if available
    reliable_kwargs = dict(
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
        showlegend=True,
    )
    if has_uncertainty:
        reliable_kwargs['error_x'] = dict(
            type='data',
            array=uncertainty_kingslake[reliable],
            visible=True,
            color='rgba(100,100,100,0.5)',
            thickness=2,
            width=3,
        )
        median_unc = np.median(uncertainty_kingslake[reliable])
        reliable_kwargs['name'] = f'Reliable ± σ (median {median_unc:.3f} m/yr)'
        reliable_kwargs['hovertemplate'] = 'Depth: %{y:.0f}m<br>Velocity: %{x:.3f} ± %{error_x.array:.4f} m/yr<br>R²: %{marker.color:.3f}<extra></extra>'
    else:
        reliable_kwargs['name'] = 'Reliable (R² ≥ 0.3)'
        reliable_kwargs['hovertemplate'] = 'Depth: %{y:.0f}m<br>Velocity: %{x:.3f} m/yr<br>R²: %{marker.color:.3f}<extra></extra>'
    
    fig.add_trace(go.Scatter(**reliable_kwargs), row=1, col=1)
    
    # Deep layers (segment-stitched tracking)
    if deep_layers is not None and deep_layers.get('available', False):
        dl_depths = deep_layers['depths']
        dl_vels = deep_layers['velocities']
        dl_r2 = deep_layers['r_squared']
        dl_tier = deep_layers['quality_tier']
        dl_nye = deep_layers['nye_velocities']
        
        # Tier 1: best detections (R²>0.90, |Δv|<0.05)
        t1 = dl_tier == 1
        if np.any(t1):
            fig.add_trace(go.Scatter(
                x=dl_vels[t1], y=dl_depths[t1],
                mode='markers',
                marker=dict(color='#d6604d', size=10, symbol='diamond',
                            line=dict(color='darkred', width=1)),
                name=f'Deep Tier 1 ({np.sum(t1)})',
                hovertemplate='Depth: %{y:.0f}m<br>v: %{x:.3f} m/yr<br>R²: %{customdata:.3f}<extra>Deep Tier 1</extra>',
                customdata=dl_r2[t1],
            ), row=1, col=1)
        
        # Tier 2: good detections (R²>0.80, |Δv|<0.10)
        t2 = dl_tier == 2
        if np.any(t2):
            fig.add_trace(go.Scatter(
                x=dl_vels[t2], y=dl_depths[t2],
                mode='markers',
                marker=dict(color='#f4a582', size=8, symbol='square',
                            line=dict(color='#d6604d', width=1)),
                name=f'Deep Tier 2 ({np.sum(t2)})',
                hovertemplate='Depth: %{y:.0f}m<br>v: %{x:.3f} m/yr<br>R²: %{customdata:.3f}<extra>Deep Tier 2</extra>',
                customdata=dl_r2[t2],
            ), row=1, col=1)
        
        # Tier 3: fair detections
        t3 = dl_tier == 3
        if np.any(t3):
            fig.add_trace(go.Scatter(
                x=dl_vels[t3], y=dl_depths[t3],
                mode='markers',
                marker=dict(color='#fddbc7', size=7, symbol='triangle-up',
                            line=dict(color='#f4a582', width=1)),
                name=f'Deep Tier 3 ({np.sum(t3)})',
                hovertemplate='Depth: %{y:.0f}m<br>v: %{x:.3f} m/yr<br>R²: %{customdata:.3f}<extra>Deep Tier 3</extra>',
                customdata=dl_r2[t3],
            ), row=1, col=1)
    
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
        
        # Add uncertainty envelope around spline if available
        if has_uncertainty:
            # Interpolate uncertainty to the depth grid
            unc_at_layers = uncertainty_kingslake[reliable]
            # Build uncertainty on the depth grid by interpolation
            sort_rel = np.argsort(depths[reliable])
            unc_interp = np.interp(depth_grid, depths[reliable][sort_rel], unc_at_layers[sort_rel])
            
            # Upper bound (go top to bottom)
            fig.add_trace(
                go.Scatter(
                    x=velocity_interp + unc_interp,
                    y=depth_grid,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip',
                ),
                row=1, col=2
            )
            # Lower bound with fill to upper
            fig.add_trace(
                go.Scatter(
                    x=velocity_interp - unc_interp,
                    y=depth_grid,
                    mode='lines',
                    line=dict(width=0),
                    fill='tonextx',
                    fillcolor='rgba(139, 92, 246, 0.2)',
                    showlegend=True,
                    name=f'Uncertainty envelope (± σ)',
                    hoverinfo='skip',
                ),
                row=1, col=2
            )
        
        # Add reliable layer data points (with uncertainty bars if available)
        interp_layer_kwargs = dict(
            x=velocities[reliable],
            y=depths[reliable],
            mode='markers',
            marker=dict(color='#3b82f6', size=8, line=dict(color='white', width=1)),
            name='Layer velocities',
            showlegend=False,
        )
        if has_uncertainty:
            interp_layer_kwargs['error_x'] = dict(
                type='data',
                array=uncertainty_kingslake[reliable],
                visible=True,
                color='rgba(59,130,246,0.6)',
                thickness=2,
                width=3,
            )
            interp_layer_kwargs['hovertemplate'] = 'Depth: %{y:.0f}m<br>Velocity: %{x:.3f} ± %{error_x.array:.4f} m/yr<extra></extra>'
        else:
            interp_layer_kwargs['hovertemplate'] = 'Depth: %{y:.0f}m<br>Velocity: %{x:.3f} m/yr<extra></extra>'

        fig.add_trace(go.Scatter(**interp_layer_kwargs), row=1, col=2)
        
        # Deep layers on interpolation panel
        if deep_layers is not None and deep_layers.get('available', False):
            fig.add_trace(go.Scatter(
                x=deep_layers['velocities'], y=deep_layers['depths'],
                mode='markers',
                marker=dict(color='#d6604d', size=7, symbol='diamond',
                            line=dict(color='white', width=0.5)),
                name='Deep layers',
                showlegend=False,
                hovertemplate='Depth: %{y:.0f}m<br>v: %{x:.3f} m/yr<extra>Deep</extra>',
            ), row=1, col=2)
        
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
    
    # Don't preload the denoised echogram - lazy load on first use instead
    # This saves startup time since the file is ~1GB
    print("Denoised echogram will be loaded on first use (lazy loading)")
    
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

    # Add deep layer entries (offset by 10000 to distinguish)
    dl = results.get('deep_layers')
    if dl and dl.get('available'):
        tier_stars = {1: '★★★', 2: '★★', 3: '★'}
        for j in range(dl['n_layers']):
            d = dl['depths'][j]
            v = dl['velocities'][j]
            r = dl['r_squared'][j]
            tier = dl['quality_tier'][j]
            stars = tier_stars.get(int(tier), '★')
            layer_options.append({
                'label': f"{d:.0f} m  Deep {stars} (v={v:.3f} m/yr, R\u00b2={r:.3f})",
                'value': 10000 + j,
            })
    
    # Options for 3D highlighting (only reliable layers)
    highlight_options = [
        {
            'label': f"{depths[i]:.0f} m (v={velocities[i]:.2f} m/yr, R²={r_squared[i]:.2f})",
            'value': i
        }
        for i in range(len(depths)) if reliable[i]
    ]

    # Add deep layer entries to highlight options (offset by 10000)
    if dl and dl.get('available'):
        tier_stars = {1: '★★★', 2: '★★', 3: '★'}
        for j in range(dl['n_layers']):
            d_dl = dl['depths'][j]
            v_dl = dl['velocities'][j]
            r_dl = dl['r_squared'][j]
            tier_dl = dl['quality_tier'][j]
            stars_dl = tier_stars.get(int(tier_dl), '★')
            highlight_options.append({
                'label': f"{d_dl:.0f} m  Deep {stars_dl} (v={v_dl:.3f}, R\u00b2={r_dl:.3f})",
                'value': 10000 + j,
            })
    
    # Create initial 3D echograms
    lambdac = 0.5608
    if 'phase' in results and 'lambdac' in results['phase']:
        lambdac = float(np.array(results['phase']['lambdac']).squeeze())
    rcoarse = apres_data['Rcoarse']

    # Initial figures (using default depth range)
    initial_3d_fig = create_3d_echogram_figure(
        apres_data,
        depths,
        highlighted_layers=None,
        depth_range=(50, 250),
        color_mode='amplitude',
    )
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
    deep_layers = results.get('deep_layers', None)
    velocity_profile_fig = create_velocity_profile_with_sea_surface(
        results, 
        sea_surface_depth=SEA_SURFACE_DEPTH,
        sea_surface_velocity=sea_surface_velocity,
        sea_surface_r_squared=sea_surface_r_squared,
        ice_thickness=SEA_SURFACE_DEPTH,
        deep_layers=deep_layers,
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
                                    {'label': 'Median', 'value': 'median'},
                                    {'label': 'SVD', 'value': 'svd'},
                                ],
                                value='original',
                                inline=True,
                                style={'marginTop': '6px'},
                                inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                            ),
                        ], style={'marginRight': '20px'}),
                        html.Div([
                            html.Label('Color:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.RadioItems(
                                id='echogram-3d-color-mode',
                                options=[
                                    {'label': 'Amplitude', 'value': 'amplitude'},
                                    {'label': 'Phase', 'value': 'phase'},
                                ],
                                value='amplitude',
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
                        html.Div([
                            html.Label('Resolution:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='resolution-3d',
                                options=[
                                    {'label': 'Low (fast)', 'value': 'low'},
                                    {'label': 'Medium', 'value': 'medium'},
                                    {'label': 'High', 'value': 'high'},
                                    {'label': 'Ultra (slow)', 'value': 'ultra'},
                                ],
                                value='medium',
                                clearable=False,
                                style={'width': '140px', 'marginTop': '5px'},
                            ),
                        ], style={'width': '160px'}),
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
                    # Shared controls for all 2D views
                    html.H3('2D Visualizations', style={'color': theme['text'], 'marginBottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.Label('View type:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.RadioItems(
                                id='view-type-2d',
                                options=[
                                    {'label': 'Amplitude', 'value': 'amplitude'},
                                    {'label': 'Phase', 'value': 'phase'},
                                    {'label': 'Layer Detection', 'value': 'detection'},
                                ],
                                value='amplitude',
                                inline=True,
                                style={'marginTop': '6px'},
                                inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                            ),
                        ], style={'marginRight': '20px'}),
                        html.Div([
                            html.Label('Processing:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.RadioItems(
                                id='denoise-toggle-2d',
                                options=[
                                    {'label': 'Original', 'value': 'original'},
                                    {'label': 'Median', 'value': 'median'},
                                    {'label': 'SVD', 'value': 'svd'},
                                ],
                                value='original',
                                inline=True,
                                style={'marginTop': '6px'},
                                inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                            ),
                        ], style={'marginRight': '20px'}),
                        html.Div([
                            html.Label('Overlay layers:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='highlight-layers-2d',
                                options=highlight_options,
                                value=[],
                                multi=True,
                                placeholder='Select layers...',
                                style={'width': '100%', 'marginTop': '5px'},
                            ),
                        ], style={'flex': '1', 'maxWidth': '300px'}),
                        html.Div([
                            html.Label('Resolution:', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='resolution-2d',
                                options=[
                                    {'label': 'Low (fast)', 'value': 'low'},
                                    {'label': 'Medium', 'value': 'medium'},
                                    {'label': 'High', 'value': 'high'},
                                    {'label': 'Full', 'value': 'full'},
                                ],
                                value='high',
                                clearable=False,
                                style={'width': '140px', 'marginTop': '5px'},
                            ),
                        ], style={'width': '160px'}),
                    ], style={'display': 'flex', 'alignItems': 'flex-start', 'marginBottom': '10px', 'flexWrap': 'wrap', 'gap': '15px'}),
                    
                    # View-specific controls (conditionally shown)
                    html.Div(id='phase-controls-2d', children=[
                        html.Label('Phase mode:', style={'fontWeight': '600', 'color': theme['text'], 'marginRight': '10px'}),
                        dcc.RadioItems(
                            id='phase-mode-2d',
                            options=[
                                {'label': 'Wrapped', 'value': 'wrapped'},
                                {'label': 'Unwrapped', 'value': 'unwrapped'},
                            ],
                            value='wrapped',
                            inline=True,
                            inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                        ),
                    ], style={'marginBottom': '15px', 'display': 'none'}),
                    
                    html.Div(id='detection-controls-2d', children=[
                        html.Label('Detection method:', style={'fontWeight': '600', 'color': theme['text'], 'marginRight': '10px'}),
                        dcc.RadioItems(
                            id='detection-method-2d',
                            options=[
                                {'label': 'Gradient', 'value': 'gradient'},
                                {'label': 'Coherence', 'value': 'coherence'},
                                {'label': 'Variance', 'value': 'variance'},
                            ],
                            value='gradient',
                            inline=True,
                            inputStyle={'marginRight': '6px', 'marginLeft': '12px'},
                        ),
                        html.P(
                            'Gradient: highlights layer edges. Coherence: highlights consistent horizontal structures. Variance: highlights areas with low vertical variation.',
                            style={'color': theme['muted'], 'fontSize': '11px', 'marginTop': '5px'},
                        ),
                    ], style={'marginBottom': '15px', 'display': 'none'}),
                    
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
                    dcc.Graph(id='lake-surface-figure-tab', figure=lake_surface_fig, style={'height': '360px'}),
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
                        html.Div([
                            html.Label('Histogram bins', style={'fontWeight': '600', 'color': theme['text']}),
                            dcc.Input(
                                id='interactive-nbins',
                                type='number',
                                value=100,
                                min=10,
                                max=500,
                                step=10,
                                style={'width': '100px', 'marginTop': '6px'},
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
            # Methods Documentation Tab
            dcc.Tab(label='Methods', children=[
                html.Div([
                    html.H2("Mathematical Methods Documentation", 
                           style={'color': theme['text'], 'marginBottom': '24px', 'fontWeight': '600'}),
                    
                    # Section 1: FMCW Radar Processing
                    html.Div([
                        html.H3("1. FMCW Radar Signal Processing", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.H4("1.1 Transmitted Signal", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "The ApRES transmits a frequency-modulated continuous wave (FMCW) chirp signal:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
$$s_{tx}(t) = A \cos\left(2\pi f_0 t + \pi K t^2\right)$$

where:
- $f_0$ = Start frequency (200 MHz)
- $K = B/T$ = Chirp rate (Hz/s)
- $B$ = Bandwidth (200 MHz)
- $T$ = Chirp duration (~1s)
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                        
                        html.H4("1.2 Received Signal & Beat Frequency", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "After reflection from a target at range R, the received signal is mixed with the transmitted signal. ",
                            "The beat frequency is proportional to range:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
$$f_b = \frac{2RK}{c_i} = \frac{2RB}{c_i T}$$

where $c_i = c_0/\sqrt{\varepsilon_r}$ is the wave speed in ice ($\varepsilon_r \approx 3.18$).
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                        
                        html.H4("1.3 Range Processing", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "Range profiles are computed via FFT of the windowed beat signal (Brennan et al., 2014):"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
**Coarse Range** (bin centers):
$$R_{coarse}[n] = \frac{n \cdot c_i}{2 B p}$$

where $p$ is the zero-padding factor (default: 8).

**Blackman Window**: Applied before FFT to reduce spectral leakage (-58 dB sidelobes):
$$w[n] = 0.42 - 0.5\cos\left(\frac{2\pi n}{N-1}\right) + 0.08\cos\left(\frac{4\pi n}{N-1}\right)$$

**Phase Correction**: Remove systematic phase offset at bin center:
$$\phi_{ref} = 2\pi f_c \tau - \frac{K\tau^2}{2}, \quad \tau = \frac{n}{Bp}$$

**Fine Range** from phase:
$$R_{fine} = \frac{\phi}{\frac{4\pi}{\lambda_c} - \frac{4 R_{coarse} K}{c_i^2}}$$
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 2: Layer Detection
                    html.Div([
                        html.H3("2. Layer Detection", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.H4("2.1 Peak Detection", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "Internal layers are detected as local maxima in the time-averaged amplitude profile:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
**Time-averaged amplitude**:
$$\bar{A}(R) = \frac{1}{N_t} \sum_{t=1}^{N_t} |S(R, t)|$$

**Peak detection criteria**:
1. Local maximum: $\bar{A}(R_i) > \bar{A}(R_{i\pm1})$
2. Prominence threshold: peak stands out from surrounding baseline
3. Minimum separation: $\Delta R > R_{min}$ (typically 2m)
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 3: Phase Tracking
                    html.Div([
                        html.H3("3. Phase-Based Layer Tracking", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.H4("3.1 Sub-bin Peak Interpolation", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "Layer peaks are tracked with sub-bin precision using parabolic interpolation:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
Given amplitudes at bins $[n-1, n, n+1]$ denoted $[A_{-1}, A_0, A_{+1}]$:

$$\delta = \frac{A_{-1} - A_{+1}}{2(A_{-1} - 2A_0 + A_{+1})}$$

**Sub-bin position**: $n_{sub} = n + \delta$

**Sub-bin range**: $R_{sub} = R_{coarse}[n] + \delta \cdot \Delta R_{bin}$
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                        
                        html.H4("3.2 Phase-based Fine Range", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "Phase is extracted at the sub-bin position and converted to fine range correction:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
**Phase extraction** (linear interpolation at sub-bin position):
$$\phi_{sub}(t) = \phi[n](t) + \delta \cdot (\phi[n+1](t) - \phi[n](t))$$

**Range from phase** (two-way travel):
$$\Delta R_{fine}(t) = \frac{\lambda_c}{4\pi} \cdot \phi_{sub}(t)$$

where $\lambda_c \approx 0.56$ m is the center wavelength in ice.

**Sensitivity**: $\lambda_c / 4\pi \approx 0.045$ m per radian, or **~0.28 m per $2\pi$ wrap**
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                        
                        html.H4("3.3 Phase Unwrapping", style={'color': theme['text'], 'marginTop': '16px'}),
                        dcc.Markdown(r'''
Phase is unwrapped to remove $2\pi$ discontinuities:

$$\phi_{unwrap}(t) = \phi(t) + 2\pi \cdot n_{wrap}(t)$$

where $n_{wrap}$ is chosen to minimize $|\phi_{unwrap}(t) - \phi_{unwrap}(t-1)|$.

**Robust unwrapping**: Uses median filtering to detect and correct wrap errors.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 4: Velocity Estimation
                    html.Div([
                        html.H3("4. Velocity Estimation", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.P([
                            "Vertical velocity at each layer is estimated from the linear trend in range change:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
**Linear regression**:
$$R(t) = v \cdot t + R_0$$

The slope $v$ gives the vertical velocity (m/day), converted to m/yr:
$$v_{yr} = v \cdot 365.25$$

**Quality metric**: $R^2$ coefficient of determination:
$$R^2 = 1 - \frac{\sum_t (R(t) - \hat{R}(t))^2}{\sum_t (R(t) - \bar{R})^2}$$

Layers with $R^2 > 0.5$ are considered reliable.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 5: Denoising
                    html.Div([
                        html.H3("5. Echogram Denoising", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.H4("5.1 Median Filter Denoising", style={'color': theme['text'], 'marginTop': '16px'}),
                        dcc.Markdown(r'''
Applied in log (dB) domain with horizontal emphasis to preserve layer structure:

1. **Convert to dB**: $A_{dB} = 10 \log_{10}(A^2)$
2. **Median filter**: kernel size $(5, 11)$ - 5 bins in depth, 11 in time
3. **Uniform smoothing**: kernel size $(1, 9)$ - time-only smoothing
4. **Convert back**: $A_{filtered} = \sqrt{10^{A_{dB}/10}}$
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                        
                        html.H4("5.2 SVD Denoising", style={'color': theme['text'], 'marginTop': '16px'}),
                        html.P([
                            "Singular Value Decomposition separates coherent signal from incoherent noise:"
                        ], style={'color': theme['text']}),
                        dcc.Markdown(r'''
The echogram matrix $\mathbf{A} \in \mathbb{R}^{N_R \times N_t}$ is decomposed:

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Low-rank approximation** (keeping $k$ components):
$$\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

**Physical interpretation**:
- Coherent layers appear in the **first few singular vectors** (large $\sigma_i$)
- Incoherent noise spreads across **many small singular values**
- With $k=2$: captures ~99.9% of layer signal, removes nearly all noise

**Block processing**: Applied in depth blocks of 500 bins for computational efficiency.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 6: Deep Layer Detection
                    html.Div([
                        html.H3("6. Deep Layer Detection", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        
                        html.P([
                            "Standard phase-coherent tracking loses sensitivity below ~785 m due to "
                            "signal attenuation and increased noise. We recover velocity estimates in "
                            "the deep ice column (785–1082 m) using an amplitude-gated, segment-stitched "
                            "phase tracking method guided by Nye model priors."
                        ], style={'color': theme['text'], 'marginBottom': '16px'}),

                        html.H4("6.1 Amplitude-Gated Segment Tracking", style={'color': theme['text'], 'marginTop': '16px'}),
                        dcc.Markdown(r'''
For each depth bin in the deep column, the method identifies time intervals where the 
amplitude exceeds a dynamic threshold:

$$A_{threshold}(z) = f \cdot \text{median}(A(z, :))$$

where $f = 1.5$ is the amplitude gating factor. Contiguous intervals above this threshold 
form **elevated segments** — time windows where the signal-to-noise ratio is sufficient 
for reliable phase tracking.

Within each segment, phase differences between consecutive measurements are unwrapped 
and converted to displacement rates using:

$$v = \frac{\lambda_c}{4\pi \Delta t} \cdot \Delta\phi$$

A linear fit to displacement vs. time yields the segment velocity. Long segments 
(≥20 measurements) are particularly valuable for constraining the velocity.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),

                        html.H4("6.2 Segment Stitching with Nye Priors", style={'color': theme['text'], 'marginTop': '16px'}),
                        dcc.Markdown(r'''
Multiple elevated segments at the same depth bin are stitched together by aligning 
their wrap offsets. The Nye model velocity provides a prior expectation:

$$v_{Nye}(z) = v_0 + \dot{\varepsilon} \cdot z$$

with $v_0 = 0.0453$ m/yr and $\dot{\varepsilon} = 5.95 \times 10^{-4}$ /yr 
(from the reliable layer fit). Wrap-offset candidates are selected as:

$$n^* = \arg\min_n \left| v_{segment} + n \cdot v_{wrap} - v_{Nye}(z) \right|$$

After offset alignment, all segments are merged and a global linear fit produces the 
final velocity estimate and $R^2$ statistic for that depth bin.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),

                        html.H4("6.3 Quality Tiers and Filtering", style={'color': theme['text'], 'marginTop': '16px'}),
                        dcc.Markdown(r'''
Deep layer detections are classified into three quality tiers:

| Tier | Criteria | Symbol |
|------|----------|--------|
| **Tier 1** (★★★) | $R^2 > 0.90$ and $|\Delta v| < 0.05$ m/yr | Red diamond |
| **Tier 2** (★★) | $R^2 > 0.80$ and $|\Delta v| < 0.10$ m/yr | Orange square |
| **Tier 3** (★) | Remaining ($R^2 > 0.70$) | Tan triangle |

where $\Delta v = v_{measured} - v_{Nye}(z)$ is the velocity residual relative to the 
Nye model. Adjacent detections within 3 m are merged into discrete layers. Velocity 
outliers ($|\Delta v| > 0.10$ m/yr) are removed.

**Validation**: Permutation tests (1000 randomised phase-difference series per bin) 
confirm that all retained layers have $p < 0.001$, ruling out spurious tracking of noise.

**Result**: 38 deep layers detected from 785–1082 m depth (9 Tier 1, 18 Tier 2, 11 Tier 3), 
with the deepest detection at 1082.3 m — only 12 m above the ice–lake interface.
                        ''', mathjax=True, style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                    # Section 7: References
                    html.Div([
                        html.H3("7. References", 
                               style={'color': theme['accent'], 'marginBottom': '12px', 'borderBottom': f"2px solid {theme['accent']}", 'paddingBottom': '8px'}),
                        dcc.Markdown('''
- **Brennan, P. V., Lok, L. B., Nicholls, K., & Corr, H. (2014)**. Phase-sensitive FMCW radar system for high-precision Antarctic ice shelf profile monitoring. *IET Radar, Sonar & Navigation*, 8(7), 776-786.

- **Nicholls, K. W., Corr, H. F. J., Stewart, C. L., Lok, L. B., Brennan, P. V., & Vaughan, D. G. (2015)**. A ground-based radar for measuring vertical strain rates and time-varying basal melt rates in ice sheets and shelves. *Journal of Glaciology*, 61(230), 1079-1087.

- **Summers, P., Peters, S., Bienert, N., et al. (2021)**. ApRES Processing Methods. *IGARSS 2021*.

- **Stewart, C. L. (2018)**. Ice-ocean interactions beneath the north-western Ross Ice Shelf, Antarctica. PhD Thesis, University of Cambridge.
                        ''', style={'backgroundColor': theme['bg'], 'padding': '16px', 'borderRadius': '8px', 'border': f"1px solid {theme['border']}"}),
                    ], style={'marginBottom': '32px'}),
                    
                ], style={
                    'margin': '18px',
                    'padding': '32px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                    'maxWidth': '900px',
                }),
            ]),
            # ── Interpretation Tab ──────────────────────────────────
            dcc.Tab(label='Interpretation', children=[
                html.Div([
                    html.H3("Scenario: Uniform Horizontal Flow over Sloping Layers",
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P(
                        "If all measured vertical velocity is caused by horizontal ice flow over "
                        "sloping internal layers, what slope would each layer need? "
                        "Adjust the assumed surface velocity to explore different scenarios.",
                        style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'},
                    ),
                    html.Div([
                        html.Label("Horizontal surface velocity (m/yr):",
                                  style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                        dcc.Slider(
                            id='interp-horizontal-velocity',
                            min=50, max=400, step=5, value=226,
                            marks={v: f'{v}' for v in [50, 100, 150, 200, 226, 250, 300, 350, 400]},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                        html.P(
                            "GPS LA17 measured 225.9 m/yr. Satellite-derived values are ~200 m/yr.",
                            style={'color': theme['muted'], 'fontSize': '12px', 'marginTop': '4px', 'fontStyle': 'italic'},
                        ),
                    ], style={'marginBottom': '24px', 'maxWidth': '700px'}),
                    html.Div([
                        html.Div([
                            html.Label("Layers to display:",
                                      style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                            dcc.Dropdown(
                                id='interp-layer-filter',
                                options=[
                                    {'label': 'All layers', 'value': 'all'},
                                    {'label': 'Reliable only (R² > 0.3)', 'value': 'reliable'},
                                    {'label': 'Unreliable only', 'value': 'unreliable'},
                                ],
                                value='reliable',
                                clearable=False,
                                style={'width': '260px', 'display': 'inline-block', 'verticalAlign': 'middle'},
                            ),
                        ], style={'display': 'inline-flex', 'alignItems': 'center', 'marginRight': '40px'}),
                        html.Div([
                            html.Label("Depth range (m):",
                                      style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                            dcc.RangeSlider(
                                id='interp-depth-range',
                                min=0, max=1100, step=10,
                                value=[0, 1100],
                                marks={v: f'{v}' for v in [0, 200, 400, 600, 800, 1000]},
                                tooltip={'placement': 'bottom', 'always_visible': True},
                            ),
                        ], style={'display': 'inline-flex', 'alignItems': 'center', 'minWidth': '450px'}),
                    ], style={'marginBottom': '20px', 'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}),
                    html.Div([
                        html.Label("Slope exaggeration in cross-section:",
                                  style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                        dcc.Slider(
                            id='interp-exaggeration',
                            min=1, max=500, step=1, value=1,
                            marks={1: '1×', 50: '50×', 100: '100×', 200: '200×', 300: '300×', 500: '500×'},
                            tooltip={'placement': 'bottom', 'always_visible': True},
                        ),
                        html.P(
                            "At 1× the true slopes are shown (very small — max ~0.17°). "
                            "Increase to make layer tilts visible in the cross-section.",
                            style={'color': theme['muted'], 'fontSize': '12px', 'marginTop': '4px', 'fontStyle': 'italic'},
                        ),
                    ], style={'marginBottom': '20px', 'maxWidth': '700px'}),
                    dcc.Graph(id='interp-slope-analysis', style={'height': '750px'}),
                ], style={
                    'margin': '20px 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                # ── Nye Model: Incompressibility ──────────────────────
                html.Div([
                    html.H3("Scenario: Nye Model (Incompressibility)",
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P([
                        "The incompressibility condition ",
                        html.Span("∂u/∂x + ∂v/∂y + ∂w/∂z = 0", style={'fontFamily': 'monospace', 'fontWeight': '600'}),
                        " requires vertical strain rate to balance horizontal strain rates. "
                        "If horizontal strain rates are uniform with depth, the vertical velocity "
                        "profile is linear (Nye model): ",
                        html.Span("w(z) = w_s + ε̇_zz · z", style={'fontFamily': 'monospace', 'fontWeight': '600'}),
                        ". Since ",
                        html.Span("∂u/∂x ≫ ∂v/∂y", style={'fontFamily': 'monospace'}),
                        " (along-flow strain dominates), ",
                        html.Span("ε̇_zz ≈ −∂u/∂x", style={'fontFamily': 'monospace', 'fontWeight': '600'}),
                        ".",
                    ], style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Fit using:",
                                  style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                        dcc.RadioItems(
                            id='nye-fit-layers',
                            options=[
                                {'label': 'Reliable layers only', 'value': 'reliable'},
                                {'label': 'All layers', 'value': 'all'},
                                {'label': 'Reliable + lake surface', 'value': 'reliable_lake'},
                                {'label': 'Reliable + deep layers', 'value': 'reliable_deep'},
                                {'label': 'Reliable + deep + lake', 'value': 'reliable_deep_lake'},
                            ],
                            value='reliable_deep_lake',
                            inline=True,
                            style={'display': 'inline-flex', 'gap': '20px'},
                            labelStyle={'color': theme['text'], 'fontSize': '14px'},
                        ),
                    ], style={'marginBottom': '20px'}),
                    dcc.Graph(id='nye-model-analysis', style={'height': '650px'}),
                ], style={
                    'margin': '20px 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                # ── Dansgaard-Johnsen Model ────────────────────────────
                html.Div([
                    html.H3("Scenario: Dansgaard-Johnsen Model",
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P([
                        "The Dansgaard-Johnsen model extends Nye by allowing strain rate to "
                        "decrease linearly to zero in a basal shear layer of thickness ",
                        html.Span("h", style={'fontFamily': 'monospace', 'fontWeight': '600'}),
                        ". Above this layer, strain rate is constant (like Nye). "
                        "This is particularly relevant over a subglacial lake where basal drag is zero. "
                        "Parameters (w\u209b, \u03b5\u0307\u2080, h) are fit automatically.",
                    ], style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Fit using:",
                                  style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                        dcc.RadioItems(
                            id='dj-fit-layers',
                            options=[
                                {'label': 'Reliable layers only', 'value': 'reliable'},
                                {'label': 'All layers', 'value': 'all'},
                                {'label': 'Reliable + lake surface', 'value': 'reliable_lake'},
                                {'label': 'Reliable + deep layers', 'value': 'reliable_deep'},
                                {'label': 'Reliable + deep + lake', 'value': 'reliable_deep_lake'},
                            ],
                            value='reliable_deep_lake',
                            inline=True,
                            style={'display': 'inline-flex', 'gap': '20px'},
                            labelStyle={'color': theme['text'], 'fontSize': '14px'},
                        ),
                    ], style={'marginBottom': '20px'}),
                    dcc.Graph(id='dj-model-analysis', style={'height': '650px'}),
                ], style={
                    'margin': '20px 18px 30px 18px',
                    'padding': '24px',
                    'backgroundColor': theme['panel'],
                    'borderRadius': '16px',
                    'border': f"1px solid {theme['border']}",
                    'boxShadow': theme['shadow'],
                }),
                # ── Lliboutry Model ───────────────────────────────────
                html.Div([
                    html.H3("Scenario: Lliboutry Shape-Function Model",
                           style={'color': theme['text'], 'marginBottom': '8px', 'fontWeight': '600'}),
                    html.P([
                        "The Lliboutry model parameterises the velocity profile with a shape exponent ",
                        html.Span("p", style={'fontFamily': 'monospace', 'fontWeight': '600'}),
                        " related to Glen's flow law. ",
                        html.Span("p = 1", style={'fontFamily': 'monospace'}),
                        " gives a linear (Nye) profile, while ",
                        html.Span("p \u2192 \u221e", style={'fontFamily': 'monospace'}),
                        " gives plug flow (all deformation at the bed). "
                        "Parameters (w\u209b, w_b, p) are fit automatically.",
                    ], style={'color': theme['muted'], 'fontSize': '14px', 'marginBottom': '16px'}),
                    html.Div([
                        html.Label("Fit using:",
                                  style={'fontWeight': '600', 'marginRight': '12px', 'color': theme['text']}),
                        dcc.RadioItems(
                            id='llib-fit-layers',
                            options=[
                                {'label': 'Reliable layers only', 'value': 'reliable'},
                                {'label': 'All layers', 'value': 'all'},
                                {'label': 'Reliable + lake surface', 'value': 'reliable_lake'},
                                {'label': 'Reliable + deep layers', 'value': 'reliable_deep'},
                                {'label': 'Reliable + deep + lake', 'value': 'reliable_deep_lake'},
                            ],
                            value='reliable_deep_lake',
                            inline=True,
                            style={'display': 'inline-flex', 'gap': '20px'},
                            labelStyle={'color': theme['text'], 'fontSize': '14px'},
                        ),
                    ], style={'marginBottom': '20px'}),
                    dcc.Graph(id='llib-model-analysis', style={'height': '650px'}),
                ], style={
                    'margin': '20px 18px 30px 18px',
                    'padding': '24px',
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

        # ── Deep layer detail (offset indices >= 10000) ──
        if layer_idx >= 10000:
            return _build_deep_layer_detail(layer_idx - 10000)

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

    def _compute_deep_layer_tracking(deep_idx):
        """Compute tracked depth over time for a deep layer using segment-stitched reconstruction.

        Returns (time_days, tracked_depths) where tracked_depths = base_depth + displacement.
        Points where tracking is unavailable are NaN.
        """
        from scipy.ndimage import uniform_filter1d

        dl = results.get('deep_layers')
        if dl is None or not dl.get('available') or deep_idx >= dl['n_layers']:
            return None, None

        depth = float(dl['depths'][deep_idx])
        nye_v = float(dl['nye_velocities'][deep_idx])

        Rcoarse = apres_data['Rcoarse']
        bin_idx = np.argmin(np.abs(Rcoarse - depth))
        raw_complex_data = apres_data.get('raw_complex')
        time_days_local = apres_data['time_days']
        if raw_complex_data is None:
            return None, None

        z = raw_complex_data[bin_idx, :]
        amp = np.abs(z)
        amp_smooth = uniform_filter1d(amp, size=15)
        amp_factor = 1.3
        threshold = amp_factor * np.median(amp_smooth)
        elevated = amp_smooth > threshold

        lambdac_local = 0.5608
        if 'phase' in results and 'lambdac' in results['phase']:
            lambdac_local = float(np.asarray(results['phase']['lambdac']).flat[0])
        wrap_period = lambdac_local / 2.0

        min_pts = 30
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(len(elevated)):
            if elevated[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not elevated[i] and in_seg:
                if i - seg_start >= min_pts:
                    segments.append((seg_start, i))
                in_seg = False
        if in_seg and len(elevated) - seg_start >= min_pts:
            segments.append((seg_start, len(elevated)))

        nye_tol = 0.15
        good_segs = []
        for s_start, s_end in segments:
            phi_seg = np.angle(z[s_start:s_end])
            phi_unwrap = np.unwrap(phi_seg)
            t_seg = time_days_local[s_start:s_end]
            disp_seg = phi_unwrap * lambdac_local / (4 * np.pi)
            slope_s, _, r_val, _, _ = stats.linregress(t_seg, disp_seg)
            v_seg = slope_s * 365.25
            r2_seg = r_val ** 2
            if abs(v_seg - nye_v) < nye_tol and r2_seg > 0.3:
                good_segs.append({
                    'start': s_start, 'end': s_end,
                    'velocity': v_seg, 'r2': r2_seg,
                    'disp': disp_seg, 'time': t_seg,
                })

        displacement = np.full(len(time_days_local), np.nan)
        if good_segs:
            good_segs.sort(key=lambda s: s['start'])
            first = good_segs[0]
            base_disp = first['disp'] - first['disp'][0]
            displacement[first['start']:first['end']] = base_disp
            for k in range(1, len(good_segs)):
                prev = good_segs[k - 1]
                curr = good_segs[k]
                last_disp = displacement[prev['end'] - 1]
                last_time = time_days_local[prev['end'] - 1]
                dt = time_days_local[curr['start']] - last_time
                nye_predicted = last_disp + (nye_v / 365.25) * dt
                curr_disp_rel = curr['disp'] - curr['disp'][0]
                n_wraps = round((nye_predicted - curr_disp_rel[0]) / wrap_period)
                offset = n_wraps * wrap_period
                displacement[curr['start']:curr['end']] = curr_disp_rel + offset

        tracked_depths = depth + displacement  # depth + displacement in meters
        return time_days_local, tracked_depths

    def _build_deep_layer_detail(deep_idx):
        """Build Layer Detail figure for a deep layer using segment-stitched reconstruction."""
        from scipy.ndimage import uniform_filter1d

        dl = results.get('deep_layers')
        if dl is None or not dl.get('available') or deep_idx >= dl['n_layers']:
            fig = go.Figure()
            fig.add_annotation(text='Deep layer data not available', showarrow=False,
                               xref='paper', yref='paper', x=0.5, y=0.5)
            return fig

        depth = float(dl['depths'][deep_idx])
        velocity = float(dl['velocities'][deep_idx])
        r_sq = float(dl['r_squared'][deep_idx])
        nye_v = float(dl['nye_velocities'][deep_idx])
        tier = int(dl['quality_tier'][deep_idx])
        tier_stars = {1: '★★★ Tier 1', 2: '★★ Tier 2', 3: '★ Tier 3'}

        # Find bin index from depth
        Rcoarse = apres_data['Rcoarse']
        bin_idx = np.argmin(np.abs(Rcoarse - depth))

        # Get raw complex time series at this bin
        raw_complex_data = apres_data.get('raw_complex')
        time_days_local = apres_data['time_days']
        if raw_complex_data is None:
            fig = go.Figure()
            fig.add_annotation(text='Raw complex data not available', showarrow=False,
                               xref='paper', yref='paper', x=0.5, y=0.5)
            return fig

        z = raw_complex_data[bin_idx, :]

        # Amplitude
        amp = np.abs(z)
        amp_db = 10 * np.log10(amp**2 + 1e-30)
        amp_smooth = uniform_filter1d(amp, size=15)

        # Amplitude threshold
        amp_factor = 1.3
        threshold = amp_factor * np.median(amp_smooth)
        elevated = amp_smooth > threshold

        # Reconstruct segment-stitched displacement
        lambdac = 0.5608
        if 'phase' in results and 'lambdac' in results['phase']:
            lambdac = float(np.asarray(results['phase']['lambdac']).flat[0])
        wrap_period = lambdac / 2.0

        # Find segments
        min_pts = 30
        segments = []
        in_seg = False
        seg_start = 0
        for i in range(len(elevated)):
            if elevated[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not elevated[i] and in_seg:
                if i - seg_start >= min_pts:
                    segments.append((seg_start, i))
                in_seg = False
        if in_seg and len(elevated) - seg_start >= min_pts:
            segments.append((seg_start, len(elevated)))

        # Track and stitch segments
        nye_tol = 0.15
        good_segs = []
        for s_start, s_end in segments:
            phi_seg = np.angle(z[s_start:s_end])
            phi_unwrap = np.unwrap(phi_seg)
            t_seg = time_days_local[s_start:s_end]
            disp_seg = phi_unwrap * lambdac / (4 * np.pi)
            slope_s, _, r_val, _, _ = stats.linregress(t_seg, disp_seg)
            v_seg = slope_s * 365.25
            r2_seg = r_val ** 2
            if abs(v_seg - nye_v) < nye_tol and r2_seg > 0.3:
                good_segs.append({
                    'start': s_start, 'end': s_end,
                    'velocity': v_seg, 'r2': r2_seg,
                    'disp': disp_seg, 'time': t_seg,
                })

        # Build displacement array
        displacement = np.full(len(time_days_local), np.nan)
        if good_segs:
            good_segs.sort(key=lambda s: s['start'])
            first = good_segs[0]
            base_disp = first['disp'] - first['disp'][0]
            displacement[first['start']:first['end']] = base_disp
            for k in range(1, len(good_segs)):
                prev = good_segs[k - 1]
                curr = good_segs[k]
                last_disp = displacement[prev['end'] - 1]
                last_time = time_days_local[prev['end'] - 1]
                dt = time_days_local[curr['start']] - last_time
                nye_predicted = last_disp + (nye_v / 365.25) * dt
                curr_disp_rel = curr['disp'] - curr['disp'][0]
                n_wraps = round((nye_predicted - curr_disp_rel[0]) / wrap_period)
                offset = n_wraps * wrap_period
                displacement[curr['start']:curr['end']] = curr_disp_rel + offset

        disp_cm = displacement * 100  # m → cm

        # Linear fit
        valid = ~np.isnan(disp_cm)
        if np.sum(valid) > 10:
            slope_fit, intercept_fit, _, _, _ = stats.linregress(time_days_local[valid], disp_cm[valid])
            fit_line = slope_fit * time_days_local + intercept_fit
        else:
            fit_line = np.full_like(time_days_local, np.nan, dtype=float)

        # Build 3-panel figure
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                f'Segment-Stitched Displacement at {depth:.0f}m',
                f'Amplitude at {depth:.0f}m',
                f'Elevated Segments',
            ),
            column_widths=[0.45, 0.30, 0.25],
        )

        # Panel 1: Displacement with fit
        fig.add_trace(
            go.Scatter(x=time_days_local, y=disp_cm, mode='markers', name='Tracked',
                       marker=dict(color='#3498db', size=2, opacity=0.6)),
            row=1, col=1,
        )
        if not np.all(np.isnan(fit_line)):
            fig.add_trace(
                go.Scatter(x=time_days_local, y=fit_line, mode='lines',
                           name=f'Fit: {velocity:.3f} m/yr',
                           line=dict(color='#e74c3c', width=2, dash='dash')),
                row=1, col=1,
            )
        # Nye prediction line
        nye_cm = nye_v / 365.25 * time_days_local * 100  # Nye displacement in cm, from 0
        if np.sum(valid) > 0:
            nye_cm = nye_cm - nye_cm[0] + (intercept_fit if not np.all(np.isnan(fit_line)) else 0)
        fig.add_trace(
            go.Scatter(x=time_days_local, y=nye_cm, mode='lines',
                       name=f'Nye: {nye_v:.3f} m/yr',
                       line=dict(color='#94a3b8', width=1.5, dash='dot')),
            row=1, col=1,
        )
        fig.update_xaxes(title='Time (days)', row=1, col=1)
        fig.update_yaxes(title='Displacement (cm)', row=1, col=1)

        # Panel 2: Amplitude
        fig.add_trace(
            go.Scatter(x=time_days_local, y=amp_db, mode='lines', name='Amplitude',
                       line=dict(color='#27ae60', width=0.8)),
            row=1, col=2,
        )
        thresh_db = 10 * np.log10(threshold**2 + 1e-30)
        fig.add_hline(y=thresh_db, line=dict(color='#f97316', dash='dash', width=1.5),
                      annotation_text=f'Threshold ({amp_factor}× median)',
                      annotation_position='top right', row=1, col=2)
        fig.update_xaxes(title='Time (days)', row=1, col=2)
        fig.update_yaxes(title='Amplitude (dB)', row=1, col=2)

        # Panel 3: Segment map
        seg_color = np.where(elevated, 1.0, 0.0)
        fig.add_trace(
            go.Scatter(x=time_days_local, y=seg_color, mode='lines',
                       fill='tozeroy', name='Elevated',
                       line=dict(color='#f97316', width=0.5),
                       fillcolor='rgba(249, 115, 22, 0.3)'),
            row=1, col=3,
        )
        for s in good_segs:
            t_mid = time_days_local[s['start']:s['end']].mean()
            fig.add_annotation(
                x=t_mid, y=1.05, text=f"R²={s['r2']:.2f}",
                showarrow=False, font=dict(size=8, color='#0ea5e9'),
                row=1, col=3,
            )
        fig.update_xaxes(title='Time (days)', row=1, col=3)
        fig.update_yaxes(title='Above threshold', tickvals=[0, 1],
                         ticktext=['Below', 'Above'], row=1, col=3)

        stars = tier_stars.get(tier, '★ Tier 3')
        n_pts = int(dl['n_tracked_pts'][deep_idx])
        fig.update_layout(
            title=dict(
                text=(
                    f'Deep Layer at {depth:.0f}m ({stars}): '
                    f'v = {velocity:.3f} m/yr, R² = {r_sq:.3f}, '
                    f'{len(good_segs)} segments, {n_pts} tracked pts'
                ),
                font=dict(size=13),
            ),
            height=380,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        )
        return fig
    
    # Callback for 3D echogram layer highlighting, depth range, and denoising
    @app.callback(
        Output('echogram-3d', 'figure'),
        Input('highlight-layers', 'value'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
        Input('denoise-toggle', 'value'),
        Input('echogram-3d-color-mode', 'value'),
        Input('resolution-3d', 'value'),
    )
    def update_3d_echogram(highlighted_indices, depth_start, depth_interval, denoise_mode, color_mode, resolution):
        if highlighted_indices is None:
            highlighted_indices = []
        # Separate deep layer indices (>= 10000) from standard
        std_indices = [i for i in highlighted_indices if i < 10000]
        deep_indices = [i - 10000 for i in highlighted_indices if i >= 10000]
        highlighted_indices = std_indices
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
        
        # Parse denoise mode - now 'original', 'median', or 'svd'
        denoise = denoise_mode not in (None, 'original')
        denoise_method = denoise_mode if denoise else 'median'  # Default to median
        color_mode = color_mode or 'amplitude'
        
        # Resolution mapping for 3D: (time_subsample, depth_subsample)
        res_map = {'low': (20, 10), 'medium': (10, 5), 'high': (5, 2), 'ultra': (2, 1)}
        time_sub, depth_sub = res_map.get(resolution or 'medium', (10, 5))
        
        depths = velocity_data['depths'].flatten()
        
        # Get range_timeseries for layer tracking
        phase_data = results.get('phase', {})
        range_ts = phase_data.get('range_timeseries', None)
        phase_time = phase_data.get('time_days', None)
        if phase_time is not None:
            phase_time = phase_time.flatten()
        
        # Get initial depths if available (for correct layer overlay positioning)
        initial_depths = phase_data.get('initial_depths', None)
        if initial_depths is not None:
            initial_depths = initial_depths.flatten()
        
        fig = create_3d_echogram_figure(
            apres_data, 
            depths, 
            highlighted_layers=highlighted_indices,
            depth_range=tuple(depth_range),
            time_subsample=time_sub,
            depth_subsample=depth_sub,
            denoise=denoise,
            denoise_method=denoise_method,
            output_dir=output_dir,
            range_timeseries=range_ts,
            phase_time=phase_time,
            color_mode=color_mode,
            initial_depths=initial_depths,
        )

        # Overlay deep layers with tracked depth path
        dl = results.get('deep_layers')
        if dl and dl.get('available') and deep_indices:
            tier_colors = {1: '#ef4444', 2: '#f97316', 3: '#d4a574'}
            tier_symbols = {1: '★★★', 2: '★★', 3: '★'}
            Rcoarse_local = apres_data['Rcoarse']
            range_img_local = apres_data['range_img']
            depth_mask_local = (Rcoarse_local >= depth_range[0]) & (Rcoarse_local <= depth_range[1])
            echo_sel = range_img_local[depth_mask_local, :][::depth_sub, ::time_sub]
            echo_db_local = 10 * np.log10(echo_sel**2 + 1e-30)
            echo_db_local = np.clip(echo_db_local, -25, 50)
            depths_sel_local = Rcoarse_local[depth_mask_local][::depth_sub]
            for di in deep_indices:
                if di >= dl['n_layers']:
                    continue
                dd = float(dl['depths'][di])
                if dd < depth_range[0] or dd > depth_range[1]:
                    continue
                tier = int(dl['quality_tier'][di])
                clr = tier_colors.get(tier, '#d4a574')
                stars = tier_symbols.get(tier, '★')
                # Use segment-stitched tracking for overlay
                t_track, d_track = _compute_deep_layer_tracking(di)
                t_vals = apres_data['time_days'][::time_sub]
                if t_track is not None and d_track is not None and not np.all(np.isnan(d_track)):
                    # Interpolate tracked depths to subsampled time grid
                    d_interp = np.interp(t_vals, t_track, d_track, left=np.nan, right=np.nan)
                    # Where tracking has NaN, use NaN for gaps
                    nan_mask = np.isnan(d_track)
                    nan_at_t = np.interp(t_vals, t_track, nan_mask.astype(float)) > 0.5
                    d_interp[nan_at_t] = np.nan
                else:
                    d_interp = np.full_like(t_vals, dd)
                # Look up z-values at tracked depth for each time step
                z_vals = np.zeros(len(t_vals))
                for ti in range(len(t_vals)):
                    d_at_t = d_interp[ti] if not np.isnan(d_interp[ti]) else dd
                    didx = np.argmin(np.abs(depths_sel_local - d_at_t))
                    z_vals[ti] = echo_db_local[min(didx, echo_db_local.shape[0]-1), ti]
                fig.add_trace(
                    go.Scatter3d(
                        x=t_vals, y=d_interp, z=z_vals + 3,
                        mode='lines',
                        line=dict(color=clr, width=8),
                        name=f'Deep {dd:.0f}m {stars}',
                        connectgaps=False,
                        hovertemplate=f'Deep layer {dd:.0f}m<br>Time: %{{x:.1f}} days<br>Depth: %{{y:.1f}} m<extra></extra>',
                    )
                )

        return fig

    @app.callback(
        Output('echogram-3d-phase', 'figure'),
        Input('highlight-layers', 'value'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
        Input('phase-wrap-toggle', 'value'),
        Input('resolution-3d', 'value'),
    )
    def update_3d_phase_echogram(highlighted_indices, depth_start, depth_interval, phase_mode, resolution):
        if highlighted_indices is None:
            highlighted_indices = []
        # Separate deep layer indices
        std_indices = [i for i in highlighted_indices if i < 10000]
        deep_indices_phase = [i - 10000 for i in highlighted_indices if i >= 10000]
        highlighted_indices = std_indices
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

        # Resolution mapping for 3D: (time_subsample, depth_subsample)
        res_map = {'low': (20, 10), 'medium': (10, 5), 'high': (5, 2), 'ultra': (2, 1)}
        time_sub, depth_sub = res_map.get(resolution or 'medium', (10, 5))

        # Get range_timeseries for layer tracking
        phase_data = results.get('phase', {})
        range_ts = phase_data.get('range_timeseries', None)
        pt = phase_data.get('time_days', None)
        if pt is not None:
            pt = pt.flatten()
        
        # Get initial depths if available (for correct layer overlay positioning)
        init_depths = phase_data.get('initial_depths', None)
        if init_depths is not None:
            init_depths = init_depths.flatten()

        fig = create_3d_phase_echogram_figure(
            apres_data,
            depths,
            lambdac,
            highlighted_layers=highlighted_indices,
            depth_range=depth_range,
            time_subsample=time_sub,
            depth_subsample=depth_sub,
            phase_mode=phase_mode,
            range_timeseries=range_ts,
            phase_time=pt,
            initial_depths=init_depths,
        )

        # Overlay deep layers on 3D phase echogram
        dl = results.get('deep_layers')
        if dl and dl.get('available') and deep_indices_phase:
            tier_colors = {1: '#ef4444', 2: '#f97316', 3: '#d4a574'}
            tier_symbols = {1: '★★★', 2: '★★', 3: '★'}
            res_map_p = {'low': (20, 10), 'medium': (10, 5), 'high': (5, 2), 'ultra': (2, 1)}
            ts_p, ds_p = res_map_p.get(resolution or 'medium', (10, 5))
            Rcoarse_local = apres_data['Rcoarse']
            range_img_local = apres_data['range_img']
            depth_mask_local = (Rcoarse_local >= depth_range[0]) & (Rcoarse_local <= depth_range[1])
            echo_sel = range_img_local[depth_mask_local, :][::ds_p, ::ts_p]
            echo_db_local = np.clip(10 * np.log10(echo_sel**2 + 1e-30), -25, 50)
            depths_sel_local = Rcoarse_local[depth_mask_local][::ds_p]
            t_vals = apres_data['time_days'][::ts_p]
            for di in deep_indices_phase:
                if di >= dl['n_layers']:
                    continue
                dd = float(dl['depths'][di])
                if dd < depth_range[0] or dd > depth_range[1]:
                    continue
                tier = int(dl['quality_tier'][di])
                clr = tier_colors.get(tier, '#d4a574')
                stars = tier_symbols.get(tier, '★')
                # Use segment-stitched tracking for overlay
                t_track, d_track = _compute_deep_layer_tracking(di)
                if t_track is not None and d_track is not None and not np.all(np.isnan(d_track)):
                    d_interp = np.interp(t_vals, t_track, d_track, left=np.nan, right=np.nan)
                    nan_mask = np.isnan(d_track)
                    nan_at_t = np.interp(t_vals, t_track, nan_mask.astype(float)) > 0.5
                    d_interp[nan_at_t] = np.nan
                else:
                    d_interp = np.full_like(t_vals, dd)
                # Look up z-values at tracked depth
                z_vals = np.zeros(len(t_vals))
                for ti in range(len(t_vals)):
                    d_at_t = d_interp[ti] if not np.isnan(d_interp[ti]) else dd
                    didx_t = np.argmin(np.abs(depths_sel_local - d_at_t))
                    z_vals[ti] = echo_db_local[min(didx_t, echo_db_local.shape[0]-1), ti]
                fig.add_trace(
                    go.Scatter3d(
                        x=t_vals, y=d_interp, z=z_vals + 3,
                        mode='lines', line=dict(color=clr, width=8),
                        name=f'Deep {dd:.0f}m {stars}',
                        connectgaps=False,
                        hovertemplate=f'Deep layer {dd:.0f}m<br>Time: %{{x:.1f}} days<br>Depth: %{{y:.1f}} m<extra></extra>',
                    )
                )

        return fig

    @app.callback(
        Output('echogram-2d', 'figure'),
        Output('phase-controls-2d', 'style'),
        Output('detection-controls-2d', 'style'),
        Input('depth-start-slider', 'value'),
        Input('depth-interval-input', 'value'),
        Input('view-type-2d', 'value'),
        Input('denoise-toggle-2d', 'value'),
        Input('highlight-layers-2d', 'value'),
        Input('phase-mode-2d', 'value'),
        Input('detection-method-2d', 'value'),
        Input('resolution-2d', 'value'),
    )
    def update_2d_views(depth_start, depth_interval, view_type, denoise_mode, 
                        highlighted_indices, phase_mode, detection_method, resolution):
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
        
        if highlighted_indices is None:
            highlighted_indices = []
        # Separate deep layer indices
        std_indices_2d = [i for i in highlighted_indices if i < 10000]
        deep_indices_2d = [i - 10000 for i in highlighted_indices if i >= 10000]
        highlighted_indices = std_indices_2d
        
        # Resolution mapping for 2D: (time_step, depth_step)
        res_map = {'low': (10, 5), 'medium': (5, 2), 'high': (2, 1), 'full': (1, 1)}
        time_step, depth_step = res_map.get(resolution or 'high', (2, 1))
        
        # Get layer tracking data
        phase_data = results.get('phase', {})
        range_ts = phase_data.get('range_timeseries', None)
        phase_time = phase_data.get('time_days', None)
        if phase_time is not None:
            phase_time = phase_time.flatten()
        
        # Get initial depths if available (for correct layer overlay positioning)
        initial_depths = phase_data.get('initial_depths', None)
        if initial_depths is not None:
            initial_depths = initial_depths.flatten()
        
        # Parse denoise mode
        denoise = denoise_mode not in (None, 'original')
        denoise_method = denoise_mode if denoise else 'median'
        
        # View-specific control styles
        phase_style = {'marginBottom': '15px', 'display': 'none'}
        detection_style = {'marginBottom': '15px', 'display': 'none'}
        
        view_type = view_type or 'amplitude'
        
        if view_type == 'phase':
            phase_style = {'marginBottom': '15px', 'display': 'flex', 'alignItems': 'center'}
            fig = create_2d_phase_figure(
                apres_data, 
                depth_range=depth_range,
                phase_mode=phase_mode or 'wrapped',
                highlighted_layers=highlighted_indices,
                layer_depths=velocity_data['depths'].flatten(),
                range_timeseries=range_ts,
                phase_time=phase_time,
                initial_depths=initial_depths,
                time_step=time_step,
                depth_step=depth_step,
            )
        elif view_type == 'detection':
            detection_style = {'marginBottom': '15px', 'display': 'block'}
            fig = create_2d_detection_overlay_figure(
                apres_data, 
                depth_range=depth_range,
                method=detection_method or 'gradient',
                denoise=denoise,
                denoise_method=denoise_method,
                output_dir=output_dir,
                highlighted_layers=highlighted_indices,
                layer_depths=velocity_data['depths'].flatten(),
                range_timeseries=range_ts,
                phase_time=phase_time,
                initial_depths=initial_depths,
                time_step=time_step,
                depth_step=depth_step,
            )
        else:
            # Default: amplitude
            fig = create_2d_echogram_figure(
                apres_data, 
                depth_range=depth_range,
                denoise=denoise,
                denoise_method=denoise_method,
                output_dir=output_dir,
                highlighted_layers=highlighted_indices,
                layer_depths=velocity_data['depths'].flatten(),
                range_timeseries=range_ts,
                phase_time=phase_time,
                initial_depths=initial_depths,
                time_step=time_step,
                depth_step=depth_step,
            )
        
        # Overlay deep layers on 2D echogram (tracked path, not constant line)
        dl = results.get('deep_layers')
        if dl and dl.get('available') and deep_indices_2d:
            tier_colors = {1: '#ef4444', 2: '#f97316', 3: '#d4a574'}
            tier_symbols = {1: '★★★', 2: '★★', 3: '★'}
            for di in deep_indices_2d:
                if di >= dl['n_layers']:
                    continue
                dd = float(dl['depths'][di])
                if dd < depth_range[0] or dd > depth_range[1]:
                    continue
                tier = int(dl['quality_tier'][di])
                clr = tier_colors.get(tier, '#d4a574')
                stars = tier_symbols.get(tier, '★')
                # Use segment-stitched tracking for the overlay
                t_track, d_track = _compute_deep_layer_tracking(di)
                if t_track is not None and d_track is not None and not np.all(np.isnan(d_track)):
                    fig.add_trace(
                        go.Scatter(
                            x=t_track,
                            y=d_track,
                            mode='lines',
                            line=dict(color=clr, width=2.5),
                            name=f'Deep {dd:.0f}m {stars}',
                            connectgaps=False,
                            hovertemplate='Time: %{x:.1f} days<br>Tracked depth: %{y:.2f} m<extra></extra>',
                        )
                    )
                else:
                    # Fallback to horizontal line if tracking fails
                    fig.add_hline(
                        y=dd,
                        line=dict(color=clr, width=2.5, dash='dash'),
                        annotation_text=f'Deep {dd:.0f}m {stars}',
                        annotation_position='right',
                        annotation_font=dict(color=clr, size=10),
                    )

        return fig, phase_style, detection_style

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
                            nbinsx=100,
                            name='Phase Δ',
                            marker=dict(color='#2563eb'),
                            opacity=0.8,
                        )
                    )
                    if sigma > 0:
                        x_line = np.linspace(values.min(), values.max(), 300)
                        pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
                        pdf_scaled = pdf * (values.size * (values.max() - values.min()) / 100)
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
        State('interactive-nbins', 'value'),
    )
    def build_interactive_histogram(n_clicks, depth_value, unwrap_value, window_m, weight_value, n_bins):
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

        # Use provided n_bins or default to 100
        n_bins = int(n_bins) if n_bins else 100
        
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=phase_diff,
                nbinsx=n_bins,
                name='Phase Δ',
                marker=dict(color='#2563eb'),
                opacity=0.8,
            )
        )
        if sigma > 0:
            x_line = np.linspace(phase_diff.min(), phase_diff.max(), 300)
            pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
            pdf_scaled = pdf * (phase_diff.size * (phase_diff.max() - phase_diff.min()) / n_bins)
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

    # ── Interpretation callback: slope scenario analysis ──────────
    @app.callback(
        Output('interp-slope-analysis', 'figure'),
        Input('interp-horizontal-velocity', 'value'),
        Input('interp-layer-filter', 'value'),
        Input('interp-depth-range', 'value'),
        Input('interp-exaggeration', 'value'),
    )
    def update_slope_interpretation(u_h, layer_filter, depth_range, exag):
        if u_h is None:
            u_h = 226.0
        u_h = float(u_h)
        if layer_filter is None:
            layer_filter = 'reliable'
        if depth_range is None:
            depth_range = [0, 1100]
        depth_min, depth_max = depth_range
        if exag is None:
            exag = 1
        exag = float(exag)

        velocity_data_local = results['velocity']
        depths_local = velocity_data_local['depths'].flatten()
        vel_local = velocity_data_local['velocities'].flatten()
        r_sq_local = velocity_data_local['r_squared'].flatten()
        rel_local = velocity_data_local['reliable'].flatten().astype(bool)
        vel_smooth = velocity_data_local['velocities_smooth'].flatten()

        # Load uncertainties if available
        has_unc = 'uncertainty_kingslake' in velocity_data_local
        if has_unc:
            unc = velocity_data_local['uncertainty_kingslake'].flatten()

        # Calculate implied slopes: θ = arctan(w / u_h)
        slopes_deg = np.degrees(np.arctan(vel_local / u_h))
        slopes_smooth_deg = np.degrees(np.arctan(vel_smooth / u_h))

        # Sea/lake surface values
        ice_thickness = SEA_SURFACE_DEPTH
        if not np.isnan(sea_surface_velocity):
            bed_slope_deg = np.degrees(np.arctan(sea_surface_velocity / u_h))
        else:
            bed_slope_deg = np.nan

        # Build figure: 3 panels
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Implied Layer Slopes",
                "Ice Column Cross-Section",
                "Slope vs Measured Velocity",
            ),
            column_widths=[0.35, 0.35, 0.30],
            horizontal_spacing=0.08,
        )

        # ─── Panel 1: Slopes vs depth ───────────────────────────
        # Unreliable
        fig.add_trace(go.Scatter(
            x=slopes_deg[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#cbd5e1', size=6),
            name='Unreliable',
            hovertemplate='Depth: %{y:.0f} m<br>Slope: %{x:.3f}°<extra></extra>',
        ), row=1, col=1)

        # Reliable, colored by R²
        fig.add_trace(go.Scatter(
            x=slopes_deg[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis',
                size=9,
                cmin=0.3, cmax=1.0,
                colorbar=dict(title='R²', x=0.30, len=0.5, y=0.5),
                line=dict(color='white', width=0.5),
            ),
            name='Reliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>Slope: %{x:.4f}°<br>R²: %{marker.color:.2f}<extra></extra>',
        ), row=1, col=1)

        # Smoothed trend
        fig.add_trace(go.Scatter(
            x=slopes_smooth_deg, y=depths_local,
            mode='lines',
            line=dict(color='#8b5cf6', width=2),
            name='Smoothed trend',
        ), row=1, col=1)

        # Bed slope marker
        if not np.isnan(bed_slope_deg):
            fig.add_trace(go.Scatter(
                x=[bed_slope_deg], y=[ice_thickness],
                mode='markers',
                marker=dict(color='#ef4444', size=14, symbol='star',
                            line=dict(color='white', width=2)),
                name=f'Lake surface ({bed_slope_deg:.3f}°)',
                hovertemplate=f'Lake surface<br>Depth: {ice_thickness:.0f} m<br>Slope: {bed_slope_deg:.4f}°<extra></extra>',
            ), row=1, col=1)

        # Zero line
        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dash', width=1), row=1, col=1)

        fig.update_xaxes(title='Implied slope (degrees)', row=1, col=1)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)

        # ─── Panel 2: Cross-section schematic ────────────────────
        # Draw a schematic cross-section of the ice column
        # The ApRES sits at the surface; layers are tilted by their implied slope
        section_width = 200  # meters horizontal extent of drawing
        half_w = section_width / 2

        # Ice surface (horizontal)
        fig.add_trace(go.Scatter(
            x=[-half_w, half_w], y=[0, 0],
            mode='lines',
            line=dict(color='#0ea5e9', width=3),
            name='Ice surface',
            showlegend=True,
            hoverinfo='skip',
        ), row=1, col=2)

        # Bed / lake surface
        if not np.isnan(bed_slope_deg):
            bed_slope_rad = np.arctan(sea_surface_velocity / u_h)
            bed_y_left = ice_thickness - half_w * np.tan(bed_slope_rad * exag)
            bed_y_right = ice_thickness + half_w * np.tan(bed_slope_rad * exag)
        else:
            bed_y_left = ice_thickness
            bed_y_right = ice_thickness
        fig.add_trace(go.Scatter(
            x=[-half_w, half_w], y=[bed_y_left, bed_y_right],
            mode='lines',
            line=dict(color='#ef4444', width=3),
            name='Lake surface',
            showlegend=True,
            hoverinfo='skip',
        ), row=1, col=2)

        # Fill ice body
        fig.add_trace(go.Scatter(
            x=[-half_w, half_w, half_w, -half_w, -half_w],
            y=[0, 0, bed_y_right, bed_y_left, 0],
            mode='lines',
            fill='toself',
            fillcolor='rgba(14, 165, 233, 0.08)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=2)

        # Fill water body beneath
        fig.add_trace(go.Scatter(
            x=[-half_w, half_w, half_w, -half_w, -half_w],
            y=[bed_y_left, bed_y_right, ice_thickness * 1.05, ice_thickness * 1.05, bed_y_left],
            mode='lines',
            fill='toself',
            fillcolor='rgba(239, 68, 68, 0.10)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=2)

        # Draw tilted internal layers based on filter
        if layer_filter == 'all':
            draw_mask = (depths_local >= depth_min) & (depths_local <= depth_max)
        elif layer_filter == 'unreliable':
            draw_mask = (~rel_local) & (depths_local >= depth_min) & (depths_local <= depth_max)
        else:  # 'reliable'
            draw_mask = rel_local & (depths_local >= depth_min) & (depths_local <= depth_max)

        draw_depths = depths_local[draw_mask]
        draw_slopes = np.arctan(vel_local[draw_mask] / u_h)
        draw_r2 = r_sq_local[draw_mask]
        draw_reliable = rel_local[draw_mask]
        draw_vel = vel_local[draw_mask]

        for j in range(len(draw_depths)):
            d = draw_depths[j]
            y_left = d - half_w * np.tan(draw_slopes[j] * exag)
            y_right = d + half_w * np.tan(draw_slopes[j] * exag)
            r2_val = draw_r2[j]
            is_rel = draw_reliable[j]

            # Color by depth (blue→purple gradient), faded if unreliable
            d_min_draw = draw_depths.min() if len(draw_depths) > 0 else 0
            d_max_draw = draw_depths.max() if len(draw_depths) > 0 else 1
            frac = (d - d_min_draw) / max(1, d_max_draw - d_min_draw)
            r_c = int(59 + frac * (139 - 59))
            g_c = int(130 + frac * (92 - 130))
            b_c = int(246 + frac * (246 - 246))
            color = f'rgb({r_c},{g_c},{b_c})' if is_rel else f'rgba({r_c},{g_c},{b_c},0.4)'

            fig.add_trace(go.Scatter(
                x=[-half_w, half_w], y=[y_left, y_right],
                mode='lines',
                line=dict(color=color, width=1.5 if is_rel else 0.8,
                          dash='solid' if is_rel else 'dot'),
                showlegend=False,
                hovertemplate=(
                    f'Depth: {d:.0f} m<br>'
                    f'Slope: {np.degrees(draw_slopes[j]):.4f}°<br>'
                    f'Velocity: {draw_vel[j]:.3f} m/yr<br>'
                    f'R²: {r2_val:.2f}<br>'
                    f"{'Reliable' if is_rel else 'Unreliable'}"
                    '<extra></extra>'
                ),
            ), row=1, col=2)

        # ApRES marker at surface center
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(color='#f59e0b', size=14, symbol='triangle-down',
                        line=dict(color='white', width=2)),
            text=['ApRES'],
            textposition='top center',
            textfont=dict(size=11, color='#f59e0b'),
            showlegend=False,
            hovertemplate='ApRES instrument<br>Surface<extra></extra>',
        ), row=1, col=2)

        # Flow arrow
        fig.add_annotation(
            x=half_w * 0.7, y=ice_thickness * 0.08,
            ax=-half_w * 0.3, ay=ice_thickness * 0.08,
            xref='x2', yref='y2', axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor='#0f172a',
        )
        fig.add_annotation(
            x=half_w * 0.3, y=ice_thickness * 0.04,
            text=f'Flow → {u_h:.0f} m/yr',
            showarrow=False,
            font=dict(size=12, color='#0f172a', family='Inter'),
            xref='x2', yref='y2',
        )

        # Label "Water" below bed
        fig.add_annotation(
            x=0, y=ice_thickness * 1.02,
            text='Subglacial lake',
            showarrow=False,
            font=dict(size=11, color='#ef4444', family='Inter'),
            xref='x2', yref='y2',
        )

        # Exaggeration annotation (only shown when > 1×)
        if exag > 1:
            fig.add_annotation(
                x=0.5, y=1.08,
                xref='x2 domain', yref='y2 domain',
                text=f'Slopes exaggerated ×{exag:.0f}',
                showarrow=False,
                font=dict(size=10, color='#94a3b8'),
            )

        fig.update_xaxes(
            title='Horizontal distance (m)',
            range=[-half_w * 1.1, half_w * 1.1],
            row=1, col=2,
        )
        fig.update_yaxes(
            title='Depth (m)',
            autorange='reversed',
            range=[-ice_thickness * 0.1, ice_thickness * 1.08],
            row=1, col=2,
        )

        # ─── Panel 3: Slope vs measured velocity ──────────────────
        fig.add_trace(go.Scatter(
            x=vel_local[rel_local], y=slopes_deg[rel_local],
            mode='markers',
            marker=dict(
                color=depths_local[rel_local],
                colorscale='Viridis',
                size=8,
                colorbar=dict(title='Depth (m)', x=1.0, len=0.5, y=0.5),
                line=dict(color='white', width=0.5),
            ),
            name='Layers',
            showlegend=False,
            hovertemplate='Velocity: %{x:.3f} m/yr<br>Slope: %{y:.4f}°<br>Depth: %{marker.color:.0f} m<extra></extra>',
        ), row=1, col=3)

        # Theoretical line: slope = arctan(v / u_h)
        v_range = np.linspace(
            min(0, float(np.nanmin(vel_local[rel_local])) - 0.05),
            float(np.nanmax(vel_local[rel_local])) + 0.05,
            100,
        )
        theory_slope = np.degrees(np.arctan(v_range / u_h))
        fig.add_trace(go.Scatter(
            x=v_range, y=theory_slope,
            mode='lines',
            line=dict(color='#94a3b8', width=1, dash='dash'),
            name=f'θ = arctan(v / {u_h:.0f})',
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=3)

        fig.add_hline(y=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=3)

        fig.update_xaxes(title='Measured vertical velocity (m/yr)', row=1, col=3)
        fig.update_yaxes(title='Implied slope (degrees)', row=1, col=3)

        # ─── Layout ──────────────────────────────────────────────
        # Summary stats annotation
        rel_slopes_deg = slopes_deg[rel_local]
        stats_text = (
            f"<b>Scenario: uniform u<sub>h</sub> = {u_h:.0f} m/yr</b><br>"
            f"Layers: {int(np.sum(rel_local))} reliable<br>"
            f"Slope range: {np.nanmin(rel_slopes_deg):.3f}° to {np.nanmax(rel_slopes_deg):.3f}°<br>"
            f"Median slope: {np.nanmedian(rel_slopes_deg):.3f}°<br>"
        )
        if not np.isnan(bed_slope_deg):
            stats_text += f"Bed slope: {bed_slope_deg:.3f}°"

        fig.add_annotation(
            x=0.02, y=0.02,
            xref='x domain', yref='y domain',
            text=stats_text,
            showarrow=False,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=6,
            align='left',
        )

        fig.update_layout(
            height=750,
            showlegend=True,
            legend=dict(x=0.01, y=0.15, bgcolor='rgba(255,255,255,0.9)',
                       bordercolor='#e2e8f0', borderwidth=1),
            template='plotly_white',
            font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
            margin=dict(l=60, r=60, t=60, b=50),
            title=dict(
                text=f'Layer Slope Interpretation — assuming u<sub>h</sub> = {u_h:.0f} m/yr',
                font=dict(size=15),
            ),
        )

        return fig

    # ── Nye Model callback: incompressibility-based linear fit ────
    @app.callback(
        Output('nye-model-analysis', 'figure'),
        Input('nye-fit-layers', 'value'),
    )
    def update_nye_model(fit_mode):
        if fit_mode is None:
            fit_mode = 'reliable_lake'

        velocity_data_local = results['velocity']
        depths_local = velocity_data_local['depths'].flatten()
        vel_local = velocity_data_local['velocities'].flatten()
        r_sq_local = velocity_data_local['r_squared'].flatten()
        rel_local = velocity_data_local['reliable'].flatten().astype(bool)
        vel_smooth = velocity_data_local['velocities_smooth'].flatten()

        ice_thickness = SEA_SURFACE_DEPTH
        dl = deep_layers  # deep layer data

        # Select data for fitting
        if fit_mode == 'all':
            fit_mask = np.ones(len(depths_local), dtype=bool)
            fit_depths = depths_local[fit_mask]
            fit_vel = vel_local[fit_mask]
        elif fit_mode == 'reliable':
            fit_mask = rel_local
            fit_depths = depths_local[fit_mask]
            fit_vel = vel_local[fit_mask]
        elif fit_mode == 'reliable_deep':
            fit_mask = rel_local
            fit_depths = depths_local[fit_mask].copy()
            fit_vel = vel_local[fit_mask].copy()
            if dl is not None and dl.get('available', False):
                fit_depths = np.concatenate([fit_depths, dl['depths']])
                fit_vel = np.concatenate([fit_vel, dl['velocities']])
        elif fit_mode == 'reliable_deep_lake':
            fit_mask = rel_local
            fit_depths = depths_local[fit_mask].copy()
            fit_vel = vel_local[fit_mask].copy()
            if dl is not None and dl.get('available', False):
                fit_depths = np.concatenate([fit_depths, dl['depths']])
                fit_vel = np.concatenate([fit_vel, dl['velocities']])
            fit_depths = np.concatenate([fit_depths, [ice_thickness]])
            fit_vel = np.concatenate([fit_vel, [sea_surface_velocity]])
        else:  # 'reliable_lake' — include lake surface point
            fit_mask = rel_local
            fit_depths = np.concatenate([depths_local[fit_mask], [ice_thickness]])
            fit_vel = np.concatenate([vel_local[fit_mask], [sea_surface_velocity]])

        # Linear fit: w(z) = w_s + ε̇_zz * z
        # Using polyfit: w = a*z + b  → a = ε̇_zz, b = w_s (surface vertical velocity)
        valid = np.isfinite(fit_depths) & np.isfinite(fit_vel)
        coeffs = np.polyfit(fit_depths[valid], fit_vel[valid], 1)
        strain_rate_zz = coeffs[0]  # dw/dz in yr⁻¹
        w_surface = coeffs[1]       # w at z=0

        # Model predictions
        z_model = np.linspace(0, ice_thickness, 200)
        w_model = w_surface + strain_rate_zz * z_model

        # Residuals for fit points
        w_predicted_all = w_surface + strain_rate_zz * depths_local
        residuals = vel_local - w_predicted_all

        # R² of the fit
        ss_res = np.sum((fit_vel[valid] - (w_surface + strain_rate_zz * fit_depths[valid]))**2)
        ss_tot = np.sum((fit_vel[valid] - np.mean(fit_vel[valid]))**2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Implied horizontal strain rates (incompressibility)
        # dw/dz = -(du/dx + dv/dy) ≈ -du/dx (since du/dx >> dv/dy)
        dudx = -strain_rate_zz

        # Build figure: 3 panels
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Vertical Velocity Profile + Nye Fit",
                "Residuals (Measured − Nye Model)",
                "Strain Rate Budget",
            ),
            column_widths=[0.40, 0.30, 0.30],
            horizontal_spacing=0.08,
        )

        # ─── Panel 1: Velocity profile with Nye fit ──────────────
        # Unreliable layers
        fig.add_trace(go.Scatter(
            x=vel_local[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            name='Unreliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<extra>Unreliable</extra>',
        ), row=1, col=1)

        # Reliable layers colored by R²
        fig.add_trace(go.Scatter(
            x=vel_local[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis',
                cmin=0, cmax=1,
                size=7,
                colorbar=dict(title='R²', x=0.38, len=0.6),
                line=dict(width=0.5, color='white'),
            ),
            name='Reliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<br>R²: %{marker.color:.2f}<extra></extra>',
        ), row=1, col=1)

        # Lake surface point
        if not np.isnan(sea_surface_velocity):
            fig.add_trace(go.Scatter(
                x=[sea_surface_velocity], y=[ice_thickness],
                mode='markers',
                marker=dict(color='#ef4444', size=12, symbol='diamond',
                            line=dict(color='white', width=2)),
                name=f'Lake surface ({sea_surface_velocity:.3f} m/yr)',
                hovertemplate=f'Lake Surface<br>Depth: {ice_thickness:.0f} m<br>w: {sea_surface_velocity:.3f} m/yr<extra></extra>',
            ), row=1, col=1)

        # Deep layers
        if dl is not None and dl.get('available', False):
            fig.add_trace(go.Scatter(
                x=dl['velocities'], y=dl['depths'],
                mode='markers',
                marker=dict(color='#d6604d', size=8, symbol='diamond',
                            line=dict(color='white', width=0.5)),
                name=f'Deep layers ({dl["n_layers"]})',
                hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<extra>Deep layer</extra>',
            ), row=1, col=1)

        # Nye model line
        fig.add_trace(go.Scatter(
            x=w_model, y=z_model,
            mode='lines',
            line=dict(color='#f59e0b', width=3, dash='dash'),
            name=f'Nye model (ε̇_zz = {strain_rate_zz:.2e} /yr)',
            hovertemplate='Depth: %{y:.0f} m<br>w_Nye: %{x:.4f} m/yr<extra>Nye model</extra>',
        ), row=1, col=1)

        # Spline interpolation (for comparison)
        fig.add_trace(go.Scatter(
            x=vel_smooth, y=depths_local,
            mode='lines',
            line=dict(color='#3b82f6', width=2, dash='dot'),
            name='Spline interpolation',
            hovertemplate='Depth: %{y:.0f} m<br>w_smooth: %{x:.3f} m/yr<extra>Spline</extra>',
        ), row=1, col=1)

        # Zero line
        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=1)

        fig.update_xaxes(title='Vertical velocity w (m/yr)', row=1, col=1)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)

        # ─── Panel 2: Residuals vs depth ─────────────────────────
        # Unreliable
        fig.add_trace(go.Scatter(
            x=residuals[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            name='Residual (unreliable)',
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>Residual: %{x:.4f} m/yr<extra>Unreliable</extra>',
        ), row=1, col=2)

        # Reliable
        fig.add_trace(go.Scatter(
            x=residuals[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis',
                cmin=0, cmax=1,
                size=7,
                showscale=False,
                line=dict(width=0.5, color='white'),
            ),
            name='Residual (reliable)',
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>Residual: %{x:.4f} m/yr<br>R²: %{marker.color:.2f}<extra></extra>',
        ), row=1, col=2)

        # Deep layer residuals
        if dl is not None and dl.get('available', False):
            dl_predicted = w_surface + strain_rate_zz * dl['depths']
            dl_residuals = dl['velocities'] - dl_predicted
            fig.add_trace(go.Scatter(
                x=dl_residuals, y=dl['depths'],
                mode='markers',
                marker=dict(color='#d6604d', size=7, symbol='diamond',
                            line=dict(color='white', width=0.5)),
                name='Residual (deep)',
                showlegend=False,
                hovertemplate='Depth: %{y:.0f} m<br>Residual: %{x:.4f} m/yr<extra>Deep</extra>',
            ), row=1, col=2)

        # Zero line
        fig.add_vline(x=0, line=dict(color='#f59e0b', dash='dash', width=2), row=1, col=2)

        # RMS residual annotation
        rel_rms = np.sqrt(np.mean(residuals[rel_local]**2))
        fig.add_annotation(
            x=0.95, y=0.05,
            xref='x2 domain', yref='y2 domain',
            text=f'RMS residual (reliable): {rel_rms:.4f} m/yr',
            showarrow=False,
            font=dict(size=11, color='#64748b'),
            bgcolor='rgba(255,255,255,0.8)',
            borderpad=4,
        )

        fig.update_xaxes(title='Residual w − w_Nye (m/yr)', row=1, col=2)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=2)

        # ─── Panel 3: Strain rate budget ─────────────────────────
        # Show the implied strain rates as a bar chart
        labels = ['ε̇_zz = ∂w/∂z', 'ε̇_xx ≈ ∂u/∂x', 'ε̇_yy ≈ ∂v/∂y']
        values = [strain_rate_zz, dudx, 0.0]  # dv/dy ≈ 0 by assumption
        colors = ['#3b82f6', '#10b981', '#94a3b8']

        fig.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f'{v:.2e}' for v in values],
            textposition='outside',
            showlegend=False,
            hovertemplate='%{x}<br>%{y:.4e} /yr<extra></extra>',
        ), row=1, col=3)

        fig.add_hline(y=0, line=dict(color='#94a3b8', width=1), row=1, col=3)

        # Verification: sum should be zero
        fig.add_annotation(
            x=0.5, y=0.95,
            xref='x3 domain', yref='y3 domain',
            text=(
                f'<b>∂u/∂x + ∂v/∂y + ∂w/∂z = 0</b><br>'
                f'ε̇_zz = {strain_rate_zz:.4e} /yr<br>'
                f'⇒ ∂u/∂x ≈ {dudx:.4e} /yr<br>'
                f'(dv/dy ≈ 0 assumed)'
            ),
            showarrow=False,
            font=dict(size=11),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=6,
        )

        fig.update_yaxes(title='Strain rate (/yr)', row=1, col=3)

        # ─── Layout ──────────────────────────────────────────────
        stats_text = (
            f"<b>Nye Model Fit</b><br>"
            f"w(z) = {w_surface:.4f} + ({strain_rate_zz:.2e}) · z<br>"
            f"R² = {r2_fit:.4f}<br>"
            f"Surface vertical velocity w_s = {w_surface:.4f} m/yr<br>"
            f"Vertical strain rate ε̇_zz = {strain_rate_zz:.4e} /yr<br>"
            f"Implied ∂u/∂x ≈ {dudx:.4e} /yr<br>"
            f"RMS residual = {rel_rms:.4f} m/yr"
        )

        fig.add_annotation(
            x=0.01, y=0.99,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=11, family='Inter'),
            align='left',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=8,
            xanchor='left', yanchor='top',
        )

        fig.update_layout(
            height=650,
            showlegend=True,
            legend=dict(x=0.01, y=0.15, bgcolor='rgba(255,255,255,0.9)',
                       bordercolor='#e2e8f0', borderwidth=1),
            template='plotly_white',
            font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
            margin=dict(l=60, r=60, t=60, b=50),
            title=dict(
                text='Nye Model: Incompressibility-derived Vertical Strain Rate',
                font=dict(size=15),
            ),
        )

        return fig

    # ── Dansgaard-Johnsen callback ───────────────────────────────
    @app.callback(
        Output('dj-model-analysis', 'figure'),
        Input('dj-fit-layers', 'value'),
    )
    def update_dj_model(fit_mode):
        if fit_mode is None:
            fit_mode = 'reliable_lake'

        velocity_data_local = results['velocity']
        depths_local = velocity_data_local['depths'].flatten()
        vel_local = velocity_data_local['velocities'].flatten()
        r_sq_local = velocity_data_local['r_squared'].flatten()
        rel_local = velocity_data_local['reliable'].flatten().astype(bool)
        vel_smooth = velocity_data_local['velocities_smooth'].flatten()
        ice_thickness = SEA_SURFACE_DEPTH
        dl = deep_layers  # deep layer data

        # Select data for fitting
        if fit_mode == 'all':
            fit_depths = depths_local.copy()
            fit_vel = vel_local.copy()
        elif fit_mode == 'reliable':
            fit_depths = depths_local[rel_local]
            fit_vel = vel_local[rel_local]
        elif fit_mode == 'reliable_deep':
            fit_depths = depths_local[rel_local].copy()
            fit_vel = vel_local[rel_local].copy()
            if dl is not None and dl.get('available', False):
                fit_depths = np.concatenate([fit_depths, dl['depths']])
                fit_vel = np.concatenate([fit_vel, dl['velocities']])
        elif fit_mode == 'reliable_deep_lake':
            fit_depths = depths_local[rel_local].copy()
            fit_vel = vel_local[rel_local].copy()
            if dl is not None and dl.get('available', False):
                fit_depths = np.concatenate([fit_depths, dl['depths']])
                fit_vel = np.concatenate([fit_vel, dl['velocities']])
            fit_depths = np.concatenate([fit_depths, [ice_thickness]])
            fit_vel = np.concatenate([fit_vel, [sea_surface_velocity]])
        else:  # 'reliable_lake'
            fit_depths = np.concatenate([depths_local[rel_local], [ice_thickness]])
            fit_vel = np.concatenate([vel_local[rel_local], [sea_surface_velocity]])

        # Dansgaard-Johnsen model:
        # w(z) = w_s + ε̇₀·z                                     for z ≤ H-h
        # w(z) = w_s + ε̇₀·(H-h) + ε̇₀/(2h)·[h² - (z-H)²]   for z > H-h
        #
        # h = kink height (thickness of basal shear layer)
        # ε̇₀ = strain rate in the upper column
        H = ice_thickness

        def dj_model(z, w_s, eps0, h_kink):
            h_kink = np.clip(h_kink, 10.0, H - 10.0)  # keep physical
            w = np.where(
                z <= H - h_kink,
                w_s + eps0 * z,
                w_s + eps0 * (H - h_kink) + eps0 / (2 * h_kink) * (h_kink**2 - (z - H)**2)
            )
            return w

        # Initial guesses from Nye fit
        nye_coeffs = np.polyfit(fit_depths, fit_vel, 1)
        p0 = [nye_coeffs[1], nye_coeffs[0], H / 3]

        try:
            popt, pcov = curve_fit(
                dj_model, fit_depths, fit_vel, p0=p0,
                bounds=([-np.inf, -np.inf, 10.0], [np.inf, np.inf, H - 10.0]),
                maxfev=10000,
            )
            w_s_fit, eps0_fit, h_kink_fit = popt
            perr = np.sqrt(np.diag(pcov))
            fit_success = True
        except Exception:
            w_s_fit, eps0_fit, h_kink_fit = p0
            perr = [np.nan, np.nan, np.nan]
            fit_success = False

        # Model predictions
        z_model = np.linspace(0, H, 300)
        w_model = dj_model(z_model, w_s_fit, eps0_fit, h_kink_fit)

        # Also compute Nye for comparison
        w_nye = nye_coeffs[1] + nye_coeffs[0] * z_model

        # Residuals
        w_predicted_all = dj_model(depths_local, w_s_fit, eps0_fit, h_kink_fit)
        residuals = vel_local - w_predicted_all

        # R²
        valid = np.isfinite(fit_depths) & np.isfinite(fit_vel)
        ss_res = np.sum((fit_vel[valid] - dj_model(fit_depths[valid], *popt))**2) if fit_success else np.nan
        ss_tot = np.sum((fit_vel[valid] - np.mean(fit_vel[valid]))**2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 and fit_success else np.nan

        # Nye R² for comparison
        ss_res_nye = np.sum((fit_vel[valid] - (nye_coeffs[1] + nye_coeffs[0] * fit_depths[valid]))**2)
        r2_nye = 1 - ss_res_nye / ss_tot if ss_tot > 0 else np.nan

        # Build figure: 3 panels
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Velocity Profile + DJ Fit",
                "Residuals (Measured − DJ Model)",
                "Strain Rate Profile",
            ),
            column_widths=[0.40, 0.30, 0.30],
            horizontal_spacing=0.08,
        )

        # ─── Panel 1: Velocity profile with DJ fit ────────────
        # Unreliable layers
        fig.add_trace(go.Scatter(
            x=vel_local[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            name='Unreliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<extra>Unreliable</extra>',
        ), row=1, col=1)

        # Reliable layers
        fig.add_trace(go.Scatter(
            x=vel_local[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis',
                cmin=0, cmax=1,
                size=7,
                colorbar=dict(title='R²', x=0.38, len=0.6),
                line=dict(width=0.5, color='white'),
            ),
            name='Reliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<br>R²: %{marker.color:.2f}<extra></extra>',
        ), row=1, col=1)

        # Deep layers
        if dl is not None and dl.get('available', False):
            fig.add_trace(go.Scatter(
                x=dl['velocities'], y=dl['depths'],
                mode='markers',
                marker=dict(color='#d6604d', size=8, symbol='diamond',
                            line=dict(color='white', width=0.5)),
                name=f'Deep layers ({dl["n_layers"]})',
                hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<extra>Deep layer</extra>',
            ), row=1, col=1)

        # Lake surface
        if not np.isnan(sea_surface_velocity):
            fig.add_trace(go.Scatter(
                x=[sea_surface_velocity], y=[ice_thickness],
                mode='markers',
                marker=dict(color='#ef4444', size=12, symbol='diamond',
                            line=dict(color='white', width=2)),
                name=f'Lake surface ({sea_surface_velocity:.3f} m/yr)',
            ), row=1, col=1)

        # Nye model (dashed, light)
        fig.add_trace(go.Scatter(
            x=w_nye, y=z_model,
            mode='lines',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            name=f'Nye model (R²={r2_nye:.4f})',
        ), row=1, col=1)

        # DJ model (solid, bold)
        fig.add_trace(go.Scatter(
            x=w_model, y=z_model,
            mode='lines',
            line=dict(color='#f59e0b', width=3),
            name=f'DJ model (R²={r2_fit:.4f})',
            hovertemplate='Depth: %{y:.0f} m<br>w_DJ: %{x:.4f} m/yr<extra>DJ model</extra>',
        ), row=1, col=1)

        # Kink depth line
        fig.add_hline(y=H - h_kink_fit, line=dict(color='#f59e0b', dash='dash', width=1.5),
                      annotation_text=f'Kink depth: {H - h_kink_fit:.0f} m (h={h_kink_fit:.0f} m)',
                      annotation_position='top right',
                      annotation_font=dict(size=10, color='#f59e0b'),
                      row=1, col=1)

        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=1)
        fig.update_xaxes(title='Vertical velocity w (m/yr)', row=1, col=1)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)

        # ─── Panel 2: Residuals ─────────────────────────
        fig.add_trace(go.Scatter(
            x=residuals[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=residuals[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis', cmin=0, cmax=1,
                size=7, showscale=False,
                line=dict(width=0.5, color='white'),
            ),
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>Residual: %{x:.4f} m/yr<extra></extra>',
        ), row=1, col=2)

        fig.add_vline(x=0, line=dict(color='#f59e0b', dash='dash', width=2), row=1, col=2)
        fig.add_hline(y=H - h_kink_fit, line=dict(color='#f59e0b', dash='dash', width=1),
                      row=1, col=2)

        rel_rms = np.sqrt(np.mean(residuals[rel_local]**2))
        fig.add_annotation(
            x=0.95, y=0.05,
            xref='x2 domain', yref='y2 domain',
            text=f'RMS residual (reliable): {rel_rms:.4f} m/yr',
            showarrow=False, font=dict(size=11, color='#64748b'),
            bgcolor='rgba(255,255,255,0.8)', borderpad=4,
        )

        fig.update_xaxes(title='Residual w − w_DJ (m/yr)', row=1, col=2)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=2)

        # ─── Panel 3: Strain rate profile ───────────────
        # DJ strain rate: constant above kink, linearly decreasing below
        z_sr = np.linspace(0, H, 300)
        eps_profile = np.where(
            z_sr <= H - h_kink_fit,
            eps0_fit,
            eps0_fit * (H - z_sr) / h_kink_fit
        )

        fig.add_trace(go.Scatter(
            x=eps_profile, y=z_sr,
            mode='lines',
            line=dict(color='#f59e0b', width=3),
            name='DJ strain rate',
            hovertemplate='Depth: %{y:.0f} m<br>ε̇_zz: %{x:.2e} /yr<extra></extra>',
        ), row=1, col=3)

        # Nye constant strain rate for comparison
        fig.add_trace(go.Scatter(
            x=[nye_coeffs[0], nye_coeffs[0]], y=[0, H],
            mode='lines',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            name='Nye (constant)',
        ), row=1, col=3)

        fig.add_hline(y=H - h_kink_fit, line=dict(color='#f59e0b', dash='dash', width=1),
                      row=1, col=3)
        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=3)

        fig.update_xaxes(title='ε̇_zz (vertical strain rate, /yr)', row=1, col=3)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=3)

        # Stats annotation
        status = 'Fit converged' if fit_success else 'Fit FAILED — showing initial guess'
        stats_text = (
            f"<b>Dansgaard-Johnsen Model</b><br>"
            f"{status}<br>"
            f"w\u209b = {w_s_fit:.4f} \u00b1 {perr[0]:.4f} m/yr<br>"
            f"\u03b5\u0307\u2080 = {eps0_fit:.4e} \u00b1 {perr[1]:.1e} /yr<br>"
            f"h (kink height) = {h_kink_fit:.0f} \u00b1 {perr[2]:.0f} m<br>"
            f"Kink depth = {H - h_kink_fit:.0f} m<br>"
            f"R\u00b2 = {r2_fit:.4f} (Nye: {r2_nye:.4f})<br>"
            f"RMS residual = {rel_rms:.4f} m/yr"
        )

        fig.add_annotation(
            x=0.01, y=0.99,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=11, family='Inter'),
            align='left',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e2e8f0', borderwidth=1, borderpad=8,
            xanchor='left', yanchor='top',
        )

        fig.update_layout(
            height=650,
            showlegend=True,
            legend=dict(x=0.01, y=0.15, bgcolor='rgba(255,255,255,0.9)',
                       bordercolor='#e2e8f0', borderwidth=1),
            template='plotly_white',
            font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
            margin=dict(l=60, r=60, t=60, b=50),
            title=dict(text='Dansgaard-Johnsen Model', font=dict(size=15)),
        )

        return fig

    # ── Lliboutry callback ─────────────────────────────────────
    @app.callback(
        Output('llib-model-analysis', 'figure'),
        Input('llib-fit-layers', 'value'),
    )
    def update_llib_model(fit_mode):
        if fit_mode is None:
            fit_mode = 'reliable_deep_lake'

        velocity_data_local = results['velocity']
        depths_local = velocity_data_local['depths'].flatten()
        vel_local = velocity_data_local['velocities'].flatten()
        r_sq_local = velocity_data_local['r_squared'].flatten()
        rel_local = velocity_data_local['reliable'].flatten().astype(bool)
        vel_smooth = velocity_data_local['velocities_smooth'].flatten()
        ice_thickness = SEA_SURFACE_DEPTH
        H = ice_thickness

        dl = results.get('deep_layers')

        # Select data for fitting
        if fit_mode == 'all':
            fit_depths = depths_local.copy()
            fit_vel = vel_local.copy()
        elif fit_mode == 'reliable':
            fit_depths = depths_local[rel_local]
            fit_vel = vel_local[rel_local]
        elif fit_mode == 'reliable_deep':
            fit_depths = np.concatenate([depths_local[rel_local], dl['depths']]) if dl else depths_local[rel_local]
            fit_vel = np.concatenate([vel_local[rel_local], dl['velocities']]) if dl else vel_local[rel_local]
        elif fit_mode == 'reliable_deep_lake':
            if dl:
                fit_depths = np.concatenate([depths_local[rel_local], dl['depths'], [ice_thickness]])
                fit_vel = np.concatenate([vel_local[rel_local], dl['velocities'], [sea_surface_velocity]])
            else:
                fit_depths = np.concatenate([depths_local[rel_local], [ice_thickness]])
                fit_vel = np.concatenate([vel_local[rel_local], [sea_surface_velocity]])
        else:  # 'reliable_lake'
            fit_depths = np.concatenate([depths_local[rel_local], [ice_thickness]])
            fit_vel = np.concatenate([vel_local[rel_local], [sea_surface_velocity]])

        # Lliboutry shape-function model:
        # w(z) = w_b + (w_s - w_b) * [1 - (p+2)/(p+1) * ζ + 1/(p+1) * ζ^(p+2)]
        # where ζ = z/H, p = shape exponent

        def llib_model(z, w_s, w_b, p_exp):
            p_exp = max(0.1, p_exp)  # keep physical
            zeta = np.clip(z / H, 0, 1)
            return w_b + (w_s - w_b) * (
                1.0 - (p_exp + 2) / (p_exp + 1) * zeta
                + 1.0 / (p_exp + 1) * zeta ** (p_exp + 2)
            )

        # Initial guesses
        nye_coeffs = np.polyfit(fit_depths, fit_vel, 1)
        w_s_guess = nye_coeffs[1]
        w_b_guess = nye_coeffs[1] + nye_coeffs[0] * H
        p0 = [w_s_guess, w_b_guess, 1.0]

        try:
            popt, pcov = curve_fit(
                llib_model, fit_depths, fit_vel, p0=p0,
                bounds=([-np.inf, -np.inf, 0.1], [np.inf, np.inf, 50.0]),
                maxfev=10000,
            )
            w_s_fit, w_b_fit, p_fit = popt
            perr = np.sqrt(np.diag(pcov))
            fit_success = True
        except Exception:
            w_s_fit, w_b_fit, p_fit = p0
            perr = [np.nan, np.nan, np.nan]
            fit_success = False

        # Model predictions
        z_model = np.linspace(0, H, 300)
        w_model = llib_model(z_model, w_s_fit, w_b_fit, p_fit)

        # Nye for comparison
        w_nye = nye_coeffs[1] + nye_coeffs[0] * z_model

        # Residuals
        w_predicted_all = llib_model(depths_local, w_s_fit, w_b_fit, p_fit)
        residuals = vel_local - w_predicted_all

        # R²
        valid = np.isfinite(fit_depths) & np.isfinite(fit_vel)
        ss_res = np.sum((fit_vel[valid] - llib_model(fit_depths[valid], *popt))**2) if fit_success else np.nan
        ss_tot = np.sum((fit_vel[valid] - np.mean(fit_vel[valid]))**2)
        r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 and fit_success else np.nan

        ss_res_nye = np.sum((fit_vel[valid] - (nye_coeffs[1] + nye_coeffs[0] * fit_depths[valid]))**2)
        r2_nye = 1 - ss_res_nye / ss_tot if ss_tot > 0 else np.nan

        # Strain rate profile: dw/dz
        # dw/dz = (w_s - w_b)/H * [-(p+2)/(p+1) + (p+2)/(p+1) * ζ^(p+1)]
        def llib_strain_rate(z, w_s, w_b, p_exp):
            p_exp = max(0.1, p_exp)
            zeta = np.clip(z / H, 0, 1)
            return (w_s - w_b) / H * (
                -(p_exp + 2) / (p_exp + 1)
                + (p_exp + 2) / (p_exp + 1) * zeta ** (p_exp + 1)
            )

        eps_profile = llib_strain_rate(z_model, w_s_fit, w_b_fit, p_fit)

        # Build figure: 3 panels
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Velocity Profile + Lliboutry Fit",
                "Residuals (Measured − Lliboutry)",
                "Strain Rate Profile",
            ),
            column_widths=[0.40, 0.30, 0.30],
            horizontal_spacing=0.08,
        )

        # ─── Panel 1: Velocity profile ──────────────────
        fig.add_trace(go.Scatter(
            x=vel_local[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            name='Unreliable layers',
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=vel_local[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis', cmin=0, cmax=1,
                size=7,
                colorbar=dict(title='R²', x=0.38, len=0.6),
                line=dict(width=0.5, color='white'),
            ),
            name='Reliable layers',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.3f} m/yr<br>R²: %{marker.color:.2f}<extra></extra>',
        ), row=1, col=1)

        if not np.isnan(sea_surface_velocity):
            fig.add_trace(go.Scatter(
                x=[sea_surface_velocity], y=[ice_thickness],
                mode='markers',
                marker=dict(color='#ef4444', size=12, symbol='diamond',
                            line=dict(color='white', width=2)),
                name=f'Lake surface ({sea_surface_velocity:.3f} m/yr)',
            ), row=1, col=1)

        # Nye
        fig.add_trace(go.Scatter(
            x=w_nye, y=z_model,
            mode='lines',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            name=f'Nye (R²={r2_nye:.4f})',
        ), row=1, col=1)

        # Lliboutry
        fig.add_trace(go.Scatter(
            x=w_model, y=z_model,
            mode='lines',
            line=dict(color='#10b981', width=3),
            name=f'Lliboutry p={p_fit:.2f} (R²={r2_fit:.4f})',
            hovertemplate='Depth: %{y:.0f} m<br>w: %{x:.4f} m/yr<extra>Lliboutry</extra>',
        ), row=1, col=1)

        # Deep layers
        if dl:
            tier_config = [
                (1, '★★★ Tier 1', 'diamond', '#ef4444', 10),
                (2, '★★ Tier 2', 'square', '#f97316', 8),
                (3, '★ Tier 3', 'triangle-up', '#d4a574', 7),
            ]
            for tier_val, tier_name, sym, clr, sz in tier_config:
                mask = dl['quality_tier'] == tier_val
                if np.any(mask):
                    fig.add_trace(go.Scatter(
                        x=dl['velocities'][mask], y=dl['depths'][mask],
                        mode='markers',
                        marker=dict(color=clr, size=sz, symbol=sym,
                                    line=dict(color='white', width=1)),
                        name=f'Deep {tier_name}',
                        hovertemplate='Depth: %{y:.1f} m<br>w: %{x:.4f} m/yr<extra></extra>',
                    ), row=1, col=1)

        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=1)
        fig.update_xaxes(title='Vertical velocity w (m/yr)', row=1, col=1)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=1)

        # ─── Panel 2: Residuals ─────────────────────────
        fig.add_trace(go.Scatter(
            x=residuals[~rel_local], y=depths_local[~rel_local],
            mode='markers',
            marker=dict(color='#94a3b8', size=5, opacity=0.4),
            showlegend=False,
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=residuals[rel_local], y=depths_local[rel_local],
            mode='markers',
            marker=dict(
                color=r_sq_local[rel_local],
                colorscale='Viridis', cmin=0, cmax=1,
                size=7, showscale=False,
                line=dict(width=0.5, color='white'),
            ),
            showlegend=False,
            hovertemplate='Depth: %{y:.0f} m<br>Residual: %{x:.4f} m/yr<extra></extra>',
        ), row=1, col=2)

        fig.add_vline(x=0, line=dict(color='#10b981', dash='dash', width=2), row=1, col=2)

        # Deep layer residuals
        if dl:
            deep_resid = dl['velocities'] - llib_model(dl['depths'], w_s_fit, w_b_fit, p_fit)
            fig.add_trace(go.Scatter(
                x=deep_resid, y=dl['depths'],
                mode='markers',
                marker=dict(color='#ef4444', size=8, symbol='diamond',
                            line=dict(color='white', width=1)),
                name='Deep layer residuals',
                showlegend=False,
                hovertemplate='Depth: %{y:.1f} m<br>Residual: %{x:.4f} m/yr<extra></extra>',
            ), row=1, col=2)

        rel_rms = np.sqrt(np.mean(residuals[rel_local]**2))
        fig.add_annotation(
            x=0.95, y=0.05,
            xref='x2 domain', yref='y2 domain',
            text=f'RMS residual (reliable): {rel_rms:.4f} m/yr',
            showarrow=False, font=dict(size=11, color='#64748b'),
            bgcolor='rgba(255,255,255,0.8)', borderpad=4,
        )

        fig.update_xaxes(title='Residual w − w_Llib (m/yr)', row=1, col=2)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=2)

        # ─── Panel 3: Strain rate profile ───────────────
        fig.add_trace(go.Scatter(
            x=eps_profile, y=z_model,
            mode='lines',
            line=dict(color='#10b981', width=3),
            name='Lliboutry strain rate',
            hovertemplate='Depth: %{y:.0f} m<br>ε̇_zz: %{x:.2e} /yr<extra></extra>',
        ), row=1, col=3)

        # Nye constant
        fig.add_trace(go.Scatter(
            x=[nye_coeffs[0], nye_coeffs[0]], y=[0, H],
            mode='lines',
            line=dict(color='#94a3b8', width=2, dash='dot'),
            name='Nye (constant)',
        ), row=1, col=3)

        fig.add_vline(x=0, line=dict(color='#94a3b8', dash='dot', width=1), row=1, col=3)
        fig.update_xaxes(title='ε̇_zz (vertical strain rate, /yr)', row=1, col=3)
        fig.update_yaxes(title='Depth (m)', autorange='reversed', row=1, col=3)

        # Physical interpretation of p
        if p_fit < 1.5:
            p_interp = 'Nearly linear (Nye-like) — distributed deformation'
        elif p_fit < 5:
            p_interp = 'Moderate basal concentration'
        elif p_fit < 15:
            p_interp = 'Strong plug flow — most deformation near bed'
        else:
            p_interp = 'Extreme plug flow — nearly all deformation at bed'

        status = 'Fit converged' if fit_success else 'Fit FAILED — showing initial guess'
        stats_text = (
            f"<b>Lliboutry Shape-Function Model</b><br>"
            f"{status}<br>"
            f"w\u209b = {w_s_fit:.4f} \u00b1 {perr[0]:.4f} m/yr<br>"
            f"w_b = {w_b_fit:.4f} \u00b1 {perr[1]:.4f} m/yr<br>"
            f"p = {p_fit:.2f} \u00b1 {perr[2]:.2f}<br>"
            f"\u2192 {p_interp}<br>"
            f"R\u00b2 = {r2_fit:.4f} (Nye: {r2_nye:.4f})<br>"
            f"RMS residual = {rel_rms:.4f} m/yr"
        )

        fig.add_annotation(
            x=0.01, y=0.99,
            xref='paper', yref='paper',
            text=stats_text,
            showarrow=False,
            font=dict(size=11, family='Inter'),
            align='left',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e2e8f0', borderwidth=1, borderpad=8,
            xanchor='left', yanchor='top',
        )

        fig.update_layout(
            height=650,
            showlegend=True,
            legend=dict(x=0.01, y=0.15, bgcolor='rgba(255,255,255,0.9)',
                       bordercolor='#e2e8f0', borderwidth=1),
            template='plotly_white',
            font=dict(family='Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'),
            margin=dict(l=60, r=60, t=60, b=50),
            title=dict(text='Lliboutry Shape-Function Model', font=dict(size=15)),
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
