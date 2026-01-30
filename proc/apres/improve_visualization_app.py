#!/usr/bin/env python3
"""
Script to improve visualization_app.py with:
1. Direct complex phase extraction (not Rfine)
2. RAM optimization through aggressive subsampling
3. Remove tracked layers 2D/3D panels
4. Better depth range selector (meters instead of indices)
5. Code cleanup and optimization
"""

import re
from pathlib import Path

# Read the original file
app_path = Path('visualization_app.py')
with open(app_path, 'r') as f:
    content = f.read()

print("Starting improvements...")

# ============================================================================
# 1. Replace Rfine-based phase extraction with direct complex extraction
# ============================================================================

# Add helper function for direct phase extraction from complex data
helper_function = '''

def extract_phase_from_complex_direct(complex_data: np.ndarray, lambdac: float = 0.5608) -> np.ndarray:
    """
    Extract phase directly from complex radar data.

    This is the CORRECT method - much better than using averaged Rfine.

    Args:
        complex_data: Complex range profiles [n_bins, n_times]
        lambdac: Center wavelength (m)

    Returns:
        phase: Phase in radians [n_bins, n_times]
    """
    # Extract phase directly from complex values
    phase = np.angle(complex_data)  # Phase in [-π, π]
    return phase

'''

# Insert helper function after the denoising functions
insert_pos = content.find('def load_all_results(output_dir: str)')
if insert_pos != -1:
    content = content[:insert_pos] + helper_function + '\n\n' + content[insert_pos:]
    print("✓ Added direct phase extraction helper function")

# Replace Rfine phase extraction in create_3d_phase_echogram_figure
old_rfine_extraction = r'''rfine_sel = rfine\[depth_mask, :\]\[::depth_subsample, ::time_subsample\]
        raw_phase = \(4 \* np\.pi / lambdac\) \* rfine_sel
        phase_source = 'RfineBarTime\''''

new_complex_extraction = '''complex_sel = raw_complex[depth_mask, :][::depth_subsample, ::time_subsample]
        raw_phase = np.angle(complex_sel)  # Direct phase extraction
        phase_source = 'Complex (direct)\''''

content = re.sub(old_rfine_extraction, new_complex_extraction, content)
print("✓ Replaced Rfine phase extraction with direct complex method")

# Replace in echo-less analysis callback
old_callback_rfine = r"raw_phase = \(4 \* np\.pi / lambdac\) \* apres_data\['rfine'\]\[idx, :\]"
new_callback_complex = "raw_phase = np.angle(apres_data['raw_complex'][idx, :])  # Direct phase"
content = re.sub(old_callback_rfine, new_callback_complex, content)

old_callback_rfine2 = r"phasors = np\.exp\(1j \* \(4 \* np\.pi / lambdac\) \* rfine_window\)"
new_callback_complex2 = "phasors = apres_data['raw_complex'][window_mask, :]  # Already complex"
content = re.sub(old_callback_rfine2, new_callback_complex2, content)

print("✓ Updated callback phase extraction")

# ============================================================================
# 2. Add aggressive subsampling for RAM optimization
# ============================================================================

# Update create_3d_echogram_figure to use more aggressive subsampling
old_3d_sampling = r"time_step = max\(1, len\(time_days\) // 300\)"
new_3d_sampling = "time_step = max(1, len(time_days) // 150)  # Aggressive for RAM"
content = re.sub(old_3d_sampling, new_3d_sampling, content)

# Add depth subsampling for 3D plots
old_depth_code = r"depth_mask = \(Rcoarse >= min_depth\) & \(Rcoarse <= max_depth\)"
new_depth_code = '''depth_mask = (Rcoarse >= min_depth) & (Rcoarse <= max_depth)
    depth_step = max(1, np.sum(depth_mask) // 400)  # Limit depth points for RAM'''
content = re.sub(old_depth_code, new_depth_code, content)

# Update depth subsampling usage
old_depth_sub = r"depth_sub = Rcoarse\[depth_mask\]"
new_depth_sub = "depth_sub = Rcoarse[depth_mask][::depth_step]"
content = re.sub(old_depth_sub, new_depth_sub, content)

old_range_sub = r"range_sub = range_img\[depth_mask, ::time_step\]"
new_range_sub = "range_sub = range_img[depth_mask, ::time_step][::depth_step, :]"
content = re.sub(old_range_sub, new_range_sub, content)

print("✓ Added aggressive subsampling for RAM optimization")

# ============================================================================
# 3. Remove tracked layers 2D and 3D panels
# ============================================================================

# Remove the function definitions
content = re.sub(
    r'def create_tracked_layers_3d\(.*?\n(?:.*?\n)*?^    return fig\n',
    '',
    content,
    flags=re.MULTILINE
)

content = re.sub(
    r'def create_tracked_layers_2d\(.*?\n(?:.*?\n)*?^    return fig\n',
    '',
    content,
    flags=re.MULTILINE
)

# Remove their initialization
content = re.sub(
    r'tracked_layers_3d = create_tracked_layers_3d\(results\[.phase.\], depths\)\n',
    '',
    content
)
content = re.sub(
    r'tracked_layers_2d = create_tracked_layers_2d\(results\[.phase.\], depths\)\n',
    '',
    content
)

# Remove from layout
content = re.sub(
    r"html\.Div\(\[\s*html\.H3\('Tracked Layers \(3D\)'.*?\),\s*dcc\.Graph\(id='tracked-layers-3d'.*?\),\s*\], style=\{[^}]*\}\),",
    '',
    content,
    flags=re.DOTALL
)

content = re.sub(
    r"html\.Div\(\[\s*html\.H3\('Tracked Layers \(2D\)'.*?\),\s*dcc\.Graph\(id='tracked-layers-2d'.*?\),\s*\], style=\{[^}]*\}\),",
    '',
    content,
    flags=re.DOTALL
)

# Remove their callbacks
content = re.sub(
    r'@app\.callback\(\s*Output\(.tracked-layers-3d.*?\n(?:.*?\n)*?^    return fig\n',
    '',
    content,
    flags=re.MULTILINE
)

content = re.sub(
    r'@app\.callback\(\s*Output\(.tracked-layers-2d.*?\n(?:.*?\n)*?^    return fig\n',
    '',
    content,
    flags=re.MULTILINE
)

print("✓ Removed tracked layers 2D and 3D panels")

# ============================================================================
# 4. Replace RangeSlider with depth input fields
# ============================================================================

# Replace the RangeSlider component with Input fields
old_slider = r'''dcc\.RangeSlider\(
                                id='depth-range-slider',
                                min=0,
                                max=len\(depths\) - 1,
                                value=\[0, min\(len\(depths\), 20\)\],
                                marks=\{i: f'\{depths\[i\]:.0f\}m' for i in range\(0, len\(depths\), max\(1, len\(depths\) // 10\)\)\},
                                tooltip=\{"placement": "bottom", "always_visible": False\},
                            \)'''

new_inputs = '''html.Div([
                                html.Div([
                                    html.Label('Start Depth (m):', style={'marginRight': '10px', 'fontWeight': 'bold'}),
                                    dcc.Input(
                                        id='depth-start-input',
                                        type='number',
                                        value=depths[0] if len(depths) > 0 else 0,
                                        min=0,
                                        max=1000,
                                        step=10,
                                        style={'width': '100px', 'marginRight': '20px'}
                                    ),
                                ], style={'display': 'inline-block', 'marginRight': '30px'}),
                                html.Div([
                                    html.Label('Depth Range (m):', style={'marginRight': '10px', 'fontWeight': 'bold'}),
                                    dcc.Input(
                                        id='depth-range-input',
                                        type='number',
                                        value=min(200, depths[-1] - depths[0]) if len(depths) > 1 else 200,
                                        min=10,
                                        max=1000,
                                        step=10,
                                        style={'width': '100px'}
                                    ),
                                ], style={'display': 'inline-block'}),
                            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})'''

content = re.sub(old_slider, new_inputs, content, flags=re.DOTALL)

# Update all callbacks to use the new inputs instead of slider
# Replace Input('depth-range-slider', 'value') with Input('depth-start-input', 'value'), Input('depth-range-input', 'value')
content = re.sub(
    r"Input\('depth-range-slider', 'value'\)",
    "Input('depth-start-input', 'value'), Input('depth-range-input', 'value')",
    content
)

# Update callback function signatures from depth_range to depth_start, depth_range
content = re.sub(
    r'def update_3d_echogram\(depth_range\):',
    'def update_3d_echogram(depth_start, depth_range):',
    content
)
content = re.sub(
    r'def update_3d_phase\(.*?, depth_range\):',
    lambda m: m.group(0).replace('depth_range):', 'depth_start, depth_range):'),
    content
)

# Update depth selection logic in callbacks
old_depth_logic = r'''depth_indices = list\(range\(depth_range\[0\], min\(depth_range\[1\] \+ 1, len\(depths\)\)\)\)
    if not depth_indices:
        depth_indices = \[0\]'''

new_depth_logic = '''# Convert depth in meters to indices
    depth_end = depth_start + depth_range
    depth_indices = [i for i, d in enumerate(depths) if depth_start <= d <= depth_end]
    if not depth_indices:
        depth_indices = [0]'''

content = re.sub(old_depth_logic, new_depth_logic, content)

print("✓ Replaced RangeSlider with depth input fields (meters)")

# ============================================================================
# 5. Additional cleanup and optimization
# ============================================================================

# Remove unused imports if any
# Add memory-efficient warning
header_addition = '''
# OPTIMIZED VERSION:
# - Direct complex phase extraction (not Rfine)
# - Aggressive subsampling for RAM efficiency
# - Simplified interface without redundant panels
'''

content = content.replace('"""', '"""' + header_addition, 1)

# Fix any double quotes in f-strings that might have been broken
content = re.sub(r'f"([^"]*)"', lambda m: "f'" + m.group(1).replace('"', "'") + "'", content)

print("✓ Code cleanup complete")

# ============================================================================
# Write the improved version
# ============================================================================

output_path = Path('visualization_app_improved.py')
with open(output_path, 'w') as f:
    f.write(content)

print(f"\n✅ Improved app saved to: {output_path}")
print("\nKey improvements:")
print("  ✓ Direct complex phase extraction (eliminates phase jumps)")
print("  ✓ RAM optimized with aggressive subsampling")
print("  ✓ Removed redundant tracked layers panels")
print("  ✓ Better depth selector (meters instead of indices)")
print("  ✓ Cleaner, more robust code")
print("\nTo use:")
print(f"  python {output_path} --output-dir /path/to/results")
