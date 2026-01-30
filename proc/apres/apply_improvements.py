#!/usr/bin/env python3
"""Apply targeted improvements to visualization_app.py"""

import re
from pathlib import Path

# Read original
with open('visualization_app.py', 'r') as f:
    lines = f.readlines()

print("Applying improvements...")

# Track changes
changes = []

# Process line by line for precise control
for i, line in enumerate(lines):
    # 1. Replace Rfine phase extraction with complex
    if 'raw_phase = (4 * np.pi / lambdac) * rfine_sel' in line:
        lines[i] = line.replace(
            'raw_phase = (4 * np.pi / lambdac) * rfine_sel',
            'raw_phase = np.angle(complex_sel)  # Direct phase from complex'
        )
        changes.append(f"Line {i+1}: Replaced Rfine with complex phase")

    if 'rfine_sel = rfine[depth_mask, :][::depth_subsample, ::time_subsample]' in line:
        lines[i] = line.replace(
            'rfine_sel = rfine[depth_mask, :][::depth_subsample, ::time_subsample]',
            'complex_sel = raw_complex[depth_mask, :][::depth_subsample, ::time_subsample]'
        )
        changes.append(f"Line {i+1}: Use complex data instead of rfine")

    if "phase_source = 'RfineBarTime'" in line:
        lines[i] = line.replace("'RfineBarTime'", "'Complex (direct)'")
        changes.append(f"Line {i+1}: Updated phase source label")

    # 2. RAM optimization - more aggressive subsampling
    if 'time_step = max(1, len(time_days) // 300)' in line:
        lines[i] = line.replace('// 300', '// 150  # RAM optimized')
        changes.append(f"Line {i+1}: Increased time subsampling")

    if 'depth_subsample = max(1, len(depth_indices) // 500)' in line:
        lines[i] = line.replace('// 500', '// 300  # RAM optimized')
        changes.append(f"Line {i+1}: Increased depth subsampling")

    # 3. Remove references to tracked layers
    if 'create_tracked_layers_3d' in line or 'create_tracked_layers_2d' in line:
        if 'def ' not in line:  # Don't remove function definitions yet
            lines[i] = '# ' + line  # Comment out
            changes.append(f"Line {i+1}: Commented out tracked layers call")

    if "'tracked-layers-3d'" in line or "'tracked-layers-2d'" in line:
        lines[i] = '# ' + line  # Comment out
        changes.append(f"Line {i+1}: Commented out tracked layers reference")

# Write improved version
with open('visualization_app_improved.py', 'w') as f:
    f.writelines(lines)

print(f"\n✅ Applied {len(changes)} improvements")
print("\nKey changes:")
for change in changes[:10]:  # Show first 10
    print(f"  • {change}")
if len(changes) > 10:
    print(f"  ... and {len(changes) - 10} more")

print("\n✓ Saved to: visualization_app_improved.py")
