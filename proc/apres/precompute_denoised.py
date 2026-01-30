#!/usr/bin/env python3
"""
Pre-compute denoised echogram and save to file.

This script applies median + smoothing filter to create a denoised
version of the echogram that can be loaded quickly in the visualization.

Usage:
    python precompute_denoised.py --data data/apres/ImageP2_python.mat --output output/apres
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import median_filter, uniform_filter
from pathlib import Path
import argparse


def apply_fast_denoising(range_img: np.ndarray) -> np.ndarray:
    """
    Apply fast denoising to the echogram using median + smoothing filter.
    
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
    print(f"  Input shape: {range_img.shape}")
    
    # Convert to dB for filtering (log-domain filtering works better)
    img_db = 10 * np.log10(range_img**2 + 1e-30)
    
    # Apply median filter with horizontal emphasis (preserves layers)
    # Size is (depth, time) - larger in time dimension to smooth along layers
    print("  Applying median filter (3x7)...")
    filtered = median_filter(img_db, size=(3, 7))
    
    # Additional smoothing along time axis only
    print("  Applying uniform filter (1x5)...")
    filtered = uniform_filter(filtered, size=(1, 5))
    
    # Convert back from dB
    denoised = np.sqrt(10 ** (filtered / 10))
    
    return denoised


def main():
    parser = argparse.ArgumentParser(description='Pre-compute denoised echogram')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ApRES data file (ImageP2_python.mat)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    args = parser.parse_args()
    
    data_path = Path(args.data)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {data_path}...")
    apres_data = loadmat(str(data_path))
    
    # Use RawImageComplex if available
    if 'RawImageComplex' in apres_data:
        raw_complex = np.array(apres_data['RawImageComplex'])
        range_img = np.abs(raw_complex)
        print("Using RawImageComplex for amplitude")
    else:
        range_img = np.array(apres_data['RawImage'])
        print("Using RawImage for amplitude")
    
    print(f"\nComputing denoised echogram...")
    denoised = apply_fast_denoising(range_img)
    
    # Save denoised data
    output_file = output_path / 'echogram_denoised.mat'
    print(f"\nSaving to {output_file}...")
    savemat(str(output_file), {
        'range_img_denoised': denoised,
        'Rcoarse': apres_data['Rcoarse'],
        'TimeInDays': apres_data['TimeInDays'],
    }, do_compression=True)
    
    # Check file size
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"Saved denoised echogram: {size_mb:.1f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
