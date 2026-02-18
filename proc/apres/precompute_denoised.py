#!/usr/bin/env python3
"""
Pre-compute denoised echograms and save to file.

This script applies two denoising methods:
1. Fast median + smoothing filter (quick, good for visualization)
2. SVD/PCA denoising (slower, better for incoherent noise removal)

Both are saved so the visualization can toggle between them.

Usage:
    python precompute_denoised.py --data data/apres/ImageP2_python.mat --output output/apres/hybrid
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import median_filter, uniform_filter
from pathlib import Path
import argparse


def apply_fast_denoising(range_img: np.ndarray) -> np.ndarray:
    """
    Apply fast denoising using median + smoothing filter.
    
    Uses a horizontally-oriented filter to preserve layer structure.
    Fast (~1 second) but less effective than SVD.
    
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
    print("  Applying median filter (5x11)...")
    filtered = median_filter(img_db, size=(5, 11))
    
    # Additional smoothing along time axis only
    print("  Applying uniform filter (1x9)...")
    filtered = uniform_filter(filtered, size=(1, 9))
    
    # Convert back from dB
    denoised = np.sqrt(10 ** (filtered / 10))
    
    return denoised


def apply_svd_denoising(
    range_img: np.ndarray,
    n_components: int = 50,
    block_size: int = 500,
) -> np.ndarray:
    """
    Apply SVD-based denoising to remove incoherent noise.
    
    SVD separates coherent signal (layers) from incoherent noise by
    keeping only the top singular value components. Layers are coherent
    across time, so they concentrate in the first components, while
    incoherent noise spreads across many small components.
    
    Uses block processing to handle large matrices efficiently.
    
    Parameters
    ----------
    range_img : np.ndarray
        Original echogram (depth x time)
    n_components : int
        Number of SVD components to keep (more = less denoising)
    block_size : int
        Number of depth bins to process at once
    
    Returns
    -------
    np.ndarray
        Denoised echogram (same shape as input)
    """
    n_depth, n_time = range_img.shape
    print(f"  Input shape: {range_img.shape}")
    print(f"  Keeping {n_components} SVD components")
    
    # Work in log domain for better dynamic range
    img_db = 10 * np.log10(range_img**2 + 1e-30)
    
    # Center the data (important for SVD)
    mean_profile = np.mean(img_db, axis=1, keepdims=True)
    img_centered = img_db - mean_profile
    
    # Process in blocks to manage memory
    n_blocks = (n_depth + block_size - 1) // block_size
    denoised_db = np.zeros_like(img_db)
    
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = min((i + 1) * block_size, n_depth)
        block = img_centered[start_idx:end_idx, :]
        
        print(f"  Processing block {i+1}/{n_blocks} (depths {start_idx}-{end_idx})...")
        
        # SVD decomposition
        try:
            U, S, Vh = np.linalg.svd(block, full_matrices=False)
            
            # Keep only top n_components
            n_keep = min(n_components, len(S))
            S_filtered = np.zeros_like(S)
            S_filtered[:n_keep] = S[:n_keep]
            
            # Reconstruct
            block_denoised = U @ np.diag(S_filtered) @ Vh
            denoised_db[start_idx:end_idx, :] = block_denoised + mean_profile[start_idx:end_idx]
            
        except np.linalg.LinAlgError as e:
            print(f"  Warning: SVD failed for block {i+1}, using original: {e}")
            denoised_db[start_idx:end_idx, :] = img_db[start_idx:end_idx, :]
    
    # Convert back from dB
    denoised = np.sqrt(10 ** (denoised_db / 10))
    
    # Clip to reasonable range (SVD can produce artifacts)
    denoised = np.clip(denoised, 0, np.max(range_img) * 1.1)
    
    return denoised


def main():
    parser = argparse.ArgumentParser(description='Pre-compute denoised echograms')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to ApRES data file (ImageP2_python.mat)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--svd-components', type=int, default=50,
                        help='Number of SVD components to keep (default: 50)')
    parser.add_argument('--skip-svd', action='store_true',
                        help='Skip SVD denoising (faster)')
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
    
    # Method 1: Fast median filter denoising
    print(f"\n{'='*60}")
    print("Method 1: Fast median filter denoising")
    print('='*60)
    denoised_fast = apply_fast_denoising(range_img)
    
    # Method 2: SVD denoising (optional)
    denoised_svd = None
    if not args.skip_svd:
        print(f"\n{'='*60}")
        print("Method 2: SVD denoising (incoherent noise removal)")
        print('='*60)
        denoised_svd = apply_svd_denoising(range_img, n_components=args.svd_components)
    
    # Save denoised data
    output_file = output_path / 'echogram_denoised.mat'
    print(f"\nSaving to {output_file}...")
    
    save_dict = {
        'range_img_denoised': denoised_fast,  # Default: fast method
        'range_img_denoised_median': denoised_fast,
        'Rcoarse': apres_data['Rcoarse'],
        'TimeInDays': apres_data['TimeInDays'],
        'denoising_method': 'median_filter',
    }
    
    if denoised_svd is not None:
        save_dict['range_img_denoised_svd'] = denoised_svd
        save_dict['svd_components'] = args.svd_components
    
    savemat(str(output_file), save_dict, do_compression=True)
    
    # Check file size
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"Saved denoised echogram: {size_mb:.1f} MB")
    
    print("\nDenoising methods saved:")
    print("  - range_img_denoised_median: Fast median filter")
    if denoised_svd is not None:
        print(f"  - range_img_denoised_svd: SVD with {args.svd_components} components")
    print("\nDone!")


if __name__ == '__main__':
    main()
