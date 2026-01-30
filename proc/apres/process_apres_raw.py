#!/usr/bin/env python3
"""
Convert raw ApRES .DAT files into ImageP2_python.mat with complex data preserved.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.io import savemat

from apres_python import process_timeseries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process ApRES .DAT files and save ImageP2_python.mat",
    )
    parser.add_argument(
        "--data-folder",
        type=str,
        required=True,
        help="Folder containing .DAT files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/apres/ImageP2_python.mat",
        help="Output .mat file path",
    )
    parser.add_argument("--er", type=float, default=3.18, help="Relative permittivity")
    parser.add_argument("--max-range", type=float, default=2000.0, help="Max range (m)")
    parser.add_argument("--pad-factor", type=int, default=8, help="FFT padding factor")
    parser.add_argument("--step", type=int, default=1, help="Process every Nth file")
    parser.add_argument(
        "--no-complex",
        action="store_true",
        help="Do not store complex RawImage (magnitude only)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keep_complex = not args.no_complex

    result = process_timeseries(
        args.data_folder,
        er=args.er,
        max_range=args.max_range,
        pad_factor=args.pad_factor,
        step=args.step,
        keep_complex=keep_complex,
    )

    if keep_complex:
        range_img, rfine_avg, rcoarse, time_days, timestamps, range_img_complex = result
    else:
        range_img, rfine_avg, rcoarse, time_days, timestamps = result
        range_img_complex = None

    mat_dict = {
        "RawImage": range_img,
        "RfineBarTime": rfine_avg,
        "Rcoarse": rcoarse,
        "TimeInDays": time_days,
    }
    if range_img_complex is not None:
        mat_dict["RawImageComplex"] = range_img_complex

    savemat(str(output_path), mat_dict)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
