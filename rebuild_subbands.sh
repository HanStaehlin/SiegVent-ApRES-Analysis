#!/bin/bash
# Rebuild the low and high sub-band .mat files for ApRES data
# This process takes around 1.5 hours on your machine and generates ~2GB sized files cleanly.

echo "Rebuilding low sub-band dataset (approx 45 mins)..."
mamba run -n siegvent2023 python proc/apres/process_apres_raw.py --data-folder data/apres/raw --output data/apres/ImageP2_low_python.mat --subband low

echo "Rebuilding high sub-band dataset (approx 45 mins)..."
mamba run -n siegvent2023 python proc/apres/process_apres_raw.py --data-folder data/apres/raw --output data/apres/ImageP2_high_python.mat --subband high

echo "Done generating sub-band files!"
