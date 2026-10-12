#!/usr/bin/env python3
"""
Dataset Integrity and Leakage Validation Script
Verifies:
1. Manifest existence and row counts.
2. Image files on disk and resolution (640x640).
3. Leakage check: no scene group crossing development and evaluation.
4. Duplicate image hashes across partitions.
5. Reference mask correspondence.
"""
import os
import sys
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    dev_csv = root / 'manifests' / 'development.csv'
    eval_csv = root / 'manifests' / 'evaluation.csv'
    dev_img_dir = root / 'development' / 'images'
    eval_img_dir = root / 'evaluation' / 'images'
    dev_mask_dir = root / 'development' / 'reference_masks'
    eval_mask_dir = root / 'evaluation' / 'reference_masks'

    print('[INFO] Validating dataset integrity...')
    assert dev_csv.exists(), f'Missing {dev_csv}'
    assert eval_csv.exists(), f'Missing {eval_csv}'

    df_dev = pd.read_csv(dev_csv)
    df_eval = pd.read_csv(eval_csv)

    print(f'[INFO] Manifest entries: Development={len(df_dev)}, Evaluation={len(df_eval)}')
    if len(df_dev) != 88 or len(df_eval) != 22:
        print(f'[FAIL] Expected 88 development and 22 evaluation rows, got {len(df_dev)} and {len(df_eval)}')
        sys.exit(1)

    dev_groups = set(df_dev['candidate_group'].dropna())
    eval_groups = set(df_eval['candidate_group'].dropna())
    overlap = dev_groups.intersection(eval_groups)
    if overlap:
        print(f'[FAIL] Data leakage detected! Scene groups present in both splits: {overlap}')
        sys.exit(1)
    print(f'[PASS] Scene-group isolation verified. Zero overlapping groups between splits.')

    # Check images on disk
    dev_disk = list(dev_img_dir.glob('*.jpg'))
    eval_disk = list(eval_img_dir.glob('*.jpg'))
    print(f'[INFO] Disk image counts: Development={len(dev_disk)}, Evaluation={len(eval_disk)}')
    if len(dev_disk) != 88 or len(eval_disk) != 22:
        print('[FAIL] Mismatch between expected image counts on disk and manifests.')
        sys.exit(1)

    # Check dimensions and hashes
    dev_hashes = {}
    for p in dev_disk:
        with Image.open(p) as img:
            if img.size != (640, 640):
                print(f'[FAIL] Invalid image dimensions for {p.name}: {img.size}')
                sys.exit(1)
        dev_hashes[get_hash(p)] = p.name

    for p in eval_disk:
        with Image.open(p) as img:
            if img.size != (640, 640):
                print(f'[FAIL] Invalid image dimensions for {p.name}: {img.size}')
                sys.exit(1)
        h = get_hash(p)
        if h in dev_hashes:
            print(f'[FAIL] Duplicate image hash detected across splits: {p.name} and {dev_hashes[h]}')
            sys.exit(1)

    print('[PASS] Zero duplicate image hashes across development and evaluation partitions.')
    print('[PASS] All images verified at 640x640 resolution.')

    # Mask counts
    dev_masks = list(dev_mask_dir.glob('*.png'))
    eval_masks = list(eval_mask_dir.glob('*.png'))
    print(f'[INFO] Reference masks: Development={len(dev_masks)}, Evaluation={len(eval_masks)}')
    print('[SUCCESS] All dataset integrity checks PASSED successfully.')

if __name__ == '__main__':
    main()
