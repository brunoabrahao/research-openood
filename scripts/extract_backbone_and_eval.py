#!/usr/bin/env python3
"""Extract backbone from wrapper network checkpoints and evaluate with standard scorers.

For methods like RotPred, ConfBranch, CIDER, NPOS that save checkpoints with
wrapper-specific keys (backbone.*, rot_fc.*, etc.), this script:
1. Loads the wrapper checkpoint
2. Extracts the backbone state_dict (stripping 'backbone.' prefix)
3. Saves a standard ResNet18 checkpoint
4. Runs eval_ood.py with the standard checkpoint

Usage:
    python extract_backbone_and_eval.py \
        --root results/cifar100_rot_net_rotpred_e100_lr0.1_default \
        --id-data cifar100 \
        --postprocessor msp
"""
import argparse
import os
import sys
import torch
import shutil
from pathlib import Path

def extract_backbone(ckpt_path, output_path):
    """Extract backbone weights from a wrapper checkpoint."""
    state = torch.load(ckpt_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    backbone_state = {}
    for k, v in state.items():
        if k.startswith('backbone.'):
            new_key = k[len('backbone.'):]
            backbone_state[new_key] = v
        elif k.startswith('fc.'):
            backbone_state[k] = v

    if not backbone_state:
        print(f"  No backbone.* keys found in {ckpt_path}, trying raw keys")
        backbone_state = state

    torch.save(backbone_state, output_path)
    return len(backbone_state)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Results directory with seed subdirs')
    parser.add_argument('--id-data', required=True, choices=['cifar10', 'cifar100', 'imagenet200'])
    parser.add_argument('--postprocessor', required=True)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--save-csv', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    extracted_root = root.parent / (root.name + '_backbone')
    extracted_root.mkdir(exist_ok=True)

    # Extract backbone for each seed
    for seed_dir in sorted(root.glob('s*')):
        if not seed_dir.is_dir():
            continue

        ckpt = seed_dir / 'best.ckpt'
        if not ckpt.exists():
            print(f"Skipping {seed_dir}: no best.ckpt")
            continue

        out_dir = extracted_root / seed_dir.name
        out_dir.mkdir(exist_ok=True)
        out_ckpt = out_dir / 'best.ckpt'

        if out_ckpt.exists():
            print(f"  {seed_dir.name}: already extracted")
        else:
            n = extract_backbone(ckpt, out_ckpt)
            print(f"  {seed_dir.name}: extracted {n} backbone keys -> {out_ckpt}")

    # Run eval_ood.py with extracted backbone
    eval_script = Path(__file__).parent / 'eval_ood.py'
    cmd = [
        sys.executable, str(eval_script),
        '--root', str(extracted_root),
        '--id-data', args.id_data,
        '--postprocessor', args.postprocessor,
        '--batch-size', str(args.batch_size),
    ]
    if args.save_csv:
        cmd.append('--save-csv')

    print(f"\nRunning: {' '.join(cmd)}")
    os.execvp(sys.executable, cmd)

if __name__ == '__main__':
    main()
