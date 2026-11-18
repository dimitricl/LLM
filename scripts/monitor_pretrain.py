#!/usr/bin/env python3
"""
Monitor pretraining progress from runs/pretrain_v8/log.jsonl.

Reads log entries and displays:
- Latest step, loss, learning rate, gradient norm
- Smoothed loss over last 500 steps
- Validation loss (if available)
- Slope over last 500 steps
- ETA based on current tokens/s
- Alerts for: NaN loss, val_loss rising for 3 consecutive evals, tokens/s drop > 20%

Pure read-only, no writes to runs/.
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


def load_log_entries(log_path: Path) -> List[Dict]:
    """Load all JSONL entries, keeping only the latest entry per step."""
    entries = []
    latest_by_step = {}

    for line in log_path.open():
        try:
            entry = json.loads(line.strip())
            step = entry.get("step")
            if step is not None:
                latest_by_step[step] = entry
        except (json.JSONDecodeError, ValueError):
            continue

    # Sort by step and return
    entries = [latest_by_step[step] for step in sorted(latest_by_step.keys())]
    return entries


def extract_metrics(entries: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Separate training and validation entries."""
    train_entries = []
    val_entries = []

    for entry in entries:
        if "val_loss" in entry:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    return train_entries, val_entries


def compute_smoothed_loss(train_entries: List[Dict], window: int = 500) -> Optional[float]:
    """Compute exponentially weighted moving average over last window steps."""
    if not train_entries:
        return None

    # Take last window entries (or all if fewer)
    recent = train_entries[-window:] if len(train_entries) >= window else train_entries

    losses = [e.get("loss") for e in recent if e.get("loss") is not None]
    if not losses:
        return None

    # Check for NaN
    if any(np.isnan(l) for l in losses):
        return np.nan

    # Simple moving average (could be EWMA)
    return np.mean(losses)


def compute_slope(train_entries: List[Dict], window: int = 500) -> Optional[float]:
    """Compute linear regression slope of loss over last window steps."""
    if len(train_entries) < 2:
        return None

    recent = train_entries[-window:] if len(train_entries) >= window else train_entries

    steps = [e.get("step") for e in recent if e.get("loss") is not None]
    losses = [e.get("loss") for e in recent if e.get("loss") is not None]

    if len(steps) < 2:
        return None

    # Linear regression: loss = a * step + b
    x = np.array(steps)
    y = np.array(losses)

    # Check for NaN
    if np.isnan(y).any():
        return np.nan

    # Slope using least squares
    slope = np.polyfit(x, y, 1)[0]
    return slope


def compute_eta(train_entries: List[Dict], max_steps: int = 4700) -> Optional[str]:
    """Compute ETA based on current tokens/s."""
    if not train_entries:
        return None

    latest = train_entries[-1]
    current_step = latest.get("step", 0)
    tokens_per_s = latest.get("tokens_per_s")

    if tokens_per_s is None or tokens_per_s == 0:
        return None

    tokens_per_step = 131072  # From config: micro_batch=4, grad_accum=64, context=512
    remaining_steps = max_steps - current_step
    remaining_tokens = remaining_steps * tokens_per_step

    seconds_remaining = remaining_tokens / tokens_per_s

    if seconds_remaining < 60:
        return f"{seconds_remaining:.0f}s"
    elif seconds_remaining < 3600:
        return f"{seconds_remaining / 60:.1f}m"
    else:
        return f"{seconds_remaining / 3600:.1f}h"


def check_val_loss_rising(val_entries: List[Dict], patience: int = 3) -> Tuple[bool, Optional[str]]:
    """Check if val_loss has risen for patience consecutive evaluations."""
    if len(val_entries) < patience:
        return False, None

    recent = val_entries[-patience:]
    val_losses = [e.get("val_loss") for e in recent if e.get("val_loss") is not None]

    if len(val_losses) < patience:
        return False, None

    # Check if each is higher than the previous
    for i in range(1, len(val_losses)):
        if val_losses[i] <= val_losses[i-1]:
            return False, None

    return True, f"val_loss rising: {val_losses}"


def check_tokens_per_s_drop(train_entries: List[Dict], threshold: float = 0.2, window: int = 100) -> Tuple[bool, Optional[str]]:
    """Check if tokens/s has dropped by more than threshold compared to average over window."""
    if len(train_entries) < window + 1:
        return False, None

    latest = train_entries[-1]
    current_tps = latest.get("tokens_per_s")

    if current_tps is None:
        return False, None

    # Average over window steps ago
    window_ago = train_entries[-(window+1)]
    baseline_tps = window_ago.get("tokens_per_s")

    if baseline_tps is None or baseline_tps == 0:
        return False, None

    drop = (baseline_tps - current_tps) / baseline_tps

    if drop > threshold:
        return True, f"tokens/s dropped {drop*100:.1f}%: {baseline_tps:.0f} -> {current_tps:.0f}"

    return False, None


def format_number(value: float, decimals: int = 4) -> str:
    """Format number, handling NaN and None."""
    if value is None:
        return "N/A"
    if np.isnan(value):
        return "NaN"
    return f"{value:.{decimals}f}"


def main():
    parser = argparse.ArgumentParser(description="Monitor pretraining progress")
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("runs/pretrain_v8/log.jsonl"),
        help="Path to log.jsonl file"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=4700,
        help="Maximum training steps for ETA calculation"
    )
    parser.add_argument(
        "--val-patience",
        type=int,
        default=3,
        help="Patience for val_loss rising alert"
    )
    parser.add_argument(
        "--tps-drop-threshold",
        type=float,
        default=0.2,
        help="Threshold for tokens/s drop alert (0.2 = 20%)"
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"Error: Log file not found: {args.log_path}")
        sys.exit(1)

    # Load entries
    entries = load_log_entries(args.log_path)
    if not entries:
        print("No valid entries in log file")
        sys.exit(1)

    train_entries, val_entries = extract_metrics(entries)

    # Get latest metrics
    if train_entries:
        latest = train_entries[-1]
        step = latest.get("step")
        tokens = latest.get("tokens")
        loss = latest.get("loss")
        lr = latest.get("lr")
        grad_norm = latest.get("grad_norm")
        tokens_per_s = latest.get("tokens_per_s")
        peak_memory_mb = latest.get("peak_memory_mb")
    else:
        print("No training entries found")
        sys.exit(1)

    # Compute derived metrics
    smoothed_loss = compute_smoothed_loss(train_entries)
    slope = compute_slope(train_entries)
    eta = compute_eta(train_entries, args.max_steps)

    # Check for alerts
    alerts = []

    # NaN check
    if loss is not None and np.isnan(loss):
        alerts.append("⚠️  NaN loss detected!")

    if smoothed_loss is not None and np.isnan(smoothed_loss):
        alerts.append("⚠️  NaN in smoothed loss!")

    # Val loss rising
    val_rising, val_msg = check_val_loss_rising(val_entries, args.val_patience)
    if val_rising:
        alerts.append(f"⚠️  {val_msg}")

    # Tokens/s drop
    tps_drop, tps_msg = check_tokens_per_s_drop(train_entries, args.tps_drop_threshold)
    if tps_drop:
        alerts.append(f"⚠️  {tps_msg}")

    # Get latest val loss
    latest_val_loss = None
    if val_entries:
        latest_val_loss = val_entries[-1].get("val_loss")

    # Print report
    print("=" * 60)
    print("Pretraining Monitor")
    print("=" * 60)
    print(f"Step:        {step} / {args.max_steps} ({step/args.max_steps*100:.1f}%)")
    print(f"Tokens:      {tokens:,} / {args.max_steps * 131072:,}")
    print(f"Loss:        {format_number(loss)}")
    if smoothed_loss is not None:
        print(f"Loss (500s): {format_number(smoothed_loss)}")
    if latest_val_loss is not None:
        print(f"Val Loss:    {format_number(latest_val_loss)}")
    print(f"LR:          {format_number(lr, 6)}")
    print(f"Grad Norm:   {format_number(grad_norm, 4)}")
    print(f"Tokens/s:    {format_number(tokens_per_s, 0)}")
    print(f"Peak Memory: {peak_memory_mb:.0f} MB")
    if slope is not None:
        print(f"Slope (500s): {format_number(slope, 6)}")
    if eta is not None:
        print(f"ETA:         {eta}")
    print("=" * 60)

    if alerts:
        print("Alerts:")
        for alert in alerts:
            print(f"  {alert}")
        print("=" * 60)

    # Progress bar
    progress = step / args.max_steps
    bar_length = 40
    filled = int(bar_length * progress)
    bar = "[" + "=" * filled + ">" + " " * (bar_length - filled) + "]"
    print(f"Progress: {bar}")


if __name__ == "__main__":
    main()
