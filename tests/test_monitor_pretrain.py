#!/usr/bin/env python3
"""Tests for monitor_pretrain.py"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from scripts.monitor_pretrain import (
    load_log_entries,
    extract_metrics,
    compute_smoothed_loss,
    compute_slope,
    compute_eta,
    check_val_loss_rising,
    check_tokens_per_s_drop,
)


def test_load_log_entries_basic():
    """Test basic loading of log entries."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"step": 10, "loss": 8.5}\n')
        f.write('{"step": 20, "loss": 8.0}\n')
        f.write('{"step": 30, "loss": 7.5}\n')
        path = Path(f.name)

    try:
        entries = load_log_entries(path)
        assert len(entries) == 3
        assert entries[0]["step"] == 10
        assert entries[2]["step"] == 30
    finally:
        path.unlink()


def test_load_log_entries_duplicates():
    """Test that only latest entry per step is kept."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"step": 10, "loss": 8.5}\n')
        f.write('{"step": 20, "loss": 8.0}\n')
        f.write('{"step": 10, "loss": 8.3}\n')  # Duplicate step, should replace
        f.write('{"step": 30, "loss": 7.5}\n')
        path = Path(f.name)

    try:
        entries = load_log_entries(path)
        assert len(entries) == 3
        assert entries[0]["step"] == 10
        assert entries[0]["loss"] == 8.3  # Latest value
    finally:
        path.unlink()


def test_extract_metrics():
    """Test separation of train and val entries."""
    entries = [
        {"step": 10, "loss": 8.5},
        {"step": 20, "loss": 8.0},
        {"step": 10, "val_loss": 8.6},
        {"step": 20, "val_loss": 8.1},
    ]

    train, val = extract_metrics(entries)
    assert len(train) == 2
    assert len(val) == 2
    assert train[0]["loss"] == 8.5
    assert val[0]["val_loss"] == 8.6


def test_compute_smoothed_loss():
    """Test smoothed loss computation."""
    entries = [
        {"step": i, "loss": 10.0 - i * 0.1}
        for i in range(10, 101, 10)
    ]

    smoothed = compute_smoothed_loss(entries, window=500)
    assert smoothed is not None
    assert smoothed < 10.0  # Should be lower than initial


def test_compute_smoothed_loss_nan():
    """Test NaN detection in smoothed loss."""
    entries = [
        {"step": i, "loss": 10.0 - i * 0.1}
        for i in range(10, 51, 10)
    ]
    entries.append({"step": 60, "loss": float("nan")})

    smoothed = compute_smoothed_loss(entries, window=500)
    assert np.isnan(smoothed)


def test_compute_slope():
    """Test slope computation."""
    entries = [
        {"step": i, "loss": 10.0 - i * 0.1}
        for i in range(10, 101, 10)
    ]

    slope = compute_slope(entries, window=500)
    assert slope is not None
    assert slope < 0  # Should be negative (loss decreasing)


def test_compute_slope_insufficient_data():
    """Test slope with insufficient data."""
    entries = [{"step": 10, "loss": 8.5}]
    slope = compute_slope(entries, window=500)
    assert slope is None


def test_compute_eta():
    """Test ETA computation."""
    entries = [
        {"step": 100, "tokens_per_s": 7870.0}
    ]

    eta = compute_eta(entries, max_steps=4700)
    assert eta is not None
    assert "h" in eta  # Should be in hours


def test_compute_eta_no_tps():
    """Test ETA with missing tokens_per_s."""
    entries = [{"step": 100}]
    eta = compute_eta(entries, max_steps=4700)
    assert eta is None


def test_check_val_loss_rising():
    """Test val loss rising detection."""
    val_entries = [
        {"step": 100, "val_loss": 8.0},
        {"step": 200, "val_loss": 8.1},
        {"step": 300, "val_loss": 8.2},
    ]

    rising, msg = check_val_loss_rising(val_entries, patience=3)
    assert rising is True
    assert "rising" in msg.lower()


def test_check_val_loss_rising_not_triggered():
    """Test val loss not rising."""
    val_entries = [
        {"step": 100, "val_loss": 8.2},
        {"step": 200, "val_loss": 8.1},
        {"step": 300, "val_loss": 8.0},
    ]

    rising, msg = check_val_loss_rising(val_entries, patience=3)
    assert rising is False
    assert msg is None


def test_check_tokens_per_s_drop():
    """Test tokens/s drop detection."""
    train_entries = [
        {"step": 100, "tokens_per_s": 7870.0},
        {"step": 200, "tokens_per_s": 7870.0},
    ] + [
        {"step": i, "tokens_per_s": 6000.0}
        for i in range(300, 310)
    ]

    drop, msg = check_tokens_per_s_drop(train_entries, threshold=0.2, window=10)
    assert drop is True
    assert "dropped" in msg.lower()


def test_check_tokens_per_s_drop_not_triggered():
    """Test tokens/s not dropping."""
    train_entries = [
        {"step": i, "tokens_per_s": 7870.0}
        for i in range(100, 310, 10)
    ]

    drop, msg = check_tokens_per_s_drop(train_entries, threshold=0.2, window=10)
    assert drop is False
    assert msg is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
