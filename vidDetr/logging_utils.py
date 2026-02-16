# Copyright (c) 2026. All Rights Reserved.
"""
Logging utilities for VideoDETR training and evaluation.

Provides structured logging via Python's standard logging module and
CSV-based metric tracking for training/validation progress. Replaces
the custom MetricLogger with a standard, file-friendly approach.

Usage:
    from vidDetr.logging_utils import setupLogging, MetricTracker

    logger = setupLogging(outputDir="vidDetr_weights/")
    tracker = MetricTracker(outputDir="vidDetr_weights/", phase="train")
    ...
    tracker.update(step=42, loss=0.5, lr=1e-4)
    tracker.writeEpochSummary(epoch=1)
"""

import csv
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional


def setupLogging(
    outputDir: Optional[str] = None,
    logFilename: str = "training.log",
    level: int = logging.INFO,
    distributed: bool = False,
    rank: int = 0
) -> logging.Logger:
    """
    Configure the root 'vidDetr' logger with console and file handlers.

    Only rank-0 logs in distributed training. The file handler uses
    unbuffered writes so lines are immediately visible with ``tail -f``.

    Args:
        outputDir: Directory for the log file. ``None`` means console only.
        logFilename: Name of the log file inside *outputDir*.
        level: Logging level (default ``logging.INFO``).
        distributed: Whether distributed training is active.
        rank: Process rank; non-zero ranks are silenced.

    Returns:
        Configured ``logging.Logger`` instance named ``'vidDetr'``.
    """
    logger = logging.getLogger("vidDetr")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers (avoid duplicate handlers on re-init)
    logger.handlers.clear()

    # Silence non-master processes
    if distributed and rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout) — always present
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(level)
    consoleHandler.setFormatter(fmt)
    logger.addHandler(consoleHandler)

    # File handler — append mode, line-buffered for ``tail -f`` friendliness
    if outputDir is not None:
        os.makedirs(outputDir, exist_ok=True)
        logPath = os.path.join(outputDir, logFilename)
        fileHandler = logging.FileHandler(logPath, mode="a", encoding="utf-8")
        fileHandler.setLevel(level)
        fileHandler.setFormatter(fmt)
        logger.addHandler(fileHandler)

    return logger


class MetricTracker:
    """
    Tracks per-batch and per-epoch metrics and persists them to CSV files.

    Two CSV files are created:

    * ``<phase>_batches.csv`` — one row per batch (step-level granularity).
    * ``<phase>_epochs.csv`` — one row per epoch (epoch-level summary).

    The ``phase`` is typically ``"train"`` or ``"val"``.

    Metrics are accumulated for the current epoch and a summary
    (mean, min, max, last) is written at epoch end.
    """

    def __init__(
        self,
        outputDir: str,
        phase: str = "train",
    ):
        """
        Args:
            outputDir: Directory where CSV files are created.
            phase: Prefix for CSV filenames (e.g. ``"train"``, ``"val"``).
        """
        self.outputDir = outputDir
        self.phase = phase
        self.logger = logging.getLogger("vidDetr")

        os.makedirs(outputDir, exist_ok=True)

        self._batchPath = os.path.join(outputDir, f"{phase}_batches.csv")
        self._epochPath = os.path.join(outputDir, f"{phase}_epochs.csv")

        # Accumulated values for the current epoch
        self._epochAccum: Dict[str, list] = defaultdict(list)
        self._currentEpoch: int = 0
        self._epochStartTime: float = 0.0

        # Track whether CSV headers have been written
        self._batchHeaderWritten = os.path.exists(self._batchPath)
        self._epochHeaderWritten = os.path.exists(self._epochPath)

    def epochStart(self, epoch: int) -> None:
        """Call at the beginning of an epoch to reset accumulators."""
        self._currentEpoch = epoch
        self._epochAccum.clear()
        self._epochStartTime = time.time()

    def update(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Record metrics for a single batch and append to the batch CSV.

        Args:
            step: Global step or batch index within the epoch.
            metrics: Dict of metric names to scalar values.
        """
        row = {"epoch": self._currentEpoch, "step": step, **metrics}

        # Accumulate for epoch summary
        for k, v in metrics.items():
            self._epochAccum[k].append(v)

        # Append to batch CSV
        self._appendCsv(self._batchPath, row, self._batchHeaderWritten)
        self._batchHeaderWritten = True

    def writeEpochSummary(self, extraMetrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Compute and write an epoch-level summary row to the epoch CSV.

        For every accumulated metric, the summary contains
        ``<metric>_mean``, ``<metric>_last``, ``<metric>_min``, and
        ``<metric>_max`` columns so the reader can distinguish between
        the epoch average and the final batch value.

        Args:
            extraMetrics: Additional metrics (e.g., num_samples) to include.

        Returns:
            Dict of mean values for each metric (the primary summary).
        """
        elapsed = time.time() - self._epochStartTime
        summary: Dict[str, Any] = {
            "epoch": self._currentEpoch,
            "epoch_time_s": round(elapsed, 1),
        }

        for k, values in self._epochAccum.items():
            if not values:
                continue
            summary[f"{k}_mean"] = sum(values) / len(values)
            summary[f"{k}_last"] = values[-1]
            summary[f"{k}_min"] = min(values)
            summary[f"{k}_max"] = max(values)

        if extraMetrics:
            summary.update(extraMetrics)

        self._appendCsv(self._epochPath, summary, self._epochHeaderWritten)
        self._epochHeaderWritten = True

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _appendCsv(path: str, row: Dict[str, Any], headerWritten: bool) -> None:
        """Append a single row to a CSV file, writing the header if needed."""
        fileExists = os.path.exists(path)
        mode = "a" if fileExists else "w"

        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            # Write header on first write, or when the file is new
            if not headerWritten or not fileExists:
                writer.writeheader()

            writer.writerow({k: _fmtValue(v) for k, v in row.items()})


def _fmtValue(v: Any) -> str:
    """Format a value for CSV output."""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
