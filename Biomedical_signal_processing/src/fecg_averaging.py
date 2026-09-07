"""Fetal ECG averaging for the final S5-to-S6 pipeline step.

After fetal QRS detection, each detected fetal beat gives us an alignment point.
This module extracts short S5 windows around those points and averages them.
The average waveform is the final S6 representation of the fetal ECG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FECGAverageResult:
    """Output of fetal ECG averaging around detected fetal QRS positions."""

    average_fecg: np.ndarray  # S6 average waveform, shape: window samples x channels.
    segments: np.ndarray  # Individual aligned fetal-beat windows before averaging.
    used_fqrs_samples: np.ndarray  # FQRS positions that had a complete window.
    rejected_fqrs_samples: np.ndarray  # FQRS positions too close to signal edges.
    before_samples: int  # Samples included before each fetal QRS.
    after_samples: int  # Samples included after each fetal QRS.
    fs: float  # Sampling frequency in Hz.
    before_s: float  # Seconds included before each fetal QRS.
    after_s: float  # Seconds included after each fetal QRS.
    baseline_s: float  # Length of the pre-QRS baseline segment removed from each beat.
    snr_gain_factor: float  # Expected SNR gain for uncorrelated noise: sqrt(number of beats).

    @property
    def relative_time_s(self) -> np.ndarray:
        """Time axis of the averaged beat, centered on the fetal QRS."""

        return (np.arange(self.average_fecg.shape[0]) - self.before_samples) / self.fs


def _as_2d_float(signals: np.ndarray) -> np.ndarray:
    """Validate that the signal array has shape samples x channels."""

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if signals.shape[0] < 3:
        raise ValueError("signals must contain at least 3 samples")
    return signals


def _validate_fqrs_samples(fqrs_samples: np.ndarray, n_samples: int) -> np.ndarray:
    """Return sorted, unique fetal QRS samples inside the signal bounds."""

    fqrs_samples = np.asarray(fqrs_samples, dtype=int)
    if fqrs_samples.ndim != 1:
        raise ValueError("fqrs_samples must be one-dimensional")
    fqrs_samples = np.unique(fqrs_samples)
    fqrs_samples = fqrs_samples[(fqrs_samples >= 0) & (fqrs_samples < n_samples)]
    if len(fqrs_samples) == 0:
        raise ValueError("fqrs_samples does not contain any valid sample")
    return fqrs_samples


def average_fetal_ecg(
    signals: np.ndarray,
    fqrs_samples: np.ndarray,
    fs: float,
    before_s: float = 0.2,
    after_s: float = 0.2,
    baseline_s: float = 0.05,
) -> FECGAverageResult:
    """Average S5 windows centered on detected fetal QRS positions.

    ``signals`` is normally S5. The function extracts one short segment around
    each detected fetal QRS. Segments that would cross the signal boundaries are
    rejected. Before averaging, a small pre-QRS baseline is subtracted from each
    segment so that slow offsets do not dominate the final average.
    """

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if before_s <= 0 or after_s <= 0:
        raise ValueError("before_s and after_s must be positive")
    if baseline_s < 0:
        raise ValueError("baseline_s must be non-negative")

    n_samples, _ = signals.shape
    fqrs_samples = _validate_fqrs_samples(fqrs_samples, n_samples)
    before_samples = int(round(before_s * fs))
    after_samples = int(round(after_s * fs))
    if before_samples <= 0 or after_samples <= 0:
        raise ValueError("before_s and after_s must select at least one sample")

    complete_window_mask = (
        (fqrs_samples >= before_samples)
        & (fqrs_samples + after_samples < n_samples)
    )
    used_fqrs = fqrs_samples[complete_window_mask]
    rejected_fqrs = fqrs_samples[~complete_window_mask]
    if len(used_fqrs) == 0:
        raise ValueError("no FQRS positions are far enough from signal edges")

    segments = np.stack(
        [
            signals[peak - before_samples : peak + after_samples + 1].copy()
            for peak in used_fqrs
        ],
        axis=0,
    )

    # Remove a local pre-QRS baseline from every segment and channel. This keeps
    # the average focused on the beat shape instead of small residual offsets.
    baseline_samples = int(round(baseline_s * fs))
    baseline_samples = min(max(baseline_samples, 0), before_samples)
    if baseline_samples > 0:
        baseline = np.mean(segments[:, :baseline_samples, :], axis=1, keepdims=True)
        segments = segments - baseline

    average_fecg = np.mean(segments, axis=0)
    snr_gain_factor = float(np.sqrt(len(used_fqrs)))

    return FECGAverageResult(
        average_fecg=average_fecg,
        segments=segments,
        used_fqrs_samples=used_fqrs.astype(int),
        rejected_fqrs_samples=rejected_fqrs.astype(int),
        before_samples=before_samples,
        after_samples=after_samples,
        fs=fs,
        before_s=before_s,
        after_s=after_s,
        baseline_s=baseline_s,
        snr_gain_factor=snr_gain_factor,
    )
