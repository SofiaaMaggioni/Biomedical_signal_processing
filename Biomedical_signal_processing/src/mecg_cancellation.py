"""Maternal ECG cancellation for the fetal ECG pipeline.

This step converts ``S4`` into ``S5``. The input signal still contains a strong
maternal ECG component. Since the maternal QRS positions were detected in the
previous step, we can estimate the maternal ECG waveform around each maternal
beat and subtract that estimate from the abdominal channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MECGCancellationResult:
    """Output of the maternal ECG cancellation step."""

    cancelled_signals: np.ndarray  # S5 signal, shape: samples x channels.
    estimated_mecg: np.ndarray  # Maternal ECG estimate subtracted from S4.
    mqrs_samples: np.ndarray  # Input maternal QRS positions in S4 sample indices.
    cancelled_mqrs_samples: np.ndarray  # MQRS positions for which a template was available.
    scale_factors: np.ndarray  # Per-beat/per-channel P, QRS, and T scale factors.
    fs: float  # Sampling frequency in Hz.
    preceding_beats: int  # Number of previous maternal beats used for each estimate.
    before_s: float  # Seconds included before each maternal R peak.
    after_s: float  # Seconds included after each maternal R peak.
    qrs_half_width_s: float  # Half-width of the QRS zone around the R peak.
    p_range_samples: tuple[int, int]  # P-wave range inside one MECG segment.
    qrs_range_samples: tuple[int, int]  # QRS range inside one MECG segment.
    t_range_samples: tuple[int, int]  # T-wave range inside one MECG segment.


def _as_2d_float(signals: np.ndarray) -> np.ndarray:
    """Validate that the signal has shape samples x channels."""

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if signals.shape[0] < 3:
        raise ValueError("signals must contain at least 3 samples")
    return signals


def _validate_mqrs_samples(mqrs_samples: np.ndarray, n_samples: int) -> np.ndarray:
    """Return sorted, unique maternal QRS samples inside the signal bounds."""

    mqrs_samples = np.asarray(mqrs_samples, dtype=int)
    if mqrs_samples.ndim != 1:
        raise ValueError("mqrs_samples must be one-dimensional")
    mqrs_samples = np.unique(mqrs_samples)
    mqrs_samples = mqrs_samples[(mqrs_samples >= 0) & (mqrs_samples < n_samples)]
    if len(mqrs_samples) == 0:
        raise ValueError("mqrs_samples does not contain any valid sample")
    return mqrs_samples


def _segment_ranges(
    fs: float,
    before_s: float,
    after_s: float,
    qrs_half_width_s: float,
) -> tuple[int, int, slice, slice, slice]:
    """Compute the sample ranges used for P, QRS, and T scaling."""

    if fs <= 0:
        raise ValueError("fs must be positive")
    if before_s <= 0 or after_s <= 0:
        raise ValueError("before_s and after_s must be positive")
    if qrs_half_width_s <= 0:
        raise ValueError("qrs_half_width_s must be positive")
    if qrs_half_width_s >= before_s or qrs_half_width_s >= after_s:
        raise ValueError("qrs_half_width_s must fit inside the MECG segment")

    before_samples = int(round(before_s * fs))
    after_samples = int(round(after_s * fs))
    qrs_half_samples = int(round(qrs_half_width_s * fs))
    segment_len = before_samples + after_samples + 1

    qrs_start = before_samples - qrs_half_samples
    qrs_stop = before_samples + qrs_half_samples + 1

    p_slice = slice(0, qrs_start)
    qrs_slice = slice(qrs_start, qrs_stop)
    t_slice = slice(qrs_stop, segment_len)
    return before_samples, after_samples, p_slice, qrs_slice, t_slice


def _design_matrix_from_average(
    average_segment: np.ndarray,
    p_slice: slice,
    qrs_slice: slice,
    t_slice: slice,
) -> np.ndarray:
    """Create a three-column matrix for separate P, QRS, and T scaling."""

    segment_len = len(average_segment)
    design = np.zeros((segment_len, 3), dtype=float)
    design[p_slice, 0] = average_segment[p_slice]
    design[qrs_slice, 1] = average_segment[qrs_slice]
    design[t_slice, 2] = average_segment[t_slice]
    return design


def cancel_maternal_ecg(
    signals: np.ndarray,
    mqrs_samples: np.ndarray,
    fs: float,
    preceding_beats: int = 10,
    before_s: float = 0.25,
    after_s: float = 0.45,
    qrs_half_width_s: float = 0.05,
) -> MECGCancellationResult:
    """Estimate and subtract the maternal ECG from ``S4``.

    For each maternal beat, the function looks at the previous
    ``preceding_beats`` maternal beats. Their aligned ECG segments are averaged
    to obtain a local maternal ECG estimate. The estimate is then split into
    P-wave, QRS-complex, and T-wave zones. Each zone gets its own least-squares
    scale factor before subtraction, because maternal ECG morphology can change
    slightly over time and across abdominal channels.
    """

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if preceding_beats <= 0:
        raise ValueError("preceding_beats must be positive")

    n_samples, n_channels = signals.shape
    mqrs_samples = _validate_mqrs_samples(mqrs_samples, n_samples)
    before_samples, after_samples, p_slice, qrs_slice, t_slice = _segment_ranges(
        fs=fs,
        before_s=before_s,
        after_s=after_s,
        qrs_half_width_s=qrs_half_width_s,
    )
    segment_len = before_samples + after_samples + 1

    estimated_sum = np.zeros_like(signals, dtype=float)
    estimate_weight = np.zeros((n_samples, 1), dtype=float)
    scale_factors: list[np.ndarray] = []
    cancelled_mqrs: list[int] = []

    for beat_idx in range(preceding_beats, len(mqrs_samples)):
        current_peak = int(mqrs_samples[beat_idx])
        current_start = current_peak - before_samples
        current_stop = current_peak + after_samples + 1
        if current_start < 0 or current_stop > n_samples:
            continue

        previous_peaks = mqrs_samples[beat_idx - preceding_beats : beat_idx]
        previous_segments = []
        for previous_peak in previous_peaks:
            previous_start = int(previous_peak) - before_samples
            previous_stop = int(previous_peak) + after_samples + 1
            if previous_start < 0 or previous_stop > n_samples:
                break
            previous_segments.append(signals[previous_start:previous_stop])

        if len(previous_segments) != preceding_beats:
            continue

        current_segment = signals[current_start:current_stop]
        average_segment = np.mean(np.stack(previous_segments, axis=0), axis=0)
        estimated_segment = np.zeros_like(current_segment, dtype=float)
        beat_scale_factors = np.zeros((n_channels, 3), dtype=float)

        for channel in range(n_channels):
            design = _design_matrix_from_average(
                average_segment[:, channel],
                p_slice=p_slice,
                qrs_slice=qrs_slice,
                t_slice=t_slice,
            )
            # Least squares chooses the three scale factors that best adapt the
            # average maternal complex to the current maternal complex.
            factors, *_ = np.linalg.lstsq(design, current_segment[:, channel], rcond=None)
            beat_scale_factors[channel] = factors
            estimated_segment[:, channel] = design @ factors

        # MECG windows can overlap at faster heart rates. We average overlapping
        # estimates so the same time sample is not subtracted twice.
        estimated_sum[current_start:current_stop] += estimated_segment
        estimate_weight[current_start:current_stop] += 1.0
        scale_factors.append(beat_scale_factors)
        cancelled_mqrs.append(current_peak)

    estimated_mecg = np.zeros_like(signals, dtype=float)
    valid = estimate_weight[:, 0] > 0
    estimated_mecg[valid] = estimated_sum[valid] / estimate_weight[valid]
    cancelled_signals = signals - estimated_mecg

    if scale_factors:
        scale_factors_array = np.stack(scale_factors, axis=0)
    else:
        scale_factors_array = np.empty((0, n_channels, 3), dtype=float)

    return MECGCancellationResult(
        cancelled_signals=cancelled_signals,
        estimated_mecg=estimated_mecg,
        mqrs_samples=mqrs_samples,
        cancelled_mqrs_samples=np.asarray(cancelled_mqrs, dtype=int),
        scale_factors=scale_factors_array,
        fs=fs,
        preceding_beats=preceding_beats,
        before_s=before_s,
        after_s=after_s,
        qrs_half_width_s=qrs_half_width_s,
        p_range_samples=(p_slice.start or 0, p_slice.stop or 0),
        qrs_range_samples=(qrs_slice.start or 0, qrs_slice.stop or 0),
        t_range_samples=(t_slice.start or 0, t_slice.stop or 0),
    )


def qrs_window_rms(
    signals: np.ndarray,
    mqrs_samples: np.ndarray,
    fs: float,
    half_width_s: float = 0.05,
) -> np.ndarray:
    """Compute RMS amplitude around maternal QRS positions for each channel."""

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if half_width_s <= 0:
        raise ValueError("half_width_s must be positive")

    n_samples, n_channels = signals.shape
    mqrs_samples = _validate_mqrs_samples(mqrs_samples, n_samples)
    half_width_samples = int(round(half_width_s * fs))
    windows = []

    for peak in mqrs_samples:
        start = int(peak) - half_width_samples
        stop = int(peak) + half_width_samples + 1
        if start >= 0 and stop <= n_samples:
            windows.append(signals[start:stop])

    if not windows:
        return np.full(n_channels, np.nan)

    stacked = np.concatenate(windows, axis=0)
    return np.sqrt(np.mean(stacked**2, axis=0))
