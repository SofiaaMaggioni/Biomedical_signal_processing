"""Pre-processing functions for the fetal ECG pipeline.

The first paper steps clean the abdominal ECG before any QRS detection:

1. Baseline wander removal: remove slow movement of the signal baseline.
2. Power-line interference cancellation: remove the periodic 50 Hz electrical
   mains disturbance and selected harmonics.
3. Sampling frequency adaptation: increase the sampling frequency before QRS
   detection and maternal ECG cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BaselineRemovalResult:
    """Output of the baseline wander removal step."""

    filtered_signals: np.ndarray  # S2 signal, shape: samples x channels.
    taps: np.ndarray  # FIR filter coefficients used to obtain S2.
    cutoff_hz: float  # High-pass cutoff frequency.
    fs: float  # Sampling frequency in Hz.


@dataclass(frozen=True)
class PowerlineRemovalResult:
    """Output of the power-line interference cancellation step."""

    cleaned_signals: np.ndarray  # S3 signal, shape: samples x channels.
    estimated_interference: np.ndarray  # Component subtracted from S2.
    frequencies_hz: tuple[float, ...]  # Frequencies removed from the signal.
    fs: float  # Sampling frequency in Hz.


@dataclass(frozen=True)
class UpsamplingResult:
    """Output of the sampling frequency adaptation step."""

    upsampled_signals: np.ndarray  # S4 signal, shape: upsampled samples x channels.
    original_fs: float  # Sampling frequency of S3 in Hz.
    target_fs: float  # Sampling frequency of S4 in Hz.
    original_samples: int  # Number of samples before upsampling.
    upsampled_samples: int  # Number of samples after upsampling.


def _ensure_odd_numtaps(numtaps: int) -> int:
    """Return an odd number of FIR taps.

    High-pass FIR filters designed with ``firwin`` are safest with an odd
    number of taps because this gives a Type-I linear phase FIR filter.
    """

    return numtaps if numtaps % 2 == 1 else numtaps + 1


def _numtaps_that_fit_signal(num_samples: int, requested_numtaps: int) -> int:
    """Choose a tap count compatible with zero-phase filtering.

    ``scipy.signal.filtfilt`` pads the signal at both ends. If the filter is
    too long for the available signal, filtering fails. Full PhysioNet records
    have 60000 samples, so the default 1001 taps is fine; this helper simply
    makes the function robust if we later test shorter snippets.
    """

    numtaps = _ensure_odd_numtaps(requested_numtaps)

    # filtfilt requires the input length to be greater than padlen, where for
    # this FIR filter padlen is approximately 3 * numtaps.
    max_numtaps = max(31, (num_samples - 1) // 3)
    max_numtaps = _ensure_odd_numtaps(max_numtaps)
    if max_numtaps > num_samples:
        max_numtaps = _ensure_odd_numtaps(max(31, num_samples // 3))

    return min(numtaps, max_numtaps)


def remove_baseline_wander(
    signals: np.ndarray,
    fs: float,
    cutoff_hz: float = 3.0,
    numtaps: int = 1001,
) -> BaselineRemovalResult:
    """Remove slow baseline drift from abdominal ECG signals.

    Parameters
    ----------
    signals:
        Raw abdominal ECG signal ``S1`` with shape ``samples x channels``.
    fs:
        Sampling frequency in Hz. For Challenge 2013 Set-A, this is 1000 Hz.
    cutoff_hz:
        High-pass cutoff. Components below this frequency are attenuated.
    numtaps:
        Length of the FIR filter. A larger value gives a sharper transition
        but needs more samples for stable zero-phase filtering.

    Returns
    -------
    BaselineRemovalResult
        ``filtered_signals`` is the baseline-corrected signal ``S2``.
    """

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist frequency")

    n_samples = signals.shape[0]
    fitted_numtaps = _numtaps_that_fit_signal(n_samples, numtaps)

    # Design a linear-phase high-pass FIR filter. pass_zero=False means that
    # low frequencies are attenuated and frequencies above cutoff_hz pass.
    taps = signal.firwin(
        fitted_numtaps,
        cutoff=cutoff_hz,
        fs=fs,
        pass_zero=False,
    )

    # filtfilt applies the filter forward and backward, removing phase delay.
    # That matters because later we compare QRS positions in time.
    filtered = signal.filtfilt(taps, [1.0], signals, axis=0)

    return BaselineRemovalResult(
        filtered_signals=filtered,
        taps=taps,
        cutoff_hz=cutoff_hz,
        fs=fs,
    )


def remove_powerline_interference(
    signals: np.ndarray,
    fs: float,
    frequencies_hz: tuple[float, ...] = (50.0, 100.0, 150.0),
) -> PowerlineRemovalResult:
    """Remove sinusoidal power-line interference from ECG signals.

    The Italian/European electrical grid oscillates at 50 Hz. That periodic
    component can leak into ECG recordings and appear as a thin, regular noise.
    Harmonics such as 100 Hz and 150 Hz can also be present.

    This function estimates, for each selected frequency and each ECG channel,
    the best-fitting sine and cosine waves. Those fitted waves represent the
    power-line disturbance, so we subtract them from the signal.

    Parameters
    ----------
    signals:
        Baseline-corrected ECG signal ``S2`` with shape ``samples x channels``.
    fs:
        Sampling frequency in Hz.
    frequencies_hz:
        Power-line frequencies to remove. The default targets 50 Hz and the
        first two harmonics.

    Returns
    -------
    PowerlineRemovalResult
        ``cleaned_signals`` is the power-line-corrected signal ``S3``.
    """

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")

    nyquist = fs / 2
    valid_frequencies = tuple(float(f) for f in frequencies_hz if 0 < float(f) < nyquist)
    if not valid_frequencies:
        raise ValueError("at least one frequency must be between 0 and Nyquist frequency")

    n_samples = signals.shape[0]
    time = np.arange(n_samples) / fs
    estimated = np.zeros_like(signals, dtype=float)

    for freq in valid_frequencies:
        angle = 2 * np.pi * freq * time

        # Any sinusoid at this frequency can be written as:
        # A*sin(2*pi*f*t) + B*cos(2*pi*f*t).
        # Least squares finds A and B for every channel at once.
        basis = np.column_stack([np.sin(angle), np.cos(angle)])
        coeffs, *_ = np.linalg.lstsq(basis, signals, rcond=None)
        estimated += basis @ coeffs

    cleaned = signals - estimated

    return PowerlineRemovalResult(
        cleaned_signals=cleaned,
        estimated_interference=estimated,
        frequencies_hz=valid_frequencies,
        fs=fs,
    )


def upsample_signals(
    signals: np.ndarray,
    fs: float,
    target_fs: float = 2000.0,
) -> UpsamplingResult:
    """Increase the sampling frequency of ECG signals.

    This step converts ``S3`` into ``S4``. It does not create new physiological
    information; it only evaluates the same signal on a denser time grid. This
    is useful before QRS detection because peak positions can be localized on
    smaller time steps.

    Parameters
    ----------
    signals:
        Power-line-corrected ECG signal ``S3`` with shape ``samples x channels``.
    fs:
        Original sampling frequency in Hz.
    target_fs:
        Desired sampling frequency in Hz. The paper pipeline uses 2000 Hz.

    Returns
    -------
    UpsamplingResult
        ``upsampled_signals`` is the resampled signal ``S4``.
    """

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if target_fs <= 0:
        raise ValueError("target_fs must be positive")
    if target_fs < fs:
        raise ValueError("target_fs must be greater than or equal to fs for upsampling")

    original_samples = signals.shape[0]
    target_samples = int(round(original_samples * target_fs / fs))

    if target_samples == original_samples:
        upsampled = signals.copy()
    else:
        # scipy.signal.resample returns exactly target_samples points along the
        # selected axis. For a14, this maps 60000 samples at 1000 Hz to 120000
        # samples at 2000 Hz while preserving the 60-second duration.
        upsampled = signal.resample(signals, target_samples, axis=0)

    return UpsamplingResult(
        upsampled_signals=upsampled,
        original_fs=fs,
        target_fs=target_fs,
        original_samples=original_samples,
        upsampled_samples=upsampled.shape[0],
    )
