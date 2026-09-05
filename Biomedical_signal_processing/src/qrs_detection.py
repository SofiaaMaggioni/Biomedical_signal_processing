"""QRS-detection helpers for the fetal ECG pipeline.

The paper uses the same general QRS detector twice:

1. On ``S4`` to find maternal QRS complexes.
2. Later, on ``S5`` to find fetal QRS complexes after maternal ECG removal.

This module starts with the maternal case. The detector follows the paper idea:
combine the abdominal channels with PCA, build a QRS template, and detect QRS
positions from the cross-correlation between the signal and that template.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class PCAFirstComponentResult:
    """Output of the multi-channel QRS enhancement step."""

    enhanced_signal: np.ndarray  # One-dimensional signal from the first PCA component.
    components: np.ndarray  # PCA component weights, shape: components x channels.
    explained_variance_ratio: np.ndarray  # Fraction of variance explained by each component.
    channel_mean: np.ndarray  # Mean removed from each input channel before PCA.
    channel_std: np.ndarray  # Standard deviation used to normalize each channel.


@dataclass(frozen=True)
class QRSTemplateResult:
    """Output of the QRS template-building step."""

    template: np.ndarray  # Average QRS shape used for cross-correlation.
    used_qrs_samples: np.ndarray  # Initial QRS candidates that contributed to the template.
    before_samples: int  # Samples included before each QRS candidate.
    after_samples: int  # Samples included after each QRS candidate.
    fs: float  # Sampling frequency in Hz.


@dataclass(frozen=True)
class MaternalQRSDetectionResult:
    """Complete output of maternal QRS detection on S4."""

    mqrs_samples: np.ndarray  # Final maternal QRS positions in S4 sample indices.
    initial_qrs_samples: np.ndarray  # Initial candidates used before template matching.
    pca_signal: np.ndarray  # First PCA component, one sample per S4 sample.
    qrs_enhanced_signal: np.ndarray  # Band-pass filtered PCA signal used for detection.
    correlation_signal: np.ndarray  # Normalized cross-correlation with the QRS template.
    qrs_template: np.ndarray  # Template used by the cross-correlation detector.
    template_before_samples: int  # Samples before the QRS peak in the template.
    template_after_samples: int  # Samples after the QRS peak in the template.
    explained_variance_ratio: np.ndarray  # PCA variance ratios.
    fs: float  # Sampling frequency in Hz.


@dataclass(frozen=True)
class FetalQRSDetectionResult:
    """Complete output of fetal QRS detection on S5."""

    fqrs_samples: np.ndarray  # Final fetal QRS positions in S5 sample indices.
    detection_signal: np.ndarray  # One-dimensional signal used to find FQRS peaks.
    filtered_signals: np.ndarray  # Band-pass filtered S5 channels.
    channel_energy: np.ndarray  # Integrated QRS energy for each S5 channel.
    threshold: float  # Peak-detection threshold applied to detection_signal.
    fs: float  # Sampling frequency in Hz.
    low_hz: float  # Lower cutoff of the fetal QRS band-pass filter.
    high_hz: float  # Upper cutoff of the fetal QRS band-pass filter.
    integration_window_s: float  # Moving-average window used for QRS energy.


def _as_2d_float(signals: np.ndarray) -> np.ndarray:
    """Validate that a signal array has shape samples x channels."""

    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if signals.shape[0] < 3:
        raise ValueError("signals must contain at least 3 samples")
    return signals


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    """Return a z-scored copy, guarding against a zero standard deviation."""

    values = np.asarray(values, dtype=float)
    scale = np.std(values)
    if scale == 0:
        return values - np.mean(values)
    return (values - np.mean(values)) / scale


def pca_first_component(signals: np.ndarray) -> PCAFirstComponentResult:
    """Combine abdominal ECG channels using the first PCA component.

    Each channel is first centered and normalized. This avoids letting one
    channel dominate the PCA only because it has a larger amplitude scale.
    The first component is the linear combination with maximum variance and,
    for this step, usually emphasizes the strong maternal ECG activity.
    """

    signals = _as_2d_float(signals)

    channel_mean = np.mean(signals, axis=0)
    centered = signals - channel_mean
    channel_std = np.std(centered, axis=0, ddof=1)
    channel_std = np.where(channel_std == 0, 1.0, channel_std)
    normalized = centered / channel_std

    # SVD is a stable way to compute PCA without adding another dependency.
    # Rows are time samples, columns are ECG channels.
    _, singular_values, components = np.linalg.svd(normalized, full_matrices=False)
    enhanced_signal = normalized @ components[0]

    eigenvalues = singular_values**2 / max(normalized.shape[0] - 1, 1)
    total_variance = np.sum(eigenvalues)
    if total_variance == 0:
        explained_variance_ratio = np.zeros_like(eigenvalues)
    else:
        explained_variance_ratio = eigenvalues / total_variance

    return PCAFirstComponentResult(
        enhanced_signal=enhanced_signal,
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        channel_mean=channel_mean,
        channel_std=channel_std,
    )


def qrs_bandpass_filter(
    signal_1d: np.ndarray,
    fs: float,
    low_hz: float = 8.0,
    high_hz: float = 40.0,
    order: int = 3,
) -> np.ndarray:
    """Keep the frequency band where QRS complexes are usually prominent."""

    signal_1d = np.asarray(signal_1d, dtype=float)
    if signal_1d.ndim != 1:
        raise ValueError("signal_1d must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("low_hz and high_hz must be between 0 and Nyquist frequency")

    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, signal_1d)


def _qrs_bandpass_filter_channels(
    signals: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> np.ndarray:
    """Apply the same QRS band-pass filter to every ECG channel."""

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError("low_hz and high_hz must be between 0 and Nyquist frequency")

    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, signals, axis=0)


def _moving_average_channels(values: np.ndarray, window_samples: int) -> np.ndarray:
    """Apply a centered moving average independently to every channel."""

    values = _as_2d_float(values)
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")

    kernel = np.ones(window_samples, dtype=float) / window_samples
    return np.column_stack(
        [
            np.convolve(values[:, channel], kernel, mode="same")
            for channel in range(values.shape[1])
        ]
    )


def detect_initial_qrs_candidates(
    qrs_enhanced_signal: np.ndarray,
    fs: float,
    max_heart_rate_bpm: float = 140.0,
    prominence_std: float = 0.8,
) -> np.ndarray:
    """Find first-pass QRS candidates from a QRS-enhanced signal.

    These candidates are not the final result. They are used to build the QRS
    template that will later be matched against the signal.
    """

    qrs_enhanced_signal = np.asarray(qrs_enhanced_signal, dtype=float)
    if qrs_enhanced_signal.ndim != 1:
        raise ValueError("qrs_enhanced_signal must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if max_heart_rate_bpm <= 0:
        raise ValueError("max_heart_rate_bpm must be positive")

    refractory_samples = max(1, int(round((60.0 / max_heart_rate_bpm) * fs)))
    prominence = prominence_std * np.std(qrs_enhanced_signal)

    # We use the absolute value because ECG polarity depends on electrode
    # placement. A QRS can appear as a positive spike or as a negative spike.
    peaks, _ = signal.find_peaks(
        np.abs(qrs_enhanced_signal),
        distance=refractory_samples,
        prominence=prominence,
    )
    return peaks.astype(int)


def build_qrs_template(
    qrs_enhanced_signal: np.ndarray,
    qrs_samples: np.ndarray,
    fs: float,
    before_s: float = 0.06,
    after_s: float = 0.08,
) -> QRSTemplateResult:
    """Build a median QRS template around candidate QRS positions."""

    qrs_enhanced_signal = np.asarray(qrs_enhanced_signal, dtype=float)
    qrs_samples = np.asarray(qrs_samples, dtype=int)
    if qrs_enhanced_signal.ndim != 1:
        raise ValueError("qrs_enhanced_signal must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if before_s <= 0 or after_s <= 0:
        raise ValueError("before_s and after_s must be positive")

    before_samples = int(round(before_s * fs))
    after_samples = int(round(after_s * fs))
    valid = qrs_samples[
        (qrs_samples >= before_samples)
        & (qrs_samples + after_samples < len(qrs_enhanced_signal))
    ]
    if len(valid) == 0:
        raise ValueError("no QRS candidates are far enough from signal edges")

    segments = np.stack(
        [
            qrs_enhanced_signal[peak - before_samples : peak + after_samples + 1]
            for peak in valid
        ]
    )

    # The median template is less affected by occasional wrong candidates than
    # a plain average template.
    template = np.median(segments, axis=0)
    template = template - np.mean(template)
    norm = np.linalg.norm(template)
    if norm == 0:
        raise ValueError("QRS template has zero energy")
    template = template / norm

    return QRSTemplateResult(
        template=template,
        used_qrs_samples=valid.astype(int),
        before_samples=before_samples,
        after_samples=after_samples,
        fs=fs,
    )


def detect_qrs_crosscorr(
    qrs_enhanced_signal: np.ndarray,
    qrs_template: np.ndarray,
    fs: float,
    max_heart_rate_bpm: float = 140.0,
    min_correlation_prominence: float = 1.5,
    min_correlation_percentile: float = 70.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect QRS locations as peaks of template cross-correlation."""

    qrs_enhanced_signal = np.asarray(qrs_enhanced_signal, dtype=float)
    qrs_template = np.asarray(qrs_template, dtype=float)
    if qrs_enhanced_signal.ndim != 1:
        raise ValueError("qrs_enhanced_signal must be one-dimensional")
    if qrs_template.ndim != 1:
        raise ValueError("qrs_template must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if max_heart_rate_bpm <= 0:
        raise ValueError("max_heart_rate_bpm must be positive")

    normalized_signal = _safe_zscore(qrs_enhanced_signal)
    normalized_template = qrs_template - np.mean(qrs_template)
    template_norm = np.linalg.norm(normalized_template)
    if template_norm == 0:
        raise ValueError("qrs_template has zero energy")
    normalized_template = normalized_template / template_norm

    # Matched filtering: high values mean that the local waveform looks like
    # the QRS template.
    correlation = signal.fftconvolve(normalized_signal, normalized_template[::-1], mode="same")
    correlation = correlation - np.median(correlation)
    correlation_std = np.std(correlation)
    if correlation_std > 0:
        correlation = correlation / correlation_std

    refractory_samples = max(1, int(round((60.0 / max_heart_rate_bpm) * fs)))
    min_height = np.percentile(correlation, min_correlation_percentile)

    peaks, _ = signal.find_peaks(
        correlation,
        distance=refractory_samples,
        height=min_height,
        prominence=min_correlation_prominence,
    )
    return peaks.astype(int), correlation


def detect_maternal_qrs(
    signals: np.ndarray,
    fs: float,
    max_heart_rate_bpm: float = 140.0,
) -> MaternalQRSDetectionResult:
    """Detect maternal QRS complexes from the upsampled signal ``S4``.

    The returned ``mqrs_samples`` are sample indices at the sampling frequency
    of ``S4``. For the current project this is 2000 Hz.
    """

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")

    pca_result = pca_first_component(signals)
    pca_signal = pca_result.enhanced_signal
    qrs_signal = qrs_bandpass_filter(pca_signal, fs=fs)

    initial_qrs = detect_initial_qrs_candidates(
        qrs_signal,
        fs=fs,
        max_heart_rate_bpm=max_heart_rate_bpm,
        prominence_std=0.8,
    )
    if len(initial_qrs) == 0:
        raise ValueError("no initial QRS candidates detected")

    # PCA polarity is arbitrary. Flip the component when most detected QRS
    # candidates are negative so plots show QRS peaks as upward deflections.
    if np.median(qrs_signal[initial_qrs]) < 0:
        pca_signal = -pca_signal
        qrs_signal = -qrs_signal

    template_result = build_qrs_template(
        qrs_signal,
        initial_qrs,
        fs=fs,
        before_s=0.06,
        after_s=0.08,
    )

    mqrs_samples, correlation = detect_qrs_crosscorr(
        qrs_signal,
        template_result.template,
        fs=fs,
        max_heart_rate_bpm=max_heart_rate_bpm,
        min_correlation_prominence=1.5,
        min_correlation_percentile=70.0,
    )

    return MaternalQRSDetectionResult(
        mqrs_samples=mqrs_samples,
        initial_qrs_samples=initial_qrs,
        pca_signal=pca_signal,
        qrs_enhanced_signal=qrs_signal,
        correlation_signal=correlation,
        qrs_template=template_result.template,
        template_before_samples=template_result.before_samples,
        template_after_samples=template_result.after_samples,
        explained_variance_ratio=pca_result.explained_variance_ratio,
        fs=fs,
    )


def detect_fetal_qrs(
    signals: np.ndarray,
    fs: float,
    low_hz: float = 10.0,
    high_hz: float = 60.0,
    integration_window_s: float = 0.035,
    threshold_percentile: float = 85.0,
    min_distance_s: float = 0.25,
    min_prominence: float = 0.2,
) -> FetalQRSDetectionResult:
    """Detect fetal QRS complexes from the maternal-cancelled signal ``S5``.

    Fetal QRS peaks are weaker than maternal QRS peaks, so the detector first
    emphasizes fast QRS-like activity in every residual channel. The channel
    energies are then combined with the median, which keeps peaks that are
    visible across channels while reducing the effect of one noisy channel.
    """

    signals = _as_2d_float(signals)
    if fs <= 0:
        raise ValueError("fs must be positive")
    if integration_window_s <= 0:
        raise ValueError("integration_window_s must be positive")
    if not 0 < threshold_percentile < 100:
        raise ValueError("threshold_percentile must be between 0 and 100")
    if min_distance_s <= 0:
        raise ValueError("min_distance_s must be positive")

    filtered = _qrs_bandpass_filter_channels(
        signals,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        order=3,
    )

    # Normalize each channel before squaring, otherwise the channel with the
    # largest amplitude would dominate the combined detector.
    centered = filtered - np.median(filtered, axis=0)
    scale = np.std(centered, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    normalized = centered / scale

    # Squaring turns both positive and negative fetal QRS spikes into positive
    # energy peaks. A short moving average smooths those peaks.
    energy = normalized**2
    window_samples = max(1, int(round(integration_window_s * fs)))
    channel_energy = _moving_average_channels(energy, window_samples)

    detection_signal = np.median(channel_energy, axis=1)
    detection_signal = detection_signal - np.median(detection_signal)
    detection_scale = np.std(detection_signal)
    if detection_scale > 0:
        detection_signal = detection_signal / detection_scale

    threshold = float(np.percentile(detection_signal, threshold_percentile))
    min_distance_samples = max(1, int(round(min_distance_s * fs)))
    peaks, _ = signal.find_peaks(
        detection_signal,
        height=threshold,
        distance=min_distance_samples,
        prominence=min_prominence,
    )

    return FetalQRSDetectionResult(
        fqrs_samples=peaks.astype(int),
        detection_signal=detection_signal,
        filtered_signals=filtered,
        channel_energy=channel_energy,
        threshold=threshold,
        fs=fs,
        low_hz=low_hz,
        high_hz=high_hz,
        integration_window_s=integration_window_s,
    )
