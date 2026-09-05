"""Plotting helpers for ECG exploration and reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def plot_record_overview(record, seconds: tuple[float, float] | None = None, save_path: str | Path | None = None):
    """Plot all abdominal ECG channels for a selected time window.

    This is used at the start of the analysis to inspect the raw abdominal ECG
    before any filtering or maternal ECG cancellation.
    """

    signals = record.signals
    time = record.time

    # By default, show the first 10 seconds. A full 60-second plot is usually
    # too dense to inspect visually.
    if seconds is None:
        start_s, end_s = 0.0, min(10.0, time[-1])
    else:
        start_s, end_s = seconds

    # Boolean mask selecting only the time interval requested by the caller.
    mask = (time >= start_s) & (time <= end_s)
    fig, axes = plt.subplots(signals.shape[1], 1, figsize=(12, 8), sharex=True)

    if signals.shape[1] == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        ax.plot(time[mask], signals[mask, ch_idx], linewidth=0.9)
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: abdominal ECG channels ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    # Saving figures from the plotting function keeps the notebook clean and
    # ensures every important plot can be reused later in the presentation.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_reference_fqrs(record, channel: int = 0, seconds: tuple[float, float] = (0.0, 10.0), save_path: str | Path | None = None):
    """Plot one channel and overlay fetal QRS reference annotations.

    This is not the output of our algorithm yet. The red markers are the
    reference annotations from PhysioNet, used later to compute performance.
    """

    start_s, end_s = seconds
    time = record.time
    signal = record.signals[:, channel]
    mask = (time >= start_s) & (time <= end_s)

    # Keep only reference fetal QRS annotations that fall inside the plotted
    # time window. The annotations are sample indices, so seconds are converted
    # to samples using fs.
    ref = record.fqrs_reference
    ref = ref[(ref >= int(start_s * record.fs)) & (ref <= int(end_s * record.fs))]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time[mask], signal[mask], linewidth=0.9, label=record.channels[channel])

    if len(ref) > 0:
        # Convert annotation samples to seconds for the x-axis. For the y-axis,
        # use the signal value at each annotated sample.
        ax.scatter(ref / record.fs, signal[ref], s=28, color="tab:red", label="Reference FQRS", zorder=3)

    ax.set_title(f"Record {record.name}: reference fetal QRS on {record.channels[channel]}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    # Optional export for figures that will be used in reports or slides.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_baseline_removal_comparison(
    record,
    filtered_signals: np.ndarray,
    seconds: tuple[float, float] = (0.0, 10.0),
    save_path: str | Path | None = None,
):
    """Plot raw signal S1 and baseline-corrected signal S2 channel by channel."""

    start_s, end_s = seconds
    time = record.time
    raw = record.signals
    filtered = np.asarray(filtered_signals)

    if filtered.shape != raw.shape:
        raise ValueError("filtered_signals must have the same shape as record.signals")

    # Select the time interval to display. A short interval shows waveform
    # details; a full interval shows how the slow baseline drift was removed.
    mask = (time >= start_s) & (time <= end_s)
    fig, axes = plt.subplots(raw.shape[1], 1, figsize=(12, 8), sharex=True)

    if raw.shape[1] == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        ax.plot(time[mask], raw[mask, ch_idx], color="0.65", linewidth=0.8, label="S1 raw")
        ax.plot(time[mask], filtered[mask, ch_idx], color="tab:blue", linewidth=0.9, label="S2 baseline removed")
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)
        if ch_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: baseline wander removal ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_filter_response(taps: np.ndarray, fs: float, cutoff_hz: float, save_path: str | Path | None = None):
    """Plot the magnitude response of the FIR high-pass filter."""

    # freqz evaluates the filter in the frequency domain. This lets us verify
    # that low frequencies are attenuated and ECG/QRS frequencies are preserved.
    w, h = signal.freqz(taps, worN=4096, fs=fs)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(w, magnitude_db, linewidth=1.2)
    ax.axvline(cutoff_hz, color="tab:red", linestyle="--", label=f"cutoff = {cutoff_hz:g} Hz")
    # scipy.signal.firwin defines the cutoff around the half-amplitude point,
    # which corresponds to about -6 dB.
    ax.axhline(-6, color="0.5", linestyle=":", label="-6 dB")
    ax.set_xlim(0, min(30, fs / 2))
    ax.set_ylim(-80, 5)
    ax.set_title("High-pass FIR filter response for baseline wander removal")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_powerline_removal_comparison(
    record,
    before_signals: np.ndarray,
    after_signals: np.ndarray,
    seconds: tuple[float, float] = (0.0, 10.0),
    save_path: str | Path | None = None,
):
    """Plot S2 and S3 channel by channel after power-line cancellation.

    ``before_signals`` is usually S2, after baseline wander removal.
    ``after_signals`` is usually S3, after 50 Hz interference cancellation.
    """

    start_s, end_s = seconds
    time = record.time
    before = np.asarray(before_signals)
    after = np.asarray(after_signals)

    if before.shape != after.shape:
        raise ValueError("before_signals and after_signals must have the same shape")
    if before.shape[0] != len(time):
        raise ValueError("signals must have the same number of samples as record.time")

    mask = (time >= start_s) & (time <= end_s)
    fig, axes = plt.subplots(before.shape[1], 1, figsize=(12, 8), sharex=True)

    if before.shape[1] == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        line_s3 = ax.plot(
            time[mask],
            after[mask, ch_idx],
            color="tab:green",
            linewidth=0.9,
            alpha=0.55,
            label="S3 power-line removed",
        )[0]
        # S2 and S3 can be almost identical in time, so S2 is drawn last as a
        # dashed line. This makes the overlap visible instead of hiding S2.
        line_s2 = ax.plot(
            time[mask],
            before[mask, ch_idx],
            color="0.1",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
            label="S2 baseline removed",
        )[0]
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)
        if ch_idx == 0:
            ax.legend(handles=[line_s2, line_s3], loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: power-line interference cancellation ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_powerline_zoom_detail(
    record,
    before_signals: np.ndarray,
    after_signals: np.ndarray,
    estimated_interference: np.ndarray,
    channel: int,
    seconds: tuple[float, float] = (0.0, 1.0),
    save_path: str | Path | None = None,
):
    """Show a zoomed S2/S3 comparison and the removed component for one channel."""

    start_s, end_s = seconds
    time = record.time
    before = np.asarray(before_signals)
    after = np.asarray(after_signals)
    estimated = np.asarray(estimated_interference)

    if before.shape != after.shape or estimated.shape != before.shape:
        raise ValueError("all signals must have the same shape")
    if before.shape[0] != len(time):
        raise ValueError("signals must have the same number of samples as record.time")

    mask = (time >= start_s) & (time <= end_s)
    channel_name = record.channels[channel]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: S2 and S3 on the real ECG scale. They usually overlap because the
    # removed component is tiny compared with the ECG peaks.
    axes[0].plot(time[mask], after[mask, channel], color="tab:green", linewidth=1.0, alpha=0.65, label="S3 after")
    axes[0].plot(time[mask], before[mask, channel], color="0.1", linestyle="--", linewidth=1.0, alpha=0.9, label="S2 before")
    axes[0].set_ylabel(channel_name)
    axes[0].set_title("S2 and S3 on the ECG amplitude scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    # Bottom: the same removed component on its own smaller scale.
    axes[1].plot(time[mask], estimated[mask, channel], color="tab:orange", linewidth=1.0)
    axes[1].set_ylabel("removed")
    axes[1].set_title("Estimated interference removed from S2")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"Record {record.name}: zoomed power-line removal detail ({channel_name}, {start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_estimated_powerline_interference(
    record,
    estimated_interference: np.ndarray,
    seconds: tuple[float, float] = (0.0, 1.0),
    save_path: str | Path | None = None,
):
    """Plot the sinusoidal component subtracted during power-line cancellation."""

    start_s, end_s = seconds
    time = record.time
    estimated = np.asarray(estimated_interference)

    if estimated.shape[0] != len(time):
        raise ValueError("estimated_interference must have the same number of samples as record.time")

    # A short window is easier to read here: at 50 Hz one cycle lasts 0.02 s,
    # so one second already contains many repetitions of the electrical noise.
    mask = (time >= start_s) & (time <= end_s)
    fig, axes = plt.subplots(estimated.shape[1], 1, figsize=(12, 8), sharex=True)

    if estimated.shape[1] == 1:
        axes = [axes]

    for ch_idx, ax in enumerate(axes):
        ax.plot(time[mask], estimated[mask, ch_idx], color="tab:orange", linewidth=0.9)
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: estimated power-line interference ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_upsampling_zoom(
    record,
    original_signals: np.ndarray,
    upsampled_signals: np.ndarray,
    original_fs: float,
    target_fs: float,
    channel: int = 0,
    seconds: tuple[float, float] = (0.0, 0.05),
    save_path: str | Path | None = None,
):
    """Show how upsampling adds intermediate samples without changing duration."""

    start_s, end_s = seconds
    original = np.asarray(original_signals)
    upsampled = np.asarray(upsampled_signals)

    if original.ndim != 2 or upsampled.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if original.shape[1] != upsampled.shape[1]:
        raise ValueError("original_signals and upsampled_signals must have the same channel count")
    if original_fs <= 0 or target_fs <= 0:
        raise ValueError("sampling frequencies must be positive")

    original_time = np.arange(original.shape[0]) / original_fs
    upsampled_time = np.arange(upsampled.shape[0]) / target_fs

    original_mask = (original_time >= start_s) & (original_time <= end_s)
    upsampled_mask = (upsampled_time >= start_s) & (upsampled_time <= end_s)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        upsampled_time[upsampled_mask],
        upsampled[upsampled_mask, channel],
        color="tab:purple",
        linewidth=1.0,
        marker=".",
        markersize=3,
        label=f"S4 resampled ({target_fs:g} Hz)",
    )
    # The original samples are drawn as points so the denser S4 sampling grid
    # is visible between them.
    ax.scatter(
        original_time[original_mask],
        original[original_mask, channel],
        color="0.15",
        s=18,
        label=f"S3 original samples ({original_fs:g} Hz)",
        zorder=3,
    )

    ax.set_title(
        f"Record {record.name}: upsampling detail on {record.channels[channel]} "
        f"({start_s:.3f}-{end_s:.3f} s)"
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(record.channels[channel])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_maternal_qrs_detection(
    record_name: str,
    pca_signal: np.ndarray,
    qrs_enhanced_signal: np.ndarray,
    correlation_signal: np.ndarray,
    mqrs_samples: np.ndarray,
    fs: float,
    seconds: tuple[float, float] = (0.0, 10.0),
    save_path: str | Path | None = None,
):
    """Plot the signals used to detect maternal QRS complexes.

    The first panel shows the first PCA component. The second panel shows the
    same component after QRS-band filtering. The third panel shows the
    cross-correlation with the maternal QRS template; its peaks are the final
    maternal QRS detections.
    """

    pca_signal = np.asarray(pca_signal)
    qrs_enhanced_signal = np.asarray(qrs_enhanced_signal)
    correlation_signal = np.asarray(correlation_signal)
    mqrs_samples = np.asarray(mqrs_samples, dtype=int)

    if pca_signal.ndim != 1 or qrs_enhanced_signal.ndim != 1 or correlation_signal.ndim != 1:
        raise ValueError("pca_signal, qrs_enhanced_signal, and correlation_signal must be one-dimensional")
    if not (len(pca_signal) == len(qrs_enhanced_signal) == len(correlation_signal)):
        raise ValueError("all detection signals must have the same length")
    if fs <= 0:
        raise ValueError("fs must be positive")

    start_s, end_s = seconds
    time = np.arange(len(pca_signal)) / fs
    mask = (time >= start_s) & (time <= end_s)
    peak_mask = (mqrs_samples / fs >= start_s) & (mqrs_samples / fs <= end_s)
    visible_peaks = mqrs_samples[peak_mask]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time[mask], pca_signal[mask], color="tab:blue", linewidth=0.9)
    if len(visible_peaks) > 0:
        axes[0].scatter(
            visible_peaks / fs,
            pca_signal[visible_peaks],
            marker="x",
            color="0.05",
            s=28,
            label="Detected MQRS",
            zorder=3,
        )
    axes[0].set_ylabel("PCA 1")
    axes[0].set_title("First PCA component")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(time[mask], qrs_enhanced_signal[mask], color="tab:orange", linewidth=0.9)
    if len(visible_peaks) > 0:
        axes[1].scatter(
            visible_peaks / fs,
            qrs_enhanced_signal[visible_peaks],
            marker="x",
            color="0.05",
            s=28,
            zorder=3,
        )
    axes[1].set_ylabel("QRS band")
    axes[1].set_title("QRS-enhanced PCA signal")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time[mask], correlation_signal[mask], color="tab:green", linewidth=0.9)
    if len(visible_peaks) > 0:
        axes[2].scatter(
            visible_peaks / fs,
            correlation_signal[visible_peaks],
            marker="x",
            color="0.05",
            s=28,
            zorder=3,
        )
    axes[2].set_ylabel("corr.")
    axes[2].set_title("Cross-correlation with maternal QRS template")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"Record {record_name}: maternal QRS detection ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_maternal_qrs_on_channels(
    record,
    signals: np.ndarray,
    fs: float,
    mqrs_samples: np.ndarray,
    seconds: tuple[float, float] = (0.0, 60.0),
    save_path: str | Path | None = None,
):
    """Plot S4 channels and overlay the detected maternal QRS positions."""

    signals = np.asarray(signals)
    mqrs_samples = np.asarray(mqrs_samples, dtype=int)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if signals.shape[1] != len(record.channels):
        raise ValueError("signals must have the same channel count as record.channels")

    start_s, end_s = seconds
    time = np.arange(signals.shape[0]) / fs
    mask = (time >= start_s) & (time <= end_s)
    peak_mask = (mqrs_samples / fs >= start_s) & (mqrs_samples / fs <= end_s)
    visible_peaks = mqrs_samples[peak_mask]

    fig, axes = plt.subplots(signals.shape[1], 1, figsize=(12, 8), sharex=True)
    if signals.shape[1] == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ch_idx, ax in enumerate(axes):
        color = colors[ch_idx % len(colors)]
        ax.plot(time[mask], signals[mask, ch_idx], color=color, linewidth=0.75, alpha=0.9)
        if len(visible_peaks) > 0:
            ax.scatter(
                visible_peaks / fs,
                signals[visible_peaks, ch_idx],
                marker="x",
                color="0.05",
                s=18,
                label="Detected MQRS" if ch_idx == 0 else None,
                zorder=3,
            )
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)
        if ch_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: detected maternal QRS on S4 ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_qrs_template(
    qrs_template: np.ndarray,
    fs: float,
    before_samples: int,
    save_path: str | Path | None = None,
):
    """Plot the maternal QRS template used for cross-correlation."""

    qrs_template = np.asarray(qrs_template)
    if qrs_template.ndim != 1:
        raise ValueError("qrs_template must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")

    time_ms = (np.arange(len(qrs_template)) - before_samples) / fs * 1000.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_ms, qrs_template, color="tab:purple", linewidth=1.4)
    ax.axvline(0, color="0.25", linestyle="--", linewidth=0.9, label="candidate center")
    ax.set_title("Maternal QRS template")
    ax.set_xlabel("Time around QRS [ms]")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_mecg_cancellation_comparison(
    record,
    before_signals: np.ndarray,
    after_signals: np.ndarray,
    fs: float,
    seconds: tuple[float, float] = (10.0, 20.0),
    save_path: str | Path | None = None,
):
    """Plot S4 and S5 channel by channel after maternal ECG cancellation."""

    before = np.asarray(before_signals)
    after = np.asarray(after_signals)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if before.shape != after.shape:
        raise ValueError("before_signals and after_signals must have the same shape")
    if before.shape[1] != len(record.channels):
        raise ValueError("signals must have the same channel count as record.channels")
    if fs <= 0:
        raise ValueError("fs must be positive")

    start_s, end_s = seconds
    time = np.arange(before.shape[0]) / fs
    mask = (time >= start_s) & (time <= end_s)

    fig, axes = plt.subplots(before.shape[1], 1, figsize=(12, 8), sharex=True)
    if before.shape[1] == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ch_idx, ax in enumerate(axes):
        after_line = ax.plot(
            time[mask],
            after[mask, ch_idx],
            color=colors[ch_idx % len(colors)],
            linewidth=0.85,
            alpha=0.9,
            label="S5 MECG cancelled",
        )[0]
        before_line = ax.plot(
            time[mask],
            before[mask, ch_idx],
            color="0.65",
            linestyle="--",
            linewidth=0.8,
            alpha=0.85,
            label="S4 before",
        )[0]
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)
        if ch_idx == 0:
            ax.legend(handles=[before_line, after_line], loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: maternal ECG cancellation ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_mecg_cancellation_detail(
    record,
    before_signals: np.ndarray,
    estimated_mecg: np.ndarray,
    after_signals: np.ndarray,
    fs: float,
    mqrs_samples: np.ndarray,
    channel: int,
    seconds: tuple[float, float] = (12.0, 14.0),
    save_path: str | Path | None = None,
):
    """Show the maternal ECG estimate and residual for one channel."""

    before = np.asarray(before_signals)
    estimated = np.asarray(estimated_mecg)
    after = np.asarray(after_signals)
    mqrs_samples = np.asarray(mqrs_samples, dtype=int)
    if before.ndim != 2 or estimated.ndim != 2 or after.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if before.shape != estimated.shape or before.shape != after.shape:
        raise ValueError("before, estimated, and after signals must have the same shape")
    if before.shape[1] != len(record.channels):
        raise ValueError("signals must have the same channel count as record.channels")
    if fs <= 0:
        raise ValueError("fs must be positive")

    start_s, end_s = seconds
    time = np.arange(before.shape[0]) / fs
    mask = (time >= start_s) & (time <= end_s)
    peak_mask = (mqrs_samples / fs >= start_s) & (mqrs_samples / fs <= end_s)
    visible_peaks = mqrs_samples[peak_mask]
    channel_name = record.channels[channel]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(time[mask], before[mask, channel], color="0.65", linestyle="--", linewidth=0.9, label="S4 before")
    axes[0].plot(time[mask], after[mask, channel], color="tab:blue", linewidth=0.95, label="S5 after")
    if len(visible_peaks) > 0:
        axes[0].scatter(
            visible_peaks / fs,
            before[visible_peaks, channel],
            marker="x",
            color="0.05",
            s=28,
            label="MQRS",
            zorder=3,
        )
    axes[0].set_ylabel(channel_name)
    axes[0].set_title("S4 before and S5 after cancellation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(time[mask], estimated[mask, channel], color="tab:orange", linewidth=0.95)
    if len(visible_peaks) > 0:
        axes[1].scatter(
            visible_peaks / fs,
            estimated[visible_peaks, channel],
            marker="x",
            color="0.05",
            s=28,
            zorder=3,
        )
    axes[1].set_ylabel("estimated")
    axes[1].set_title("Estimated maternal ECG subtracted from S4")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(time[mask], after[mask, channel], color="tab:green", linewidth=0.95)
    if len(visible_peaks) > 0:
        axes[2].scatter(
            visible_peaks / fs,
            after[visible_peaks, channel],
            marker="x",
            color="0.05",
            s=28,
            zorder=3,
        )
    axes[2].set_ylabel("S5")
    axes[2].set_title("Residual signal after maternal ECG cancellation")
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f"Record {record.name}: MECG cancellation detail on {channel_name} ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_fetal_detection_signal(
    record_name: str,
    detection_signal: np.ndarray,
    detected_fqrs_samples: np.ndarray,
    reference_fqrs_samples: np.ndarray,
    fs: float,
    seconds: tuple[float, float] = (10.0, 20.0),
    threshold: float | None = None,
    save_path: str | Path | None = None,
):
    """Plot the one-dimensional signal used for fetal QRS detection."""

    detection_signal = np.asarray(detection_signal)
    detected = np.asarray(detected_fqrs_samples, dtype=int)
    reference = np.asarray(reference_fqrs_samples, dtype=int)
    if detection_signal.ndim != 1:
        raise ValueError("detection_signal must be one-dimensional")
    if fs <= 0:
        raise ValueError("fs must be positive")

    start_s, end_s = seconds
    time = np.arange(len(detection_signal)) / fs
    mask = (time >= start_s) & (time <= end_s)
    detected_mask = (detected / fs >= start_s) & (detected / fs <= end_s)
    reference_mask = (reference / fs >= start_s) & (reference / fs <= end_s)
    visible_detected = detected[detected_mask]
    visible_reference = reference[reference_mask]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time[mask], detection_signal[mask], color="tab:blue", linewidth=1.0, label="FQRS detection signal")
    if threshold is not None:
        ax.axhline(threshold, color="0.35", linestyle="--", linewidth=0.9, label="threshold")
    if len(visible_reference) > 0:
        ax.scatter(
            visible_reference / fs,
            detection_signal[visible_reference],
            facecolors="none",
            edgecolors="tab:red",
            s=52,
            linewidths=1.3,
            label="Reference FQRS",
            zorder=3,
        )
    if len(visible_detected) > 0:
        ax.scatter(
            visible_detected / fs,
            detection_signal[visible_detected],
            marker="x",
            color="0.05",
            s=36,
            label="Detected FQRS",
            zorder=4,
        )

    ax.set_title(f"Record {record_name}: fetal QRS detection signal ({start_s:.1f}-{end_s:.1f} s)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Detector value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_fetal_qrs_on_channels(
    record,
    signals: np.ndarray,
    fs: float,
    detected_fqrs_samples: np.ndarray,
    reference_fqrs_samples: np.ndarray,
    seconds: tuple[float, float] = (10.0, 20.0),
    save_path: str | Path | None = None,
):
    """Plot S5 channels with detected and reference fetal QRS positions."""

    signals = np.asarray(signals)
    detected = np.asarray(detected_fqrs_samples, dtype=int)
    reference = np.asarray(reference_fqrs_samples, dtype=int)
    if signals.ndim != 2:
        raise ValueError("signals must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if signals.shape[1] != len(record.channels):
        raise ValueError("signals must have the same channel count as record.channels")

    start_s, end_s = seconds
    time = np.arange(signals.shape[0]) / fs
    mask = (time >= start_s) & (time <= end_s)
    detected_mask = (detected / fs >= start_s) & (detected / fs <= end_s)
    reference_mask = (reference / fs >= start_s) & (reference / fs <= end_s)
    visible_detected = detected[detected_mask]
    visible_reference = reference[reference_mask]

    fig, axes = plt.subplots(signals.shape[1], 1, figsize=(12, 8), sharex=True)
    if signals.shape[1] == 1:
        axes = [axes]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for ch_idx, ax in enumerate(axes):
        color = colors[ch_idx % len(colors)]
        ax.plot(time[mask], signals[mask, ch_idx], color=color, linewidth=0.8, alpha=0.9)
        if len(visible_reference) > 0:
            ax.scatter(
                visible_reference / fs,
                signals[visible_reference, ch_idx],
                facecolors="none",
                edgecolors="tab:red",
                s=42,
                linewidths=1.2,
                label="Reference FQRS" if ch_idx == 0 else None,
                zorder=3,
            )
        if len(visible_detected) > 0:
            ax.scatter(
                visible_detected / fs,
                signals[visible_detected, ch_idx],
                marker="x",
                color="0.05",
                s=24,
                label="Detected FQRS" if ch_idx == 0 else None,
                zorder=4,
            )
        ax.set_ylabel(record.channels[ch_idx])
        ax.grid(True, alpha=0.25)
        if ch_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Record {record.name}: detected vs reference fetal QRS on S5 ({start_s:.1f}-{end_s:.1f} s)")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, axes


def plot_average_fecg(
    record,
    average_fecg: np.ndarray,
    fs: float,
    before_samples: int,
    used_beats: int,
    snr_gain_factor: float,
    save_path: str | Path | None = None,
):
    """Plot the final averaged fetal ECG waveform, called S6."""

    average_fecg = np.asarray(average_fecg)
    if average_fecg.ndim != 2:
        raise ValueError("average_fecg must have shape samples x channels")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if average_fecg.shape[1] != len(record.channels):
        raise ValueError("average_fecg must have the same channel count as record.channels")

    time_ms = (np.arange(average_fecg.shape[0]) - before_samples) / fs * 1000.0
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ch_idx, channel_name in enumerate(record.channels):
        ax.plot(
            time_ms,
            average_fecg[:, ch_idx],
            color=colors[ch_idx % len(colors)],
            linewidth=1.4,
            label=channel_name,
        )

    # All segments are aligned at 0 ms. If the averaging worked, the fetal QRS
    # should be the most visible feature around this vertical line.
    ax.axvline(0, color="0.2", linestyle="--", linewidth=0.9, label="FQRS alignment")
    ax.set_title(
        f"Record {record.name}: average FECG / S6 "
        f"(N_av={used_beats}, expected SNR gain={snr_gain_factor:.1f}x)"
    )
    ax.set_xlabel("Time relative to detected FQRS [ms]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_fecg_segments_vs_average(
    record,
    segments: np.ndarray,
    average_fecg: np.ndarray,
    fs: float,
    before_samples: int,
    channel: int = 0,
    max_segments: int = 30,
    save_path: str | Path | None = None,
):
    """Show noisy individual fetal beats against their final average."""

    segments = np.asarray(segments)
    average_fecg = np.asarray(average_fecg)
    if segments.ndim != 3:
        raise ValueError("segments must have shape beats x samples x channels")
    if average_fecg.ndim != 2:
        raise ValueError("average_fecg must have shape samples x channels")
    if segments.shape[1:] != average_fecg.shape:
        raise ValueError("segments and average_fecg must have matching sample/channel dimensions")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if not 0 <= channel < average_fecg.shape[1]:
        raise ValueError("channel is outside the available channel range")

    time_ms = (np.arange(average_fecg.shape[0]) - before_samples) / fs * 1000.0
    visible_count = min(max_segments, segments.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    for beat_idx in range(visible_count):
        ax.plot(
            time_ms,
            segments[beat_idx, :, channel],
            color="0.7",
            linewidth=0.55,
            alpha=0.35,
        )
    ax.plot(
        time_ms,
        average_fecg[:, channel],
        color="tab:blue",
        linewidth=2.0,
        label="S6 average",
    )
    ax.axvline(0, color="0.2", linestyle="--", linewidth=0.9, label="FQRS alignment")
    ax.set_title(
        f"Record {record.name}: individual FECG segments vs average "
        f"({record.channels[channel]})"
    )
    ax.set_xlabel("Time relative to detected FQRS [ms]")
    ax.set_ylabel(record.channels[channel])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_power_spectrum_comparison(
    before_signals: np.ndarray,
    after_signals: np.ndarray,
    fs: float,
    channel_name: str,
    channel: int = 0,
    target_frequencies: tuple[float, ...] = (50.0, 100.0, 150.0),
    max_hz: float = 180.0,
    removed_interference: np.ndarray | None = None,
    save_path: str | Path | None = None,
):
    """Compare signal power before and after removing power-line interference."""

    before = np.asarray(before_signals)
    after = np.asarray(after_signals)
    if before.shape != after.shape:
        raise ValueError("before_signals and after_signals must have the same shape")
    removed = None
    if removed_interference is not None:
        removed = np.asarray(removed_interference)
        if removed.shape != before.shape:
            raise ValueError("removed_interference must have the same shape as before_signals")

    # Welch's method estimates how much power the signal has at each frequency.
    # Peaks around 50, 100, or 150 Hz indicate power-line contamination.
    nperseg = min(4096, before.shape[0])
    freq_before, power_before = signal.welch(before[:, channel], fs=fs, nperseg=nperseg)
    freq_after, power_after = signal.welch(after[:, channel], fs=fs, nperseg=nperseg)
    if removed is not None:
        freq_removed, power_removed = signal.welch(removed[:, channel], fs=fs, nperseg=nperseg)

    power_before_db = 10 * np.log10(np.maximum(power_before, 1e-18))
    power_after_db = 10 * np.log10(np.maximum(power_after, 1e-18))
    if removed is not None:
        power_removed_db = 10 * np.log10(np.maximum(power_removed, 1e-18))

    mask = freq_before <= max_hz
    fig, ax = plt.subplots(figsize=(11, 4))
    line_s3 = ax.plot(freq_after[mask], power_after_db[mask], color="tab:green", linewidth=1.0, alpha=0.85, label="S3 after")[0]
    # Draw S2 last and dashed because it often overlaps S3 almost perfectly.
    line_s2 = ax.plot(freq_before[mask], power_before_db[mask], color="0.25", linestyle="--", linewidth=0.9, alpha=0.85, label="S2 before")[0]
    line_removed = None
    if removed is not None:
        removed_mask = freq_removed <= max_hz
        line_removed = ax.plot(
            freq_removed[removed_mask],
            power_removed_db[removed_mask],
            color="tab:orange",
            linestyle=":",
            linewidth=1.3,
            label="estimated interference",
        )[0]

    for freq in target_frequencies:
        if freq <= max_hz:
            ax.axvline(freq, color="tab:red", linestyle="--", alpha=0.75)
            ax.text(freq, ax.get_ylim()[1], f"{freq:g} Hz", color="tab:red", ha="center", va="top", fontsize=8)

    ax.set_title(f"Power spectrum before/after power-line removal ({channel_name})")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power [dB]")
    ax.grid(True, alpha=0.25)
    legend_handles = [line_s2, line_s3]
    if line_removed is not None:
        legend_handles.append(line_removed)
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


def plot_batch_evaluation_summary(performance_table, save_path: str | Path | None = None):
    """Plot the multi-record FQRS detection results from the batch evaluation.

    The figure is meant for the final presentation: the top panel shows the
    main performance percentages, while the bottom panel shows whether the
    detector found a similar number of fetal beats compared with the reference.
    """

    required_columns = [
        "record",
        "reference_fqrs",
        "detected_fqrs",
        "sensitivity",
        "positive_predictive_value",
        "f1_score",
    ]
    missing_columns = [column for column in required_columns if column not in performance_table]
    if missing_columns:
        raise ValueError(f"performance_table is missing columns: {missing_columns}")
    if len(performance_table) == 0:
        raise ValueError("performance_table must contain at least one record")

    records = [str(record) for record in performance_table["record"]]
    x = np.arange(len(records))

    fig, (ax_metrics, ax_counts) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )

    metric_specs = [
        ("sensitivity", "Sensitivity", "tab:blue"),
        ("positive_predictive_value", "PPV", "tab:green"),
        ("f1_score", "F1-score", "tab:red"),
    ]
    bar_width = 0.24
    for metric_idx, (column, label, color) in enumerate(metric_specs):
        values = np.asarray(performance_table[column], dtype=float) * 100.0
        offset = (metric_idx - 1) * bar_width
        bars = ax_metrics.bar(x + offset, values, width=bar_width, color=color, alpha=0.85, label=label)

        # Label only the F1 bars because this is the easiest single metric to
        # discuss in a 10-minute presentation.
        if column == "f1_score":
            for bar, value in zip(bars, values):
                ax_metrics.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 2.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax_metrics.axhline(80, color="0.45", linestyle="--", linewidth=0.9, alpha=0.7)
    ax_metrics.text(len(records) - 0.55, 82, "80%", color="0.35", fontsize=8, va="bottom")
    ax_metrics.set_ylabel("Performance [%]")
    ax_metrics.set_ylim(0, 110)
    ax_metrics.set_title("Batch evaluation: FQRS detection performance across Set-A records")
    ax_metrics.grid(True, axis="y", alpha=0.25)
    ax_metrics.legend(loc="upper left", ncol=3)

    count_width = 0.34
    reference_counts = np.asarray(performance_table["reference_fqrs"], dtype=float)
    detected_counts = np.asarray(performance_table["detected_fqrs"], dtype=float)
    ax_counts.bar(x - count_width / 2, reference_counts, width=count_width, color="0.45", alpha=0.75, label="Reference FQRS")
    ax_counts.bar(x + count_width / 2, detected_counts, width=count_width, color="tab:orange", alpha=0.85, label="Detected FQRS")
    ax_counts.set_ylabel("FQRS count")
    ax_counts.set_xticks(x)
    ax_counts.set_xticklabels(records)
    ax_counts.set_xlabel("Record")
    ax_counts.grid(True, axis="y", alpha=0.25)
    ax_counts.legend(loc="upper right", ncol=2)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, (ax_metrics, ax_counts)
