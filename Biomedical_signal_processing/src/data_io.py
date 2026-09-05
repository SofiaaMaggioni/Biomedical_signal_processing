"""Data loading helpers for PhysioNet Challenge 2013 records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import wfdb


PHYSIONET_SET_A = "challenge-2013/1.0.0/set-a"


@dataclass(frozen=True)
class FetalEcgRecord:
    """Container for one non-invasive fetal ECG record.

    Keeping the data in one object makes later notebooks easier to read:
    the signal, sampling frequency, time axis, channel labels and reference
    annotations always travel together.
    """

    name: str
    signals: np.ndarray  # ECG samples with shape: samples x channels.
    fs: float  # Sampling frequency in Hz.
    time: np.ndarray  # Time axis in seconds, one value per sample.
    channels: list[str]  # Channel names read from the WFDB header.
    fqrs_reference: np.ndarray  # Reference fetal QRS positions, in sample indices.


def load_record(record_name: str, pn_dir: str = PHYSIONET_SET_A) -> FetalEcgRecord:
    """Load one Set-A record and its fetal QRS reference annotations.

    Parameters
    ----------
    record_name:
        Record id such as "a14".
    pn_dir:
        PhysioNet directory. Defaults to Challenge 2013 Set-A.

    Returns
    -------
    FetalEcgRecord
        Signals are returned as ``samples x channels``.
    """

    # WFDB reads the pair .dat + .hea. The .dat file contains the binary
    # signal samples; the .hea file tells WFDB how to interpret them.
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)

    # The "fqrs" annotator contains the fetal QRS reference positions for
    # Set-A. WFDB returns them as sample indices, not seconds.
    annotations = wfdb.rdann(record_name, "fqrs", pn_dir=pn_dir)

    if record.p_signal is None:
        raise ValueError(f"Record {record_name} does not contain physical signals.")

    # record.p_signal already contains calibrated physical values and is shaped
    # as samples x channels, which is convenient for plotting and filtering.
    signals = np.asarray(record.p_signal, dtype=float)
    fs = float(record.fs)

    # Build a time vector in seconds so plots can use real time instead of
    # sample number. For a14 this runs from 0 to about 60 s.
    time = np.arange(signals.shape[0]) / fs

    # Use the names from the WFDB header when available; otherwise create
    # simple fallback labels.
    channels = list(record.sig_name or [f"ch{i + 1}" for i in range(signals.shape[1])])
    fqrs_reference = np.asarray(annotations.sample, dtype=int)

    return FetalEcgRecord(
        name=record_name,
        signals=signals,
        fs=fs,
        time=time,
        channels=channels,
        fqrs_reference=fqrs_reference,
    )


def list_set_a_records() -> list[str]:
    """Return the standard Set-A record ids used in the Challenge 2013 dataset."""

    # Set-A records are named a01, a02, ..., a25.
    return [f"a{i:02d}" for i in range(1, 26)]
