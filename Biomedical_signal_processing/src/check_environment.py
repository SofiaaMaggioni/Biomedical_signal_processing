"""Quick environment check for the fetal ECG project."""

from __future__ import annotations

import importlib


PACKAGES = [
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "wfdb",
    "sklearn",
    "jupyterlab",
    "openpyxl",
]


def check_imports() -> None:
    """Verify that all Python packages needed by the project import correctly."""

    print("Checking Python packages...")
    for package in PACKAGES:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"  OK {package}: {version}")


def check_physionet_access() -> None:
    """Verify that WFDB can reach PhysioNet and read one short reference record."""

    print("\nChecking PhysioNet access with record a14...")
    import wfdb

    try:
        # Read only the first second of signal to keep the environment check
        # fast. This still verifies that .dat and .hea can be accessed.
        record = wfdb.rdrecord(
            "a14",
            pn_dir="challenge-2013/1.0.0/set-a",
            sampfrom=0,
            sampto=1000,
        )

        # Read reference fetal QRS annotations in the first 10 seconds. This
        # verifies that the .fqrs annotation file is available too.
        ann = wfdb.rdann(
            "a14",
            "fqrs",
            pn_dir="challenge-2013/1.0.0/set-a",
            sampfrom=0,
            sampto=10000,
        )
    except Exception as exc:  # Network access may be unavailable.
        print(f"  PhysioNet check skipped/failed: {exc}")
        return

    print(f"  OK record fs: {record.fs} Hz")
    print(f"  OK signal shape: {record.p_signal.shape}")
    print(f"  OK reference FQRS loaded: {len(ann.sample)} annotations in first 10 s")


def main() -> None:
    check_imports()
    check_physionet_access()


if __name__ == "__main__":
    main()
