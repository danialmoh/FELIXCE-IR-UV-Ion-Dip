"""
Shared dataset loading utility for pages that need data from the .pkl.gz bundle.
Usage:
    from packages.load_dataset import ensure_dataset_loaded
    ensure_dataset_loaded()  # shows UI and stops if data is not available
"""
import os
import gzip
import pickle
import configparser
import streamlit as st
import numpy as np
import pandas as pd


def _get_default_path():
    """Try to get the default .pkl.gz path from defaults.ini."""
    _defaults_file = r'./.streamlit/defaults.ini'
    _default_dir = ""
    if os.path.exists(_defaults_file):
        _cfg = configparser.ConfigParser()
        _cfg.read(_defaults_file)
        try:
            _default_dir = _cfg.get('Import Data', 'file_directory')
        except configparser.Error:
            pass
    if _default_dir:
        return os.path.join(_default_dir, "baseline_corrected_full_dataset.pkl.gz")
    return ""


def ensure_dataset_loaded(
    require_keys=None,
    compute_megasum=False,
    page_key_prefix="_shared",
):
    """
    Check if required session-state keys are populated.
    If not, show a 'Load Dataset' UI and populate them from a .pkl.gz file.

    Parameters
    ----------
    require_keys : list[str] or None
        Session-state keys that must be present. Defaults to the standard set:
        ["x_mass", "compilation_baseline_corrected_data", "unique_wavenumbers"]
    compute_megasum : bool
        If True and MegaSum is missing, compute it from the loaded bundle.
    page_key_prefix : str
        Prefix for widget keys to avoid duplicates across pages.

    Returns
    -------
    bool
        True if all required keys are now available, False (with st.stop()) otherwise.
    """
    if require_keys is None:
        require_keys = [
            "x_mass",
            "compilation_baseline_corrected_data",
            "unique_wavenumbers",
        ]

    # Check if everything is already available
    all_present = all(st.session_state.get(k) is not None for k in require_keys)
    if compute_megasum:
        all_present = all_present and st.session_state.get("MegaSum") is not None

    if all_present:
        return True

    # --- Show fallback loading UI ---
    st.info(
        "No data in session. You can either run Sections 1–2 first, "
        "or load a previously exported dataset below."
    )

    st.markdown("### 📂 Load Saved Dataset")
    st.caption(
        "Load a `baseline_corrected_full_dataset.pkl.gz` file exported from "
        "**Section 2.1**. This lets you skip steps 1–2 entirely."
    )

    _default_path = _get_default_path()
    load_path = st.text_input(
        "Path to exported dataset (.pkl.gz)",
        value=_default_path,
        key=f"{page_key_prefix}_load_path",
        help="Full path to the .pkl.gz file exported by Section 2.1's 'Export full dataset' button.",
    )

    if st.button("📥 Load Dataset", type="primary", key=f"{page_key_prefix}_load_btn"):
        if not load_path or not os.path.exists(load_path):
            st.error(f"❌ File not found: `{load_path}`")
        else:
            try:
                with gzip.open(load_path, "rb") as f:
                    bundle = pickle.load(f)

                # Populate standard keys
                st.session_state["x_mass"] = bundle["x_mass"]
                st.session_state["compilation_baseline_corrected_data"] = bundle[
                    "compilation_baseline_corrected_data"
                ]
                st.session_state["unique_wavenumbers"] = bundle["unique_wavenumbers"]
                st.session_state["plot_columnIndex_withoutIR"] = bundle.get(
                    "plot_columnIndex_withoutIR", -2
                )
                st.session_state["plot_columnIndex_withIR"] = bundle.get(
                    "plot_columnIndex_withIR", -1
                )

                # Also set compiled_data as alias (used by 4.1_MegaSum)
                st.session_state["compiled_data"] = bundle[
                    "compilation_baseline_corrected_data"
                ]

                # Compute MegaSum if requested or if it's missing
                if compute_megasum or st.session_state.get("MegaSum") is None:
                    _compute_megasum_from_bundle(bundle)

                n_wn = len(bundle["unique_wavenumbers"])
                n_mz = len(bundle["x_mass"])
                st.success(
                    f"✅ Loaded {n_wn} wavenumbers × {n_mz} m/z bins "
                    f"from `{os.path.basename(load_path)}`"
                )
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load: {e}")

    st.stop()
    return False


def _compute_megasum_from_bundle(bundle):
    """
    Compute MegaSum (sum of all mass spectra across wavenumbers) from the bundle data.
    """
    compilation = bundle["compilation_baseline_corrected_data"]
    unique_wavenumbers = bundle["unique_wavenumbers"]
    x_mass = bundle["x_mass"]

    # Concatenate all wavenumber DataFrames
    frames = []
    for wn in unique_wavenumbers:
        if wn in compilation:
            frames.append(compilation[wn])

    if not frames:
        return

    MegaTable = pd.concat(frames, axis=1)

    # Sum signals: even columns = without IR, odd columns = with IR
    signal_withoutIR = MegaTable.iloc[:, 0::2].sum(axis=1)
    signal_withIR = MegaTable.iloc[:, 1::2].sum(axis=1)

    # Simple baseline correction using first 10% of points
    n_baseline = max(5, int(len(signal_withoutIR) * 0.1))
    baseline_withoutIR = np.mean(signal_withoutIR.iloc[:n_baseline])
    baseline_withIR = np.mean(signal_withIR.iloc[:n_baseline])

    # Use baseline_range_indices if available
    baseline_range_indices = st.session_state.get("baseline_range_indices")
    if baseline_range_indices is not None:
        idx = np.asarray(baseline_range_indices, dtype=int)
        baseline_withoutIR = np.mean(signal_withoutIR.iloc[idx])
        baseline_withIR = np.mean(signal_withIR.iloc[idx])

    new_table = pd.DataFrame({
        "signal_withoutIR": signal_withoutIR,
        "signal_withIR": signal_withIR,
        "baseline_corrected_signal_withoutIR": signal_withoutIR - baseline_withoutIR,
        "baseline_corrected_signal_withIR": signal_withIR - baseline_withIR,
    })

    st.session_state["MegaSum"] = pd.concat([MegaTable, new_table], axis=1)
