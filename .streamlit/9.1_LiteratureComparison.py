import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from scipy.signal import find_peaks, savgol_filter
import os
import glob
import gzip
import pickle
from packages.load_dataset import _NumpyCompatUnpickler
import configparser
import zipfile
import io
from datetime import datetime

st.set_page_config(page_title="Literature Comparison", layout="wide")
st.title("📚 Literature Comparison")
st.caption(
    "Compare Danial's detected masses and spectra against published results. "
    "Choose exactly which data to pull from the pipeline or load from files."
)

# ─── JDX (JCAMP-DX) file parser ───────────────────────────────────────────────
def parse_jdx(file_content):
    """Parse a JCAMP-DX (.jdx) file and return metadata dict, wavenumber array, intensity array."""
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8", errors="replace")
    lines = file_content.splitlines()
    meta = {}
    data_start = None
    xfactor = 1.0
    yfactor = 1.0
    for i, line in enumerate(lines):
        line_s = line.strip()
        if line_s.startswith("##"):
            key_val = line_s[2:]
            if "=" in key_val:
                key, val = key_val.split("=", 1)
                key = key.strip().upper()
                val = val.strip()
                meta[key] = val
                if key == "XFACTOR":
                    xfactor = float(val)
                elif key == "YFACTOR":
                    yfactor = float(val)
                elif key == "XYDATA":
                    data_start = i + 1
        elif line_s.startswith("$$"):
            continue  # comment

    if data_start is None:
        raise ValueError("No ##XYDATA= line found in JDX file.")

    # Parse packed X++(Y..Y) format
    x_values = []
    y_values = []
    firstx = float(meta.get("FIRSTX", 0))
    lastx = float(meta.get("LASTX", 0))
    npoints = int(meta.get("NPOINTS", 0))

    for line in lines[data_start:]:
        line_s = line.strip()
        if line_s.startswith("##"):
            break  # end of data block
        if not line_s or line_s.startswith("$$"):
            continue
        parts = line_s.split()
        if len(parts) < 2:
            continue
        try:
            x_start = float(parts[0]) * xfactor
            for j, ystr in enumerate(parts[1:]):
                y_values.append(float(ystr) * yfactor)
        except ValueError:
            continue

    # Reconstruct x values: evenly spaced from FIRSTX to LASTX
    if npoints > 0 and len(y_values) > 0:
        x_values = np.linspace(firstx, lastx, len(y_values))
    else:
        x_values = np.arange(len(y_values), dtype=float)

    return meta, np.array(x_values), np.array(y_values)

# ─── Workflow mode selector ────────────────────────────────────────────────────
_workflow_mode = st.radio(
    "Workflow mode",
    ["Full workflow (mass + spectral comparison)", "Quick spectral comparison only"],
    key="_lit_workflow_mode",
    horizontal=True,
    help="'Quick' skips mass loading (Sections 1–3) and jumps straight to spectral comparison.",
)
_quick_mode = (_workflow_mode == "Quick spectral comparison only")

# ════════════════════════════════════════════════════════════════════════════════
# QUICK MODE — Skip mass comparison, go straight to spectral comparison
# ════════════════════════════════════════════════════════════════════════════════
if _quick_mode:
    st.markdown("---")
    st.markdown("## Quick Spectral Comparison")
    st.caption(
        "Directly compare your IR spectrum with NIST / literature spectra. "
        "Supports CSV and **JDX (JCAMP-DX)** files from NIST."
    )

    _q_col1, _q_col2 = st.columns(2)

    # ── Your spectrum ────────────────────────────────────────────────────
    with _q_col1:
        st.markdown("**Your spectrum (Danial)**")
        _q_your_src = st.radio(
            "Source", ["From session state", "Upload CSV"],
            key="_q_your_src", horizontal=True,
        )
        your_wn = None
        your_intensity = None

        if _q_your_src == "From session state":
            _xm = st.session_state.get("x_mass")
            _comp = st.session_state.get("compilation_baseline_corrected_data")
            _wns = st.session_state.get("unique_wavenumbers")
            _pc_wo = st.session_state.get("plot_columnIndex_withoutIR", -2)
            _pc_w = st.session_state.get("plot_columnIndex_withIR", -1)
            if _xm is not None and _comp is not None and _wns is not None:
                _q_mz = st.number_input("m/z to extract IR for", value=128.0, step=0.1, key="_q_mz")
                _q_hw = st.number_input("Integration half-width (amu)", value=0.3, min_value=0.05,
                                        max_value=2.0, step=0.05, key="_q_hw")
                _sel_mask = (np.asarray(_xm) >= _q_mz - _q_hw) & (np.asarray(_xm) <= _q_mz + _q_hw)
                _n_in = int(_sel_mask.sum())
                if _n_in > 0:
                    _wn_sorted = sorted(_wns)
                    _wn_arr = np.array(_wn_sorted, dtype=float)
                    _depl = np.zeros(len(_wn_sorted))
                    for _i, _wn in enumerate(_wn_sorted):
                        _df_wn = _comp[_wn]
                        _wo = _df_wn.iloc[:, _pc_wo].values[_sel_mask].sum()
                        _w = _df_wn.iloc[:, _pc_w].values[_sel_mask].sum()
                        if _wo > 0:
                            _depl[_i] = 1.0 - (_w / _wo)
                        else:
                            _depl[_i] = np.nan
                    with np.errstate(invalid="ignore", divide="ignore"):
                        _ln = -np.log(np.clip(1.0 - _depl, 1e-6, None))
                    your_wn = _wn_arr
                    your_intensity = _ln
                    st.success(f"✅ Computed IR spectrum from {_n_in} m/z bins × {len(_wn_sorted)} wn")
                else:
                    st.warning(f"No m/z bins within ±{_q_hw} of {_q_mz:.1f}")
            else:
                st.info("No dataset in session. Load a .pkl.gz or run sections 1–2 first, or upload a CSV.")
        else:
            _q_your_file = st.file_uploader(
                "Upload your spectrum CSV (columns: wavenumber, intensity)",
                type=["csv"], key="_q_your_csv",
            )
            if _q_your_file:
                _df = pd.read_csv(_q_your_file)
                if len(_df.columns) >= 2:
                    your_wn = _df.iloc[:, 0].values
                    your_intensity = _df.iloc[:, 1].values
                    st.success(f"Loaded {len(your_wn)} points")

    # ── Literature / NIST spectra ────────────────────────────────────────
    with _q_col2:
        st.markdown("**Literature / NIST spectra**")
        _q_lit_files = st.file_uploader(
            "Upload literature spectra (CSV or JDX/JCAMP-DX from NIST)",
            type=["csv", "jdx", "dx", "jcamp"],
            key="_q_lit_upload",
            accept_multiple_files=True,
        )

        lit_spectra = []
        if _q_lit_files:
            for _f in _q_lit_files:
                _fname = _f.name
                _ext = os.path.splitext(_fname)[1].lower()
                try:
                    _f.seek(0)
                    _raw = _f.read()
                    if _ext in (".jdx", ".dx", ".jcamp"):
                        _meta, _wn, _it = parse_jdx(_raw)
                        _title = _meta.get("TITLE", _fname)
                        _molform = _meta.get("MOLFORM", "")
                        _cas = _meta.get("CAS REGISTRY NO", "")
                        _display_name = f"{_title}"
                        if _molform:
                            _display_name += f" ({_molform})"
                        if _cas:
                            _display_name += f" [CAS {_cas}]"
                        # Convert transmittance to absorbance if needed
                        _yunits = _meta.get("YUNITS", "").upper()
                        if "TRANSMITTANCE" in _yunits:
                            with np.errstate(invalid="ignore", divide="ignore"):
                                _it = -np.log10(np.clip(_it, 1e-6, None))
                            _display_name += " (→ absorbance)"
                        lit_spectra.append({"name": _display_name, "wn": _wn, "intensity": _it})
                        st.success(f"📄 JDX: {_display_name} — {len(_wn)} pts, "
                                   f"{float(_wn.min()):.0f}–{float(_wn.max()):.0f} cm⁻¹")
                    else:
                        _df = pd.read_csv(io.BytesIO(_raw))
                        if len(_df.columns) >= 2:
                            lit_spectra.append({
                                "name": _fname,
                                "wn": _df.iloc[:, 0].values,
                                "intensity": _df.iloc[:, 1].values,
                            })
                            st.success(f"📄 CSV: {_fname} — {len(_df)} pts")
                except Exception as _e:
                    st.warning(f"Could not parse {_fname}: {_e}")

    # ── Plot overlay (reuse the same plotting logic) ─────────────────────
    if your_wn is not None or len(lit_spectra) > 0:
        _opt_col1, _opt_col2, _opt_col3, _opt_col4 = st.columns([1, 1, 1, 1])
        with _opt_col1:
            _norm = st.checkbox("Normalize to [0, 1]", value=True, key="_q_spec_norm")
        with _opt_col2:
            _sg_on = st.checkbox("Savitzky–Golay smoothing (your data)", value=False, key="_q_spec_sg")
        with _opt_col3:
            _sg_win = st.number_input(
                "SG window length (odd)", value=11, min_value=3, max_value=201, step=2,
                key="_q_spec_sg_win", disabled=not _sg_on,
            )
        with _opt_col4:
            _sg_poly = st.number_input(
                "SG polyorder", value=3, min_value=1, max_value=7, step=1,
                key="_q_spec_sg_poly", disabled=not _sg_on,
            )
        _show_raw = st.checkbox(
            "Show raw (un-smoothed) as faint line", value=True,
            key="_q_spec_show_raw", disabled=not _sg_on,
        )
        _rc1, _rc2, _rc3 = st.columns([1, 1, 1])
        with _rc1:
            _ridge_on = st.checkbox("Ridge plot (stack vertically)", value=True, key="_q_spec_ridge")
        with _rc2:
            _ridge_gap = st.slider(
                "Ridge spacing", min_value=0.2, max_value=2.0, value=1.0, step=0.1,
                key="_q_spec_ridge_gap", disabled=not _ridge_on,
            )
        with _rc3:
            _ridge_fill = st.checkbox("Fill under ridge curves", value=False, key="_q_spec_ridge_fill",
                                       disabled=not _ridge_on)

        # Per-spectrum shift controls
        _shift_on = st.checkbox("Per-spectrum wavenumber shift", value=False, key="_q_shift_on")
        _shifts = {}
        if _shift_on:
            _shift_cols = st.columns(min(len(lit_spectra) + 1, 6))
            _all_names = ["Danial"] + [s["name"] for s in lit_spectra]
            for _si, _sn in enumerate(_all_names):
                with _shift_cols[_si % len(_shift_cols)]:
                    _shifts[_sn] = st.number_input(
                        f"Shift: {_sn[:25]}", value=0.0, step=1.0,
                        key=f"_q_shift_{_si}", format="%.1f",
                    )

        # Helper functions
        def _apply_sg(y):
            if not _sg_on:
                return y
            _y_in = np.asarray(y, dtype=float).copy()
            _w = _sg_win if _sg_win % 2 == 1 else _sg_win + 1
            _p = min(_sg_poly, _w - 1)
            if len(_y_in) < _w:
                return _y_in
            return savgol_filter(_y_in, window_length=_w, polyorder=_p)

        def _normalize(y):
            if not _norm:
                return y
            _arr = np.asarray(y, dtype=float)
            _finite = _arr[np.isfinite(_arr)]
            if len(_finite) == 0 or np.ptp(_finite) == 0:
                return _arr
            return (_arr - np.nanmin(_arr)) / np.ptp(_finite)

        # Build rows
        _rows = []
        _lit_palette = [
            "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        if your_wn is not None:
            _y_raw_d = np.asarray(your_intensity, dtype=float).copy()
            _y_sm_d = _apply_sg(_y_raw_d)
            _rows.append({
                "name": "Danial" + (" (SG)" if _sg_on else ""),
                "color": "#1f77b4",
                "wn": your_wn + _shifts.get("Danial", 0.0),
                "y": _normalize(_y_sm_d),
                "raw_y": _normalize(_y_raw_d) if (_sg_on and _show_raw) else None,
            })
        for _idx, _spec in enumerate(lit_spectra):
            _clr = _lit_palette[_idx % len(_lit_palette)]
            _y_lit = np.asarray(_spec["intensity"], dtype=float)
            _rows.append({
                "name": _spec["name"],
                "color": _clr,
                "wn": np.asarray(_spec["wn"], dtype=float) + _shifts.get(_spec["name"], 0.0),
                "y": _normalize(_y_lit),
                "raw_y": None,
            })

        # Plot
        fig = go.Figure()
        for _ri, _row in enumerate(_rows):
            _offset = _ri * _ridge_gap if _ridge_on else 0
            _y_plot = _row["y"] + _offset
            if _row["raw_y"] is not None:
                fig.add_trace(go.Scatter(
                    x=_row["wn"], y=_row["raw_y"] + _offset,
                    mode="lines", line=dict(color=_row["color"], width=0.5, dash="dot"),
                    name=_row["name"] + " (raw)", opacity=0.4, showlegend=False,
                ))
            _fillcolor = None
            if _ridge_on and _ridge_fill:
                _r, _g, _b = int(_row["color"][1:3], 16), int(_row["color"][3:5], 16), int(_row["color"][5:7], 16)
                _fillcolor = f"rgba({_r},{_g},{_b},0.15)"
            fig.add_trace(go.Scatter(
                x=_row["wn"], y=_y_plot, mode="lines",
                line=dict(color=_row["color"], width=2),
                name=_row["name"],
                fill="toself" if _fillcolor else None,
                fillcolor=_fillcolor,
            ))

        fig.update_layout(
            title="Quick Spectral Comparison",
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity" + (" (stacked)" if _ridge_on else ""),
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # PNG export
        with st.expander("📐 PNG export settings"):
            _pc1, _pc2, _pc3 = st.columns(3)
            with _pc1:
                _png_w = st.number_input("Width (px)", value=1400, step=100, key="_q_png_w")
            with _pc2:
                _png_h = st.number_input("Height (px)", value=600, step=100, key="_q_png_h")
            with _pc3:
                _png_s = st.number_input("Scale", value=2.0, step=0.5, key="_q_png_s")
        _png_bytes = fig.to_image(format="png", width=_png_w, height=_png_h, scale=_png_s)
        # Build filename from loaded spectrum names
        _name_parts = []
        if your_wn is not None:
            _name_parts.append("Danial")
        for _ls in lit_spectra:
            _safe = os.path.splitext(_ls["name"])[0]
            _safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in _safe)[:40]
            _name_parts.append(_safe)
        _png_fname = ("__".join(_name_parts) if _name_parts else "spectral_comparison") + ".png"
        st.download_button(
            "📥 Download PNG",
            data=_png_bytes,
            file_name=_png_fname,
            mime="image/png",
        )
    else:
        st.info("Upload spectra above to see the comparison plot.")

    st.stop()  # ← Don't run the full workflow below

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DANIAL'S DATA  (full workflow)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 1. Danial's Data")
st.caption("Choose where to load Danial's detected masses from.")

your_source = st.radio(
    "Data source",
    [
        "Paste / type manually",
        "Load .pkl.gz dataset & detect peaks here",
        "Session state — IR-active masses (4.0 Misc)",
        "Session state — Peak detection (4.2 PeakDetection)",
        "Session state — Mass-channel IR results (4.0 Misc Tab 3)",
        "Load from CSV file",
    ],
    key="_lit_your_source",
    horizontal=False,
)

your_masses_df = None

if your_source == "Paste / type manually":
    _manual_input = st.text_area(
        "Enter m/z values (comma-separated)",
        placeholder="128.1, 152.0, 154.1, 178.0, 202.0",
        height=80,
        key="_lit_manual_masses",
    )
    _your_label = st.text_input("Label for Danial's data", value="Danial", key="_lit_your_label")
    if _manual_input.strip():
        try:
            _vals = [float(x.strip()) for x in _manual_input.split(",") if x.strip()]
            your_masses_df = pd.DataFrame({"m/z": sorted(_vals), "source": _your_label})
        except ValueError:
            st.error("Could not parse values. Use comma-separated numbers.")

elif your_source == "Load .pkl.gz dataset & detect peaks here":
    # ── Locate default path ─────────────────────────────────────────────────
    _default_dir = ""
    _defaults_file = r'./.streamlit/defaults.ini'
    if os.path.exists(_defaults_file):
        _cfg = configparser.ConfigParser()
        _cfg.read(_defaults_file)
        try:
            _default_dir = _cfg.get('Import Data', 'file_directory')
        except configparser.Error:
            pass
    _default_pkl = os.path.join(_default_dir, "baseline_corrected_full_dataset.pkl.gz") if _default_dir else ""

    _pkl_path = st.text_input(
        "Path to baseline_corrected_full_dataset.pkl.gz",
        value=st.session_state.get("_lit_pkl_path", _default_pkl),
        key="_lit_pkl_path",
        help="The full dataset exported from Section 2.1 'Export full dataset'.",
    )

    # Load button
    _load_col1, _load_col2 = st.columns([1, 3])
    with _load_col1:
        _do_load = st.button("📥 Load dataset", key="_lit_do_load_pkl",
                              type="primary" if not st.session_state.get("_lit_pkl_loaded") else "secondary")

    if _do_load:
        if not _pkl_path or not os.path.exists(_pkl_path):
            st.error(f"❌ File not found: `{_pkl_path}`")
        else:
            try:
                with gzip.open(_pkl_path, "rb") as f:
                    bundle = _NumpyCompatUnpickler(f).load()
                st.session_state["x_mass"] = bundle["x_mass"]
                st.session_state["compilation_baseline_corrected_data"] = bundle["compilation_baseline_corrected_data"]
                st.session_state["unique_wavenumbers"] = bundle["unique_wavenumbers"]
                st.session_state["plot_columnIndex_withoutIR"] = bundle.get("plot_columnIndex_withoutIR", -2)
                st.session_state["plot_columnIndex_withIR"] = bundle.get("plot_columnIndex_withIR", -1)
                st.session_state["_lit_pkl_loaded"] = True
                st.success(
                    f"✅ Loaded {len(bundle['unique_wavenumbers'])} wavenumbers × "
                    f"{len(bundle['x_mass'])} m/z bins"
                )
            except Exception as e:
                st.error(f"❌ Failed to load: {e}")

    # ── Peak detection controls (only if loaded) ────────────────────────────
    if st.session_state.get("_lit_pkl_loaded"):
        st.markdown("#### Peak detection settings")

        _x_mass_local = st.session_state["x_mass"]
        _comp_local = st.session_state["compilation_baseline_corrected_data"]
        _wns_local = st.session_state["unique_wavenumbers"]
        _pc_wo = st.session_state["plot_columnIndex_withoutIR"]
        _pc_w = st.session_state["plot_columnIndex_withIR"]

        _pd_col1, _pd_col2, _pd_col3 = st.columns(3)
        with _pd_col1:
            _wn_min = st.number_input(
                "Wavenumber min (cm⁻¹)", value=float(min(_wns_local)),
                step=10.0, key="_lit_wn_min",
            )
        with _pd_col2:
            _wn_max = st.number_input(
                "Wavenumber max (cm⁻¹)", value=float(max(_wns_local)),
                step=10.0, key="_lit_wn_max",
            )
        with _pd_col3:
            _noise_floor = st.number_input(
                "Noise floor", value=0.001, min_value=0.0,
                step=0.0005, format="%.5f", key="_lit_noise_floor",
                help="Mask m/z bins where mean |without-IR signal| is below this.",
            )

        _pp_col1, _pp_col2 = st.columns(2)
        with _pp_col1:
            _prom = st.number_input(
                "Min prominence", value=0.05, min_value=0.0,
                step=0.01, format="%.4f", key="_lit_prom",
            )
        with _pp_col2:
            _dist = st.number_input(
                "Min distance between peaks (m/z)", value=1.0,
                min_value=0.1, step=0.5, key="_lit_dist",
            )

        if st.button("🔍 Detect IR-active masses", type="primary", key="_lit_detect"):
            with st.spinner("Building matrices and detecting peaks…"):
                _wn_list = sorted([wn for wn in _wns_local if _wn_min <= float(wn) <= _wn_max])
                _n_wn = len(_wn_list)
                _n_mz = len(_x_mass_local)
                _mat_wo = np.zeros((_n_wn, _n_mz))
                _mat_w = np.zeros((_n_wn, _n_mz))
                for i, wn in enumerate(_wn_list):
                    _data_wn = _comp_local[wn]
                    _mat_wo[i, :] = _data_wn.iloc[:, _pc_wo].values
                    _mat_w[i, :] = _data_wn.iloc[:, _pc_w].values
                _mat_delta = _mat_wo - _mat_w
                _baseline = np.mean(np.abs(_mat_wo), axis=0)
                _mask = _baseline >= _noise_floor

                _mz_active = _x_mass_local[_mask]
                _mean_abs_delta = np.mean(np.abs(_mat_delta[:, _mask]), axis=0)

                # Peak detection
                if len(_mz_active) > 1:
                    _avg_spacing = np.mean(np.diff(_mz_active))
                    _min_dist_idx = max(1, int(_dist / _avg_spacing))
                else:
                    _min_dist_idx = 1
                _peak_idx, _peak_props = find_peaks(_mean_abs_delta, prominence=_prom, distance=_min_dist_idx)

                if len(_peak_idx) > 0:
                    _peaks_df = pd.DataFrame({
                        "m/z": _mz_active[_peak_idx],
                        "Mean |ΔI|": _mean_abs_delta[_peak_idx],
                        "Prominence": _peak_props["prominences"],
                    }).sort_values("m/z").reset_index(drop=True)
                    st.session_state["_lit_inline_peaks"] = _peaks_df
                    st.success(f"✅ Detected {len(_peak_idx)} IR-active masses.")
                else:
                    st.session_state["_lit_inline_peaks"] = None
                    st.warning("⚠️ No peaks detected. Try lowering prominence.")

        # Show & use detected peaks
        if st.session_state.get("_lit_inline_peaks") is not None:
            _peaks_df = st.session_state["_lit_inline_peaks"]
            st.dataframe(_peaks_df, use_container_width=True, height=250)
            your_masses_df = _peaks_df[["m/z"]].copy()
            your_masses_df["source"] = "Danial"
            your_masses_df["signal"] = _peaks_df["Mean |ΔI|"].values
            your_masses_df["prominence"] = _peaks_df["Prominence"].values

elif your_source == "Session state — IR-active masses (4.0 Misc)":
    _dp = st.session_state.get("_detected_peaks")
    if _dp is not None and len(_dp) > 0:
        st.success(f"Found {len(_dp)} IR-active masses from 4.0 Misc.")
        st.dataframe(_dp, use_container_width=True, height=200)
        your_masses_df = _dp[["m/z"]].copy()
        your_masses_df["source"] = "IR-active (4.0)"
        if "Mean |ΔI|" in _dp.columns:
            your_masses_df["signal"] = _dp["Mean |ΔI|"].values
        if "Prominence" in _dp.columns:
            your_masses_df["prominence"] = _dp["Prominence"].values
    else:
        st.warning("No IR-active masses found in session state. Run detection in **4.0 Misc** first.")

elif your_source == "Session state — Peak detection (4.2 PeakDetection)":
    _dp2 = st.session_state.get("detected_peaks_df")
    if _dp2 is not None and len(_dp2) > 0:
        st.success(f"Found {len(_dp2)} peaks from 4.2 PeakDetection.")
        st.dataframe(_dp2, use_container_width=True, height=200)
        _mz_col = "m/z" if "m/z" in _dp2.columns else _dp2.columns[0]
        your_masses_df = _dp2[[_mz_col]].copy()
        your_masses_df.columns = ["m/z"]
        your_masses_df["source"] = "PeakDet (4.2)"
    else:
        st.warning("No peak detection data in session state. Run **4.2 PeakDetection** first.")

elif your_source == "Session state — Mass-channel IR results (4.0 Misc Tab 3)":
    _fr = st.session_state.get("_frag_results")
    if _fr is not None and len(_fr) > 0:
        _centers = [v["center"] for v in _fr.values()]
        st.success(f"Found {len(_centers)} mass channels with IR spectra from 4.0 Misc Tab 3.")
        your_masses_df = pd.DataFrame({"m/z": sorted(_centers), "source": "IR spectra (4.0 Tab3)"})
        your_masses_df["has_IR_spectrum"] = True
    else:
        st.warning("No mass-channel IR results in session state. Compute in **4.0 Misc → Tab 3** first.")

elif your_source == "Load from CSV file":
    # Try to find output folder
    _default_dir = ""
    _defaults_file = r'./.streamlit/defaults.ini'
    if os.path.exists(_defaults_file):
        _cfg = configparser.ConfigParser()
        _cfg.read(_defaults_file)
        try:
            _default_dir = _cfg.get('Import Data', 'file_directory')
        except configparser.Error:
            pass

    _csv_path = st.text_input(
        "Path to CSV file with Danial's masses",
        value=os.path.join(_default_dir, "output", "") if _default_dir else "",
        key="_lit_csv_path",
        help="CSV must have a column named 'm/z'. Other columns are optional.",
    )
    if _csv_path and os.path.isfile(_csv_path):
        try:
            _loaded = pd.read_csv(_csv_path)
            if "m/z" in _loaded.columns:
                st.success(f"Loaded {len(_loaded)} rows from `{os.path.basename(_csv_path)}`")
                st.dataframe(_loaded, use_container_width=True, height=200)
                your_masses_df = _loaded.copy()
                if "source" not in your_masses_df.columns:
                    your_masses_df["source"] = os.path.basename(_csv_path)
            else:
                st.error("CSV must contain a column named `m/z`.")
        except Exception as e:
            st.error(f"Failed to load: {e}")
    elif _csv_path and not os.path.isfile(_csv_path):
        # Show available CSVs in directory if it's a folder
        if os.path.isdir(_csv_path):
            _csvs = sorted(glob.glob(os.path.join(_csv_path, "*.csv")))
            if _csvs:
                st.info(f"Found {len(_csvs)} CSV files in this folder:")
                for f in _csvs:
                    st.caption(f"`{os.path.basename(f)}`")

# Store for later sections
if your_masses_df is not None and len(your_masses_df) > 0:
    your_masses_df = your_masses_df.sort_values("m/z").reset_index(drop=True)
    st.session_state["_lit_your_masses"] = your_masses_df
    st.markdown(f"**✅ {len(your_masses_df)} masses loaded from Danial's data**")
elif "_lit_your_masses" in st.session_state:
    your_masses_df = st.session_state["_lit_your_masses"]
    st.caption(f"Using previously loaded data: {len(your_masses_df)} masses")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LITERATURE DATA
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 2. Literature Data")
st.caption(
    "Upload one or more CSV files with published mass assignments. "
    "Each CSV needs at least a `m/z` column. Optional columns: "
    "`assignment`, `composition`, `confidence`, `has_DFT`, `has_exp_IR`."
)

# Initialize literature storage
if "_lit_papers" not in st.session_state:
    st.session_state["_lit_papers"] = {}

n_papers = st.number_input(
    "Number of literature sources to compare",
    min_value=1, max_value=10, value=2 if len(st.session_state["_lit_papers"]) == 0 else len(st.session_state["_lit_papers"]),
    step=1, key="_lit_n_papers",
)

for i in range(int(n_papers)):
    with st.expander(f"📄 Literature source {i+1}", expanded=(i == 0 or i < len(st.session_state["_lit_papers"]))):
        # Set default labels
        default_labels = [
            "Lemmens et al. 2020 (pure naphthalene)",
            "Loru et al. 2022 (naphthalene + acetonitrile)"
        ]
        default_label = default_labels[i] if i < len(default_labels) else f"Paper {i+1}"
        
        # Check if this is a new session or if label already exists
        if f"_lit_paper_label_{i}" not in st.session_state or st.session_state[f"_lit_paper_label_{i}"] == f"Paper {i+1}":
            st.session_state[f"_lit_paper_label_{i}"] = default_label
        
        _label = st.text_input(
            "Reference label",
            value=st.session_state[f"_lit_paper_label_{i}"],
            key=f"_lit_paper_label_{i}",
            help="Short name for this source, e.g. 'Lemmens 2020'",
        )

        # Set default input method to "paste" for first two sources
        if f"_lit_paper_method_{i}" not in st.session_state and i < 2:
            st.session_state[f"_lit_paper_method_{i}"] = "Paste masses"
        
        _input_method = st.radio(
            "Input method",
            ["Upload CSV", "Paste masses"],
            key=f"_lit_paper_method_{i}",
            horizontal=True,
        )

        _paper_df = None

        if _input_method == "Upload CSV":
            _uploaded = st.file_uploader(
                f"Upload CSV for {_label}",
                type=["csv", "tsv", "txt"],
                key=f"_lit_upload_{i}",
            )
            if _uploaded is not None:
                try:
                    _paper_df = pd.read_csv(_uploaded)
                    if "m/z" not in _paper_df.columns:
                        # Try common alternatives
                        for alt in ["mz", "mass", "M/Z", "Mass", "m_z"]:
                            if alt in _paper_df.columns:
                                _paper_df = _paper_df.rename(columns={alt: "m/z"})
                                break
                    if "m/z" in _paper_df.columns:
                        st.success(f"Loaded {len(_paper_df)} entries for {_label}")
                        st.dataframe(_paper_df, use_container_width=True, height=200)
                    else:
                        st.error(f"No `m/z` column found. Available: {list(_paper_df.columns)}")
                        _paper_df = None
                except Exception as e:
                    st.error(f"Failed to parse: {e}")

        else:  # Paste masses
            # Set default mass values for first two sources
            default_masses = [
                "36, 37, 50, 52, 61, 63, 74, 87, 98, 126, 128, 141, 150, 152, 154, 165, 176, 178, 189, 200, 202, 228, 252, 326",
                "126, 128, 140, 141, 142, 150, 151, 152, 153, 167, 176, 178"
            ]
            default_mass = default_masses[i] if i < len(default_masses) else ""
            
            # Set default value in session state if not already set
            if f"_lit_paste_{i}" not in st.session_state and i < 2:
                st.session_state[f"_lit_paste_{i}"] = default_mass
            
            _paste = st.text_area(
                f"Paste m/z values for {_label} (comma-separated)",
                height=80,
                key=f"_lit_paste_{i}",
                placeholder="126, 128, 141, 150, 152, 154, 176, 178, 202",
            )
            if _paste.strip():
                try:
                    _vals = [float(x.strip()) for x in _paste.split(",") if x.strip()]
                    _paper_df = pd.DataFrame({"m/z": sorted(_vals)})
                except ValueError:
                    st.error("Could not parse values.")

        if _paper_df is not None and len(_paper_df) > 0:
            _paper_df["_source"] = _label
            st.session_state["_lit_papers"][_label] = _paper_df

# Show summary of loaded papers
_loaded_papers = {k: v for k, v in st.session_state["_lit_papers"].items() if v is not None and len(v) > 0}
if _loaded_papers:
    st.markdown("### Loaded literature sources")
    for name, df in _loaded_papers.items():
        _cols = [c for c in df.columns if c != "_source"]
        st.caption(f"**{name}**: {len(df)} masses — columns: {', '.join(_cols)}")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MASS COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 3. Mass Comparison")

if your_masses_df is None or len(your_masses_df) == 0:
    st.info("Load Danial's data in Section 1 first.")
    st.stop()

if not _loaded_papers:
    st.info("Load at least one literature source in Section 2 first.")
    st.stop()

# Settings
_cmp_col1, _cmp_col2 = st.columns(2)
with _cmp_col1:
    tolerance = st.number_input(
        "Matching tolerance (amu)",
        value=0.5, min_value=0.01, max_value=5.0, step=0.1,
        key="_lit_tolerance",
        help="Two masses are considered a match if |m/z_Danial − m/z_lit| ≤ this value.",
    )
with _cmp_col2:
    round_display = st.number_input(
        "Round m/z display to (decimals)",
        value=1, min_value=0, max_value=4, step=1,
        key="_lit_round",
    )

if st.button("🔍 Compare Masses", type="primary", key="_lit_compare"):
    # Collect all unique masses
    all_sources = {"Danial": your_masses_df["m/z"].values}
    for name, df in _loaded_papers.items():
        all_sources[name] = df["m/z"].values

    source_names = list(all_sources.keys())

    # Build unique mass list (merge within tolerance)
    all_mz = np.concatenate(list(all_sources.values()))
    all_mz_sorted = np.sort(np.unique(all_mz))

    # Cluster masses within tolerance
    clusters = []
    used = set()
    for mz in all_mz_sorted:
        if mz in used:
            continue
        cluster = all_mz_sorted[np.abs(all_mz_sorted - mz) <= tolerance]
        for c in cluster:
            used.add(c)
        clusters.append(cluster)

    # Build comparison table
    rows = []
    for cluster in clusters:
        representative = round(float(np.mean(cluster)), round_display)
        row = {"m/z": representative}

        # Check which sources have this mass
        for src_name, src_mz in all_sources.items():
            matches = src_mz[np.abs(src_mz - representative) <= tolerance]
            if len(matches) == 0:
                # Also check against any cluster member
                found = False
                for c in cluster:
                    if np.any(np.abs(src_mz - c) <= tolerance):
                        found = True
                        break
                row[src_name] = "✓" if found else "—"
            else:
                row[src_name] = "✓"

            # Add exact value if found
            best_match = src_mz[np.argmin(np.abs(src_mz - representative))] if np.any(np.abs(src_mz - representative) <= tolerance * 2) else None
            if best_match is not None and np.abs(best_match - representative) <= tolerance:
                row[f"{src_name} (exact)"] = round(float(best_match), 2)

        # Add assignment from literature if available
        for name, df in _loaded_papers.items():
            if "assignment" in df.columns:
                _match_idx = np.where(np.abs(df["m/z"].values - representative) <= tolerance)[0]
                if len(_match_idx) > 0:
                    row["assignment"] = df.iloc[_match_idx[0]]["assignment"]
                    break
            if "composition" in df.columns:
                _match_idx = np.where(np.abs(df["m/z"].values - representative) <= tolerance)[0]
                if len(_match_idx) > 0:
                    row["composition"] = df.iloc[_match_idx[0]]["composition"]

        # Classify
        in_yours = row.get("Danial") == "✓"
        in_any_lit = any(row.get(n) == "✓" for n in source_names if n != "Danial")
        in_all = all(row.get(n) == "✓" for n in source_names)

        if in_all:
            row["status"] = "🟢 Common (all)"
        elif in_yours and in_any_lit:
            row["status"] = "🔵 Shared"
        elif in_yours and not in_any_lit:
            row["status"] = "🟡 New (Danial only)"
        elif not in_yours and in_any_lit:
            row["status"] = "🔴 Missing (lit only)"
        else:
            row["status"] = "⚪ Other"

        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values("m/z").reset_index(drop=True)

    st.session_state["_lit_comparison"] = comparison_df
    st.session_state["_lit_source_names"] = source_names

# Display results
if "_lit_comparison" in st.session_state:
    comparison_df = st.session_state["_lit_comparison"]
    source_names = st.session_state["_lit_source_names"]

    # Filter
    _filter = st.radio(
        "Filter",
        ["All", "🟢 Common (all)", "🔵 Shared", "🟡 New (Danial only)", "🔴 Missing (lit only)"],
        horizontal=True,
        key="_lit_filter",
    )
    if _filter != "All":
        _status_key = _filter
        _filtered = comparison_df[comparison_df["status"] == _status_key]
    else:
        _filtered = comparison_df

    # Summary counts
    _counts = comparison_df["status"].value_counts()
    _sum_cols = st.columns(len(_counts))
    for idx, (status, count) in enumerate(_counts.items()):
        with _sum_cols[idx]:
            st.metric(status, count)

    # Table — show only the key columns first
    _display_cols = ["m/z", "status"] + [n for n in source_names]
    _extra_cols = [c for c in _filtered.columns if c not in _display_cols and c not in [f"{n} (exact)" for n in source_names]]
    _show_cols = _display_cols + _extra_cols

    st.dataframe(
        _filtered[[c for c in _show_cols if c in _filtered.columns]],
        use_container_width=True,
        height=min(600, len(_filtered) * 40 + 60),
    )
    
    # Create a visual summary plot for mass comparison
    if st.checkbox("📊 Generate summary plot", value=True, key="_lit_show_summary_plot"):
        fig_summary = go.Figure()
        
        # Create a scatter plot of masses with status colors
        status_colors = {
            "🟢 Common (all)": "green",
            "🔵 Shared": "blue", 
            "🟡 Danial only": "orange",
            "🔴 Literature only": "red"
        }
        
        for status, color in status_colors.items():
            status_data = comparison_df[comparison_df["status"] == status]
            if len(status_data) > 0:
                fig_summary.add_trace(go.Scatter(
                    x=status_data["m/z"],
                    y=[status] * len(status_data),
                    mode='markers',
                    name=status,
                    marker=dict(color=color, size=8),
                    text=status_data["m/z"].round(3),
                    hovertemplate="m/z: %{x}<br>Status: %{y}<extra></extra>"
                ))
        
        fig_summary.update_layout(
            title="Mass Comparison Summary",
            xaxis_title="m/z",
            yaxis_title="Status",
            height=400,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_summary, use_container_width=True)
        with st.expander("⚙️ PNG export settings", expanded=False):
            _sm_c1, _sm_c2, _sm_c3 = st.columns(3)
            _sm_w = _sm_c1.number_input("Width (px)", 400, 6000, 1400, 100, key="_lit_sm_w")
            _sm_h = _sm_c2.number_input("Height (px)", 300, 4000, 700, 50, key="_lit_sm_h")
            _sm_s = _sm_c3.number_input("Scale (DPI ×)", 1.0, 4.0, 2.0, 0.5, key="_lit_sm_s")
        try:
            _summary_png = fig_summary.to_image(format="png", width=int(_sm_w), height=int(_sm_h), scale=float(_sm_s))
            st.download_button(
                "⬇️ Download this plot (PNG)",
                data=_summary_png,
                file_name=f"mass_comparison_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                key="_lit_dl_summary_inline",
            )
        except Exception as _e:
            st.caption(f"⚠️ PNG export unavailable: {_e}")

    # Download
    _dl_col1, _dl_col2 = st.columns(2)
    with _dl_col1:
        st.download_button(
            "⬇️ Download comparison table (CSV)",
            data=comparison_df.to_csv(index=False),
            file_name="mass_comparison.csv",
            mime="text/csv",
            key="_lit_dl_csv",
        )

    # ────────────────────────────────────────────────────────────────────────
    # Visual: mass spectrum overlay with markers
    # ────────────────────────────────────────────────────────────────────────
    st.markdown("### Mass Position Overlay")
    st.caption("Vertical markers for each source at their detected m/z positions.")

    fig_overlay = go.Figure()
    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # If MegaSum available, show as background
    _megasum = st.session_state.get("MegaSum")
    _xmass = st.session_state.get("x_mass")
    if _megasum is not None and _xmass is not None:
        _plot_col = st.session_state.get("plot_columnIndex_withoutIR", -2)
        try:
            _sig = _megasum.iloc[:, _plot_col].values
            fig_overlay.add_trace(go.Scatter(
                x=_xmass, y=_sig / np.max(np.abs(_sig)),
                mode="lines", name="Mass spectrum",
                line=dict(color="lightgray", width=1),
                opacity=0.5,
            ))
        except Exception:
            pass

    all_sources_data = {"Danial": your_masses_df["m/z"].values}
    for name, df in _loaded_papers.items():
        all_sources_data[name] = df["m/z"].values

    # User option: show m/z labels on each marker (also rendered in PNG export)
    _show_mz_labels = st.checkbox(
        "Show m/z value labels on each marker",
        value=True, key="_lit_overlay_show_labels",
        help="Useful for the static PNG export.",
    )

    _y_levels = []
    _y_labels = []
    for idx, (src_name, src_mz) in enumerate(all_sources_data.items()):
        c = _colors[idx % len(_colors)]
        _y_level = 0.5 + idx * 2.5  # bigger spacing between sources so vertical labels don't overlap
        _y_levels.append(_y_level)
        _y_labels.append(src_name)

        # baseline line for this source spanning the data range
        if len(src_mz) > 0:
            _x_min = float(np.nanmin([np.nanmin(v) for v in all_sources_data.values() if len(v) > 0]))
            _x_max = float(np.nanmax([np.nanmax(v) for v in all_sources_data.values() if len(v) > 0]))
            fig_overlay.add_trace(go.Scatter(
                x=[_x_min, _x_max], y=[_y_level, _y_level],
                mode="lines", line=dict(color=c, width=1, dash="dot"),
                opacity=0.3, showlegend=False, hoverinfo="skip",
            ))

        # vertical sticks
        _xs, _ys = [], []
        for mz in src_mz:
            _xs.extend([mz, mz, None])
            _ys.extend([_y_level - 0.4, _y_level + 0.4, None])
        fig_overlay.add_trace(go.Scatter(
            x=_xs, y=_ys, mode="lines",
            line=dict(color=c, width=2),
            name=f"{src_name} ({len(src_mz)})",
            hovertext=[f"{src_name}: {mz:.2f}" for mz in src_mz for _ in range(3)],
            hoverinfo="text",
        ))

        # m/z value labels — staggered alternately above/below the row
        # so neighboring numbers don't overlap. Horizontal text now (no rotation).
        if _show_mz_labels and len(src_mz) > 0:
            _sorted_mz = sorted(src_mz)
            for _j, mz in enumerate(_sorted_mz):
                # alternate offsets: even -> just above stick, odd -> further above
                if _j % 2 == 0:
                    _y_lbl = _y_level + 0.55
                else:
                    _y_lbl = _y_level - 0.55
                fig_overlay.add_annotation(
                    x=mz, y=_y_lbl,
                    text=f"{mz:.0f}",
                    showarrow=False,
                    font=dict(size=11, color=c),
                    xanchor="center",
                    yanchor="bottom" if _j % 2 == 0 else "top",
                )

    # Compute a denser set of x-ticks based on union of all masses
    _all_masses = sorted({float(m) for v in all_sources_data.values() for m in v})
    fig_overlay.update_layout(
        title=f"Mass Position Overlay — {len(all_sources_data)} sources, {len(_all_masses)} unique m/z",
        xaxis=dict(
            title="m/z",
            tickmode="array",
            tickvals=_all_masses,
            ticktext=[f"{m:.0f}" for m in _all_masses],
            tickangle=-90,
            tickfont=dict(size=10),
            showgrid=True, gridcolor="rgba(128,128,128,0.15)",
        ),
        yaxis=dict(
            tickvals=_y_levels,
            ticktext=_y_labels,
            range=[-0.5, max(_y_levels) + 2.2],
            showgrid=False, zeroline=False,
        ),
        height=max(450, 200 * len(all_sources_data) + 220),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(
            l=max(200, 9 * max((len(s) for s in _y_labels), default=0) + 40),
            r=30, t=80, b=160,
        ),
    )

    st.plotly_chart(fig_overlay, use_container_width=True)
    # Auto-suggest dimensions based on data
    _ov_w_auto = max(1600, 55 * len(_all_masses) + 700) + max(0, 9 * max((len(s) for s in _y_labels), default=0) - 200)
    _ov_h_auto = max(650, 240 * len(all_sources_data) + 320)
    with st.expander("⚙️ PNG export settings", expanded=False):
        _ov_c1, _ov_c2, _ov_c3 = st.columns(3)
        _ov_w = _ov_c1.number_input("Width (px)", 400, 8000, int(_ov_w_auto), 100, key="_lit_ov_w")
        _ov_h = _ov_c2.number_input("Height (px)", 300, 6000, int(_ov_h_auto), 50, key="_lit_ov_h")
        _ov_s = _ov_c3.number_input("Scale (DPI ×)", 1.0, 4.0, 2.0, 0.5, key="_lit_ov_s")
    try:
        _overlay_png = fig_overlay.to_image(format="png", width=int(_ov_w), height=int(_ov_h), scale=float(_ov_s))
        st.download_button(
            "⬇️ Download this plot (PNG)",
            data=_overlay_png,
            file_name=f"mass_position_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            key="_lit_dl_overlay_inline",
        )
    except Exception as _e:
        st.caption(f"⚠️ PNG export unavailable: {_e}")

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SPECTRAL COMPARISON (for matched masses)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 4. Spectral Comparison")
st.caption(
    "For masses found in both Danial's data and literature, overlay Danial's IR spectrum "
    "with their published spectrum. Select a mass, then load/upload the spectra to compare."
)

# Get common masses
if "_lit_comparison" not in st.session_state:
    st.info("Run the mass comparison in Section 3 first.")
    st.stop()

comparison_df = st.session_state["_lit_comparison"]

_show_all = st.checkbox(
    "Show all masses (including Danial-only and lit-only)",
    value=False, key="_lit_show_all_masses",
    help="By default only masses found in BOTH sources are shown, since spectral overlay needs both. "
         "Enable this to also pick Danial-only or literature-only masses (only one side of the overlay will populate).",
)

if _show_all:
    _common = comparison_df.copy()
else:
    _common = comparison_df[comparison_df["status"].isin(["🟢 Common (all)", "🔵 Shared"])]

if len(_common) == 0:
    st.warning(
        "No masses available. Run the comparison in Section 3 first, or enable "
        "'Show all masses' if nothing matched between Danial's data and literature."
    )
    st.stop()

_common_mz = _common["m/z"].values

def _fmt_mz(x):
    _row = _common[_common["m/z"] == x].iloc[0]
    _status = _row.get("status", "")
    _assign = ""
    if "assignment" in _common.columns and pd.notna(_row.get("assignment")):
        _assign = f" — {_row['assignment']}"
    return f"{_status}  m/z {x:.1f}{_assign}"

selected_mz = st.selectbox(
    "Select a mass to view / compare spectra",
    options=_common_mz,
    format_func=_fmt_mz,
    key="_lit_spec_mz",
)

st.markdown(f"### Comparing spectra at m/z {selected_mz:.1f}")

_spec_col1, _spec_col2 = st.columns(2)

# Danial's spectrum
with _spec_col1:
    st.markdown("**Danial's spectrum**")
    _your_spec_source = st.radio(
        "Source",
        ["From session state (4.0 Misc Tab 3)", "Upload CSV"],
        key="_lit_your_spec_src",
        horizontal=True,
    )

    your_wn = None
    your_intensity = None

    if _your_spec_source == "From session state (4.0 Misc Tab 3)":
        _fr = st.session_state.get("_frag_results", {})
        # Find matching channel
        _matched_key = None
        for key, val in _fr.items():
            if abs(val["center"] - selected_mz) <= st.session_state.get("_lit_tolerance", 0.5):
                _matched_key = key
                break
        if _matched_key:
            _ch = _fr[_matched_key]
            your_wn = np.array(_ch["wn"])
            your_intensity = np.array(_ch["ln_depletion"])
            st.success(f"Loaded IR spectrum for {_matched_key}")
        else:
            st.info(
                f"No precomputed IR spectrum for m/z ≈ {selected_mz:.1f}. "
                f"Trying to build one from the loaded dataset…"
            )
            # ── Fallback: compute inline from raw matrices ─────────────────
            _xm = st.session_state.get("x_mass")
            _comp = st.session_state.get("compilation_baseline_corrected_data")
            _wns = st.session_state.get("unique_wavenumbers")
            _pc_wo = st.session_state.get("plot_columnIndex_withoutIR", -2)
            _pc_w = st.session_state.get("plot_columnIndex_withIR", -1)
            if _xm is not None and _comp is not None and _wns is not None:
                _half_w = st.number_input(
                    "Integration half-width around m/z (amu)",
                    value=0.3, min_value=0.05, max_value=2.0, step=0.05,
                    key=f"_lit_inline_halfw_{selected_mz}",
                )
                _sel_mask = (np.asarray(_xm) >= selected_mz - _half_w) & (np.asarray(_xm) <= selected_mz + _half_w)
                _n_in = int(_sel_mask.sum())
                if _n_in == 0:
                    st.warning(f"No m/z bins within ±{_half_w} of {selected_mz:.1f}.")
                else:
                    _wn_sorted = sorted(_wns)
                    _wn_arr = np.array(_wn_sorted, dtype=float)
                    _depl = np.zeros(len(_wn_sorted))
                    for i, wn in enumerate(_wn_sorted):
                        _df_wn = _comp[wn]
                        _wo = _df_wn.iloc[:, _pc_wo].values[_sel_mask].sum()
                        _w = _df_wn.iloc[:, _pc_w].values[_sel_mask].sum()
                        if _wo > 0:
                            _depl[i] = 1.0 - (_w / _wo)
                        else:
                            _depl[i] = np.nan
                    # −ln(1 − depletion) for typical IR spectrum representation
                    with np.errstate(invalid="ignore", divide="ignore"):
                        _ln = -np.log(np.clip(1.0 - _depl, 1e-6, None))
                    your_wn = _wn_arr
                    your_intensity = _ln
                    st.success(
                        f"✅ Computed IR spectrum inline from {_n_in} m/z bins "
                        f"× {len(_wn_sorted)} wavenumbers."
                    )
            else:
                st.warning(
                    "No IR spectrum in `_frag_results` and no raw dataset in session. "
                    "Run **4.0 Misc → Tab 3** or load a `.pkl.gz` in Section 1."
                )
    else:
        _your_spec_file = st.file_uploader(
            "Upload Danial's spectrum CSV (columns: wavenumber, intensity)",
            type=["csv"], key="_lit_your_spec_upload",
        )
        if _your_spec_file:
            _df = pd.read_csv(_your_spec_file)
            if len(_df.columns) >= 2:
                your_wn = _df.iloc[:, 0].values
                your_intensity = _df.iloc[:, 1].values
                st.success(f"Loaded {len(your_wn)} points")

# Literature spectrum
with _spec_col2:
    st.markdown("**Literature spectrum**")
    
    # Multi-file upload (CSV or JDX)
    _lit_spec_files = st.file_uploader(
        "Upload literature spectrum files (CSV or JDX/JCAMP-DX from NIST)",
        type=["csv", "jdx", "dx", "jcamp"], key="_lit_lit_spec_upload", accept_multiple_files=True,
    )
    
    # Store uploaded files in session state for download functionality
    if "_lit_uploaded_spectra" not in st.session_state:
        st.session_state["_lit_uploaded_spectra"] = {}
    
    # List of {name, wn, intensity} dicts for each selected literature spectrum
    lit_spectra = []

    if _lit_spec_files:
        st.write(f"**{len(_lit_spec_files)} file(s) uploaded.**")

        # Multi-select with "all by default"
        _all_names = [f.name for f in _lit_spec_files]
        _select_all = st.checkbox(
            "Compare against ALL uploaded files",
            value=True, key="_lit_compare_all",
        )
        if _select_all:
            selected_file_names = _all_names
            st.caption(f"Using all {len(_all_names)} files for overlay.")
        else:
            selected_file_names = st.multiselect(
                "Select spectra to overlay in comparison:",
                options=_all_names,
                default=_all_names,
                key="_lit_selected_spectra_multi",
            )

        # Load the selected files
        for _lit_spec_file in _lit_spec_files:
            if _lit_spec_file.name in selected_file_names:
                _fname = _lit_spec_file.name
                _ext = os.path.splitext(_fname)[1].lower()
                try:
                    _lit_spec_file.seek(0)
                    _raw = _lit_spec_file.read()

                    if _ext in (".jdx", ".dx", ".jcamp"):
                        _meta, _wn, _it = parse_jdx(_raw)
                        _title = _meta.get("TITLE", _fname)
                        _molform = _meta.get("MOLFORM", "")
                        _cas = _meta.get("CAS REGISTRY NO", "")
                        _display_name = f"{_title}"
                        if _molform:
                            _display_name += f" ({_molform})"
                        if _cas:
                            _display_name += f" [CAS {_cas}]"
                        _yunits = _meta.get("YUNITS", "").upper()
                        if "TRANSMITTANCE" in _yunits:
                            with np.errstate(invalid="ignore", divide="ignore"):
                                _it = -np.log10(np.clip(_it, 1e-6, None))
                            _display_name += " (→ absorbance)"
                        lit_spectra.append({"name": _display_name, "wn": _wn, "intensity": _it})
                        st.session_state["_lit_uploaded_spectra"][_display_name] = {
                            "data": pd.DataFrame({"wavenumber": _wn, "intensity": _it}),
                            "wn": _wn, "intensity": _it,
                        }
                    else:
                        _df = pd.read_csv(io.BytesIO(_raw))
                        if len(_df.columns) >= 2:
                            _wn = _df.iloc[:, 0].values
                            _it = _df.iloc[:, 1].values
                            lit_spectra.append({"name": _fname, "wn": _wn, "intensity": _it})
                            st.session_state["_lit_uploaded_spectra"][_fname] = {
                                "data": _df, "wn": _wn, "intensity": _it,
                            }
                except Exception as _e:
                    st.warning(f"Could not parse {_fname}: {_e}")
                    continue

        if lit_spectra:
            st.success(f"Loaded {len(lit_spectra)} literature spectrum/spectra for overlay.")

        # ── Optional: sum of selected literature spectra ───────────────
        # Reset report each rerun; will be repopulated below if sum is enabled.
        st.session_state.pop("_lit_sum_report", None)
        if len(lit_spectra) >= 2:
            _sum_on = st.checkbox(
                "➕ Plot sum of literature spectra",
                value=False, key="_lit_sum_on",
                help="Adds an extra trace/row that is the sum of the chosen spectra "
                     "(interpolated onto a common wavenumber grid).",
            )
            if _sum_on:
                _names_loaded = [s["name"] for s in lit_spectra]
                _sum_picks = st.multiselect(
                    "Select files to sum:",
                    options=_names_loaded,
                    default=_names_loaded,
                    key="_lit_sum_picks",
                )
                _normalize_before_sum = st.checkbox(
                    "Normalize each spectrum to [0, 1] before summing",
                    value=True, key="_lit_sum_normalize_first",
                )
                _sum_display_mode = st.radio(
                    "Display mode",
                    ["Sum + individuals", "Only the sum"],
                    horizontal=True, key="_lit_sum_display_mode",
                )
                if len(_sum_picks) >= 2:
                    _picked_specs = [s for s in lit_spectra if s["name"] in _sum_picks]
                    # Build a common wavenumber grid over the overlap region
                    _wn_min = max(float(np.nanmin(s["wn"])) for s in _picked_specs)
                    _wn_max = min(float(np.nanmax(s["wn"])) for s in _picked_specs)
                    if _wn_max > _wn_min:
                        # Use the densest grid among the picks for resolution
                        _n_pts = max(len(s["wn"]) for s in _picked_specs)
                        _common_wn = np.linspace(_wn_min, _wn_max, _n_pts)
                        _sum_intensity = np.zeros_like(_common_wn)
                        for s in _picked_specs:
                            _wn_arr = np.asarray(s["wn"], dtype=float)
                            _it_arr = np.asarray(s["intensity"], dtype=float)
                            # ensure sorted for np.interp
                            _order = np.argsort(_wn_arr)
                            _wn_s = _wn_arr[_order]
                            _it_s = _it_arr[_order]
                            if _normalize_before_sum:
                                _rng = np.nanmax(_it_s) - np.nanmin(_it_s)
                                if _rng > 0:
                                    _it_s = (_it_s - np.nanmin(_it_s)) / _rng
                            _sum_intensity += np.interp(_common_wn, _wn_s, _it_s)
                        _sum_row = {
                            "name": f"SUM ({len(_picked_specs)} spectra)",
                            "wn": _common_wn,
                            "intensity": _sum_intensity,
                        }
                        if _sum_display_mode == "Only the sum":
                            # drop the individual spectra that went into the sum
                            lit_spectra = [
                                s for s in lit_spectra if s["name"] not in _sum_picks
                            ]
                        lit_spectra.append(_sum_row)
                        # Save metadata for the report shown below the plot
                        st.session_state["_lit_sum_report"] = {
                            "files": list(_sum_picks),
                            "wn_min": float(_wn_min),
                            "wn_max": float(_wn_max),
                            "n_pts": int(_n_pts),
                            "normalized": bool(_normalize_before_sum),
                            "display_mode": _sum_display_mode,
                        }
                        st.info(
                            f"Added sum trace over wavenumber overlap "
                            f"[{_wn_min:.1f}, {_wn_max:.1f}] cm⁻¹."
                        )
                    else:
                        st.warning(
                            "The selected spectra have no overlapping wavenumber range; "
                            "cannot compute sum."
                        )
                elif len(_sum_picks) < 2:
                    st.caption("Select at least 2 files to sum.")


# Plot overlay
if your_wn is not None or len(lit_spectra) > 0:
    _opt_col1, _opt_col2, _opt_col3, _opt_col4 = st.columns([1, 1, 1, 1])
    with _opt_col1:
        _norm = st.checkbox("Normalize to [0, 1]", value=True, key="_lit_spec_norm")
    with _opt_col2:
        _sg_on = st.checkbox("Savitzky–Golay smoothing", value=False, key="_lit_spec_sg")
    with _opt_col3:
        _sg_win = st.number_input(
            "SG window length (odd)", value=11, min_value=3, max_value=201, step=2,
            key="_lit_spec_sg_win", disabled=not _sg_on,
        )
    with _opt_col4:
        _sg_poly = st.number_input(
            "SG polyorder", value=3, min_value=1, max_value=7, step=1,
            key="_lit_spec_sg_poly", disabled=not _sg_on,
        )
    _show_raw = st.checkbox(
        "Show raw (un-smoothed) as faint line", value=True,
        key="_lit_spec_show_raw", disabled=not _sg_on,
    )
    _rc1, _rc2, _rc3 = st.columns([1, 1, 1])
    with _rc1:
        _ridge_on = st.checkbox(
            "Ridge plot (stack vertically)", value=True, key="_lit_spec_ridge",
            help="Offset each spectrum vertically so they don't overlap. Forces normalization.",
        )
    with _rc2:
        _ridge_gap = st.slider(
            "Ridge spacing", min_value=0.2, max_value=2.0, value=1.0, step=0.1,
            key="_lit_spec_ridge_gap", disabled=not _ridge_on,
        )
    with _rc3:
        _ridge_fill = st.checkbox(
            "Fill under curves", value=True,
            key="_lit_spec_ridge_fill", disabled=not _ridge_on,
        )
    if _ridge_on:
        _norm = True  # ridge requires normalization to look sensible

    def _apply_sg(y):
        if not _sg_on:
            return y
        _w = int(_sg_win)
        if _w % 2 == 0:
            _w += 1
        _p = int(_sg_poly)
        if _w <= _p or _w > len(y):
            st.warning(f"SG window ({_w}) must be > polyorder ({_p}) and ≤ {len(y)} points. Skipping smoothing.")
            return y
        _y_in = y.copy()
        _finite = np.isfinite(_y_in)
        if _finite.sum() < _w:
            return y
        _y_in[~_finite] = np.interp(
            np.flatnonzero(~_finite), np.flatnonzero(_finite), _y_in[_finite]
        ) if _finite.any() else _y_in
        return savgol_filter(_y_in, window_length=_w, polyorder=_p)

    def _normalize(y):
        if not _norm:
            return y
        _finite = y[np.isfinite(y)]
        if len(_finite) == 0 or np.ptp(_finite) == 0:
            return y
        return (y - np.nanmin(y)) / np.ptp(_finite)

    # Build a unified list of rows: [{name, color, wn, y, raw_y (optional)}]
    _rows = []
    _lit_palette = [
        "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22",
    ]
    if your_wn is not None:
        _y_raw_d = np.asarray(your_intensity, dtype=float).copy()
        _y_sm_d = _apply_sg(_y_raw_d)
        _rows.append({
            "name": "Danial" + (" (SG)" if _sg_on else ""),
            "color": "#1f77b4",
            "wn": your_wn,
            "y": _normalize(_y_sm_d),
            "raw_y": _normalize(_y_raw_d) if (_sg_on and _show_raw) else None,
        })
    # NOTE: smoothing applied ONLY to Danial's data.
    for _idx, _spec in enumerate(lit_spectra):
        _name_short = _spec["name"].rsplit(".", 1)[0]
        _rows.append({
            "name": _name_short,
            "color": _lit_palette[_idx % len(_lit_palette)],
            "wn": _spec["wn"],
            "y": _normalize(np.asarray(_spec["intensity"], dtype=float)),
            "raw_y": None,
        })

    # ── Per-spectrum wavenumber shift controls ─────────────────────────
    with st.expander("↔️ Shift spectra (cm⁻¹)", expanded=False):
        st.caption(
            "Shift each spectrum horizontally to align peaks. "
            "Positive = move to higher wavenumbers."
        )
        _shifts = {}
        _ncols = min(3, max(1, len(_rows)))
        _cols = st.columns(_ncols)
        for _i, _r in enumerate(_rows):
            with _cols[_i % _ncols]:
                _shifts[_r["name"]] = st.number_input(
                    f"{_r['name']}",
                    value=0.0, step=1.0, format="%.2f",
                    key=f"_lit_shift_{_i}_{_r['name']}",
                )
        if st.button("Reset all shifts to 0", key="_lit_reset_shifts"):
            for _i, _r in enumerate(_rows):
                st.session_state[f"_lit_shift_{_i}_{_r['name']}"] = 0.0
            st.rerun()

    # Apply shifts (and store the original wn for the report if needed)
    for _r in _rows:
        _sh = float(_shifts.get(_r["name"], 0.0))
        if _sh != 0.0:
            _r["wn"] = np.asarray(_r["wn"], dtype=float) + _sh
            _r["name"] = f"{_r['name']} (Δ{_sh:+.1f})"

    fig_spec = go.Figure()

    def _hex_to_rgba(hex_color, alpha=0.25):
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    if _ridge_on and len(_rows) > 0:
        # Stack bottom-to-top: first row (Danial) at top so it's visually prominent.
        _ordered = list(reversed(_rows))
        _tickvals = []
        _ticktext = []
        for _i, _r in enumerate(_ordered):
            _offset = _i * float(_ridge_gap)
            _y_off = np.asarray(_r["y"], dtype=float) + _offset
            if _ridge_fill:
                # baseline trace for fill
                fig_spec.add_trace(go.Scatter(
                    x=_r["wn"], y=np.full_like(_r["wn"], _offset, dtype=float),
                    mode="lines", line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                _fill_rgba = _hex_to_rgba(_r["color"], 0.25)
                fig_spec.add_trace(go.Scatter(
                    x=_r["wn"], y=_y_off, mode="lines",
                    name=_r["name"],
                    line=dict(color=_r["color"], width=2),
                    fill="tonexty",
                    fillcolor=_fill_rgba,
                ))
            else:
                fig_spec.add_trace(go.Scatter(
                    x=_r["wn"], y=_y_off, mode="lines",
                    name=_r["name"], line=dict(color=_r["color"], width=2),
                ))
            _tickvals.append(_offset)
            _ticktext.append(_r["name"])
    else:
        # Classic overlay
        for _r in _rows:
            if _r["raw_y"] is not None:
                fig_spec.add_trace(go.Scatter(
                    x=_r["wn"], y=_r["raw_y"], mode="lines",
                    name=f"{_r['name']} (raw)",
                    line=dict(color=_r["color"], width=1, dash="dot"),
                    opacity=0.4,
                ))
            fig_spec.add_trace(go.Scatter(
                x=_r["wn"], y=_r["y"], mode="lines",
                name=_r["name"], line=dict(color=_r["color"], width=2),
            ))

    # Title: include count + names if few, else just count
    _lit_count = len(lit_spectra)
    if _lit_count == 0:
        _title = f"Spectral comparison — m/z {selected_mz:.1f}"
    elif _lit_count == 1:
        _title = (
            f"Spectral comparison — m/z {selected_mz:.1f} "
            f"(Danial vs {lit_spectra[0]['name'].rsplit('.', 1)[0]})"
        )
    else:
        _title = (
            f"Spectral comparison — m/z {selected_mz:.1f} "
            f"(Danial vs {_lit_count} literature spectra)"
        )

    if _ridge_on and len(_rows) > 0:
        fig_spec.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis=dict(
                tickvals=_tickvals,
                ticktext=_ticktext,
                showgrid=False,
                zeroline=False,
            ),
            title=_title,
            height=max(300, 110 * len(_rows) + 120),
            showlegend=False,
            margin=dict(
                l=max(180, 9 * max((len(t) for t in _ticktext), default=0) + 40),
                r=20, t=60, b=50,
            ),
        )
    else:
        fig_spec.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity (norm.)" if _norm else "Intensity",
            title=_title,
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
    
    st.plotly_chart(fig_spec, use_container_width=True)

    # Sum report (only shown if a sum was actually computed)
    _sum_rep = st.session_state.get("_lit_sum_report")
    if _sum_rep:
        _files_md = "\n".join(f"  - `{f}`" for f in _sum_rep["files"])
        st.markdown(
            f"""**🧮 Sum trace report**

- **Display mode:** {_sum_rep['display_mode']}
- **Files included ({len(_sum_rep['files'])}):**
{_files_md}
- **Common wavenumber range:** {_sum_rep['wn_min']:.1f} – {_sum_rep['wn_max']:.1f} cm⁻¹
- **Grid points:** {_sum_rep['n_pts']}
- **Normalized to [0, 1] before summing:** {'yes' if _sum_rep['normalized'] else 'no'}
"""
        )

    try:
        _sp_h_auto = max(600, 160 * len(_rows) + 180) if _ridge_on else 800
        _sp_w_auto = 1600
        with st.expander("⚙️ PNG export settings", expanded=False):
            _sp_c1, _sp_c2, _sp_c3 = st.columns(3)
            _sp_w = _sp_c1.number_input("Width (px)", 400, 8000, _sp_w_auto, 100, key="_lit_sp_w")
            _sp_h = _sp_c2.number_input("Height (px)", 300, 6000, int(_sp_h_auto), 50, key="_lit_sp_h")
            _sp_s = _sp_c3.number_input("Scale (DPI ×)", 1.0, 4.0, 2.0, 0.5, key="_lit_sp_s")
        _spec_png = fig_spec.to_image(format="png", width=int(_sp_w), height=int(_sp_h), scale=float(_sp_s))
        st.download_button(
            "⬇️ Download this plot (PNG)",
            data=_spec_png,
            file_name=f"spectral_comparison_mz_{selected_mz:.1f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            key="_lit_dl_spec_inline",
        )
    except Exception as _e:
        st.caption(f"⚠️ PNG export unavailable: {_e}")

