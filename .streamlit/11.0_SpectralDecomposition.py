"""
11.0 Spectral Decomposition — NNLS Mixture Analysis
====================================================
Interactive Streamlit page for decomposing an experimental IR spectrum
into a linear combination of DFT reference spectra using non-negative
least squares (NNLS) with polynomial baseline.

Integrates with the Mass Identity Workbench (10.0) via session state
or accepts independent data uploads.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import io
import os
import re
import glob
import json
import configparser
from pathlib import Path
from datetime import datetime

from packages.SpectralDecomposition import (
    compute_diagnostics,
    pearson_derivative_scores,
    cosine_scores,
    fit_nnls,
    forward_stepwise,
    exhaustive_search,
    select_model,
    run_bootstrap,
    bootstrap_summary,
    rank_stability_matrix,
    peak_residual_analysis,
    scaling_sensitivity,
    run_full_pipeline,
)
from packages.DFT_Parsers import parse_dft_file, broaden_spectrum_felix

st.set_page_config(page_title="Spectral Decomposition", layout="wide")
st.title("🧬 Spectral Decomposition — NNLS Mixture Analysis")
st.caption(
    "Decompose an experimental IR spectrum into a weighted sum of DFT reference "
    "spectra. Uses non-negative least squares (NNLS) with polynomial baseline, "
    "forward stepwise selection, BIC/CV model selection, and block bootstrap "
    "confidence intervals."
)

st.info(
    "**Important caveat:** Spectral weights reflect contributions weighted by "
    "the product n_i × σ_UV(i)(ν_UV) × φ_ion(i). They are **not** isomer "
    "populations. Absence claims cannot be made for UV-silent candidates.",
    icon="⚠️",
)

# ════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA INPUT
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 1. Data Input")

_src_tab1, _src_tab2 = st.tabs(["📡 From Mass Identity (session)", "📂 Upload Files"])

exp_wn, exp_norm = None, None
molecules, dft_matrix = [], None

with _src_tab1:
    _mid_wn = st.session_state.get("_mid_your_wn")
    _mid_int = st.session_state.get("_mid_your_intensity")
    _mid_mz = st.session_state.get("_mid_selected_mz")
    if _mid_wn is not None:
        st.success(f"✅ Experimental spectrum available from Mass Identity "
                   f"(m/z {_mid_mz:.1f}, {len(_mid_wn)} points)")
        if st.checkbox("Use this spectrum", value=True, key="_sd_use_mid"):
            exp_wn = np.asarray(_mid_wn, dtype=float)
            exp_norm_raw = np.asarray(_mid_int, dtype=float)

            # Independent smoothing control
            _smooth_opt = st.radio(
                "Pre-processing", ["Raw (no smoothing)", "Savitzky-Golay smoothing"],
                horizontal=True, key="_sd_smooth_mode",
                help="Smoothing is applied here independently of Mass Identity settings.")
            if "Savitzky" in _smooth_opt:
                from scipy.signal import savgol_filter as _savgol
                _sg_c1, _sg_c2 = st.columns(2)
                with _sg_c1:
                    _sg_window = st.slider("Window length (pts)", 5, 51, 11, step=2,
                                           key="_sd_sg_window",
                                           help="Must be odd. Larger = smoother.")
                with _sg_c2:
                    _sg_poly = st.slider("Polynomial order", 1, 5, 3,
                                         key="_sd_sg_poly",
                                         help="Higher preserves sharper features.")
                exp_norm_raw = _savgol(exp_norm_raw, _sg_window, _sg_poly)
                st.caption(f"Applied Savitzky-Golay (window={_sg_window}, poly={_sg_poly})")

            _emax = np.nanmax(np.abs(exp_norm_raw))
            exp_norm = exp_norm_raw / _emax if _emax > 0 else exp_norm_raw
    else:
        st.warning("No experimental spectrum in session. Use the Mass Identity "
                   "Workbench (10.0) first, or upload files in the next tab.")

with _src_tab2:
    _exp_file = st.file_uploader("Upload experimental spectrum CSV",
                                  type=["csv", "txt", "dat"],
                                  key="_sd_exp_upload",
                                  help="Two columns: wavenumber, intensity")
    if _exp_file is not None:
        _edf = pd.read_csv(_exp_file, sep=None, engine="python")
        if len(_edf.columns) >= 2:
            exp_wn = _edf.iloc[:, 0].values.astype(float)
            _raw = _edf.iloc[:, 1].values.astype(float)
            _emax = np.nanmax(np.abs(_raw))
            exp_norm = _raw / _emax if _emax > 0 else _raw
            st.success(f"✅ Loaded experimental spectrum: {len(exp_wn)} points, "
                       f"range [{exp_wn.min():.0f}, {exp_wn.max():.0f}] cm⁻¹")

# DFT library upload
st.markdown("### DFT Reference Library")

# ── DFT processing parameters ──
st.markdown("#### DFT Processing Parameters")
_dft_proc_c1, _dft_proc_c2, _dft_proc_c3 = st.columns(3)
with _dft_proc_c1:
    _dft_freq_scale = st.number_input(
        "Frequency scaling factor", value=1.00, min_value=0.80,
        max_value=1.20, step=0.01, format="%.3f", key="_sd_freq_scale",
        help="Applied to raw stick spectra (.out/.log). Set to 1.0 if "
             "frequencies are already scaled.")
with _dft_proc_c2:
    _dft_bw_frac = st.number_input(
        "FELIX bandwidth fraction", value=0.0053, min_value=0.001,
        max_value=0.05, step=0.001, format="%.4f", key="_sd_bw_frac",
        help="FWHM = bw_frac × ν. FELIX typical: 0.003–0.01. "
             "0.0053 = 0.53% of center frequency.")
with _dft_proc_c3:
    _csv_is_convolved = st.selectbox(
        "CSV files are…", ["Already convolved (use as-is)", "Raw sticks (broaden)"],
        key="_sd_csv_mode",
        help="If CSVs contain stick spectra (few lines), select 'Raw sticks'. "
             "If they are pre-broadened continuous spectra, select 'Already convolved'.")

st.caption(
    f"**Raw .out/.log:** frequencies × {_dft_freq_scale:.3f}, then broadened with "
    f"FWHM = {_dft_bw_frac*100:.2f}% × ν  \n"
    f"**CSV:** {'interpolated only (no re-broadening)' if 'Already' in _csv_is_convolved else 'treated as sticks → scaled & broadened'}"
)

_dft_mode = st.radio("DFT source", ["Upload files", "Scan directory"],
                     horizontal=True, key="_sd_dft_mode")

if _dft_mode == "Upload files":
    _dft_files = st.file_uploader(
        "Upload DFT spectra (Gaussian .out/.log, CSV, or broadened CSV)",
        type=["out", "log", "csv", "txt", "dat"],
        accept_multiple_files=True, key="_sd_dft_upload",
    )
    if _dft_files and exp_wn is not None:
        _mol_list, _dft_list = [], []
        _seen_cids = set()
        _skipped = []
        for _f in sorted(_dft_files, key=lambda x: x.name):
            _fname = _f.name
            _fname_lower = _fname.lower()
            _ext = os.path.splitext(_fname)[1].lower()

            # Auto-skip stick files when in "Already convolved" mode
            if "Already" in _csv_is_convolved and "_sticks" in _fname_lower:
                _skipped.append((_fname, "stick file (skipped in convolved mode)"))
                continue

            _raw = _f.read()

            # Extract CID and name from filename pattern CID_<number>_<name>...
            _cid_match = re.match(r"CID_(\d+)_(.+?)(?:_scaled|_unscaled|$)",
                                  os.path.splitext(_fname)[0])
            if _cid_match:
                _file_cid = _cid_match.group(1)
                _file_name = _cid_match.group(2).replace("_", " ")
            else:
                _file_cid = str(len(_mol_list))
                _file_name = os.path.splitext(_fname)[0]

            # CID deduplication — skip if already loaded
            if _file_cid in _seen_cids:
                _skipped.append((_fname, f"duplicate CID {_file_cid}"))
                continue
            _seen_cids.add(_file_cid)

            if _ext in (".out", ".log"):
                _text = _raw.decode("utf-8", errors="replace") if isinstance(_raw, bytes) else _raw
                freqs, intens, meta = parse_dft_file(_text, _fname)
                if freqs is not None and len(freqs) > 0:
                    # Apply frequency scaling
                    freqs_scaled = np.asarray(freqs, dtype=float) * _dft_freq_scale
                    wn_broad, int_broad = broaden_spectrum_felix(
                        freqs_scaled, intens, bw_frac=_dft_bw_frac)
                    _dft_interp = np.interp(exp_wn, wn_broad, int_broad, left=0.0, right=0.0)
                    _name = _file_name
                    if "method" in meta:
                        _name += f" ({meta['method']})"
                    _mol_list.append({"cid": _file_cid, "name": _name, "file": _fname})
                    _dft_list.append(_dft_interp)

            elif _ext in (".csv", ".txt", ".dat"):
                _df = pd.read_csv(io.BytesIO(_raw) if isinstance(_raw, bytes) else io.StringIO(_raw),
                                  sep=None, engine="python")
                if len(_df.columns) >= 2:
                    _csv_wn = _df.iloc[:, 0].values.astype(float)
                    _csv_int = _df.iloc[:, 1].values.astype(float)
                    # Auto-detect transmittance
                    _col2 = str(_df.columns[1]).lower()
                    if "transmittance" in _col2 or "trans" in _col2 or "%t" in _col2:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            _csv_int = -np.log10(np.clip(_csv_int / 100.0, 1e-6, None))

                    if "Raw sticks" in _csv_is_convolved:
                        _csv_wn_scaled = _csv_wn * _dft_freq_scale
                        wn_broad, int_broad = broaden_spectrum_felix(
                            _csv_wn_scaled, _csv_int, bw_frac=_dft_bw_frac)
                        _dft_interp = np.interp(exp_wn, wn_broad, int_broad, left=0.0, right=0.0)
                    else:
                        _csv_wn_scaled = _csv_wn * _dft_freq_scale if abs(_dft_freq_scale - 1.0) > 1e-4 else _csv_wn
                        _dft_interp = np.interp(exp_wn, _csv_wn_scaled, _csv_int, left=0.0, right=0.0)

                    _mol_list.append({"cid": _file_cid, "name": _file_name, "file": _fname})
                    _dft_list.append(_dft_interp)

        if _dft_list:
            molecules = _mol_list
            dft_matrix = np.array(_dft_list)

        # Summary of loaded files
        if _mol_list or _skipped:
            with st.expander(f"� Loaded {len(_mol_list)} spectra, skipped {len(_skipped)}", expanded=False):
                if _mol_list:
                    _load_df = pd.DataFrame([
                        {"CID": m["cid"], "Name": m["name"], "File": m["file"]}
                        for m in _mol_list
                    ])
                    st.dataframe(_load_df, hide_index=True, use_container_width=True)
                if _skipped:
                    st.markdown("**Skipped files:**")
                    for _sf, _reason in _skipped:
                        st.caption(f"⏭️ `{_sf}` — {_reason}")

elif _dft_mode == "Scan directory":
    _dft_dir = st.text_input("Path to DFT spectra directory",
                              key="_sd_dft_dir",
                              help="Directory containing scaled DFT CSV files "
                                   "(e.g. CID_*_scaled_0.95.csv)")
    _scale_pattern = st.text_input("Scaling factor in filenames", value="0.95",
                                    key="_sd_scale_pat")
    if _dft_dir and os.path.isdir(_dft_dir) and exp_wn is not None:
        pattern = f"CID_*_scaled_{_scale_pattern}.csv"
        conv_files = sorted(glob.glob(os.path.join(_dft_dir, pattern)))
        if not conv_files:
            # Also try generic CSVs
            conv_files = sorted(glob.glob(os.path.join(_dft_dir, "*.csv")))

        _mol_list, _dft_list = [], []
        for cf in conv_files:
            bn = os.path.basename(cf)
            m = re.match(rf"CID_(\d+)_(.+)_scaled_{re.escape(_scale_pattern)}\.csv", bn)
            if m:
                cid, name = m.group(1), m.group(2).replace("_", " ")
            else:
                cid = str(len(_mol_list))
                name = os.path.splitext(bn)[0]
            try:
                df = pd.read_csv(cf, comment="#", header=None,
                                 names=["Wavenumber", "Fundamentals", "Overtones",
                                        "Combinations", "Total"])
                dft_wn = df["Wavenumber"].values
                dft_tot = df["Total"].values
            except Exception:
                df = pd.read_csv(cf, sep=None, engine="python")
                if len(df.columns) < 2:
                    continue
                dft_wn = df.iloc[:, 0].values.astype(float)
                dft_tot = df.iloc[:, 1].values.astype(float)

            # Apply frequency scaling if not 1.0
            if abs(_dft_freq_scale - 1.0) > 1e-4:
                dft_wn = dft_wn * _dft_freq_scale
            dft_interp = np.interp(exp_wn, dft_wn, dft_tot, left=0.0, right=0.0)
            _mol_list.append({"cid": cid, "name": name, "file": cf})
            _dft_list.append(dft_interp)

        if _dft_list:
            molecules = _mol_list
            dft_matrix = np.array(_dft_list)
            st.success(f"✅ Loaded {len(molecules)} DFT spectra from `{_dft_dir}`")
        else:
            st.warning(f"No matching files found in `{_dft_dir}`")

# Structure images (optional)
_img_dir = st.text_input("Structure images directory (optional)",
                          key="_sd_img_dir",
                          help="Image files (PNG/JPG) named by CID or molecule name. "
                               "Tries: CID_<cid>.png, <name>.png, <filename_stem>.png")
if _img_dir and os.path.isdir(_img_dir):
    _found_imgs = [f for f in os.listdir(_img_dir)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    st.caption(f"📂 Found {len(_found_imgs)} image(s) in directory")

# Show summary
if exp_wn is not None and dft_matrix is not None:
    st.markdown("### Data Summary")
    _sc1, _sc2, _sc3 = st.columns(3)
    _sc1.metric("Experimental points", len(exp_wn))
    _sc2.metric("DFT candidates", len(molecules))
    _sc3.metric("Wavenumber range",
                f"{exp_wn.min():.0f}–{exp_wn.max():.0f} cm⁻¹")

    # Preview plot
    with st.expander("📊 Preview: Experimental + DFT library", expanded=False):
        _fig_prev = go.Figure()
        _fig_prev.add_trace(go.Scatter(x=exp_wn, y=exp_norm, mode="lines",
                                        name="Experimental", line=dict(color="black", width=2)))
        _palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8"]
        for i, mol in enumerate(molecules):
            _dft_norm = dft_matrix[i] / np.nanmax(np.abs(dft_matrix[i])) if np.nanmax(np.abs(dft_matrix[i])) > 0 else dft_matrix[i]
            _fig_prev.add_trace(go.Scatter(
                x=exp_wn, y=_dft_norm, mode="lines", opacity=0.5,
                name=f"{mol['name'][:30]}", line=dict(width=1, color=_palette[i % len(_palette)]),
            ))
        _fig_prev.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Normalized Intensity",
            height=400, showlegend=True,
            legend=dict(font=dict(size=8), orientation="v"),
        )
        st.plotly_chart(_fig_prev, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════════
if exp_wn is not None and dft_matrix is not None:
    st.markdown("---")
    st.markdown("## 2. Pipeline Configuration")

    _cfg_c1, _cfg_c2, _cfg_c3, _cfg_c4 = st.columns(4)
    with _cfg_c1:
        _fwhm = st.number_input("FWHM (cm⁻¹)", value=10.0, min_value=1.0,
                                 max_value=50.0, step=1.0, key="_sd_fwhm")
        _poly_order = st.selectbox("Baseline polynomial order", [0, 1, 2],
                                    index=1, key="_sd_poly")
    with _cfg_c2:
        _scale_factor = st.number_input("DFT scaling factor", value=0.95,
                                         min_value=0.80, max_value=1.10,
                                         step=0.01, format="%.2f",
                                         key="_sd_scale")
        _max_k_fwd = st.number_input("Max k (forward stepwise)", value=12,
                                      min_value=2, max_value=30, key="_sd_maxk")
    with _cfg_c3:
        _max_k_exh = st.number_input("Max k (exhaustive)", value=5,
                                      min_value=2, max_value=8, key="_sd_maxk_exh")
        _n_cv = st.number_input("CV blocks", value=10, min_value=3,
                                 max_value=20, key="_sd_ncv")
    with _cfg_c4:
        _n_boot = st.number_input("Bootstrap iterations", value=1000,
                                   min_value=100, max_value=5000, step=100,
                                   key="_sd_nboot")
        _run_sensitivity = st.checkbox("Run scaling sensitivity", value=False,
                                        key="_sd_run_sens")

    if _run_sensitivity:
        _sens_factors_str = st.text_input(
            "Scaling factors to test (comma-separated)",
            value="0.94, 0.95, 0.96, 0.97", key="_sd_sens_factors")
        _sens_factors = [float(x.strip()) for x in _sens_factors_str.split(",")
                         if x.strip()]
    else:
        _sens_factors = []

    # ════════════════════════════════════════════════════════════════════
    # SECTION 3 — RUN PIPELINE
    # ════════════════════════════════════════════════════════════════════
    st.markdown("---")

    if st.button("🚀 Run Decomposition Pipeline", type="primary",
                 key="_sd_run", use_container_width=True):

        _progress_bar = st.progress(0, text="Starting...")
        _log_container = st.container()
        _log_expander = _log_container.expander("📋 Pipeline Log (live)", expanded=True)
        _log_lines = []
        _step_map = {
            "diagnostics": 0.05, "ranking": 0.10, "full_nnls": 0.15,
            "stepwise": 0.40, "model_selection": 0.50,
            "exhaustive": 0.55, "final_fit": 0.60,
            "bootstrap": 0.85, "peak_residuals": 0.90,
            "sensitivity": 0.95, "done": 1.0,
        }
        _log_placeholder = _log_expander.empty()

        def _progress(step_name, detail):
            pct = _step_map.get(step_name, 0.5)
            _progress_bar.progress(pct, text=f"{step_name}: {detail}")
            _log_lines.append(f"`[{pct*100:5.1f}%]` **{step_name}** — {detail}")
            # Show last 30 lines to keep it readable
            _log_placeholder.markdown("\n\n".join(_log_lines[-30:]))

        import time as _time
        _t0 = _time.time()
        _log_lines.append(f"`[  0.0%]` **start** — Pipeline started "
                          f"({len(molecules)} candidates, {len(exp_wn)} pts)")
        _log_placeholder.markdown("\n\n".join(_log_lines))

        results = run_full_pipeline(
            exp_wn, exp_norm, dft_matrix, molecules,
            fwhm_cm=_fwhm,
            scaling_factor=_scale_factor,
            poly_order=_poly_order,
            max_k_forward=_max_k_fwd,
            max_k_exhaustive=_max_k_exh,
            n_cv_blocks=_n_cv,
            n_bootstrap=_n_boot,
            scaling_test_factors=_sens_factors if _run_sensitivity else None,
            progress_callback=_progress,
        )

        _elapsed = _time.time() - _t0
        _log_lines.append(f"`[100.0%]` **done** — Completed in {_elapsed:.1f}s "
                          f"(R² = {results['final_fit']['r2']:.4f}, "
                          f"k = {results['best_k']})")
        _log_placeholder.markdown("\n\n".join(_log_lines[-30:]))

        _progress_bar.progress(1.0, text=f"Done! ({_elapsed:.1f}s)")
        st.session_state["_sd_results"] = results
        st.session_state["_sd_exp_wn"] = exp_wn
        st.session_state["_sd_exp_norm"] = exp_norm
        st.session_state["_sd_molecules"] = molecules
        st.session_state["_sd_dft_matrix"] = dft_matrix
        st.rerun()

# ════════════════════════════════════════════════════════════════════════
# SECTION 4 — RESULTS DISPLAY
# ════════════════════════════════════════════════════════════════════════
if "_sd_results" in st.session_state:
    results = st.session_state["_sd_results"]
    exp_wn = st.session_state["_sd_exp_wn"]
    exp_norm = st.session_state["_sd_exp_norm"]
    molecules = st.session_state["_sd_molecules"]
    dft_matrix = st.session_state["_sd_dft_matrix"]

    _palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8"]

    st.markdown("---")
    st.markdown("## Results")

    # ── Tab layout for results ──
    (tab_diag, tab_rank, tab_nnls, tab_model, tab_fit,
     tab_boot, tab_peaks, tab_sens, tab_export) = st.tabs([
        "0. Diagnostics", "1. Ranking", "2. Full NNLS",
        "3. Model Selection", "4. Best Fit", "5. Bootstrap",
        "6. Peak Residuals", "7. Sensitivity", "8. Export",
    ])

    # ── TAB 0: DIAGNOSTICS ─────────────────────────────────────────
    with tab_diag:
        diag = results["diagnostics"]
        _d1, _d2, _d3, _d4 = st.columns(4)
        _d1.metric("cond(A)", f"{diag['cond_number']:.2e}")
        _d2.metric("Max cosine pair", f"{diag['max_cosine']:.4f}")
        _d3.metric("Pairs cos > 0.90", diag["n_pairs_above_90"])
        _d4.metric("Clusters", f"{diag['n_clusters']} / {len(molecules)}")

        if diag["cond_number"] > 1e4:
            st.warning("⚠️ High condition number — weights may be unstable "
                       "between similar isomers.")

        i_mc, j_mc = diag["max_pair"]
        st.caption(f"Most similar pair: **{molecules[i_mc]['name']}** vs "
                   f"**{molecules[j_mc]['name']}** (cos = {diag['max_cosine']:.4f})")

        # Gram matrix heatmap
        st.markdown("#### Gram Matrix (DFT–DFT Cosine Similarity)")
        _gram = diag["gram_matrix"]
        _labels = [f"{m['cid']}: {m['name'][:20]}" for m in molecules]
        _fig_gram = go.Figure(data=go.Heatmap(
            z=_gram, x=_labels, y=_labels,
            colorscale="RdYlBu_r", zmin=0, zmax=1,
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>cos = %{z:.4f}<extra></extra>",
        ))
        _fig_gram.update_layout(height=max(400, 15 * len(molecules)),
                                 xaxis=dict(tickfont=dict(size=7), tickangle=90),
                                 yaxis=dict(tickfont=dict(size=7)))
        st.plotly_chart(_fig_gram, use_container_width=True)

        # Cluster table
        _cl = diag["cluster_labels"]
        _cluster_data = []
        for cl_id in sorted(set(_cl)):
            members = [i for i, c in enumerate(_cl) if c == cl_id]
            if len(members) > 1:
                for m in members:
                    _cluster_data.append({
                        "Cluster": cl_id,
                        "CID": molecules[m]["cid"],
                        "Name": molecules[m]["name"],
                    })
        if _cluster_data:
            st.markdown("#### Collinear Clusters (cos > 0.90)")
            st.dataframe(pd.DataFrame(_cluster_data), hide_index=True)

    # ── TAB 1: RANKING ─────────────────────────────────────────────
    with tab_rank:
        pearson_sc = results["pearson_scores"]
        cosine_sc = results["cosine_scores"]
        pearson_rank = np.argsort(pearson_sc)[::-1]
        cosine_rank = np.argsort(cosine_sc)[::-1]

        _rc1, _rc2 = st.columns(2)
        with _rc1:
            st.markdown("#### Pearson (smoothed 1st derivative)")
            _top_n = min(15, len(molecules))
            _tp = pearson_rank[:_top_n]
            _fig_p = go.Figure(go.Bar(
                y=[f"{molecules[i]['cid']}: {molecules[i]['name'][:25]}" for i in _tp],
                x=pearson_sc[_tp], orientation="h",
                marker_color="teal",
                hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
            ))
            _fig_p.update_layout(height=400, yaxis=dict(autorange="reversed"),
                                  xaxis_title="Pearson correlation",
                                  margin=dict(l=10))
            st.plotly_chart(_fig_p, use_container_width=True)

        with _rc2:
            st.markdown("#### Cosine Similarity")
            _tc = cosine_rank[:_top_n]
            _fig_c = go.Figure(go.Bar(
                y=[f"{molecules[i]['cid']}: {molecules[i]['name'][:25]}" for i in _tc],
                x=cosine_sc[_tc], orientation="h",
                marker_color="steelblue",
                hovertemplate="%{y}<br>Score: %{x:.4f}<extra></extra>",
            ))
            _fig_c.update_layout(height=400, yaxis=dict(autorange="reversed"),
                                  xaxis_title="Cosine similarity",
                                  margin=dict(l=10))
            st.plotly_chart(_fig_c, use_container_width=True)

        # Combined ranking table
        _rank_rows = []
        for i in range(len(molecules)):
            _rank_rows.append({
                "CID": molecules[i]["cid"],
                "Name": molecules[i]["name"],
                "Pearson (∂)": f"{pearson_sc[i]:.4f}",
                "Cosine": f"{cosine_sc[i]:.4f}",
                "Pearson Rank": int(np.where(pearson_rank == i)[0][0]) + 1,
                "Cosine Rank": int(np.where(cosine_rank == i)[0][0]) + 1,
            })
        st.dataframe(pd.DataFrame(_rank_rows).sort_values("Pearson Rank"),
                     hide_index=True, use_container_width=True)

    # ── TAB 2: FULL NNLS ──────────────────────────────────────────
    with tab_nnls:
        fn = results["full_nnls"]
        _fn1, _fn2, _fn3 = st.columns(3)
        _fn1.metric("R² (full model)", f"{fn['r2']:.4f}")
        _fn2.metric("Non-zero", int(np.sum(fn['coeffs'] > 0)))
        _fn3.metric("RSS", f"{fn['rss']:.4f}")

        _nnls_rank = np.argsort(fn["weights"])[::-1]
        _nnls_rows = []
        for r, idx in enumerate(_nnls_rank):
            if fn["coeffs"][idx] <= 0:
                break
            _nnls_rows.append({
                "Rank": r + 1,
                "CID": molecules[idx]["cid"],
                "Name": molecules[idx]["name"],
                "Spectral Weight": f"{fn['weights'][idx]:.1%}",
                "Coefficient": f"{fn['coeffs'][idx]:.5f}",
            })
        if _nnls_rows:
            st.dataframe(pd.DataFrame(_nnls_rows), hide_index=True,
                         use_container_width=True)

        # Bar chart of weights
        _nz = [i for i in _nnls_rank if fn["coeffs"][i] > 0]
        if _nz:
            _fig_wt = go.Figure(go.Bar(
                x=[f"{molecules[i]['cid']}" for i in _nz],
                y=[fn["weights"][i] * 100 for i in _nz],
                marker_color=[_palette[j % len(_palette)] for j in range(len(_nz))],
                hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
            ))
            _fig_wt.update_layout(xaxis_title="CID", yaxis_title="Spectral Weight (%)",
                                   height=350)
            st.plotly_chart(_fig_wt, use_container_width=True)

    # ── TAB 3: MODEL SELECTION ─────────────────────────────────────
    with tab_model:
        sw_history = results["stepwise_history"]
        bic_vals = results["bic_vals"]
        cv_vals = results["cv_vals"]
        best_k_bic = results["best_k_bic"]
        best_k_cv = results["best_k_cv"]
        best_k = results["best_k"]
        ss_tot = results["ss_tot"]

        _ms1, _ms2, _ms3 = st.columns(3)
        _ms1.metric("Best k (BIC)", best_k_bic)
        _ms2.metric("Best k (CV)", best_k_cv)
        _ms3.metric("Selected k", best_k)

        ks = list(range(1, len(sw_history) + 1))
        r2s = [1 - h[1] / ss_tot for h in sw_history]

        _fig_ms = make_subplots(rows=1, cols=3,
                                 subplot_titles=["R² vs k", "BIC vs k", "CV Error vs k"])

        _fig_ms.add_trace(go.Scatter(x=ks, y=r2s, mode="lines+markers",
                                      marker=dict(color="darkgreen", size=7),
                                      name="R²"), row=1, col=1)
        _fig_ms.add_vline(x=best_k, line_dash="dash", line_color="red",
                           row=1, col=1, annotation_text=f"k={best_k}")

        _fig_ms.add_trace(go.Scatter(x=ks, y=bic_vals, mode="lines+markers",
                                      marker=dict(color="crimson", size=7),
                                      name="BIC"), row=1, col=2)
        _fig_ms.add_vline(x=best_k_bic, line_dash="dash", line_color="gray",
                           row=1, col=2, annotation_text=f"k={best_k_bic}")

        _fig_ms.add_trace(go.Scatter(x=ks, y=cv_vals, mode="lines+markers",
                                      marker=dict(color="darkorange", size=7),
                                      name="CV Error"), row=1, col=3)
        _fig_ms.add_vline(x=best_k_cv, line_dash="dash", line_color="gray",
                           row=1, col=3, annotation_text=f"k={best_k_cv}")

        _fig_ms.update_xaxes(title_text="k", row=1, col=1)
        _fig_ms.update_xaxes(title_text="k", row=1, col=2)
        _fig_ms.update_xaxes(title_text="k", row=1, col=3)
        _fig_ms.update_yaxes(title_text="R²", row=1, col=1)
        _fig_ms.update_yaxes(title_text="BIC", row=1, col=2)
        _fig_ms.update_yaxes(title_text="CV Error", row=1, col=3)
        _fig_ms.update_layout(height=400, showlegend=False)
        st.plotly_chart(_fig_ms, use_container_width=True)

        # Stepwise history table
        with st.expander("Forward stepwise history"):
            _sw_rows = []
            for step, (sel, rss) in enumerate(sw_history):
                added = sel[-1]
                _sw_rows.append({
                    "k": step + 1,
                    "Added CID": molecules[added]["cid"],
                    "Added Name": molecules[added]["name"],
                    "RSS": f"{rss:.4f}",
                    "R²": f"{1 - rss / ss_tot:.4f}",
                    "BIC": f"{bic_vals[step]:.2f}",
                    "CV Error": f"{cv_vals[step]:.4f}",
                })
            st.dataframe(pd.DataFrame(_sw_rows), hide_index=True,
                         use_container_width=True)

    # ── TAB 4: BEST FIT ───────────────────────────────────────────
    with tab_fit:
        ff = results["final_fit"]
        best_sel = results["best_selection"]

        _ff1, _ff2, _ff3 = st.columns(3)
        _ff1.metric("R²", f"{ff['r2']:.4f}")
        _ff2.metric("Components (k)", results["best_k"])
        _ff3.metric("RSS", f"{ff['rss']:.4f}")

        # Decomposition plot
        _fig_fit = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.75, 0.25], vertical_spacing=0.06,
                                  subplot_titles=["Decomposition", "Residuals"])

        _fig_fit.add_trace(go.Scatter(
            x=exp_wn, y=exp_norm, mode="lines", name="Experimental",
            line=dict(color="black", width=2),
        ), row=1, col=1)

        _fig_fit.add_trace(go.Scatter(
            x=exp_wn, y=ff["reconstruction"], mode="lines",
            name=f"Fit (R²={ff['r2']:.3f})",
            line=dict(color="red", width=1.5, dash="dash"),
        ), row=1, col=1)

        # Individual components
        for ci, idx in enumerate(best_sel):
            contrib = dft_matrix[idx] * ff["coeffs"][ci]
            _c = _palette[ci % len(_palette)]
            _fig_fit.add_trace(go.Scatter(
                x=exp_wn, y=contrib, mode="lines", fill="tozeroy",
                name=f"{molecules[idx]['cid']}: {molecules[idx]['name'][:25]} "
                     f"({ff['weights'][ci]:.0%})",
                line=dict(color=_c, width=1), opacity=0.4,
            ), row=1, col=1)

        # Residuals
        _fig_fit.add_trace(go.Scatter(
            x=exp_wn, y=ff["residuals"], mode="lines", name="Residual",
            line=dict(color="gray", width=1), fill="tozeroy",
            fillcolor="rgba(128,128,128,0.2)",
        ), row=2, col=1)
        _fig_fit.add_hline(y=0, line_dash="dash", line_color="gray",
                            line_width=0.5, row=2, col=1)

        _fig_fit.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
        _fig_fit.update_yaxes(title_text="Intensity", row=1, col=1)
        _fig_fit.update_yaxes(title_text="Residual", row=2, col=1)
        _fig_fit.update_layout(height=600, legend=dict(font=dict(size=8)))
        st.plotly_chart(_fig_fit, use_container_width=True)

        # Component weight table
        _comp_rows = []
        for ci, idx in enumerate(best_sel):
            _comp_rows.append({
                "CID": molecules[idx]["cid"],
                "Name": molecules[idx]["name"],
                "Weight": f"{ff['weights'][ci]:.1%}",
                "Coefficient": f"{ff['coeffs'][ci]:.5f}",
            })
        st.dataframe(pd.DataFrame(_comp_rows), hide_index=True,
                     use_container_width=True)

        st.caption(f"Baseline: const = {ff['baseline'][0]:.5f}" +
                   (f", linear = {ff['baseline'][1]:.5f}" if len(ff['baseline']) > 1 else ""))

    # ── TAB 5: BOOTSTRAP ──────────────────────────────────────────
    with tab_boot:
        boot_w = results["bootstrap_weights"]
        best_sel = results["best_selection"]
        n_sel = len(best_sel)

        # Bootstrap distributions
        st.markdown("#### Weight Distributions")
        _fig_box = go.Figure()
        for ci in range(n_sel):
            idx = best_sel[ci]
            _fig_box.add_trace(go.Box(
                y=boot_w[:, ci] * 100,
                name=f"{molecules[idx]['cid']}",
                marker_color=_palette[ci % len(_palette)],
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
        _fig_box.update_layout(
            yaxis_title="Spectral Weight (%)", height=450,
            title=f"Block Bootstrap (n={boot_w.shape[0]})",
            showlegend=False,
        )
        st.plotly_chart(_fig_box, use_container_width=True)

        # Summary table
        _bs = results["bootstrap_summary"]
        _bs_rows = []
        for b in _bs:
            _bs_rows.append({
                "CID": b["cid"],
                "Name": b["name"],
                "Median Weight": f"{b['weight_median']:.1%}",
                "95% CI": f"[{b['ci_lo']:.1%}, {b['ci_hi']:.1%}]",
                "Selection Freq": f"{b['sel_freq']:.0%}",
            })
        st.dataframe(pd.DataFrame(_bs_rows), hide_index=True,
                     use_container_width=True)

        # Rank stability
        st.markdown("#### Pairwise Rank Stability")
        _rstab = results["rank_stability"]
        _rlabels = [f"{molecules[idx]['cid']}" for idx in best_sel]
        _fig_rstab = go.Figure(data=go.Heatmap(
            z=_rstab, x=_rlabels, y=_rlabels,
            colorscale="RdYlGn", zmin=0, zmax=1,
            hovertemplate="P(%{y} > %{x}) = %{z:.0%}<extra></extra>",
            text=np.vectorize(lambda v: f"{v:.0%}")(_rstab),
            texttemplate="%{text}",
        ))
        _fig_rstab.update_layout(height=350, title="P(row > column)")
        st.plotly_chart(_fig_rstab, use_container_width=True)

        # Warnings
        _warnings = []
        for i in range(n_sel):
            for j in range(i + 1, n_sel):
                if 0.3 < _rstab[i, j] < 0.7:
                    _warnings.append(
                        f"⚠️ {molecules[best_sel[i]]['cid']} vs "
                        f"{molecules[best_sel[j]]['cid']}: "
                        f"rank order not robust ({_rstab[i, j]:.0%})")
        if _warnings:
            for w in _warnings:
                st.warning(w)
        else:
            st.success("✓ All pairwise rank orders are robust (> 70% or < 30%)")

    # ── TAB 6: PEAK RESIDUALS ─────────────────────────────────────
    with tab_peaks:
        pa = results["peak_analysis"]
        if len(pa["peak_wn"]) > 0:
            _pk_rows = []
            for w, e, f, r in zip(pa["peak_wn"], pa["peak_exp"],
                                   pa["peak_fit"], pa["peak_resid"]):
                rel = abs(r / e) * 100 if abs(e) > 0.01 else 0
                _pk_rows.append({
                    "cm⁻¹": f"{w:.1f}",
                    "Experimental": f"{e:.3f}",
                    "Fit": f"{f:.3f}",
                    "Residual": f"{r:+.3f}",
                    "Rel Error %": f"{rel:.1f}",
                })

            _pk1, _pk2 = st.columns(2)
            _pk1.metric("Peaks detected", len(pa["peak_wn"]))
            _pk2.metric("Mean |residual|",
                        f"{np.mean(np.abs(pa['peak_resid'])):.4f}")

            st.dataframe(pd.DataFrame(_pk_rows), hide_index=True,
                         use_container_width=True)

            # Peak residual plot
            ff = results["final_fit"]
            _fig_pk = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    row_heights=[0.7, 0.3], vertical_spacing=0.06)
            _fig_pk.add_trace(go.Scatter(x=exp_wn, y=exp_norm, mode="lines",
                                          name="Exp", line=dict(color="black")),
                               row=1, col=1)
            _fig_pk.add_trace(go.Scatter(x=exp_wn, y=ff["reconstruction"],
                                          mode="lines", name="Fit",
                                          line=dict(color="red", dash="dash")),
                               row=1, col=1)
            _fig_pk.add_trace(go.Scatter(
                x=pa["peak_wn"], y=pa["peak_exp"], mode="markers",
                name="Peaks", marker=dict(color="blue", size=8, symbol="diamond"),
            ), row=1, col=1)
            _fig_pk.add_trace(go.Scatter(
                x=exp_wn, y=ff["residuals"], mode="lines",
                name="Residual", line=dict(color="gray"),
            ), row=2, col=1)
            _fig_pk.add_trace(go.Scatter(
                x=pa["peak_wn"], y=pa["peak_resid"], mode="markers",
                name="Peak residual", marker=dict(color="red", size=7),
            ), row=2, col=1)
            _fig_pk.add_hline(y=0, line_dash="dash", line_color="gray",
                               row=2, col=1)
            _fig_pk.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
            _fig_pk.update_layout(height=500)
            st.plotly_chart(_fig_pk, use_container_width=True)
        else:
            st.info("No peaks detected above threshold.")

    # ── TAB 7: SENSITIVITY ────────────────────────────────────────
    with tab_sens:
        if results["sensitivity"]:
            _sens = results["sensitivity"]
            _fig_sens = go.Figure()
            _scales = [r["scale"] for r in _sens]
            _r2s = [r["r2"] for r in _sens]
            _fig_sens.add_trace(go.Scatter(
                x=_scales, y=_r2s, mode="lines+markers",
                marker=dict(color="purple", size=10),
                line=dict(color="purple", width=2),
                hovertemplate="Scale: %{x:.2f}<br>R²: %{y:.4f}<extra></extra>",
            ))
            for r in _sens:
                _fig_sens.add_annotation(
                    x=r["scale"], y=r["r2"],
                    text=f"k={r['best_k']}<br>CID {r['top_cid']}<br>{r['top_weight']:.0%}",
                    showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(size=9),
                )
            _fig_sens.update_layout(
                xaxis_title="DFT Scaling Factor", yaxis_title="R²",
                title="Sensitivity: R² vs Scaling Factor", height=450,
            )
            st.plotly_chart(_fig_sens, use_container_width=True)

            _sens_rows = []
            for r in _sens:
                _sens_rows.append({
                    "Scale": r["scale"],
                    "Best k": r["best_k"],
                    "R²": f"{r['r2']:.4f}",
                    "CV Error": f"{r['cv_err']:.4f}",
                    "Top CID": r["top_cid"],
                    "Top Weight": f"{r['top_weight']:.1%}",
                })
            st.dataframe(pd.DataFrame(_sens_rows), hide_index=True,
                         use_container_width=True)

            _top_cids = [r["top_cid"] for r in _sens]
            if len(set(_top_cids)) == 1:
                st.success(f"✓ Top assignment stable across scales: CID {_top_cids[0]}")
            else:
                st.warning(f"⚠️ Top assignment varies: {_top_cids}")
        else:
            st.info("Scaling sensitivity was not run. Enable it in the configuration.")

    # ── TAB 8: EXPORT ─────────────────────────────────────────────
    with tab_export:
        st.markdown("#### Summary CSV")
        ff = results["final_fit"]
        best_sel = results["best_selection"]
        pearson_sc = results["pearson_scores"]
        cosine_sc = results["cosine_scores"]
        pearson_rank = np.argsort(pearson_sc)[::-1]
        cosine_rank = np.argsort(cosine_sc)[::-1]
        nnls_rank = np.argsort(results["full_nnls"]["weights"])[::-1]

        _export_rows = []
        for i in range(len(molecules)):
            _export_rows.append({
                "CID": molecules[i]["cid"],
                "Name": molecules[i]["name"],
                "Pearson_Derivative": round(float(pearson_sc[i]), 5),
                "Cosine_Similarity": round(float(cosine_sc[i]), 5),
                "NNLS_Weight_Pct": round(float(results["full_nnls"]["weights"][i] * 100), 3),
                "Pearson_Rank": int(np.where(pearson_rank == i)[0][0]) + 1,
                "Cosine_Rank": int(np.where(cosine_rank == i)[0][0]) + 1,
                "NNLS_Rank": int(np.where(nnls_rank == i)[0][0]) + 1,
                "In_Best_Model": "Yes" if i in best_sel else "No",
                "Cluster_ID": results["diagnostics"]["cluster_labels"][i],
            })
        _export_df = pd.DataFrame(_export_rows).sort_values(
            "NNLS_Weight_Pct", ascending=False)
        st.dataframe(_export_df, hide_index=True, use_container_width=True)

        _csv_export = _export_df.to_csv(index=False)
        st.download_button("📥 Download Summary CSV", data=_csv_export,
                           file_name=f"spectral_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv", key="_sd_dl_csv")

        # PDF export
        st.markdown("#### Publication PDF (8 pages)")
        if st.button("🖨️ Generate PDF Report", key="_sd_gen_pdf"):
            _buf_pdf = io.BytesIO()
            _cmap = plt.colormaps["tab10"]
            _img_dir_path = st.session_state.get("_sd_img_dir", "")

            with PdfPages(_buf_pdf) as pdf:
                # ─── Page 1: Rankings ───────────────────────────────────
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                _top_n = min(15, len(molecules))
                _tp = pearson_rank[:_top_n]
                ax1.barh(range(_top_n), pearson_sc[_tp], color="teal", edgecolor="none")
                ax1.set_yticks(range(_top_n))
                ax1.set_yticklabels([f"CID {molecules[i]['cid']}: {molecules[i]['name'][:25]}" for i in _tp], fontsize=7)
                ax1.set_xlabel("Pearson (smoothed ∂/∂ν)")
                ax1.set_title("Pearson on First Derivative", fontweight="bold", fontsize=9)
                ax1.invert_yaxis()
                _tc = cosine_rank[:_top_n]
                ax2.barh(range(_top_n), cosine_sc[_tc], color="steelblue", edgecolor="none")
                ax2.set_yticks(range(_top_n))
                ax2.set_yticklabels([f"CID {molecules[i]['cid']}: {molecules[i]['name'][:25]}" for i in _tc], fontsize=7)
                ax2.set_xlabel("Cosine Similarity")
                ax2.set_title("Cosine (raw spectra)", fontweight="bold", fontsize=9)
                ax2.invert_yaxis()
                plt.tight_layout()
                pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 2: Forward stepwise R², BIC, CV ──────────────
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                ks = list(range(1, len(sw_history) + 1))
                r2s = [1 - h[1] / ss_tot for h in sw_history]
                axes[0].plot(ks, r2s, "o-", color="darkgreen", lw=2, ms=5)
                axes[0].axvline(best_k, color="crimson", ls="--", lw=1, label=f"CV k={best_k}")
                axes[0].set_xlabel("k"); axes[0].set_ylabel("R²")
                axes[0].set_title("Forward Stepwise R²"); axes[0].legend()
                axes[0].set_ylim(bottom=0)
                axes[1].plot(ks, bic_vals, "s-", color="crimson", lw=2, ms=5)
                axes[1].axvline(best_k_bic, color="gray", ls="--", label=f"BIC k={best_k_bic}")
                axes[1].set_xlabel("k"); axes[1].set_ylabel(f"BIC (n_eff={results['n_eff']})")
                axes[1].set_title("BIC (bandwidth-corrected)"); axes[1].legend()
                axes[2].plot(ks, cv_vals, "D-", color="darkorange", lw=2, ms=5)
                axes[2].axvline(best_k_cv, color="gray", ls="--", label=f"CV k={best_k_cv}")
                axes[2].set_xlabel("k"); axes[2].set_ylabel("CV Error (RSS)")
                axes[2].set_title("Blocked CV"); axes[2].legend()
                plt.tight_layout()
                pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 3: Best fit + residual ───────────────────────
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                                gridspec_kw={"height_ratios": [3, 1]})
                ax1.plot(exp_wn, exp_norm, "k-", lw=1.2, label="Experimental")
                ax1.plot(exp_wn, ff["reconstruction"], "r-", lw=1.0, alpha=0.85,
                         label=f"Fit (k={results['best_k']}, R²={ff['r2']:.3f})")
                for ci, idx in enumerate(best_sel):
                    contrib = dft_matrix[idx] * ff["coeffs"][ci]
                    ax1.fill_between(exp_wn, 0, contrib, alpha=0.2,
                                     color=_cmap(ci % 10),
                                     label=f"CID {molecules[idx]['cid']} ({ff['weights'][ci]:.0%})")
                ax1.set_xlim(exp_wn.min(), exp_wn.max())
                ax1.set_ylabel("Intensity (norm.)")
                ax1.set_title(f"{results['best_k']}-Component Decomposition "
                              "(spectral weights, NOT populations)",
                              fontweight="bold", fontsize=9)
                ax1.legend(fontsize=6, loc="upper right", ncol=2)
                ax2.plot(exp_wn, ff["residuals"], "k-", lw=0.8)
                ax2.axhline(0, color="gray", ls="--", lw=0.5)
                ax2.fill_between(exp_wn, 0, ff["residuals"], alpha=0.25, color="gray")
                _pa = results["peak_analysis"]
                if len(_pa["peak_wn"]) > 0:
                    ax2.scatter(_pa["peak_wn"], _pa["peak_resid"], c="red", s=15,
                               zorder=5, label="Peak residuals")
                    ax2.legend(fontsize=7)
                ax2.set_xlim(exp_wn.min(), exp_wn.max())
                ax2.set_xlabel("Wavenumber (cm⁻¹)"); ax2.set_ylabel("Residual")
                plt.tight_layout()
                pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 4: Bootstrap distributions ───────────────────
                fig, ax = plt.subplots(figsize=(10, 5))
                bp = ax.boxplot([boot_w[:, i] * 100 for i in range(n_sel)],
                                vert=True, patch_artist=True,
                                medianprops=dict(color="black", lw=1.5))
                _sel_freq = np.mean(boot_w > 1e-4, axis=0)
                for patch, ci in zip(bp["boxes"], range(n_sel)):
                    patch.set_facecolor(_cmap(ci % 10)); patch.set_alpha(0.5)
                ax.set_xticklabels([f"CID {molecules[idx]['cid']}" for idx in best_sel],
                                   fontsize=7, rotation=45, ha="right")
                ax.set_ylabel("Spectral Weight (%)")
                ax.set_title(f"Block Bootstrap (n={boot_w.shape[0]}, "
                             f"block={results['block_len']}pt)", fontweight="bold")
                for i in range(n_sel):
                    ax.annotate(f"sel {_sel_freq[i]:.0%}",
                                (i + 1, np.percentile(boot_w[:, i], 97.5) * 100),
                                fontsize=6, ha="center", va="bottom", color="red")
                plt.tight_layout()
                pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 5: Sensitivity ───────────────────────────────
                if results["sensitivity"]:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    _sens = results["sensitivity"]
                    _scales = [r["scale"] for r in _sens]
                    _r2s_s = [r["r2"] for r in _sens]
                    ax.plot(_scales, _r2s_s, "o-", color="purple", lw=2, ms=8)
                    for r in _sens:
                        ax.annotate(f"k={r['best_k']}\nCID {r['top_cid']}",
                                    (r["scale"], r["r2"]),
                                    textcoords="offset points", xytext=(0, 12),
                                    fontsize=7, ha="center")
                    ax.set_xlabel("DFT Scaling Factor")
                    ax.set_ylabel("R² (best model)")
                    ax.set_title("Sensitivity: R² vs Scaling Factor", fontweight="bold")
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 6+: Selected structures with spectra ─────────
                _n_best = len(best_sel)
                _panels_per_page = 4
                _n_struct_pages = int(np.ceil(_n_best / _panels_per_page))
                for sp in range(_n_struct_pages):
                    start_i = sp * _panels_per_page
                    end_i = min(start_i + _panels_per_page, _n_best)
                    n_pan = end_i - start_i
                    fig = plt.figure(figsize=(12, 3.2 * n_pan))
                    gs = GridSpec(n_pan, 2, width_ratios=[3, 1], wspace=0.05, hspace=0.35)
                    for pi, ci in enumerate(range(start_i, end_i)):
                        idx = best_sel[ci]
                        mol = molecules[idx]
                        color = _cmap(ci % 10)
                        # Spectrum panel
                        ax_sp = fig.add_subplot(gs[pi, 0])
                        ax_sp.plot(exp_wn, exp_norm, color="0.5", lw=0.7, alpha=0.5, label="Exp")
                        contrib = dft_matrix[idx] * ff["coeffs"][ci]
                        ax_sp.fill_between(exp_wn, 0, contrib, alpha=0.3, color=color)
                        _dft_max = dft_matrix[idx].max()
                        _dft_show = dft_matrix[idx] / _dft_max if _dft_max > 0 else dft_matrix[idx]
                        ax_sp.plot(exp_wn, _dft_show, color=color, lw=1.0, label="DFT (norm.)")
                        lo_ci = np.percentile(boot_w[:, ci], 2.5)
                        hi_ci = np.percentile(boot_w[:, ci], 97.5)
                        ax_sp.set_xlim(exp_wn.min(), exp_wn.max())
                        ax_sp.set_ylim(bottom=-0.05)
                        ax_sp.text(0.01, 0.95,
                                   f"CID {mol['cid']}\n{mol['name'][:40]}\n"
                                   f"Wt: {ff['weights'][ci]:.1%} "
                                   f"[{lo_ci:.1%}–{hi_ci:.1%}]  sel: {_sel_freq[ci]:.0%}",
                                   transform=ax_sp.transAxes, fontsize=7, va="top",
                                   fontweight="bold", color=color)
                        ax_sp.spines["top"].set_visible(False)
                        ax_sp.spines["right"].set_visible(False)
                        if pi == n_pan - 1:
                            ax_sp.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
                        else:
                            ax_sp.set_xticklabels([])
                        ax_sp.set_ylabel("Int.", fontsize=8)
                        # Image panel
                        ax_im = fig.add_subplot(gs[pi, 1])
                        ax_im.axis("off")
                        _img_found = None
                        if _img_dir_path and os.path.isdir(_img_dir_path):
                            # Try multiple naming patterns
                            _name_stem = os.path.splitext(mol.get("file", mol["name"]))[0]
                            _name_stem = os.path.basename(_name_stem)
                            _candidates = [
                                f"CID_{mol['cid']}.png",
                                f"CID_{mol['cid']}.jpg",
                                f"CID_{mol['cid']}.jpeg",
                                f"{mol['name']}.png",
                                f"{mol['name']}.jpg",
                                f"{_name_stem}.png",
                                f"{_name_stem}.jpg",
                            ]
                            for _cand in _candidates:
                                _try = os.path.join(_img_dir_path, _cand)
                                if os.path.exists(_try):
                                    _img_found = _try
                                    break
                            # Fallback: case-insensitive partial match
                            if not _img_found:
                                _cid_lower = mol['cid'].lower()
                                _name_lower = mol['name'].lower().replace(" ", "_")
                                for _f in os.listdir(_img_dir_path):
                                    _fl = _f.lower()
                                    if not _fl.endswith((".png", ".jpg", ".jpeg")):
                                        continue
                                    if _cid_lower in _fl or _name_lower in _fl:
                                        _img_found = os.path.join(_img_dir_path, _f)
                                        break
                        if _img_found:
                            try:
                                from matplotlib.image import imread as _imread
                                img = _imread(_img_found)
                                ax_im.imshow(img, aspect="equal")
                            except Exception:
                                ax_im.text(0.5, 0.5, "Error loading", ha="center",
                                           va="center", transform=ax_im.transAxes, fontsize=8)
                        else:
                            ax_im.text(0.5, 0.5, "No image", ha="center",
                                       va="center", transform=ax_im.transAxes, fontsize=8)
                    fig.suptitle(f"Selected Components (page {sp+1}/{_n_struct_pages}) — "
                                 "spectral weights, NOT populations",
                                 fontsize=10, fontweight="bold", y=0.99)
                    pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 7: Gram matrix heatmap ──────────────────────
                fig, ax = plt.subplots(figsize=(10, 9))
                _gram_full = results["diagnostics"]["gram_matrix"]
                im = ax.imshow(_gram_full, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks(range(len(molecules)))
                ax.set_yticks(range(len(molecules)))
                _lbl = [molecules[i]["cid"] for i in range(len(molecules))]
                ax.set_xticklabels(_lbl, fontsize=4, rotation=90)
                ax.set_yticklabels(_lbl, fontsize=4)
                ax.set_title("DFT–DFT Cosine Similarity (Gram Matrix)", fontweight="bold")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                pdf.savefig(fig, dpi=150); plt.close(fig)

                # ─── Page 8: Peak-resolved residuals ──────────────────
                if len(_pa["peak_wn"]) > 0:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                                    gridspec_kw={"height_ratios": [2, 1]})
                    ax1.plot(exp_wn, exp_norm, "k-", lw=1.0, label="Experimental")
                    ax1.plot(exp_wn, ff["reconstruction"], "r-", lw=0.8, label="Fit")
                    ax1.scatter(_pa["peak_wn"], _pa["peak_exp"], c="blue", s=20,
                                zorder=5, marker="v", label="Detected peaks")
                    ax1.set_xlim(exp_wn.min(), exp_wn.max())
                    ax1.set_ylabel("Intensity")
                    ax1.set_title(f"Peak-Resolved Analysis ({len(_pa['peak_wn'])} peaks, "
                                  f"mean |resid| = {np.mean(np.abs(_pa['peak_resid'])):.4f})",
                                  fontweight="bold", fontsize=9)
                    ax1.legend(fontsize=7)
                    ax2.stem(_pa["peak_wn"], _pa["peak_resid"], linefmt="r-",
                             markerfmt="ro", basefmt="k-")
                    ax2.axhline(0, color="gray", ls="--", lw=0.5)
                    ax2.set_xlim(exp_wn.min(), exp_wn.max())
                    ax2.set_xlabel("Wavenumber (cm⁻¹)")
                    ax2.set_ylabel("Peak Residual")
                    plt.tight_layout()
                    pdf.savefig(fig, dpi=150); plt.close(fig)

            _buf_pdf.seek(0)
            st.download_button("📥 Download PDF Report", data=_buf_pdf,
                               file_name=f"spectral_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf", key="_sd_dl_pdf")
            st.success("✅ PDF generated (8 pages)!")
