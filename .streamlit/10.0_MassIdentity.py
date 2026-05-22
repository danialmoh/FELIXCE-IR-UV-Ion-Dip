"""
Mass Identity Workbench — per-mass assignment with formula matching,
spectral comparison, structure, and publication-ready figure export.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os
import json
import configparser
from pathlib import Path
from datetime import datetime

from packages.load_dataset import ensure_dataset_loaded
from packages.ReportManager import add_plot_to_report_button, init_report_session
from packages.DFT_Parsers import parse_dft_file, broaden_spectrum_felix
from packages.PCC_Scoring import (
    preprocess_spectrum,
    find_optimal_scaling_factor,
)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

init_report_session()

st.set_page_config(page_title="Mass Identity Workbench", layout="wide")
st.title("🔬 Mass Identity Workbench")
st.caption(
    "Go mass-by-mass: view the mass spectrum, pick a mass, extract its IR, "
    "match candidate formulas, compare against reference spectra (NIST / DFT), "
    "upload a molecular structure, and export a publication-ready figure."
)

# ════════════════════════════════════════════════════════════════════════════════
# PERSISTENT ASSIGNMENT STORE — loads from JSON on disk at startup
# ════════════════════════════════════════════════════════════════════════════════
def _assignments_json_path():
    """Return path to the persistent assignments JSON file, or None."""
    _file_dir = st.session_state.get("file_directory", "")
    if not _file_dir:
        _defaults_file = r'./.streamlit/defaults.ini'
        if os.path.exists(_defaults_file):
            _cfg = configparser.ConfigParser()
            _cfg.read(_defaults_file)
            try:
                _file_dir = _cfg.get('Import Data', 'file_directory')
            except configparser.Error:
                pass
    if not _file_dir:
        return None
    return Path(_file_dir) / "output" / "assignments" / "assignments.json"

if "_mass_assignments" not in st.session_state:
    # Try loading from disk first
    _json_p = _assignments_json_path()
    if _json_p is not None and _json_p.exists():
        try:
            with open(_json_p, "r") as _jf:
                st.session_state["_mass_assignments"] = json.load(_jf)
        except Exception:
            st.session_state["_mass_assignments"] = {}
    else:
        st.session_state["_mass_assignments"] = {}


# ════════════════════════════════════════════════════════════════════════════════
# JDX PARSER (same as 9.1)
# ════════════════════════════════════════════════════════════════════════════════
def parse_jdx(file_content):
    """Parse a JCAMP-DX (.jdx) file → (metadata dict, wavenumber array, intensity array)."""
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
            continue
    if data_start is None:
        raise ValueError("No ##XYDATA= line found in JDX file.")
    y_values = []
    firstx = float(meta.get("FIRSTX", 0))
    lastx = float(meta.get("LASTX", 0))
    npoints = int(meta.get("NPOINTS", 0))
    for line in lines[data_start:]:
        line_s = line.strip()
        if line_s.startswith("##"):
            break
        if not line_s or line_s.startswith("$$"):
            continue
        parts = line_s.split()
        if len(parts) < 2:
            continue
        try:
            for ystr in parts[1:]:
                y_values.append(float(ystr) * yfactor)
        except ValueError:
            continue
    if npoints > 0 and len(y_values) > 0:
        x_values = np.linspace(firstx, lastx, len(y_values))
    else:
        x_values = np.arange(len(y_values), dtype=float)
    return meta, np.array(x_values), np.array(y_values)


# ════════════════════════════════════════════════════════════════════════════════
# FORMULA MATCHING ENGINE (standalone, same logic as 4.2 but self-contained)
# ════════════════════════════════════════════════════════════════════════════════
Br79_mass = 78.9183
Br81_mass = 80.9163
C_mass = 12.0000
H_mass = 1.007825


def calc_rdb(n_C, n_H, n_Br=0):
    return (2 * n_C + 2 - n_H - n_Br) / 2.0


def _make_entry(formula, calc_mass, target_mz, n_C, n_H, n_Br=0, ctype="", extra=""):
    rdb = calc_rdb(n_C, n_H, n_Br)
    return {
        "formula": formula,
        "type": ctype,
        "calc_mass": calc_mass,
        "mass_error_Da": calc_mass - target_mz,
        "RDB": rdb,
        "H/C": n_H / n_C if n_C > 0 else 0,
        "n_C": n_C,
        "n_H": n_H,
        "n_Br": n_Br,
        "extra": extra,
    }


def generate_candidates(target_mz, tol=0.5, min_C=1, max_C=30,
                        hc_min=0.3, hc_max=2.5, rdb_max=15,
                        include_br=True, check_br_isotope=True):
    """Generate CH and optionally CHBr candidates for a single m/z."""
    results = []

    def _valid(n_C, n_H, n_Br=0):
        if n_H < 1:
            return False
        rdb = calc_rdb(n_C, n_H, n_Br)
        if rdb < 0 or rdb > rdb_max:
            return False
        hc = n_H / n_C if n_C > 0 else 0
        if hc < hc_min or hc > hc_max:
            return False
        return True

    # ── CH candidates ────────────────────────────────────────────────────
    for n_C in range(min_C, max_C + 1):
        for u in range(0, rdb_max + 1):
            n_H = 2 * n_C + 2 - 2 * u
            if n_H < 1:
                break
            calc = n_C * C_mass + n_H * H_mass
            if abs(calc - target_mz) <= tol and _valid(n_C, n_H):
                label = "PAH" if u >= 4 else ("Unsaturated" if u >= 1 else "Alkane")
                results.append(_make_entry(
                    f"C{n_C}H{n_H}", calc, target_mz, n_C, n_H,
                    ctype=f"CH ({label})", extra=f"u={u}",
                ))

    # ── CHBr candidates ──────────────────────────────────────────────────
    if include_br:
        br_masses = [(Br79_mass, "⁷⁹Br")]
        if check_br_isotope:
            br_masses.append((Br81_mass, "⁸¹Br"))
        for br_m, br_label in br_masses:
            for k in range(1, 4):  # up to 3 Br atoms
                for n_C in range(min_C, max_C + 1):
                    for u in range(0, rdb_max + 1):
                        n_H = 2 * n_C + 2 - 2 * u - k
                        if n_H < 1:
                            break
                        calc = n_C * C_mass + n_H * H_mass + k * br_m
                        if abs(calc - target_mz) <= tol and _valid(n_C, n_H, k):
                            results.append(_make_entry(
                                f"C{n_C}H{n_H}Br{k if k > 1 else ''}",
                                calc, target_mz, n_C, n_H, n_Br=k,
                                ctype=f"CHBr{k}", extra=f"u={u}, {br_label}",
                            ))

    # ── Deduplicate by formula + Br label, keep best mass error ──────────
    seen = {}
    for c in results:
        key = (c["formula"], c["extra"])
        if key not in seen or abs(c["mass_error_Da"]) < abs(seen[key]["mass_error_Da"]):
            seen[key] = c
    results = sorted(seen.values(), key=lambda x: abs(x["mass_error_Da"]))
    return results


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA & INTERACTIVE MASS SPECTRUM
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 1. Dataset & Mass Spectrum")

ensure_dataset_loaded(
    require_keys=["x_mass", "compilation_baseline_corrected_data", "unique_wavenumbers"],
    compute_megasum=True,
    page_key_prefix="_mid",
)

x_mass = st.session_state["x_mass"]
compilation = st.session_state["compilation_baseline_corrected_data"]
unique_wavenumbers = st.session_state["unique_wavenumbers"]
plot_col_wo = st.session_state.get("plot_columnIndex_withoutIR", -2)
plot_col_w = st.session_state.get("plot_columnIndex_withIR", -1)
MegaSum = st.session_state.get("MegaSum")

# Build the total mass spectrum from MegaSum (baseline-corrected without-IR)
if MegaSum is not None:
    ms_y = MegaSum.iloc[:, -2].values  # baseline_corrected_signal_withoutIR
else:
    ms_y = np.zeros(len(x_mass))

_mz_range = st.columns(2)
with _mz_range[0]:
    mz_lo = st.number_input("m/z min", value=float(x_mass.min()), step=5.0, key="_mid_mz_lo")
with _mz_range[1]:
    mz_hi = st.number_input("m/z max", value=float(x_mass.max()), step=5.0, key="_mid_mz_hi")

_mz_mask = (x_mass >= mz_lo) & (x_mass <= mz_hi)
_x_disp = x_mass[_mz_mask]
_y_disp = ms_y[_mz_mask]

# Peak detection on the displayed mass spectrum
_pd_col1, _pd_col2 = st.columns(2)
with _pd_col1:
    _prom = st.number_input("Peak prominence", value=float(np.nanmax(np.abs(_y_disp)) * 0.05),
                            min_value=0.0, step=0.001, format="%.4f", key="_mid_prom")
with _pd_col2:
    _dist = st.number_input("Min peak distance (m/z)", value=2.0, min_value=0.1, step=0.5, key="_mid_dist")

_avg_sp = np.mean(np.diff(_x_disp)) if len(_x_disp) > 1 else 1.0
_dist_idx = max(1, int(_dist / _avg_sp))
_peaks, _props = find_peaks(np.nan_to_num(_y_disp, nan=0), prominence=_prom, distance=_dist_idx)

fig_ms = go.Figure()
fig_ms.add_trace(go.Scatter(
    x=_x_disp, y=_y_disp, mode="lines", name="Mass spectrum",
    line=dict(color="#1f77b4", width=1.5),
))
if len(_peaks) > 0:
    fig_ms.add_trace(go.Scatter(
        x=_x_disp[_peaks], y=_y_disp[_peaks], mode="markers+text",
        marker=dict(color="red", size=7, symbol="triangle-up"),
        text=[f"{m:.1f}" for m in _x_disp[_peaks]],
        textposition="top center", textfont=dict(size=9),
        name=f"Detected ({len(_peaks)})",
    ))
fig_ms.update_layout(
    xaxis_title="m/z", yaxis_title="Intensity (a.u.)",
    title="Mass Spectrum (MegaSum, baseline-corrected without-IR)",
    height=400,
)
st.plotly_chart(fig_ms, use_container_width=True)

detected_mz_list = sorted(_x_disp[_peaks].tolist()) if len(_peaks) > 0 else []
st.caption(f"Detected {len(detected_mz_list)} peaks: {', '.join(f'{m:.1f}' for m in detected_mz_list)}")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SELECT MASS & EXTRACT IR
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 2. Select Mass & Extract IR Spectrum")

_sel_col1, _sel_col2, _sel_col3 = st.columns([2, 1, 1])
with _sel_col1:
    if detected_mz_list:
        selected_mz = st.selectbox(
            "m/z to investigate",
            options=detected_mz_list,
            format_func=lambda x: f"{x:.1f}",
            key="_mid_sel_mz",
        )
    else:
        selected_mz = st.number_input("m/z to investigate", value=128.0, step=0.1, key="_mid_sel_mz_man")
with _sel_col2:
    half_width = st.number_input("Integration half-width (amu)", value=0.3,
                                  min_value=0.05, max_value=2.0, step=0.05, key="_mid_hw")
with _sel_col3:
    _sg_on = st.checkbox("Apply SG smoothing", value=False, key="_mid_sg")
    _sg_w = st.number_input("SG window", value=7, min_value=3, max_value=51, step=2,
                            key="_mid_sg_w", disabled=not _sg_on)

# Extract IR spectrum for the selected mass
wn_sorted = sorted(unique_wavenumbers)
wn_arr = np.array(wn_sorted, dtype=float)
sel_mask = (np.asarray(x_mass) >= selected_mz - half_width) & (np.asarray(x_mass) <= selected_mz + half_width)
n_bins = int(sel_mask.sum())

if n_bins == 0:
    st.error(f"No m/z bins within ±{half_width} of {selected_mz:.1f}")
    st.stop()

int_without = np.zeros(len(wn_sorted))
int_with = np.zeros(len(wn_sorted))
for i, wn in enumerate(wn_sorted):
    df_wn = compilation[wn]
    int_without[i] = df_wn.iloc[sel_mask, plot_col_wo].values.sum()
    int_with[i] = df_wn.iloc[sel_mask, plot_col_w].values.sum()

# Compute depletion quantities
with np.errstate(divide="ignore", invalid="ignore"):
    depletion = int_with / int_without
    ln_depletion = -np.log(np.clip(depletion, 1e-10, None))

# Optional smoothing
def _smooth(y):
    if not _sg_on or len(y) < _sg_w:
        return y
    w = _sg_w if _sg_w % 2 == 1 else _sg_w + 1
    return savgol_filter(y, window_length=w, polyorder=min(3, w - 1))

# Two-panel plot: IR on/off, −ln(depletion)
fig_ir = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=[
        f"Integrated Signal (m/z {selected_mz:.1f} ± {half_width})",
        "−ln(depletion)",
    ],
)
fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(int_without), mode="lines",
                             name="Without IR", line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(int_with), mode="lines",
                             name="With IR", line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1)
fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(ln_depletion), mode="lines",
                             name="−ln(depl)", line=dict(color="#d62728", width=2)), row=2, col=1)
fig_ir.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
fig_ir.update_yaxes(title_text="Signal (a.u.)", row=1, col=1)
fig_ir.update_yaxes(title_text="−ln(depl)", row=2, col=1)
fig_ir.update_layout(height=550, showlegend=True, legend=dict(orientation="h", y=1.02))
st.plotly_chart(fig_ir, use_container_width=True)

# Store the extracted spectrum for later use (raw, unsmoothed for independent smoothing in Section 4)
st.session_state["_mid_your_wn"] = wn_arr
st.session_state["_mid_your_intensity"] = ln_depletion
st.session_state["_mid_selected_mz"] = selected_mz

st.success(f"✅ Extracted IR from {n_bins} m/z bins × {len(wn_sorted)} wavenumber steps")

# Quick assessment: is this a real signal?
_max_depl = np.nanmax(np.abs(ln_depletion))
_mean_depl = np.nanmean(np.abs(ln_depletion))
_snr = _max_depl / _mean_depl if _mean_depl > 0 else 0
st.caption(f"Signal assessment: max |−ln(depl)| = {_max_depl:.4f}, mean = {_mean_depl:.4f}, "
           f"peak/mean ratio = {_snr:.1f}×")
if _snr < 3:
    st.warning("⚠️ Low signal-to-noise ratio. This mass channel may not have a real IR signature.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FORMULA MATCHING
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 3. Candidate Formula Matching")
st.caption(f"Finding CH and CHBr candidates for **m/z = {selected_mz:.2f}**")

_fm_col1, _fm_col2, _fm_col3, _fm_col4 = st.columns(4)
with _fm_col1:
    _fm_tol = st.number_input("Mass tolerance (Da)", value=0.5, min_value=0.05, max_value=2.0,
                               step=0.05, key="_mid_fm_tol")
    _fm_min_C = int(st.number_input("Min C", value=1, min_value=1, max_value=50, key="_mid_fm_minC"))
with _fm_col2:
    _fm_max_C = int(st.number_input("Max C", value=30, min_value=_fm_min_C, max_value=100, key="_mid_fm_maxC"))
    _fm_rdb_max = int(st.number_input("Max RDB", value=15, min_value=0, max_value=50, key="_mid_fm_rdb"))
with _fm_col3:
    _fm_hc_min = st.number_input("Min H/C", value=0.3, min_value=0.0, max_value=5.0, step=0.1, key="_mid_hc_min")
    _fm_hc_max = st.number_input("Max H/C", value=2.5, min_value=_fm_hc_min, max_value=5.0, step=0.1, key="_mid_hc_max")
with _fm_col4:
    _fm_inc_br = st.checkbox("Include CHBr candidates", value=True, key="_mid_inc_br")
    _fm_br_iso = st.checkbox("Check ⁸¹Br isotopologue", value=True, key="_mid_br_iso")
    _fm_br_pair = st.checkbox("Require Br isotope pair", value=True, key="_mid_br_pair",
                               help="Reject CHBr if complementary isotopologue peak not detected.")

candidates = generate_candidates(
    selected_mz, tol=_fm_tol,
    min_C=_fm_min_C, max_C=_fm_max_C,
    hc_min=_fm_hc_min, hc_max=_fm_hc_max,
    rdb_max=_fm_rdb_max,
    include_br=_fm_inc_br,
    check_br_isotope=_fm_br_iso,
)

# Br isotope pair validation
if _fm_br_pair and detected_mz_list:
    _det_arr = np.array(detected_mz_list)
    validated = []
    for c in candidates:
        if c["n_Br"] > 0:
            shift = 2.0 * c["n_Br"]
            has_partner = any(
                np.any(np.abs(_det_arr - (selected_mz + d)) <= _fm_tol)
                for d in [+shift, -shift]
            )
            if not has_partner:
                c["extra"] += " ⚠️ no isotope pair"
        validated.append(c)
    candidates = validated

if candidates:
    _ch = [c for c in candidates if "CHBr" not in c["type"]]
    _chbr_ok = [c for c in candidates if "CHBr" in c["type"] and "⚠️" not in c.get("extra", "")]
    _chbr_bad = [c for c in candidates if "CHBr" in c["type"] and "⚠️" in c.get("extra", "")]

    def _cand_df(cands):
        return pd.DataFrame([{
            "Formula": c["formula"],
            "Type": c["type"],
            "Calc Mass": f"{c['calc_mass']:.4f}",
            "Error (Da)": f"{c['mass_error_Da']:+.4f}",
            "RDB": f"{c['RDB']:.1f}",
            "H/C": f"{c['H/C']:.2f}",
            "Notes": c["extra"],
        } for c in cands])

    if _ch:
        st.markdown(f"### CH candidates ({len(_ch)})")
        st.dataframe(_cand_df(_ch), width='stretch', hide_index=True)
    if _chbr_ok:
        st.markdown(f"### CHBr candidates — validated ({len(_chbr_ok)})")
        st.dataframe(_cand_df(_chbr_ok), width='stretch', hide_index=True)
    if _chbr_bad:
        with st.expander(f"CHBr candidates — rejected (no isotope pair, {len(_chbr_bad)})"):
            st.dataframe(_cand_df(_chbr_bad), width='stretch', hide_index=True)

    # Let user pick an assignment
    _all_formulas = [c["formula"] for c in candidates if "⚠️" not in c.get("extra", "")]
    _all_formulas = list(dict.fromkeys(_all_formulas))  # deduplicate preserving order
    if _all_formulas:
        _chosen = st.selectbox("Select working assignment", ["— none —"] + _all_formulas,
                               key="_mid_assign_formula")
        if _chosen != "— none —":
            st.session_state["_mid_chosen_formula"] = _chosen
            st.success(f"Working assignment: **{_chosen}** for m/z {selected_mz:.1f}")
else:
    st.info("No candidate formulas found. Try increasing the tolerance or carbon range.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — REFERENCE SPECTRUM COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 4. Reference Spectrum Comparison")
st.caption(
    "Upload NIST JDX, DFT output, or CSV spectra to compare against your extracted IR."
)

_ref_files = st.file_uploader(
    "Upload reference spectra (JDX / DFT output / CSV)",
    type=["jdx", "dx", "jcamp", "out", "log", "csv", "txt", "dat", "stk"],
    accept_multiple_files=True, key="_mid_ref_upload",
)

ref_spectra = []  # list of {name, wn, intensity}

if _ref_files:
    for _f in _ref_files:
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
                _dname = _title
                if _molform:
                    _dname += f" ({_molform})"
                if _cas:
                    _dname += f" [CAS {_cas}]"
                yunits = _meta.get("YUNITS", "").upper()
                if "TRANSMITTANCE" in yunits:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        _it = -np.log10(np.clip(_it, 1e-6, None))
                    _dname += " (→ abs)"
                ref_spectra.append({"name": _dname, "wn": _wn, "intensity": _it, "source": "NIST"})
                st.success(f"📄 JDX: {_dname} — {len(_wn)} pts")

            elif _ext in (".out", ".log"):
                _text = _raw.decode("utf-8", errors="replace") if isinstance(_raw, bytes) else _raw
                freqs, intens, meta = parse_dft_file(_text, _fname)
                if freqs is not None and len(freqs) > 0:
                    # Broaden the stick spectrum
                    _bw = st.session_state.get("_mid_bw_frac", 0.007)
                    wn_broad, int_broad = broaden_spectrum_felix(freqs, intens, bw_frac=_bw)
                    _dname = f"DFT: {_fname}"
                    if "method" in meta:
                        _dname += f" ({meta['method']})"
                    ref_spectra.append({"name": _dname, "wn": wn_broad, "intensity": int_broad,
                                        "source": "DFT", "freqs": freqs, "intens": intens})
                    st.success(f"🧪 DFT: {_fname} — {len(freqs)} modes → broadened")
                else:
                    st.warning(f"Could not parse DFT modes from {_fname}")

            elif _ext in (".csv", ".txt", ".dat"):
                _df = pd.read_csv(io.BytesIO(_raw) if isinstance(_raw, bytes) else io.StringIO(_raw),
                                  sep=None, engine="python")
                if len(_df.columns) >= 2:
                    _csv_wn = _df.iloc[:, 0].values.astype(float)
                    _csv_int = _df.iloc[:, 1].values.astype(float)
                    # Auto-detect transmittance and convert to absorbance
                    _col2_name = str(_df.columns[1]).lower()
                    if "transmittance" in _col2_name or "trans" in _col2_name or "%t" in _col2_name:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            _csv_int = -np.log10(np.clip(_csv_int / 100.0, 1e-6, None))
                        st.info(f"ℹ️ Detected transmittance column → converted to absorbance (−log₁₀(T))")
                    _sparse = len(_df) < 100  # treat as stick/peak list if < 100 points
                    if _sparse:
                        # Broaden peak list into continuous spectrum
                        _bw = st.session_state.get("_mid_bw_frac", 0.007)
                        _csv_wn_broad, _csv_int_broad = broaden_spectrum_felix(
                            _csv_wn, _csv_int, bw_frac=_bw)
                        ref_spectra.append({
                            "name": f"{_fname} (broadened)",
                            "wn": _csv_wn_broad,
                            "intensity": _csv_int_broad,
                            "source": "CSV",
                            "sparse": False,
                        })
                        st.success(f"📄 CSV: {_fname} — {len(_df)} peaks → broadened to spectrum")
                    else:
                        ref_spectra.append({
                            "name": _fname,
                            "wn": _csv_wn,
                            "intensity": _csv_int,
                            "source": "CSV",
                            "sparse": False,
                        })
                        st.success(f"📄 CSV: {_fname} — {len(_df)} pts (continuous)")

            elif _ext == ".stk":
                _text = _raw.decode("utf-8", errors="replace") if isinstance(_raw, bytes) else _raw
                # Try plain two-column stick format first (freq\tintensity or freq,intensity)
                _stk_freqs, _stk_intens = [], []
                for _line in _text.splitlines():
                    _line = _line.strip()
                    if not _line or _line.startswith("#"):
                        continue
                    _parts = _line.replace(",", "\t").split()
                    if len(_parts) >= 2:
                        try:
                            _stk_freqs.append(float(_parts[0]))
                            _stk_intens.append(float(_parts[1]))
                        except ValueError:
                            continue
                if len(_stk_freqs) > 0:
                    freqs = np.array(_stk_freqs)
                    intens = np.array(_stk_intens)
                else:
                    # Fallback to DFT parser (e.g. ORCA .ir.stk)
                    freqs, intens, _ = parse_dft_file(_text, _fname)
                if freqs is not None and len(freqs) > 0:
                    _bw = st.session_state.get("_mid_bw_frac", 0.007)
                    wn_broad, int_broad = broaden_spectrum_felix(freqs, intens, bw_frac=_bw)
                    ref_spectra.append({"name": f"DFT: {_fname}", "wn": wn_broad, "intensity": int_broad,
                                        "source": "DFT", "freqs": freqs, "intens": intens})
                    st.success(f"🧪 DFT stick: {_fname} — {len(freqs)} modes → broadened")

        except Exception as _e:
            st.warning(f"Could not parse {_fname}: {_e}")

# DFT broadening control
if any(s.get("source") == "DFT" for s in ref_spectra):
    _bw_frac = st.number_input("FELIX bandwidth fraction (FWHM/ν)", value=0.007,
                                min_value=0.001, max_value=0.05, step=0.001,
                                format="%.4f", key="_mid_bw_frac",
                                help="Frequency-proportional Gaussian FWHM = bw_frac × ν")

# ── Spectral overlay plot ─────────────────────────────────────────────────────
your_wn = st.session_state.get("_mid_your_wn")
your_intensity = st.session_state.get("_mid_your_intensity")

if your_wn is not None or ref_spectra:
    _oc1, _oc2, _oc3, _oc4 = st.columns(4)
    with _oc1:
        _normalize = st.checkbox("Normalize to [0,1]", value=True, key="_mid_norm")
    with _oc2:
        _ridge = st.checkbox("Ridge (stacked) layout", value=True, key="_mid_ridge")
        _ridge_gap = 1.0
        if _ridge:
            _ridge_gap = st.slider("Ridge spacing", 0.2, 2.0, 1.0, 0.1, key="_mid_rgap")
    with _oc3:
        _smooth_ref = st.checkbox("Smooth experimental", value=True, key="_mid_smooth_ref",
                                   help="Apply Savitzky-Golay smoothing to your experimental spectrum")
        _smooth_ref_w = 7
        if _smooth_ref:
            _smooth_ref_w = st.number_input("SG window", value=7, min_value=3,
                                             max_value=51, step=2, key="_mid_sg_ref_w")
    with _oc4:
        _show_peaks = st.checkbox("Show reference peaks", value=True, key="_mid_show_peaks")
        _peak_prom_pct = 0.05
        if _show_peaks:
            _peak_prom_pct = st.number_input("Peak prominence (%)", value=5.0,
                                              min_value=0.5, max_value=50.0, step=0.5,
                                              key="_mid_peak_prom_pct") / 100.0

    def _norm(y):
        if not _normalize:
            return y
        a = np.asarray(y, dtype=float)
        f = a[np.isfinite(a)]
        if len(f) == 0 or np.ptp(f) == 0:
            return a
        return (a - np.nanmin(a)) / np.ptp(f)

    def _smooth_spectrum(wn, y):
        """Apply SG smoothing to a reference spectrum."""
        if not _smooth_ref or len(y) < _smooth_ref_w:
            return y
        w = _smooth_ref_w if _smooth_ref_w % 2 == 1 else _smooth_ref_w + 1
        return savgol_filter(np.asarray(y, dtype=float), window_length=w, polyorder=min(3, w - 1))

    # Determine x-axis range from experimental spectrum
    _xmin, _xmax = None, None
    if your_wn is not None:
        _xmin = float(np.nanmin(your_wn))
        _xmax = float(np.nanmax(your_wn))

    fig_cmp = go.Figure()
    _palette = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
                "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    _traces = []

    if your_wn is not None:
        _your_smoothed = _smooth_spectrum(your_wn, your_intensity)
        _traces.append(("Your IR (−ln depl)", your_wn, _norm(_your_smoothed), "#d62728", False, False))
    for _si, _s in enumerate(ref_spectra):
        _c = _palette[(_si + 1) % len(_palette)]
        _ref_wn = np.asarray(_s["wn"])
        _ref_int = np.asarray(_s["intensity"], dtype=float)
        # Clip reference to experimental range
        if _xmin is not None:
            _mask = (_ref_wn >= _xmin) & (_ref_wn <= _xmax)
            _ref_wn = _ref_wn[_mask]
            _ref_int = _ref_int[_mask]
        _is_sparse = _s.get("sparse", False)
        _traces.append((_s["name"], _ref_wn, _norm(_ref_int), _c, True, _is_sparse))

    # unpack experimental trace (5-tuple, no sparse flag)
    _traces[0] = _traces[0] + (False,) if len(_traces[0]) == 5 else _traces[0]

    for _ri, (_tname, _twn, _ty, _tc, _is_ref, _is_sparse) in enumerate(_traces):
        _off = _ri * _ridge_gap if _ridge else 0
        if _is_sparse:
            # Render as vertical sticks (bar)
            fig_cmp.add_trace(go.Bar(
                x=_twn, y=_ty + _off, name=_tname,
                marker_color=_tc, opacity=0.8, width=3,
            ))
        else:
            fig_cmp.add_trace(go.Scatter(
                x=_twn, y=_ty + _off, mode="lines",
                line=dict(color=_tc, width=2), name=_tname,
            ))
        # Peak detection on reference spectra → vertical lines
        if _is_ref and _show_peaks and len(_ty) > 3:
            _prom_val = _peak_prom_pct * float(np.nanmax(_ty) - np.nanmin(_ty))
            _ref_peaks, _ = find_peaks(np.nan_to_num(_ty, nan=0), prominence=max(_prom_val, 1e-6))
            if len(_ref_peaks) > 0:
                for _pi in _ref_peaks:
                    fig_cmp.add_vline(
                        x=float(_twn[_pi]),
                        line=dict(color=_tc, width=1, dash="dot"),
                        opacity=0.5,
                        annotation_text=f"{_twn[_pi]:.0f}",
                        annotation_position="top",
                        annotation=dict(font_size=8, font_color=_tc, textangle=-90),
                    )

    # Limit x-axis to experimental range
    _layout_xrange = None
    if _xmin is not None:
        _layout_xrange = [_xmin, _xmax]

    fig_cmp.update_layout(
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity" + (" (stacked)" if _ridge else ""),
        title=f"Spectral Comparison — m/z {selected_mz:.1f}",
        height=550,
        xaxis=dict(range=_layout_xrange),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Show detected peak positions in a table
    if _show_peaks and ref_spectra:
        _peak_rows = []
        for _si, _s in enumerate(ref_spectra):
            _ref_wn = np.asarray(_s["wn"])
            _ref_int = np.asarray(_s["intensity"], dtype=float)
            if _xmin is not None:
                _mask = (_ref_wn >= _xmin) & (_ref_wn <= _xmax)
                _ref_wn = _ref_wn[_mask]
                _ref_int = _ref_int[_mask]
            _ref_int_n = _norm(_ref_int)
            if len(_ref_int_n) > 3:
                _prom_val = _peak_prom_pct * float(np.nanmax(_ref_int_n) - np.nanmin(_ref_int_n))
                _rp, _rpr = find_peaks(np.nan_to_num(_ref_int_n, nan=0), prominence=max(_prom_val, 1e-6))
                for _pi in _rp:
                    _peak_rows.append({
                        "Reference": _s["name"][:40],
                        "Peak (cm⁻¹)": f"{_ref_wn[_pi]:.1f}",
                        "Rel. Intensity": f"{_ref_int_n[_pi]:.3f}",
                    })
        if _peak_rows:
            with st.expander(f"📍 Detected reference peaks ({len(_peak_rows)})"):
                st.dataframe(pd.DataFrame(_peak_rows), width='stretch', hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MOLECULAR STRUCTURE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 5. Molecular Structure")

_struct_col1, _struct_col2 = st.columns(2)

with _struct_col1:
    st.markdown("**Upload structure image (PNG/SVG)**")
    _struct_img = st.file_uploader("Structure image", type=["png", "jpg", "jpeg", "svg"],
                                    key="_mid_struct_img")
    if _struct_img:
        st.image(_struct_img, caption="Uploaded structure", width=300)
        st.session_state["_mid_structure_img"] = _struct_img.getvalue()

with _struct_col2:
    st.markdown("**Or generate from SMILES**")
    if HAS_RDKIT:
        _smiles = st.text_input("SMILES string", key="_mid_smiles",
                                 placeholder="c1cccc2ccccc12  (naphthalene)")
        if _smiles:
            try:
                mol = Chem.MolFromSmiles(_smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(400, 300))
                    st.image(img, caption=f"SMILES: {_smiles}", width=300)
                    # Save to session
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.session_state["_mid_structure_img"] = buf.getvalue()
                    # Compute properties
                    _mw = Descriptors.ExactMolWt(mol)
                    _mf = Chem.rdMolDescriptors.CalcMolFormula(mol)
                    st.caption(f"**{_mf}** — Exact mass: {_mw:.4f} Da")
                else:
                    st.error("Invalid SMILES")
            except Exception as _e:
                st.error(f"RDKit error: {_e}")
    else:
        st.info("Install `rdkit` to enable SMILES → structure rendering.")


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VERDICT & EXPORT
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 6. Assignment Verdict & Export")

_v_col1, _v_col2 = st.columns(2)
with _v_col1:
    _verdict = st.selectbox("Verdict", ["Confirmed", "Tentative", "Unassigned", "Fragment"],
                             key="_mid_verdict")
with _v_col2:
    _notes = st.text_area("Notes", key="_mid_notes", height=80,
                           placeholder="e.g. matches DFT for 1-ethynylnaphthalene, C-H stretch region excellent")

_chosen_formula = st.session_state.get("_mid_chosen_formula", "")

def _get_default_output_dir():
    """Resolve the default output directory from session state or defaults.ini."""
    _file_dir = st.session_state.get("file_directory", "")
    if not _file_dir:
        _defaults_file = r'./.streamlit/defaults.ini'
        if os.path.exists(_defaults_file):
            _cfg = configparser.ConfigParser()
            _cfg.read(_defaults_file)
            try:
                _file_dir = _cfg.get('Import Data', 'file_directory')
            except configparser.Error:
                pass
    if not _file_dir:
        return ""
    return str(Path(_file_dir) / "output")

_default_out = _get_default_output_dir()
_output_path = st.text_input(
    "Output folder", value=_default_out, key="_mid_output_path",
    help="Where assignment files will be saved. Change to any folder you like.",
)

def _get_output_dir():
    if not _output_path or not _output_path.strip():
        return None
    return Path(_output_path.strip())

if st.button("💾 Save Assignment", type="primary", key="_mid_save"):
    _entry = {
        "mz": float(selected_mz),
        "formula": _chosen_formula,
        "verdict": _verdict,
        "notes": _notes,
        "timestamp": datetime.now().isoformat(),
    }
    st.session_state["_mass_assignments"][f"{selected_mz:.2f}"] = _entry

    # ── Persist to disk ──
    _out = _get_output_dir()
    if _out is None:
        st.warning("⚠️ No output directory configured — saved to session only.")
    else:
        _safe_mz = f"{selected_mz:.1f}".replace(".", "p")
        _safe_form = (_chosen_formula or "unknown").replace(" ", "_")
        if ref_spectra:
            _ref_tag = "_".join(
                os.path.splitext(s["name"])[0].replace("/", "-").replace("\\", "-").replace(" ", "_")[:30]
                for s in ref_spectra
            )[:80]
        else:
            _ref_tag = "no_ref"
        _assign_dir = _out / "assignments" / f"mz{_safe_mz}_{_safe_form}_{_ref_tag}"
        _assign_dir.mkdir(parents=True, exist_ok=True)

        _saved_files = []

        # 1) Combined CSV: experimental + interpolated references on same grid
        _exp_wn = st.session_state.get("_mid_your_wn")
        _exp_int = st.session_state.get("_mid_your_intensity")
        if _exp_wn is not None:
            _combined = pd.DataFrame({
                "wavenumber_cm-1": _exp_wn,
                "experimental_neg_ln_depl": _exp_int,
            })
            # Interpolate each reference onto the experimental wavenumber grid
            for _ri, _rs in enumerate(ref_spectra):
                _rname = _rs["name"].replace("/", "-").replace("\\", "-").replace(" ", "_")[:50]
                _rwn = np.asarray(_rs["wn"], dtype=float)
                _rint = np.asarray(_rs["intensity"], dtype=float)
                _interp = np.interp(_exp_wn, _rwn, _rint, left=0.0, right=0.0)
                _combined[f"ref{_ri+1}_{_rname}"] = _interp
            _spectra_fname = f"mz{_safe_mz}_{_safe_form}_{_verdict}.csv"
            _combined_path = _assign_dir / _spectra_fname
            _combined.to_csv(_combined_path, index=False)
            _saved_files.append(_spectra_fname)

        # 2) DFT stick spectra (raw frequencies) if available
        for _ri, _rs in enumerate(ref_spectra):
            if _rs.get("freqs") is not None:
                _rname = _rs["name"].replace("/", "-").replace("\\", "-").replace(" ", "_")[:50]
                _stick_df = pd.DataFrame({
                    "frequency_cm-1": _rs["freqs"],
                    "intensity_km_mol": _rs["intens"],
                })
                _stick_path = _assign_dir / f"ref{_ri+1}_{_rname}_sticks.csv"
                _stick_df.to_csv(_stick_path, index=False)
                _saved_files.append(str(_stick_path.name))

        # 3) Comparison plot PNG
        if _exp_wn is not None:
            try:
                _fig_s, _ax_s = plt.subplots(figsize=(10, 5))
                _yr_s = _exp_int / np.nanmax(np.abs(_exp_int)) if np.nanmax(np.abs(_exp_int)) > 0 else _exp_int
                _ax_s.plot(_exp_wn, _yr_s, color="#d62728", lw=2, label="Experimental")
                for _ri, _rs in enumerate(ref_spectra):
                    _c = _palette[(_ri + 1) % len(_palette)]
                    _rwn = np.asarray(_rs["wn"])
                    _rint = np.asarray(_rs["intensity"], dtype=float)
                    _rint_n = _rint / np.nanmax(np.abs(_rint)) if np.nanmax(np.abs(_rint)) > 0 else _rint
                    _ax_s.plot(_rwn, _rint_n, color=_c, lw=1.5, label=_rs["name"][:40])
                _ax_s.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
                _ax_s.set_ylabel("Normalized Intensity", fontsize=11)
                _ax_s.set_title(f"m/z {selected_mz:.1f} — {_chosen_formula or '?'} ({_verdict})",
                                fontsize=12, fontweight="bold")
                _ax_s.legend(fontsize=8)
                _ax_s.grid(True, alpha=0.3)
                _fig_s.tight_layout()
                _plot_path = _assign_dir / "comparison_plot.png"
                _fig_s.savefig(_plot_path, dpi=300, bbox_inches="tight")
                plt.close(_fig_s)
                _saved_files.append("comparison_plot.png")
            except Exception as _pe:
                st.warning(f"⚠️ Could not save plot: {_pe}")

        # 4) Structure image if available
        if "_mid_structure_img" in st.session_state:
            _struct_path = _assign_dir / "structure.png"
            _struct_path.write_bytes(st.session_state["_mid_structure_img"])
            _saved_files.append("structure.png")

        st.success(
            f"✅ Saved m/z {selected_mz:.1f} → {_chosen_formula} ({_verdict})\n\n"
            f"📁 `{_assign_dir}`\n\n"
            f"Files: {', '.join(_saved_files)}"
        )

    # ── Save master assignments JSON (persistent across sessions) ──
    _json_p = _assignments_json_path()
    if _json_p is not None:
        _json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(_json_p, "w") as _jf:
            json.dump(st.session_state["_mass_assignments"], _jf, indent=2)
    # Also save CSV copy for easy viewing
    _out2 = _get_output_dir()
    if _out2 is not None:
        _all_assign = st.session_state.get("_mass_assignments", {})
        if _all_assign:
            _master_df = pd.DataFrame(list(_all_assign.values()))
            _master_path = _out2 / "assignments" / "all_assignments.csv"
            _master_path.parent.mkdir(parents=True, exist_ok=True)
            _master_df.to_csv(_master_path, index=False)

# Show all assignments
_assignments = st.session_state.get("_mass_assignments", {})
if _assignments:
    st.markdown("### All Assignments")
    _adf = pd.DataFrame(list(_assignments.values()))
    st.dataframe(_adf, width='stretch', hide_index=True)

    # Export
    _csv_buf = _adf.to_csv(index=False)
    st.download_button("📥 Download assignments CSV", data=_csv_buf,
                        file_name=f"mass_assignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv", key="_mid_dl_csv")

# ── Composite figure export ──────────────────────────────────────────────────
if your_wn is not None:
    st.markdown("### Publication Figure Export")
    with st.expander("📐 Figure settings"):
        _fig_w = st.number_input("Width (inches)", value=10.0, step=0.5, key="_mid_fig_w")
        _fig_h = st.number_input("Height (inches)", value=8.0, step=0.5, key="_mid_fig_h")
        _fig_dpi = st.number_input("DPI", value=300, step=50, key="_mid_fig_dpi")

    if st.button("🖼️ Generate Composite Figure", key="_mid_gen_fig"):
        has_struct = "_mid_structure_img" in st.session_state

        if has_struct:
            fig_pub, axes = plt.subplots(2, 2, figsize=(_fig_w, _fig_h),
                                          gridspec_kw={"width_ratios": [3, 1], "height_ratios": [1, 1]})
            ax_ms = axes[0, 0]
            ax_struct = axes[0, 1]
            ax_ir = axes[1, 0]
            ax_info = axes[1, 1]
        else:
            fig_pub, axes = plt.subplots(2, 1, figsize=(_fig_w, _fig_h))
            ax_ms = axes[0]
            ax_ir = axes[1]
            ax_struct = None
            ax_info = None

        # Panel 1: Mass spectrum with selected mass highlighted
        ax_ms.plot(_x_disp, _y_disp, color="#1f77b4", lw=1)
        ax_ms.axvline(selected_mz, color="red", ls="--", lw=1, alpha=0.7)
        ax_ms.axvspan(selected_mz - half_width, selected_mz + half_width,
                       color="red", alpha=0.1)
        ax_ms.set_xlabel("m/z", fontsize=11)
        ax_ms.set_ylabel("Intensity (a.u.)", fontsize=11)
        ax_ms.set_title("Mass Spectrum", fontsize=12, fontweight="bold")
        ax_ms.annotate(f"m/z {selected_mz:.1f}", xy=(selected_mz, 0),
                        xytext=(selected_mz + 5, np.nanmax(_y_disp) * 0.8),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        fontsize=10, color="red")

        # Panel 2: IR spectrum + references (matching Section 4 settings)
        _pub_exp = _smooth_spectrum(your_wn, your_intensity) if _smooth_ref else your_intensity
        _yr = _norm(_pub_exp) if _normalize else _pub_exp
        _pub_off_0 = 0  # experimental is trace index 0
        ax_ir.plot(your_wn, _yr + (_pub_off_0 * _ridge_gap if _ridge else 0),
                   color="#d62728", lw=2, label="Experimental")
        _pub_xmin, _pub_xmax = float(np.nanmin(your_wn)), float(np.nanmax(your_wn))
        for _si, _s in enumerate(ref_spectra):
            _c = _palette[(_si + 1) % len(_palette)]
            _rwn = np.asarray(_s["wn"])
            _rint = np.asarray(_s["intensity"], dtype=float)
            # Clip to experimental range
            _rmask = (_rwn >= _pub_xmin) & (_rwn <= _pub_xmax)
            _rwn = _rwn[_rmask]
            _rint = _rint[_rmask]
            _ry = _norm(_rint) if _normalize else _rint
            _pub_off = (_si + 1) * _ridge_gap if _ridge else 0
            ax_ir.plot(_rwn, _ry + _pub_off, color=_c, lw=1.5, label=_s["name"][:40])
            # Peak vertical lines on reference
            if _show_peaks and len(_ry) > 3:
                _pv = _peak_prom_pct * float(np.nanmax(_ry) - np.nanmin(_ry))
                _rpeaks, _ = find_peaks(np.nan_to_num(_ry, nan=0), prominence=max(_pv, 1e-6))
                for _pi in _rpeaks:
                    ax_ir.axvline(_rwn[_pi], color=_c, ls=":", lw=0.8, alpha=0.5)
                    ax_ir.annotate(f"{_rwn[_pi]:.0f}", xy=(_rwn[_pi], _ry[_pi] + _pub_off),
                                   fontsize=6, color=_c, ha="center", va="bottom", rotation=90)
        ax_ir.set_xlim(_pub_xmin, _pub_xmax)
        ax_ir.set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
        ax_ir.set_ylabel("Intensity (norm.)" if _normalize else "Intensity", fontsize=11)
        ax_ir.set_title(f"IR Spectrum — m/z {selected_mz:.1f}", fontsize=12, fontweight="bold")

        # Panel 3: Structure image
        if ax_struct is not None and has_struct:
            from PIL import Image
            _img_data = st.session_state["_mid_structure_img"]
            _pil = Image.open(io.BytesIO(_img_data))
            ax_struct.imshow(_pil)
            ax_struct.axis("off")
            ax_struct.set_title("Structure", fontsize=12, fontweight="bold")

        # Panel 4: Info text (+ legend under it)
        if ax_info is not None:
            ax_info.axis("off")
            _info_lines = [
                f"m/z: {selected_mz:.2f}",
                f"Formula: {_chosen_formula or '—'}",
                f"Verdict: {_verdict}",
                f"RDB: {calc_rdb(*[c['n_C'] for c in candidates if c['formula'] == _chosen_formula][:1] or [0], *[c['n_H'] for c in candidates if c['formula'] == _chosen_formula][:1] or [0]):.1f}" if _chosen_formula else "",
            ]
            if _notes:
                _info_lines.append(f"\nNotes: {_notes}")
            ax_info.text(0.05, 0.40, "\n".join(_info_lines), transform=ax_info.transAxes,
                         fontsize=10, va="top", ha="left", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            # Legend under info text
            ax_info.legend(*ax_ir.get_legend_handles_labels(), loc="upper left",
                          fontsize=7, frameon=True, bbox_to_anchor=(0.0, 0.0))
        else:
            # No info panel: legend on IR panel
            ax_ir.legend(fontsize=8, loc="upper right")

        fig_pub.tight_layout()

        # Save and offer download
        _buf = io.BytesIO()
        fig_pub.savefig(_buf, format="png", dpi=_fig_dpi, bbox_inches="tight")
        _buf.seek(0)
        st.image(_buf, caption="Composite figure preview", width='stretch')

        _safe_mz = f"{selected_mz:.0f}"
        _safe_form = _chosen_formula.replace(" ", "_") if _chosen_formula else "unknown"
        _pub_fname = f"mass_identity_mz{_safe_mz}_{_safe_form}_{datetime.now().strftime('%Y%m%d')}.png"
        st.download_button("📥 Download Figure", data=_buf.getvalue(),
                            file_name=_pub_fname, mime="image/png", key="_mid_dl_fig")
        plt.close(fig_pub)
