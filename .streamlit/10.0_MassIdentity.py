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
# AXIS HELPERS — dense ticks with an eV value under every label
# ════════════════════════════════════════════════════════════════════════════════
_CM1_TO_EV = 1.239841984e-4   # eV per cm⁻¹  (E[eV] = ν̃[cm⁻¹] × hc/e)
_NM_EV = 1239.841984          # E[eV] = _NM_EV / λ[nm]


def _dense_tickvals(lo, hi, spacing):
    """Tick positions from lo→hi on a multiple of `spacing`."""
    if hi < lo:
        lo, hi = hi, lo
    _start = np.ceil(lo / spacing) * spacing
    return np.arange(_start, hi + spacing * 1e-6, spacing)


def _wn_ticktext(vals):
    """Two-line labels: wavenumber over its eV equivalent (for Plotly)."""
    return [f"{v:.0f}<br>{v * _CM1_TO_EV:.3f} eV" for v in vals]


def _wl_ticktext(vals):
    """Two-line labels: wavelength over its photon-energy eV (for Plotly)."""
    return [(f"{v:.1f}<br>{_NM_EV / v:.2f} eV" if v > 0 else f"{v:.1f}")
            for v in vals]


def _apply_wn_plotly(fig, lo, hi, spacing=50.0, **kw):
    """Wavenumber x-axis: ticks every `spacing` cm⁻¹ (plain labels, no eV)."""
    _vals = _dense_tickvals(lo, hi, spacing)
    fig.update_xaxes(tickmode="array", tickvals=_vals,
                     ticktext=[f"{v:.0f}" for v in _vals],
                     tickfont=dict(size=9), **kw)


def _apply_wl_plotly(fig, lo, hi, **kw):
    """Wavelength x-axis: adaptive-density ticks with eV under each nm."""
    _span = abs(hi - lo)
    if _span <= 10:
        _spacing = 0.5
    elif _span <= 25:
        _spacing = 1.0
    elif _span <= 50:
        _spacing = 2.0
    else:
        _spacing = 5.0
    _vals = _dense_tickvals(lo, hi, _spacing)
    fig.update_xaxes(tickmode="array", tickvals=_vals, ticktext=_wl_ticktext(_vals),
                     tickfont=dict(size=8), **kw)


def _apply_wn_mpl(ax, spacing=50.0, fontsize=7):
    """Matplotlib wavenumber axis: ticks every `spacing` cm⁻¹ (plain labels)."""
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    ax.xaxis.set_major_locator(MultipleLocator(spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(spacing / 2.0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0f}"))
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)


def _apply_wl_mpl(ax, fontsize=7):
    """Matplotlib wavelength axis: adaptive-density ticks + eV under each.
    Call AFTER set_xlim so the tick range is known."""
    from matplotlib.ticker import MultipleLocator, FuncFormatter
    _lo, _hi = ax.get_xlim()
    _span = abs(_hi - _lo)
    if _span <= 10:
        _spacing = 0.5
    elif _span <= 25:
        _spacing = 1.0
    elif _span <= 50:
        _spacing = 2.0
    else:
        _spacing = 5.0
    ax.xaxis.set_major_locator(MultipleLocator(_spacing))
    ax.xaxis.set_minor_locator(MultipleLocator(_spacing / 2.0))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _p: (f"{x:.1f}\n{_NM_EV / x:.2f} eV"
                                     if x > 0 else f"{x:.1f}")))
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.set_xlabel("Wavelength (nm)  /  Energy (eV)", fontsize=10)

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

    # Convert wavelength units to wavenumbers if necessary
    _xunits = meta.get("XUNITS", "").upper().replace("-", "")
    if _xunits in ("MICROMETERS", "MICRON", "MICRONS", "UM"):
        x_values = 10000.0 / x_values
        # Sort into ascending wavenumber order
        _order = np.argsort(x_values)
        x_values = x_values[_order]
        y_values = [y_values[i] for i in _order]

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
    _sg_w = st.number_input("SG window", value=7, min_value=3, max_value=101, step=2,
                            key="_mid_sg_w", disabled=not _sg_on)

# ── Isotope (multi-peak) integration ─────────────────────────────────────────
# Elements like Br (⁷⁹Br/⁸¹Br) and Cl (³⁵Cl/³⁷Cl) split a fragment into several
# isotopologue peaks separated by ~2 amu. Summing all of them recovers the full
# ion signal and improves the depletion S/N.
_iso_c1, _iso_c2, _iso_c3 = st.columns([1, 1, 1])
with _iso_c1:
    _iso_on = st.checkbox("Integrate isotopic peaks", value=False, key="_mid_iso_on",
                          help="Sum several isotopologue peaks (e.g. ⁷⁹Br + ⁸¹Br) into one IR trace.")
with _iso_c2:
    _iso_spacing = st.number_input("Isotope spacing (amu)", value=2.0, min_value=0.5,
                                   max_value=10.0, step=0.5, key="_mid_iso_sp",
                                   disabled=not _iso_on,
                                   help="Mass gap between adjacent isotope peaks (Br/Cl → 2).")
with _iso_c3:
    _iso_n = int(st.number_input("Number of peaks", value=2, min_value=1, max_value=8,
                                 step=1, key="_mid_iso_n", disabled=not _iso_on,
                                 help="How many peaks to sum, starting from the selected m/z upward."))

# Build the list of peak centres to integrate
if _iso_on and _iso_n > 1:
    iso_centers = [selected_mz + k * _iso_spacing for k in range(_iso_n)]
else:
    iso_centers = [selected_mz]

def _build_iso_mask(_centers, _hw):
    _xm = np.asarray(x_mass)
    _m = np.zeros(len(_xm), dtype=bool)
    for _c in _centers:
        _m |= (_xm >= _c - _hw) & (_xm <= _c + _hw)
    return _m

# Extract IR spectrum for the selected mass
wn_sorted = sorted(unique_wavenumbers)
wn_arr = np.array(wn_sorted, dtype=float)
sel_mask = _build_iso_mask(iso_centers, half_width)
n_bins = int(sel_mask.sum())

if n_bins == 0:
    st.error(f"No m/z bins within ±{half_width} of {selected_mz:.1f}")
    st.stop()

if _iso_on and _iso_n > 1:
    st.caption(
        "🧬 Integrating isotopic peaks at m/z "
        + ", ".join(f"{_c:.1f}" for _c in iso_centers)
        + f" (±{half_width} each)"
    )

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
        (f"Integrated Signal ({len(iso_centers)} isotopic peaks @ "
         + "/".join(f"{_c:.0f}" for _c in iso_centers) + f", ±{half_width})")
        if (_iso_on and _iso_n > 1) else
        f"Integrated Signal (m/z {selected_mz:.1f} ± {half_width})",
        "−ln(depletion)",
    ],
)
# Raw (non-smoothed) lines drawn first in grey so they sit underneath the smoothed traces
if _sg_on:
    fig_ir.add_trace(go.Scatter(x=wn_arr, y=int_without, mode="lines",
                                 name="Without IR (raw)", line=dict(color="#c9c9c9", width=1)), row=1, col=1)
    fig_ir.add_trace(go.Scatter(x=wn_arr, y=int_with, mode="lines",
                                 name="With IR (raw)", line=dict(color="#dcdcdc", width=1)), row=1, col=1)
    fig_ir.add_trace(go.Scatter(x=wn_arr, y=ln_depletion, mode="lines",
                                 name="−ln(depl) (raw)", line=dict(color="#c9c9c9", width=1)), row=2, col=1)

fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(int_without), mode="lines",
                             name="Without IR" + (" (smoothed)" if _sg_on else ""),
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(int_with), mode="lines",
                             name="With IR" + (" (smoothed)" if _sg_on else ""),
                             line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1)
fig_ir.add_trace(go.Scatter(x=wn_arr, y=_smooth(ln_depletion), mode="lines",
                             name="−ln(depl)" + (" (smoothed)" if _sg_on else ""),
                             line=dict(color="#d62728", width=2)), row=2, col=1)
fig_ir.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
fig_ir.update_yaxes(title_text="Signal (a.u.)", row=1, col=1)
fig_ir.update_yaxes(title_text="−ln(depl)", row=2, col=1)
fig_ir.update_layout(height=550, showlegend=True, legend=dict(orientation="h", y=1.02))
# Ticks every 50 cm⁻¹ (plain labels, no eV for IR axes)
if len(wn_arr) > 1:
    _apply_wn_plotly(fig_ir, float(np.nanmin(wn_arr)), float(np.nanmax(wn_arr)),
                     spacing=50.0, row=2, col=1)
st.plotly_chart(fig_ir, use_container_width=True)

# Store the extracted spectrum for later use (raw, unsmoothed for independent smoothing in Section 4)
st.session_state["_mid_your_wn"] = wn_arr
st.session_state["_mid_your_intensity"] = ln_depletion
st.session_state["_mid_int_without"] = int_without
st.session_state["_mid_int_with"] = int_with
st.session_state["_mid_selected_mz"] = selected_mz

# ── Download extracted IR spectrum (raw + smoothed) ───────────────────────────
_dl_df = pd.DataFrame({
    "wavenumber_cm-1": wn_arr,
    "int_without_IR": int_without,
    "int_with_IR": int_with,
    "neg_ln_depletion": ln_depletion,
})
if _sg_on:
    _dl_df["int_without_IR_smoothed"] = _smooth(int_without)
    _dl_df["int_with_IR_smoothed"] = _smooth(int_with)
    _dl_df["neg_ln_depletion_smoothed"] = _smooth(ln_depletion)

_iso_tag = f"_iso{len(iso_centers)}" if (_iso_on and _iso_n > 1) else ""
_sg_tag = f"_sg{_sg_w}" if _sg_on else "_raw"
st.download_button(
    "📥 Download IR spectrum (CSV)",
    data=_dl_df.to_csv(index=False).encode("utf-8"),
    file_name=f"IR_mz{selected_mz:.1f}_hw{half_width}{_iso_tag}{_sg_tag}.csv",
    mime="text/csv",
    key="_mid_dl_ir",
    help="Raw columns always included; smoothed columns added when SG smoothing is on.",
)

st.success(f"✅ Extracted IR from {n_bins} m/z bins × {len(wn_sorted)} wavenumber steps")

# Quick assessment: is this a real signal?
_max_depl = np.nanmax(np.abs(ln_depletion))
_mean_depl = np.nanmean(np.abs(ln_depletion))
_snr = _max_depl / _mean_depl if _mean_depl > 0 else 0
st.caption(f"Signal assessment: max |−ln(depl)| = {_max_depl:.4f}, mean = {_mean_depl:.4f}, "
           f"peak/mean ratio = {_snr:.1f}×")
if _snr < 3:
    st.warning("⚠️ Low signal-to-noise ratio. This mass channel may not have a real IR signature.")

# ── Band integration & feature robustness ─────────────────────────────────────
with st.expander("📐 Band Integration & Feature Robustness — how much depletion is real?"):
    st.caption(
        "Integrates whole features instead of matching individual peaks. "
        "**All numbers are computed on the raw −ln(depletion)** — the light outline "
        "smoothing below is used *only* to find where a band starts and ends, never "
        "for areas or significance. The noise estimate comes from how much neighbouring "
        "raw points differ, so the 'is it real?' score doesn't inherit the false "
        "confidence (or fake peaks) that smoothing can introduce."
    )

    _bi_c1, _bi_c2, _bi_c3 = st.columns(3)
    with _bi_c1:
        _bi_k = st.number_input("Detection threshold (× noise σ)", value=1.5,
                                min_value=0.5, max_value=5.0, step=0.25, key="_mid_bi_k",
                                help="A band is outlined where the signal stays above this many σ.")
    with _bi_c2:
        _bi_dw = int(st.number_input("Outline window (pts — detection only)", value=1,
                                     min_value=1, max_value=31, step=2, key="_mid_bi_dw",
                                     help="Light moving average used ONLY to outline band edges. "
                                          "Set to 1 to detect on fully raw data."))
    with _bi_c3:
        _bi_merge = st.number_input("Merge gaps smaller than (cm⁻¹)", value=15.0,
                                    min_value=0.0, max_value=100.0, step=5.0, key="_mid_bi_merge")

    _bi_y = np.asarray(ln_depletion, dtype=float)
    _bi_y = np.nan_to_num(_bi_y, nan=0.0)

    # Noise σ of the RAW trace, from point-to-point differences (median-based, so
    # real bands — which vary slowly — barely bias it). No smoothing involved.
    _bi_d = np.diff(_bi_y)
    _bi_sigma = 1.4826 * float(np.nanmedian(np.abs(_bi_d - np.nanmedian(_bi_d)))) / np.sqrt(2)

    if _bi_sigma <= 0 or len(_bi_y) < 5:
        st.warning("Not enough data (or zero noise estimate) to run band analysis.")
    else:
        st.caption(f"Raw noise floor: σ = {_bi_sigma:.4f} in −ln(depl) "
                   f"(≈ {(1 - np.exp(-_bi_sigma)) * 100:.1f}% depletion per point).")

        # Outline (detection only) — plain moving average, or raw if window = 1
        if _bi_dw > 1:
            _bi_outline = np.convolve(_bi_y, np.ones(_bi_dw) / _bi_dw, mode="same")
        else:
            _bi_outline = _bi_y

        # Contiguous regions above threshold
        _bi_above = _bi_outline > _bi_k * _bi_sigma
        _bi_segs = []
        _bi_start = None
        for _i, _flag in enumerate(_bi_above):
            if _flag and _bi_start is None:
                _bi_start = _i
            elif not _flag and _bi_start is not None:
                _bi_segs.append([_bi_start, _i - 1])
                _bi_start = None
        if _bi_start is not None:
            _bi_segs.append([_bi_start, len(_bi_above) - 1])

        # Merge segments separated by small gaps
        _bi_merged = []
        for _seg in _bi_segs:
            if _bi_merged and (wn_arr[_seg[0]] - wn_arr[_bi_merged[-1][1]]) < _bi_merge:
                _bi_merged[-1][1] = _seg[1]
            else:
                _bi_merged.append(_seg)
        # Drop bands with too few points for a meaningful area
        _bi_merged = [s for s in _bi_merged if (s[1] - s[0] + 1) >= 3]

        if not _bi_merged:
            st.info("No bands found above the threshold — try lowering it.")
        else:
            _bi_rows = []
            for _i0, _i1 in _bi_merged:
                _sl = slice(_i0, _i1 + 1)
                _N = _i1 - _i0 + 1
                _dnu = (wn_arr[_i1] - wn_arr[_i0]) / max(_N - 1, 1)

                # Area and its uncertainty — raw points, independent-noise model
                _A = float(np.trapz(_bi_y[_sl], wn_arr[_sl]))
                _sA = _bi_sigma * _dnu * np.sqrt(_N)
                _z = _A / _sA if _sA > 0 else 0.0

                # Physically meaningful % depletion: integrated raw ion counts
                _swo = float(int_without[_sl].sum())
                _swi = float(int_with[_sl].sum())
                _d_pct = (1.0 - _swi / _swo) * 100.0 if _swo > 0 else float("nan")

                _ctr = float(wn_arr[_i0 + int(np.argmax(_bi_y[_sl]))])
                _verdict = "✅ robust" if _z > 5 else ("⚠️ marginal" if _z > 2 else "❌ noise")
                _bi_rows.append({
                    "Band (cm⁻¹)": f"{wn_arr[_i0]:.0f}–{wn_arr[_i1]:.0f}",
                    "Center (cm⁻¹)": f"{_ctr:.0f}",
                    "Depletion (%)": f"{_d_pct:.1f}",
                    "Area (cm⁻¹)": f"{_A:.3f} ± {_sA:.3f}",
                    "z": f"{_z:.1f}",
                    "Verdict": _verdict,
                    "_z_num": _z, "_i0": _i0, "_i1": _i1, "_d": _d_pct,
                })
            _bi_rows.sort(key=lambda r: -r["_z_num"])

            # Persist detected bands (wn ranges + verdict) for use in Section 4
            st.session_state["_mid_bands"] = [
                {
                    "lo": float(wn_arr[_r["_i0"]]),
                    "hi": float(wn_arr[_r["_i1"]]),
                    "center": float(wn_arr[_r["_i0"] + int(np.argmax(_bi_y[_r["_i0"]:_r["_i1"] + 1]))]),
                    "z": float(_r["_z_num"]),
                    "depletion_pct": float(_r["_d"]),
                    "verdict": _r["Verdict"],
                }
                for _r in _bi_rows
            ]
            st.session_state["_mid_bands_mz"] = float(selected_mz)

            # Plot: raw trace with shaded, verdict-coloured bands
            _bi_fig = go.Figure()
            _bi_fig.add_trace(go.Scatter(x=wn_arr, y=_bi_y, mode="lines", name="−ln(depl) raw",
                                         line=dict(color="#888888", width=1)))
            if _bi_dw > 1:
                _bi_fig.add_trace(go.Scatter(x=wn_arr, y=_bi_outline, mode="lines",
                                             name=f"outline (w={_bi_dw}, detection only)",
                                             line=dict(color="#1f77b4", width=1, dash="dot")))
            _bi_fig.add_hline(y=_bi_k * _bi_sigma, line_width=1, line_dash="dash",
                              line_color="black",
                              annotation_text=f"{_bi_k:g}σ", annotation_position="top left")
            _bi_colors = {"✅ robust": "rgba(44,160,44,0.25)",
                          "⚠️ marginal": "rgba(255,165,0,0.25)",
                          "❌ noise": "rgba(214,39,40,0.15)"}
            for _r in _bi_rows:
                _bi_fig.add_vrect(x0=wn_arr[_r["_i0"]], x1=wn_arr[_r["_i1"]],
                                  fillcolor=_bi_colors[_r["Verdict"]], line_width=0,
                                  annotation_text=f"{_r['_d']:.0f}% (z={_r['_z_num']:.1f})",
                                  annotation_position="top", annotation_font_size=9)
            _bi_fig.update_layout(height=380, xaxis_title="Wavenumber (cm⁻¹)",
                                  yaxis_title="−ln(depl)",
                                  legend=dict(orientation="h", y=1.05))
            _apply_wn_plotly(_bi_fig, float(np.nanmin(wn_arr)), float(np.nanmax(wn_arr)),
                             spacing=50.0)
            st.plotly_chart(_bi_fig, use_container_width=True)

            _bi_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                                   for r in _bi_rows])
            st.dataframe(_bi_df, hide_index=True, use_container_width=True)
            st.caption(
                "Focus interpretation on ✅ bands (z > 5); treat ⚠️ (2 < z < 5) as tentative. "
                "The z-score is valid because it is computed on raw, uncorrelated points. "
                "Note: only the *edges* of each band depend on the detection outline — if a "
                "⚠️ band disappears when you set the outline window to 1, don't trust it."
            )
            st.download_button(
                "📥 Download band table (CSV)",
                data=_bi_df.to_csv(index=False).encode("utf-8"),
                file_name=f"band_integration_mz{selected_mz:.1f}.csv", mime="text/csv",
                key="_mid_bi_dl",
            )

# ════════════════════════════════════════════════════════════════════════════════
# BATCH MASS VERDICT SUMMARY — run the same band test on every detected mass
# ════════════════════════════════════════════════════════════════════════════════

with st.expander("Generate verdicts for all detected masses", expanded=False):
    _bv_c1, _bv_c2, _bv_c3 = st.columns(3)
    with _bv_c1:
        _bv_hw = st.number_input("Half-width (amu)", value=0.3, min_value=0.05,
                                 max_value=2.0, step=0.05, key="_mid_bv_hw")
        _bv_k = st.number_input("Detection threshold (× σ)", value=1.5, min_value=0.5,
                                max_value=5.0, step=0.25, key="_mid_bv_k")
    with _bv_c2:
        _bv_dw = int(st.number_input("Outline window (pts)", value=1, min_value=1,
                                     max_value=31, step=2, key="_mid_bv_dw"))
        _bv_merge = st.number_input("Merge gaps (cm⁻¹)", value=15.0, min_value=0.0,
                                    max_value=100.0, step=5.0, key="_mid_bv_merge")
    with _bv_c3:
        _bv_iso = st.checkbox("Integrate isotopic peaks", value=False, key="_mid_bv_iso")
        _bv_iso_sp = st.number_input("Isotope spacing (amu)", value=2.0, min_value=0.5,
                                     max_value=10.0, step=0.5, key="_mid_bv_iso_sp",
                                     disabled=not _bv_iso)
        _bv_iso_n = int(st.number_input("Number of peaks", value=2, min_value=1, max_value=8,
                                        step=1, key="_mid_bv_iso_n", disabled=not _bv_iso))

    if not detected_mz_list:
        st.warning("No detected masses yet. Adjust peak detection in Section 1.")

    if st.button("▶️ Run batch verdict", key="_mid_bv_run", disabled=not bool(detected_mz_list)):
        _bv_rows = []
        _wn_arr_bv = np.array(sorted(unique_wavenumbers), dtype=float)
        _n_total = len(detected_mz_list)
        _prog = st.progress(0.0, text="Evaluating masses...")
        for _j, _mz in enumerate(detected_mz_list):
            _centers_bv = ([_mz + _k * _bv_iso_sp for _k in range(_bv_iso_n)]
                           if (_bv_iso and _bv_iso_n > 1) else [_mz])
            _mask_bv = _build_iso_mask(_centers_bv, _bv_hw)

            if not _mask_bv.any():
                _bv_rows.append({
                    "m/z": _mz,
                    "Verdict": "no bins",
                    "max z": 0.0,
                    "# bands": 0,
                    "# robust": 0,
                    "Best band (cm⁻¹)": "",
                    "Depletion (%)": float("nan"),
                    "noise σ": float("nan"),
                    "max |−ln(depl)|": float("nan"),
                })
                _prog.progress((_j + 1) / _n_total)
                continue

            _v_wo = np.zeros(len(_wn_arr_bv))
            _v_wi = np.zeros(len(_wn_arr_bv))
            for _ii, _wn in enumerate(_wn_arr_bv):
                _df_bv = compilation[_wn]
                _v_wo[_ii] = _df_bv.iloc[_mask_bv, plot_col_wo].values.sum()
                _v_wi[_ii] = _df_bv.iloc[_mask_bv, plot_col_w].values.sum()

            with np.errstate(divide="ignore", invalid="ignore"):
                _v_ln = -np.log(np.clip(_v_wi / _v_wo, 1e-10, None))
            _v_ln = np.nan_to_num(_v_ln, nan=0.0)

            _v_d = np.diff(_v_ln)
            _v_sigma = (1.4826 * float(np.nanmedian(np.abs(_v_d - np.nanmedian(_v_d))))
                        / np.sqrt(2))

            if _v_sigma <= 0 or len(_v_ln) < 5:
                _bv_rows.append({
                    "m/z": _mz,
                    "Verdict": "no data",
                    "max z": 0.0,
                    "# bands": 0,
                    "# robust": 0,
                    "Best band (cm⁻¹)": "",
                    "Depletion (%)": float("nan"),
                    "noise σ": _v_sigma,
                    "max |−ln(depl)|": float("nan"),
                })
                _prog.progress((_j + 1) / _n_total)
                continue

            if _bv_dw > 1:
                _v_outline = np.convolve(_v_ln, np.ones(_bv_dw) / _bv_dw, mode="same")
            else:
                _v_outline = _v_ln

            _v_above = _v_outline > _bv_k * _v_sigma
            _v_segs = []
            _v_start = None
            for _vi, _flag in enumerate(_v_above):
                if _flag and _v_start is None:
                    _v_start = _vi
                elif not _flag and _v_start is not None:
                    _v_segs.append([_v_start, _vi - 1])
                    _v_start = None
            if _v_start is not None:
                _v_segs.append([_v_start, len(_v_above) - 1])

            _v_merged = []
            for _seg in _v_segs:
                if _v_merged and (_wn_arr_bv[_seg[0]] - _wn_arr_bv[_v_merged[-1][1]]) < _bv_merge:
                    _v_merged[-1][1] = _seg[1]
                else:
                    _v_merged.append(_seg)
            _v_merged = [s for s in _v_merged if (s[1] - s[0] + 1) >= 3]

            _n_robust = 0
            _best_z = 0.0
            _best_verdict = "no bands"
            _best_band = ""
            _best_d = float("nan")
            for _i0, _i1 in _v_merged:
                _sl = slice(_i0, _i1 + 1)
                _N = _i1 - _i0 + 1
                _dnu = (_wn_arr_bv[_i1] - _wn_arr_bv[_i0]) / max(_N - 1, 1)

                _A = float(np.trapz(_v_ln[_sl], _wn_arr_bv[_sl]))
                _sA = _v_sigma * _dnu * np.sqrt(_N)
                _z = _A / _sA if _sA > 0 else 0.0

                _swo = float(_v_wo[_sl].sum())
                _swi = float(_v_wi[_sl].sum())
                _d_pct = (1.0 - _swi / _swo) * 100.0 if _swo > 0 else float("nan")

                _ctr = float(_wn_arr_bv[_i0 + int(np.argmax(_v_ln[_sl]))])
                _verdict = ("✅ robust" if _z > 5 else
                            ("⚠️ marginal" if _z > 2 else "❌ noise"))
                if _z > _best_z:
                    _best_z = _z
                    _best_verdict = _verdict
                    _best_band = f"{_wn_arr_bv[_i0]:.0f}–{_wn_arr_bv[_i1]:.0f} (@{_ctr:.0f})"
                    _best_d = _d_pct
                if _z > 5:
                    _n_robust += 1

            _max_abs = float(np.nanmax(np.abs(_v_ln)))
            _bv_rows.append({
                "m/z": _mz,
                "Verdict": _best_verdict if _v_merged else "no bands",
                "max z": _best_z,
                "# bands": len(_v_merged),
                "# robust": _n_robust,
                "Best band (cm⁻¹)": _best_band,
                "Depletion (%)": _best_d,
                "noise σ": _v_sigma,
                "max |−ln(depl)|": _max_abs,
            })
            _prog.progress((_j + 1) / _n_total)

        _prog.empty()
        _bv_df = pd.DataFrame(_bv_rows)
        st.session_state["_mid_bv_df"] = _bv_df
        st.success(f"Batch verdict complete for {_n_total} mass(es).")

    _bv_df = st.session_state.get("_mid_bv_df")
    if _bv_df is not None:
        st.markdown("**Verdict table**")
        st.dataframe(_bv_df, hide_index=True, use_container_width=True)

        _robust = (_bv_df["Verdict"] == "✅ robust").sum()
        _marginal = (_bv_df["Verdict"] == "⚠️ marginal").sum()
        _noise = (_bv_df["Verdict"].isin(["❌ noise", "no bands", "no bins", "no data"])).sum()
        st.caption(
            f"Summary: {_robust} robust | {_marginal} marginal | {_noise} rejected/noise "
            f"out of {len(_bv_df)} detected masses. "
            "Focus assignments on robust masses; use marginal ones only with additional evidence."
        )

        st.download_button(
            "📥 Download verdict table (CSV)",
            data=_bv_df.to_csv(index=False).encode("utf-8"),
            file_name="mass_verdict_summary.csv", mime="text/csv",
            key="_mid_bv_dl",
        )

        # Copy-paste paragraph for the user's report
        _report_para = (
            f"A band-integration robustness test (threshold = {_bv_k:g}×σ, "
            f"outline window = {_bv_dw:d} pts, merge gaps < {_bv_merge:.1f} cm⁻¹, "
            f"m/z half-width = ±{_bv_hw:.2f} amu"
            + (f", isotope integration {_bv_iso_n:d} peaks spaced by {_bv_iso_sp:.1f} amu"
               if (_bv_iso and _bv_iso_n > 1) else "") + 
            f") was applied to all {len(_bv_df)} detected masses. "
            f"{_robust} mass channel(s) showed at least one robust depletion band (z > 5), "
            f"{_marginal} showed only marginal bands (2 < z < 5), and {_noise} "
            f"showed no statistically significant bands. "
            "Masses with no robust bands were ruled out as lacking a definitive vibrational signature."
        )
        st.text_area("Report-ready paragraph (copy this)", value=_report_para, height=120,
                     key="_mid_bv_report_para", help="Copy into your paper/report methods section.")

# ── Half-width sweep ──────────────────────────────────────────────────────────
with st.expander("🔍 Integration half-width sweep — find the optimal value"):
    st.caption(
        "Overlay −ln(depletion) traces for a range of half-widths to identify "
        "which value maximises signal quality without cross-contaminating neighbours."
    )
    _sw_col1, _sw_col2, _sw_col3, _sw_col4 = st.columns(4)
    with _sw_col1:
        _sw_lo = st.number_input("Min half-width (amu)", value=0.05, min_value=0.01,
                                  max_value=2.0, step=0.05, key="_mid_sw_lo")
    with _sw_col2:
        _sw_hi = st.number_input("Max half-width (amu)", value=0.80, min_value=0.05,
                                  max_value=5.0, step=0.05, key="_mid_sw_hi")
    with _sw_col3:
        _sw_n = int(st.number_input("Number of steps", value=8, min_value=2,
                                     max_value=20, step=1, key="_mid_sw_n"))
    with _sw_col4:
        _sw_norm = st.checkbox("Normalise traces", value=True, key="_mid_sw_norm")
        _sw_ridge = st.checkbox("Ridge layout", value=False, key="_mid_sw_ridge")
        _sw_ridge_gap = 1.0
        if _sw_ridge:
            _sw_ridge_gap = st.slider("Ridge gap", 0.2, 2.0, 0.8, 0.1, key="_mid_sw_rgap")

    _sw_widths = np.linspace(_sw_lo, _sw_hi, _sw_n)
    _sw_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#bcbd22", "#17becf", "#aec7e8",
        "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
        "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79", "#637939",
    ]

    _sw_summary_rows = []
    _sw_numeric = []  # parallel list of raw floats for auto-selection
    fig_sw = go.Figure()

    for _swi, _sw in enumerate(_sw_widths):
        _sw_mask = _build_iso_mask(iso_centers, _sw)
        _sw_bins = int(_sw_mask.sum())
        if _sw_bins == 0:
            continue
        _sw_wo = np.zeros(len(wn_sorted))
        _sw_wi = np.zeros(len(wn_sorted))
        for _ii, _wn in enumerate(wn_sorted):
            _df_wn = compilation[_wn]
            _sw_wo[_ii] = _df_wn.iloc[_sw_mask, plot_col_wo].values.sum()
            _sw_wi[_ii] = _df_wn.iloc[_sw_mask, plot_col_w].values.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            _sw_dep = -np.log(np.clip(_sw_wi / _sw_wo, 1e-10, None))

        _sw_max = np.nanmax(np.abs(_sw_dep))
        _sw_mean = np.nanmean(np.abs(_sw_dep))
        _sw_snr = _sw_max / _sw_mean if _sw_mean > 0 else 0
        # Captured mass-peak area = total without-IR ion counts inside the window.
        # This rises monotonically with the half-width and plateaus once the
        # entire mass peak is enclosed — a robust, artifact-proof coverage metric.
        _sw_area = float(np.nansum(_sw_wo))
        _sw_summary_rows.append({
            "Half-width (amu)": f"{_sw:.3f}",
            "m/z bins": _sw_bins,
            "captured peak area": f"{_sw_area:.4g}",
            "max |−ln(depl)|": f"{_sw_max:.4f}",
            "mean |−ln(depl)|": f"{_sw_mean:.4f}",
            "peak/mean SNR": f"{_sw_snr:.1f}",
        })
        _sw_numeric.append({"hw": _sw, "bins": _sw_bins, "snr": _sw_snr,
                            "maxd": _sw_max, "area": _sw_area})

        _sw_y = _sw_dep.copy()
        if _sw_norm:
            _rng = np.nanmax(_sw_y) - np.nanmin(_sw_y)
            if _rng > 0:
                _sw_y = (_sw_y - np.nanmin(_sw_y)) / _rng
        _sw_off = _swi * _sw_ridge_gap if _sw_ridge else 0
        _sw_color = _sw_palette[_swi % len(_sw_palette)]
        fig_sw.add_trace(go.Scatter(
            x=wn_arr, y=_sw_y + _sw_off,
            mode="lines",
            name=f"±{_sw:.3f} amu ({_sw_bins} bins)",
            line=dict(color=_sw_color, width=1.8),
        ))

    fig_sw.update_layout(
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="−ln(depl)" + (" [norm.]" if _sw_norm else "") + (" [stacked]" if _sw_ridge else ""),
        title=f"Half-width sweep — m/z {selected_mz:.1f}",
        height=480,
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
    )
    if len(wn_arr) > 1:
        _apply_wn_plotly(fig_sw, float(np.nanmin(wn_arr)), float(np.nanmax(wn_arr)),
                         spacing=50.0)
    st.plotly_chart(fig_sw, use_container_width=True)

    if _sw_summary_rows:
        st.markdown("**Summary table — signal quality vs. half-width**")
        _sw_df = pd.DataFrame(_sw_summary_rows)
        st.dataframe(_sw_df, hide_index=True, use_container_width=True)

        # ── Auto-select optimal half-width (peak-area knee) ───────────────
        # The window must be wide enough to capture the FULL mass peak, then
        # stop. We select on the CAPTURED MASS-PEAK AREA (total without-IR ion
        # counts inside the window). As the window widens it first fills the
        # peak (large area gains) and then only accumulates a slowly-sloping
        # baseline / neighbouring-peak tail (small, roughly constant gains).
        #
        # Because that baseline tail means the area never truly flattens, a
        # "fraction of the maximum area" rule drifts out into the tail and picks
        # far too wide a window. Instead we find the KNEE: the smallest width
        # after which each further step adds < `_knee_gain` of the current area
        # (peak captured; only baseline creep remains), requiring at least
        # `_min_frac` of the peak to already be enclosed as a safety floor.
        #
        # NOTE: max|−ln(depl)| is deliberately NOT used for selection — a single
        # sharp artifact (e.g. a spike near 700 cm⁻¹) makes it non-monotonic and
        # can hijack the choice. peak/mean "SNR" is likewise biased toward narrow
        # windows. Both are shown for information only.
        if len(_sw_numeric) >= 2:
            _sw_maxds  = np.array([r["maxd"] for r in _sw_numeric])
            _sw_hws    = np.array([r["hw"]   for r in _sw_numeric])
            _sw_snrs   = np.array([r["snr"]  for r in _sw_numeric])
            _sw_areas  = np.array([r["area"] for r in _sw_numeric], dtype=float)

            # Cumulative-max so a noisy dip cannot distort the monotonic area.
            _sw_area_env = np.maximum.accumulate(_sw_areas)
            _sw_area_full = _sw_area_env[-1]

            _knee_gain = 0.02   # next step adds < 2 % → peak captured
            _min_frac  = 0.90   # but only after ≥ 90 % of the peak is enclosed

            # Knee: smallest width whose step to the next adds < _knee_gain,
            # provided at least _min_frac of the fully-captured area is reached.
            _opt_idx = len(_sw_numeric) - 1
            for _si in range(len(_sw_area_env) - 1):
                _gain = ((_sw_area_env[_si + 1] - _sw_area_env[_si])
                         / max(_sw_area_env[_si], 1e-12))
                _frac_here = (_sw_area_env[_si] / _sw_area_full
                              if _sw_area_full > 0 else 0.0)
                if _gain < _knee_gain and _frac_here >= _min_frac:
                    _opt_idx = _si
                    break

            _opt_hw   = float(_sw_hws[_opt_idx])
            _opt_max  = float(_sw_maxds[_opt_idx])
            _opt_area = float(_sw_areas[_opt_idx])
            _opt_frac = (_sw_area_env[_opt_idx] / _sw_area_full) if _sw_area_full > 0 else 0.0
            _kneed = _opt_idx < len(_sw_numeric) - 1

            if _kneed:
                st.success(
                    f"**Suggested optimal half-width: ±{_opt_hw:.3f} amu** "
                    f"(captures {_opt_frac * 100:.0f} % of the mass-peak area; "
                    f"the peak is enclosed here — wider windows add <{_knee_gain * 100:.0f} % "
                    f"more area, i.e. only baseline / neighbour tail)"
                )
            else:
                st.warning(
                    f"**Suggested optimal half-width: ±{_opt_hw:.3f} amu** "
                    f"(captures {_opt_frac * 100:.0f} % of the swept peak area) — "
                    f"*no clear knee within the swept range; the peak may still be "
                    f"growing (increase the max half-width) or be masked by a strong "
                    f"baseline slope.*"
                )
            st.caption(
                f"ℹ️ At this width: max|−ln(depl)| = {_opt_max:.4f}, "
                f"peak/mean ratio = {float(_sw_snrs[_opt_idx]):.1f}× "
                f"(shown for reference only; not used for selection)."
            )

            _opt_mask = _build_iso_mask(iso_centers, _opt_hw)
            _opt_pad = max(1.0, 1.5 * _opt_hw)
            _opt_mz_lo = max(float(np.min(x_mass)), min(iso_centers) - _opt_hw - _opt_pad)
            _opt_mz_hi = min(float(np.max(x_mass)), max(iso_centers) + _opt_hw + _opt_pad)
            _opt_display = (x_mass >= _opt_mz_lo) & (x_mass <= _opt_mz_hi)
            _opt_fig = go.Figure()
            _opt_fig.add_trace(go.Scatter(
                x=x_mass[_opt_display], y=ms_y[_opt_display], mode="lines",
                name="Mass spectrum", line=dict(color="#1f77b4", width=1.5),
            ))
            _opt_fig.add_trace(go.Scatter(
                x=x_mass[_opt_mask], y=ms_y[_opt_mask], mode="markers",
                name=f"Included bins ({int(_opt_mask.sum())})",
                marker=dict(color="#d62728", size=6),
            ))
            for _center in iso_centers:
                _opt_fig.add_vrect(
                    x0=_center - _opt_hw, x1=_center + _opt_hw,
                    fillcolor="rgba(214,39,40,0.18)", line_width=1,
                    line_color="#d62728",
                )
                _opt_fig.add_vline(
                    x=_center, line_width=1, line_dash="dash", line_color="#d62728",
                    annotation_text=f"{_center:.3f}", annotation_position="top",
                )
            _opt_fig.update_layout(
                title=f"Mass-spectrum coverage at suggested ±{_opt_hw:.3f} amu",
                xaxis_title="m/z", yaxis_title="Intensity (a.u.)", height=360,
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(_opt_fig, use_container_width=True)
            st.caption(
                "Red shading and markers show every m/z bin included in the suggested "
                "integration window" + ("s." if len(iso_centers) > 1 else ".")
            )

            if st.button(f"⬆️ Apply {_opt_hw:.3f} amu as integration half-width",
                         key="_mid_sw_apply"):
                st.session_state["_mid_hw"] = _opt_hw
                st.rerun()


# ── Savitzky-Golay window sweep ───────────────────────────────────────────────
with st.expander("🌊 Smoothing (SG window) sweep — avoid fake peaks & over-smoothing"):
    st.caption(
        "IR-UV ion-dip spectra are noisy. Too small a window keeps noise (spurious peaks); "
        "too large smears or invents peaks. The optimal window is the **largest** one whose "
        "residual (raw − smoothed) is still **white noise** — i.e. no real signal has leaked "
        "into the residual yet."
    )
    _gw_col1, _gw_col2, _gw_col3 = st.columns(3)
    with _gw_col1:
        _gw_max = st.number_input("Max SG window (odd)", value=31, min_value=5,
                                   max_value=101, step=2, key="_mid_gw_max")
    with _gw_col2:
        _gw_poly = st.number_input("Polynomial order", value=3, min_value=1,
                                    max_value=5, step=1, key="_mid_gw_poly")
    with _gw_col3:
        _gw_acthr = st.slider("Residual whiteness threshold |r₁|", 0.05, 0.50, 0.20, 0.05,
                              key="_mid_gw_acthr",
                              help="Max allowed lag-1 autocorrelation of the residual. "
                                   "Above this, real peak structure is leaking into the residual "
                                   "(over-smoothing).")

    _gw_sig = np.asarray(ln_depletion, dtype=float)
    _gw_n = len(_gw_sig)

    if _gw_n < 7:
        st.info("Spectrum too short for a meaningful SG sweep.")
    else:
        # Candidate odd windows from 3 up to min(_gw_max, n) — window must exceed poly order
        _gw_hi = int(min(_gw_max, _gw_n if _gw_n % 2 == 1 else _gw_n - 1))
        _gw_windows = [w for w in range(3, _gw_hi + 1, 2) if w > _gw_poly]

        # Peak prominence baseline: a small fraction of the raw dynamic range
        _gw_rng = float(np.nanmax(_gw_sig) - np.nanmin(_gw_sig))
        _gw_prom = max(0.05 * _gw_rng, 1e-6)

        _gw_rows, _gw_numeric = [], []
        _gw_fig = go.Figure()
        _gw_fig.add_trace(go.Scatter(
            x=wn_arr, y=_gw_sig, mode="lines", name="raw",
            line=dict(color="#bbbbbb", width=1),
        ))
        _gw_pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79"]

        for _gi, _w in enumerate(_gw_windows):
            _po = min(int(_gw_poly), _w - 1)
            _sm = savgol_filter(_gw_sig, window_length=_w, polyorder=_po)
            _res = _gw_sig - _sm

            # Lag-1 autocorrelation of the residual (white noise → ~0)
            _rc = _res - np.nanmean(_res)
            _den = float(np.nansum(_rc * _rc))
            _r1 = float(np.nansum(_rc[:-1] * _rc[1:]) / _den) if _den > 0 else 0.0

            _noise = float(np.nanstd(_res))
            _npk = int(len(find_peaks(np.nan_to_num(_sm, nan=0.0), prominence=_gw_prom)[0]))

            _gw_rows.append({
                "SG window": _w,
                "Peaks": _npk,
                "Noise σ (resid)": f"{_noise:.4f}",
                "Residual r₁": f"{_r1:+.3f}",
                "OK?": "⚠️ leak" if _r1 > _gw_acthr else "✅",
            })
            _gw_numeric.append({"w": _w, "peaks": _npk, "noise": _noise, "r1": _r1})

            # Only plot a handful to keep the figure readable
            if _gi % max(1, len(_gw_windows) // 8) == 0 or _w == _gw_windows[-1]:
                _gw_fig.add_trace(go.Scatter(
                    x=wn_arr, y=_sm, mode="lines", name=f"w={_w}",
                    line=dict(color=_gw_pal[_gi % len(_gw_pal)], width=1.8),
                ))

        _gw_fig.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="−ln(depl)",
            title=f"SG window sweep — m/z {selected_mz:.1f}",
            height=460, legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        )
        if len(wn_arr) > 1:
            _apply_wn_plotly(_gw_fig, float(np.nanmin(wn_arr)), float(np.nanmax(wn_arr)),
                             spacing=50.0)
        st.plotly_chart(_gw_fig, use_container_width=True)

        if _gw_rows:
            st.markdown("**Summary — noise, peaks & residual whiteness vs. SG window**")
            st.dataframe(pd.DataFrame(_gw_rows), hide_index=True, use_container_width=True)

            # ── Auto-select optimal SG window ─────────────────────────────────
            # PRIMARY criterion — peak-count knee: as the window grows, spurious
            # noise peaks vanish and the detected-peak count drops, then settles
            # on a plateau (the real peaks). The optimum is the SMALLEST window
            # that reaches this plateau (minimal smoothing that kills the noise).
            # GUARD — signal leakage: a *positive* residual autocorrelation
            # (r₁ > threshold) means real peak structure is leaking into the
            # residual = over-smoothing. Never recommend at/after that onset.
            _gw_r1s = np.array([r["r1"]    for r in _gw_numeric])
            _gw_ws  = np.array([r["w"]     for r in _gw_numeric])
            _gw_pks = np.array([r["peaks"] for r in _gw_numeric])
            _N = len(_gw_numeric)

            # Onset of over-smoothing (first window with positive leakage)
            _leak = np.where(_gw_r1s > _gw_acthr)[0]
            _leak_idx = int(_leak[0]) if len(_leak) > 0 else _N

            # Peak-count knee: first index whose count equals the count two
            # steps later (a short stable run), searched only up to the leak.
            _search_hi = max(1, _leak_idx)
            _knee = None
            for _k in range(_search_hi - 1):
                _run_end = min(_k + 2, _N - 1)
                if _gw_pks[_k] == _gw_pks[_run_end]:
                    _knee = _k
                    break

            if _knee is None:
                # No clear plateau before leakage → take the window just before
                # leakage onset (or the middle of the swept range as a fallback).
                _knee = (_leak_idx - 1) if _leak_idx < _N else _N // 2
                _knee = max(0, min(_knee, _N - 1))
                _reason = ("just before over-smoothing sets in"
                           if _leak_idx < _N else
                           "mid-range (no clear plateau or leakage detected)")
            else:
                _reason = "peak count plateaus here (noise peaks gone, real peaks intact)"

            _gw_opt = int(_gw_ws[_knee])
            _leak_note = (f" Over-smoothing (signal leak) begins at w={int(_gw_ws[_leak_idx])}."
                          if _leak_idx < _N else
                          " No over-smoothing detected within the swept range.")

            st.success(
                f"**Suggested SG window: {_gw_opt}** "
                f"(polyorder {min(int(_gw_poly), _gw_opt - 1)}) — {_reason}; "
                f"detects {int(_gw_pks[_knee])} peaks, residual r₁ = {_gw_r1s[_knee]:+.3f}."
                f"{_leak_note}"
            )
            if st.button(f"⬆️ Apply SG window = {_gw_opt} (and enable smoothing)",
                         key="_mid_gw_apply"):
                st.session_state["_mid_sg"] = True
                st.session_state["_mid_sg_w"] = _gw_opt
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2b — REMPI CROSS-CHECK (load bundle exported from 8.3)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
with st.expander("🟣 REMPI Cross-Check — view your REMPI data without leaving this page"):
    st.caption(
        "Load a `REMPI_dataset.pkl.gz` bundle exported from **Section 8.3** "
        "(or reuse REMPI data already in this session) to check the summed mass "
        "spectrum and a 1D action spectrum for any mass."
    )

    def _rempi_parse_wl(_cols):
        _out = []
        for _c in _cols:
            _s = str(_c)
            if _s.startswith("bc_"):
                _s = _s[3:]
            try:
                _out.append(float(_s))
            except ValueError:
                _mm = __import__("re").search(r"[\d.]+", _s)
                _out.append(float(_mm.group()) if _mm else np.nan)
        return np.array(_out)

    # ── Acquire REMPI data: session state first, else load from pkz ───────────
    _rempi_x = st.session_state.get("rempi_x_mass")
    _rempi_df = st.session_state.get("rempi_baseline_corrected")
    _rempi_mname = st.session_state.get("rempi_molecule_name", "")
    _rempi_mmass = st.session_state.get("rempi_molecule_mass")

    # Bundle loaded earlier on this page takes precedence if present
    _rb = st.session_state.get("_mid_rempi_bundle")
    if _rb is not None:
        _rempi_x = _rb["x_mass"]
        _rempi_df = _rb["corrected_df"]
        _rempi_mname = _rb.get("molecule_name", "")
        _rempi_mmass = _rb.get("molecule_mass")

    _have_session_rempi = (_rempi_x is not None) and (_rempi_df is not None)

    # ── Auto-load from default path if nothing in session yet ─────────────────
    _def_dir = st.session_state.get("file_directory", "") or st.session_state.get("rempi_file_directory", "")
    _def_pkz = ""
    if _def_dir:
        _cand = Path(_def_dir) / "output" / "REMPI_dataset.pkl.gz"
        _def_pkz = str(_cand)

    if not _have_session_rempi and _def_pkz and os.path.isfile(_def_pkz):
        try:
            import gzip as _gzip_auto, pickle as _pkl_auto
            with _gzip_auto.open(_def_pkz, "rb") as _f_auto:
                _loaded_auto = _pkl_auto.load(_f_auto)
            st.session_state["_mid_rempi_bundle"] = _loaded_auto
            _rb = _loaded_auto
            _rempi_x = _rb["x_mass"]
            _rempi_df = _rb["corrected_df"]
            _rempi_mname = _rb.get("molecule_name", "")
            _rempi_mmass = _rb.get("molecule_mass")
            _have_session_rempi = True
        except Exception:
            pass  # fall through to manual loader below

    if _have_session_rempi:
        st.success("✅ Using REMPI data currently available "
                   + ("(loaded bundle)" if _rb is not None else "(live session)"))

    # File loader (always available, e.g. to override or when nothing in session)
    _rempi_path = st.text_input(
        "Path to REMPI bundle (.pkl.gz)", value=_def_pkz, key="_mid_rempi_path",
        help="Exported by Section 8.3 → 'Export REMPI bundle (.pkl.gz)'. "
             "Auto-loaded from the default location if the file exists.",
    )
    if st.button("📥 Load REMPI bundle", key="_mid_rempi_load"):
        if not _rempi_path or not os.path.exists(_rempi_path):
            st.error(f"❌ File not found: `{_rempi_path}`")
        else:
            try:
                import gzip, pickle
                with gzip.open(_rempi_path, "rb") as _f:
                    _loaded = pickle.load(_f)
                st.session_state["_mid_rempi_bundle"] = _loaded
                st.success(f"✅ Loaded REMPI bundle from `{os.path.basename(_rempi_path)}`")
                st.rerun()
            except Exception as _e:
                st.error(f"❌ Failed to load: {_e}")

    if not _have_session_rempi:
        st.info("No REMPI data loaded yet. Export a bundle from Section 8.3, then load it above.")
        # No REMPI available this run — clear any stale cached arrays so the
        # Save Assignment block does not write an outdated action spectrum.
        st.session_state["_mid_rempi_ready"] = None
    else:
        _rempi_x = np.asarray(_rempi_x)
        _wl_cols = [c for c in _rempi_df.columns if c != "Summed"]
        _wl_vals = _rempi_parse_wl(_wl_cols)

        # Cache the essentials so the Save Assignment block can extract the
        # action spectrum for the assigned mass without re-loading the bundle.
        st.session_state["_mid_rempi_ready"] = {
            "x": _rempi_x,
            "Z": _rempi_df[_wl_cols].to_numpy(),
            "wl_half": _wl_vals / 2.0,
        }

        # ── Summed mass spectrum ─────────────────────────────────────────────
        st.markdown("**REMPI summed mass spectrum (baseline-corrected)**")
        if "Summed" in _rempi_df.columns:
            _summed = np.asarray(_rempi_df["Summed"].values, dtype=float)
        else:
            _summed = _rempi_df[_wl_cols].to_numpy().sum(axis=1)

        _ms_c1, _ms_c2 = st.columns(2)
        with _ms_c1:
            _ms_lo = st.number_input("Mass min (amu)", value=float(np.nanmin(_rempi_x)),
                                     key="_mid_rempi_ms_lo")
        with _ms_c2:
            _ms_hi = st.number_input("Mass max (amu)", value=float(np.nanmax(_rempi_x)),
                                     key="_mid_rempi_ms_hi")
        _ms_mask = (_rempi_x >= _ms_lo) & (_rempi_x <= _ms_hi)

        _fig_ms = go.Figure()
        _fig_ms.add_trace(go.Scatter(x=_rempi_x[_ms_mask], y=_summed[_ms_mask],
                                     mode="lines", line=dict(color="#7b2cbf", width=1.5),
                                     name="REMPI summed"))
        # Mark the mass currently selected in Section 2
        _fig_ms.add_vline(x=selected_mz, line_width=1.5, line_dash="dash",
                          line_color="green",
                          annotation_text=f"IR m/z {selected_mz:.1f}",
                          annotation_position="top")
        if _rempi_mmass is not None:
            try:
                _fig_ms.add_vline(x=float(_rempi_mmass), line_width=1.5, line_dash="dot",
                                  line_color="orange",
                                  annotation_text=f"{_rempi_mname} ({float(_rempi_mmass):.0f})",
                                  annotation_position="bottom")
            except (TypeError, ValueError):
                pass
        _fig_ms.update_layout(height=320, showlegend=False,
                              xaxis_title="Mass (amu)", yaxis_title="Intensity (a.u.)",
                              margin=dict(t=20, b=40))
        st.plotly_chart(_fig_ms, use_container_width=True)

        # ── 1D action spectrum for a chosen mass ─────────────────────────────
        st.markdown("**1D action spectrum (intensity vs wavelength)**")
        _as_c1, _as_c2 = st.columns([3, 1])
        with _as_c1:
            _as_mass = st.number_input(
                "Mass to extract (amu)",
                min_value=float(np.nanmin(_rempi_x)), max_value=float(np.nanmax(_rempi_x)),
                value=float(np.clip(selected_mz, np.nanmin(_rempi_x), np.nanmax(_rempi_x))),
                step=0.1, key="_mid_rempi_as_mass",
                help="Defaults to the m/z selected in Section 2.",
            )
        with _as_c2:
            _as_tol = st.number_input("± tolerance (amu)", min_value=0.1, max_value=10.0,
                                      value=0.5, step=0.1, key="_mid_rempi_as_tol")

        _as_idx = np.where(np.abs(_rempi_x - _as_mass) <= _as_tol)[0]
        if len(_as_idx) == 0:
            st.warning(f"No mass points within ±{_as_tol} amu of {_as_mass:.1f}.")
        elif len(_wl_cols) == 0:
            st.warning("No per-wavelength columns found in this REMPI bundle.")
        else:
            _Z = _rempi_df[_wl_cols].to_numpy()
            _as_int = _Z[_as_idx, :].mean(axis=0)
            _wl_half = _wl_vals / 2
            _order = np.argsort(_wl_half)
            _wl_half_s = _wl_half[_order]
            _as_int_s = np.asarray(_as_int)[_order]

            _fig_as = go.Figure()
            _fig_as.add_trace(go.Scatter(x=_wl_half_s, y=_as_int_s, mode="lines",
                                         line=dict(color="#1f77b4", width=2),
                                         name=f"m/z {_as_mass:.1f}"))
            _fig_as.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
            _fig_as.update_layout(height=320, showlegend=False,
                                  xaxis_title="Wavelength (nm)  /  Energy (eV)",
                                  yaxis_title="Ion intensity (a.u.)",
                                  margin=dict(t=20, b=40))
            # Ticks every 0.5/1 nm with photon-energy eV under each label
            if len(_wl_half_s) > 1:
                _apply_wl_plotly(_fig_as, float(np.nanmin(_wl_half_s)),
                                 float(np.nanmax(_wl_half_s)))
            st.plotly_chart(_fig_as, use_container_width=True)

            _pk = int(np.nanargmax(_as_int_s))
            st.caption(
                f"Peak: {np.nanmax(_as_int_s):.4f} at {_wl_half_s[_pk]:.2f} nm (λ/2) "
                f"| mean {np.nanmean(_as_int_s):.4f} | {len(_as_idx)} mass bins averaged"
            )

            _as_csv = pd.DataFrame({
                "wavelength_half_nm": _wl_half_s,
                "intensity_au": _as_int_s,
            }).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download action spectrum (CSV)", data=_as_csv,
                file_name=f"REMPI_action_m{_as_mass:.1f}.csv", mime="text/csv",
                key="_mid_rempi_as_dl",
            )


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
                    # Apply harmonic frequency scaling factor
                    _scale = float(st.session_state.get("_mid_dft_scale", 1.0))
                    freqs = np.asarray(freqs, dtype=float) * _scale
                    # Broaden the stick spectrum
                    _bw = st.session_state.get("_mid_bw_frac", 0.007)
                    wn_broad, int_broad = broaden_spectrum_felix(freqs, intens, bw_frac=_bw)
                    _dname = f"DFT: {_fname}"
                    if "method" in meta:
                        _dname += f" ({meta['method']})"
                    if abs(_scale - 1.0) > 1e-9:
                        _dname += f" ×{_scale:.4f}"
                    ref_spectra.append({"name": _dname, "wn": wn_broad, "intensity": int_broad,
                                        "source": "DFT", "freqs": freqs, "intens": intens,
                                        "scale": _scale})
                    st.success(f"🧪 DFT: {_fname} — {len(freqs)} modes → broadened"
                               + (f" (scaled ×{_scale:.4f})" if abs(_scale - 1.0) > 1e-9 else ""))
                else:
                    st.warning(f"Could not parse DFT modes from {_fname}")

            elif _ext in (".csv", ".txt", ".dat"):
                _text = _raw.decode("utf-8", errors="replace") if isinstance(_raw, bytes) else _raw
                # Skip leading '#' comment lines so files like
                #   # Freq_MD (cm^-1) Inten_MD (Normalized intensity)
                #   300.0 3.1340e-03
                # work without manual editing.
                _lines = _text.splitlines()
                _first_data = 0
                for _line in _lines:
                    if _line.strip().startswith("#"):
                        _first_data += 1
                    else:
                        break
                _clean_text = "\n".join(_lines[_first_data:])
                _df = pd.read_csv(io.StringIO(_clean_text), sep=None, engine="python")
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
                    _scale = float(st.session_state.get("_mid_dft_scale", 1.0))
                    freqs = np.asarray(freqs, dtype=float) * _scale
                    _bw = st.session_state.get("_mid_bw_frac", 0.007)
                    wn_broad, int_broad = broaden_spectrum_felix(freqs, intens, bw_frac=_bw)
                    _dname = f"DFT: {_fname}"
                    if abs(_scale - 1.0) > 1e-9:
                        _dname += f" ×{_scale:.4f}"
                    ref_spectra.append({"name": _dname, "wn": wn_broad, "intensity": int_broad,
                                        "source": "DFT", "freqs": freqs, "intens": intens,
                                        "scale": _scale})
                    st.success(f"🧪 DFT stick: {_fname} — {len(freqs)} modes → broadened"
                               + (f" (scaled ×{_scale:.4f})" if abs(_scale - 1.0) > 1e-9 else ""))

        except Exception as _e:
            st.warning(f"Could not parse {_fname}: {_e}")

# DFT broadening / frequency scaling controls
if any(s.get("source") == "DFT" for s in ref_spectra):
    _dc1, _dc2 = st.columns(2)
    with _dc1:
        _bw_frac = st.number_input("FELIX bandwidth fraction (FWHM/ν)", value=0.007,
                                    min_value=0.001, max_value=0.05, step=0.001,
                                    format="%.4f", key="_mid_bw_frac",
                                    help="Frequency-proportional Gaussian FWHM = bw_frac × ν")
    with _dc2:
        _dft_scale = st.number_input("DFT frequency scaling factor", value=1.000,
                                      min_value=0.500, max_value=1.500, step=0.001,
                                      format="%.4f", key="_mid_dft_scale",
                                      help="Multiplies all DFT harmonic frequencies. "
                                           "1.0000 = no scaling (default). Typical values: "
                                           "~0.96–0.98 for B3LYP harmonic frequencies.")

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
        _traces.append(("Experimental (−ln depl)", your_wn, _norm(_your_smoothed), "#d62728", False, False))
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

    # Raw (non-smoothed) experimental IR drawn first, in grey, so it sits underneath.
    # The experimental trace is index 0 → ridge offset 0, so raw aligns with it.
    if your_wn is not None and _smooth_ref:
        fig_cmp.add_trace(go.Scatter(
            x=your_wn, y=_norm(your_intensity), mode="lines",
            line=dict(color="#c9c9c9", width=1), name="Experimental (raw)",
        ))

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
    if _xmin is not None and _xmax > _xmin:
        _apply_wn_plotly(fig_cmp, _xmin, _xmax, spacing=50.0)
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

    # ── Band-matched reference filter ─────────────────────────────────────────
    with st.expander("🎯 Keep only references matching detected bands"):
        st.caption(
            "Uses the bands found in **Section 2 → Band Integration**. A reference is "
            "kept only if at least one of its peaks lands inside a detected band window. "
            "Upload many DFT/NIST files and only the relevant ones are overlaid."
        )
        _bands = st.session_state.get("_mid_bands", [])
        _bands_mz = st.session_state.get("_mid_bands_mz")

        if not _bands:
            st.info("No detected bands yet. Open **Section 2 → 📐 Band Integration** first "
                    "to detect bands for this mass.")
        elif not ref_spectra:
            st.info("Upload reference spectra above to filter them against the detected bands.")
        else:
            if _bands_mz is not None and abs(float(_bands_mz) - float(selected_mz)) > 1e-6:
                st.warning(f"⚠️ Bands were detected for m/z {_bands_mz:.1f}, but the current "
                           f"mass is {selected_mz:.1f}. Re-run band integration for this mass.")

            _bm_c1, _bm_c2, _bm_c3 = st.columns(3)
            with _bm_c1:
                _bm_which = st.radio("Use bands", ["Robust only", "Robust + marginal", "All"],
                                     index=0, key="_mid_bm_which")
            with _bm_c2:
                _bm_tol = st.number_input("Match tolerance (cm⁻¹)", value=10.0, min_value=0.0,
                                          max_value=50.0, step=1.0, key="_mid_bm_tol",
                                          help="A peak counts as matching if it falls within "
                                               "the band window widened by this tolerance.")
            with _bm_c3:
                _bm_prom = st.number_input("Ref peak prominence (%)", value=5.0, min_value=0.5,
                                           max_value=50.0, step=0.5, key="_mid_bm_prom") / 100.0

            if _bm_which == "Robust only":
                _sel_bands = [b for b in _bands if b["verdict"].startswith("✅")]
            elif _bm_which == "Robust + marginal":
                _sel_bands = [b for b in _bands if not b["verdict"].startswith("❌")]
            else:
                _sel_bands = list(_bands)

            if not _sel_bands:
                st.warning("No bands of the selected quality. Try a lower quality filter.")
            else:
                _wins = [(b["lo"] - _bm_tol, b["hi"] + _bm_tol) for b in _sel_bands]

                def _peaks_in_bands(_wn, _it):
                    _wn = np.asarray(_wn, dtype=float)
                    _it = np.asarray(_it, dtype=float)
                    if len(_it) < 4:
                        return []
                    _n = _norm(_it)
                    _pv = _bm_prom * float(np.nanmax(_n) - np.nanmin(_n))
                    _pk, _ = find_peaks(np.nan_to_num(_n, nan=0), prominence=max(_pv, 1e-6))
                    _hits = []
                    for _pi in _pk:
                        _pw = _wn[_pi]
                        for _bi, (_wlo, _whi) in enumerate(_wins):
                            if _wlo <= _pw <= _whi:
                                _hits.append((_pw, _bi))
                                break
                    return _hits

                _matched, _match_rows = [], []
                for _s in ref_spectra:
                    _hits = _peaks_in_bands(_s["wn"], _s["intensity"])
                    if _hits:
                        _matched.append(_s)
                        _bset = sorted({f"{_sel_bands[_bi]['center']:.0f}" for _, _bi in _hits})
                        _match_rows.append({
                            "Reference": _s["name"][:50],
                            "Matched bands (cm⁻¹)": ", ".join(_bset),
                            "# peaks in bands": len(_hits),
                        })

                st.markdown(
                    f"**{len(_matched)} of {len(ref_spectra)} references** match "
                    f"{len(_sel_bands)} band window(s)."
                )

                if not _matched:
                    st.info("No references had peaks inside the detected bands. "
                            "Try widening the tolerance or relaxing the band quality.")
                else:
                    st.dataframe(pd.DataFrame(_match_rows), hide_index=True,
                                 use_container_width=True)

                    _bm_norm_stack = st.checkbox("Ridge (stacked) layout", value=True,
                                                 key="_mid_bm_ridge")
                    _bm_fig = go.Figure()
                    _bm_pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
                               "#e377c2", "#bcbd22", "#17becf", "#d62728", "#7f7f7f"]

                    # Shade the band windows
                    for _b in _sel_bands:
                        _bm_fig.add_vrect(x0=_b["lo"], x1=_b["hi"],
                                          fillcolor="rgba(44,160,44,0.12)", line_width=0,
                                          annotation_text=f"{_b['center']:.0f}",
                                          annotation_position="top", annotation_font_size=9)

                    # Experimental trace at offset 0
                    if your_wn is not None:
                        _bm_fig.add_trace(go.Scatter(
                            x=your_wn, y=_norm(your_intensity), mode="lines",
                            name="Experimental", line=dict(color="#d62728", width=2)))

                    for _mi, _s in enumerate(_matched):
                        _mwn = np.asarray(_s["wn"], dtype=float)
                        _mit = np.asarray(_s["intensity"], dtype=float)
                        if _xmin is not None:
                            _mm = (_mwn >= _xmin) & (_mwn <= _xmax)
                            _mwn, _mit = _mwn[_mm], _mit[_mm]
                        _off = (_mi + 1) * 1.0 if _bm_norm_stack else 0
                        _bm_fig.add_trace(go.Scatter(
                            x=_mwn, y=_norm(_mit) + _off, mode="lines",
                            name=_s["name"][:40],
                            line=dict(color=_bm_pal[_mi % len(_bm_pal)], width=1.5)))

                    _bm_fig.update_layout(
                        height=500, xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Intensity" + (" (stacked)" if _bm_norm_stack else ""),
                        title=f"Band-matched references — m/z {selected_mz:.1f}",
                        xaxis=dict(range=_layout_xrange),
                        legend=dict(orientation="h", y=1.02, xanchor="right", x=1))
                    if _layout_xrange is not None:
                        _apply_wn_plotly(_bm_fig, _layout_xrange[0], _layout_xrange[1],
                                         spacing=50.0)
                    st.plotly_chart(_bm_fig, use_container_width=True)

                    st.download_button(
                        "📥 Download match table (CSV)",
                        data=pd.DataFrame(_match_rows).to_csv(index=False).encode("utf-8"),
                        file_name=f"band_matched_refs_mz{selected_mz:.1f}.csv",
                        mime="text/csv", key="_mid_bm_dl")

                    # Persist for the Save Assignment block
                    st.session_state["_mid_bm_data"] = {
                        "matched": _matched,
                        "sel_bands": _sel_bands,
                        "ridge": _bm_norm_stack,
                        "xrange": _layout_xrange,
                        "mz": float(selected_mz),
                    }


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MOLECULAR STRUCTURES (multi-molecule, per-mass persistent)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 5. Molecular Structures")

# Each mass gets its own isolated SMILES list so switching masses never
# overwrites candidates you've already entered for another mass.
# Rows carry a STABLE id so widget keys never shift when a row is removed.
import re as _re5
_mz_tag       = str(selected_mz).replace('.', 'p')
_mz_key       = f"_mid_smiles_{_mz_tag}"       # list of {id, name, smiles}
_mz_dismissed = f"_mid_dismissed_{_mz_tag}"    # set of ref names the user removed
_mz_counter   = f"_mid_rowid_{_mz_tag}"        # monotonic id counter

def _clean_ref_name(_raw):
    _n = _raw
    for _pfx in ("DFT: ", "NIST: "):
        if _n.startswith(_pfx):
            _n = _n[len(_pfx):]
    _n = _re5.sub(r"\s*\[CAS[^\]]*\]", "", _n).strip()
    _n = _re5.sub(r"\.(jdx|dx|out|log|csv|stk)$", "", _n, flags=_re5.IGNORECASE).strip()
    return _n

# ── Init state for this mass ─────────────────────────────────────────────────
if _mz_counter not in st.session_state:
    st.session_state[_mz_counter] = 0
if _mz_dismissed not in st.session_state:
    st.session_state[_mz_dismissed] = set()

def _new_row(name="", smiles=""):
    _rid = st.session_state[_mz_counter]
    st.session_state[_mz_counter] += 1
    return {"id": _rid, "name": name, "smiles": smiles}

if _mz_key not in st.session_state:
    _seed = [_new_row(name=_clean_ref_name(_rs["name"]))
             for _rs in ref_spectra if _clean_ref_name(_rs["name"])]
    st.session_state[_mz_key] = _seed if _seed else [_new_row()]

_slist = st.session_state[_mz_key]

# ── Migrate rows from older versions that have no 'id' ───────────────────────
for _e in _slist:
    if "id" not in _e:
        _e["id"] = st.session_state[_mz_counter]
        st.session_state[_mz_counter] += 1

# ── Auto-add newly uploaded refs (skip ones the user dismissed) ──────────────
_present = {e["name"] for e in _slist}
for _rs in ref_spectra:
    _rn = _clean_ref_name(_rs["name"])
    if _rn and _rn not in _present and _rn not in st.session_state[_mz_dismissed]:
        _blank = next((e for e in _slist if not e["name"] and not e["smiles"]), None)
        if _blank is not None:
            _blank["name"] = _rn
        else:
            _slist.append(_new_row(name=_rn))
        _present.add(_rn)

# ── Add / clear controls ──────────────────────────────────────────────────────
_sadd_col1, _sadd_col2, _sadd_col3 = st.columns([2, 1, 1])
with _sadd_col1:
    st.markdown(f"**Candidate molecules for m/z {selected_mz:.1f}** — add one row per isomer")
with _sadd_col2:
    if st.button("➕ Add molecule", key=f"_mid_add_mol_{_mz_tag}"):
        _slist.append(_new_row())
        st.rerun()
with _sadd_col3:
    if st.button("🗑️ Clear all", key=f"_mid_clear_mols_{_mz_tag}"):
        # Dismiss every current ref-derived name so they don't auto-return
        st.session_state[_mz_dismissed].update(e["name"] for e in _slist if e["name"])
        st.session_state[_mz_key] = [_new_row()]
        st.rerun()

# ── Editable rows (keyed by stable id) ───────────────────────────────────────
_to_remove = None
for _mentry in _slist:
    _rid = _mentry["id"]
    _rc1, _rc2, _rc3 = st.columns([2, 4, 1])
    with _rc1:
        _mentry["name"] = st.text_input(
            "Label", value=_mentry["name"],
            placeholder="Candidate",
            key=f"_mid_mol_name_{_mz_tag}_{_rid}",
            label_visibility="collapsed",
        )
    with _rc2:
        _mentry["smiles"] = st.text_input(
            "SMILES", value=_mentry["smiles"],
            placeholder="e.g. c1cccc2ccccc12  (naphthalene)",
            key=f"_mid_mol_smiles_{_mz_tag}_{_rid}",
            label_visibility="collapsed",
        )
    with _rc3:
        if len(_slist) > 1:
            if st.button("✕", key=f"_mid_mol_del_{_mz_tag}_{_rid}", help="Remove this row"):
                _to_remove = _rid

if _to_remove is not None:
    _removed = next((e for e in _slist if e["id"] == _to_remove), None)
    if _removed and _removed["name"]:
        # Remember the name so the auto-add loop won't bring it back
        st.session_state[_mz_dismissed].add(_removed["name"])
    st.session_state[_mz_key] = [e for e in _slist if e["id"] != _to_remove]
    st.rerun()

# ── Render grid of structures ─────────────────────────────────────────────────
_valid_mols = []  # (index, label, mol, formula, exact_mass, png_bytes)

if HAS_RDKIT:
    _nonempty = [e for e in _slist if e["smiles"].strip()]
    if _nonempty:
        _ncols = min(len(_nonempty), 4)
        _grid_cols = st.columns(_ncols)
        for _gi, _entry in enumerate(_nonempty):
            _label = _entry["name"].strip() or f"Candidate {_gi + 1}"
            try:
                _mol = Chem.MolFromSmiles(_entry["smiles"].strip())
                if _mol is None:
                    raise ValueError("Invalid SMILES")
                _img = Draw.MolToImage(_mol, size=(320, 240))
                _buf = io.BytesIO()
                _img.save(_buf, format="PNG")
                _png = _buf.getvalue()
                _mw  = Descriptors.ExactMolWt(_mol)
                _mf  = Chem.rdMolDescriptors.CalcMolFormula(_mol)
                _valid_mols.append((_gi, _label, _mol, _mf, _mw, _png))
                with _grid_cols[_gi % _ncols]:
                    st.image(_img, caption=f"**{_label}**", use_container_width=True)
                    st.caption(f"{_mf} | {_mw:.4f} Da")
            except Exception as _se:
                with _grid_cols[_gi % _ncols]:
                    st.error(f"{_label}: {_se}")
    else:
        st.info("Enter at least one SMILES string above to render structures.")
else:
    st.info("Install `rdkit` to enable SMILES → structure rendering.")

# ── Pick active structures for export / composite figure ─────────────────────
if _valid_mols:
    _mol_labels = [f"{v[1]} ({v[3]})" for v in _valid_mols]
    _active_labels = st.multiselect(
        "Structures for export (used in saved PNG & composite figure) — pick one or more, or leave blank for all",
        options=_mol_labels,
        default=_mol_labels,
        key="_mid_active_mol",
    )
    # Empty selection → treat as All
    _export_mols = [v for v in _valid_mols if f"{v[1]} ({v[3]})" in (_active_labels or _mol_labels)]
    st.session_state["_mid_export_mols"] = _export_mols

    # Choose which single structure is the "active" one (used as the single
    # image in the saved PNG and the A4 page when the grid option is off).
    if _export_mols:
        _export_labels = [f"{v[1]} ({v[3]})" for v in _export_mols]
        _primary_label = st.selectbox(
            "Active (primary) structure — shown as the single structure in the saved PNG & A4 page",
            options=_export_labels, key="_mid_primary_mol",
            help="Pick which structure represents this mass. The A4 page can show "
                 "just this one, or the full grid (toggle in the Save section).",
        )
        _primary = next((v for v in _export_mols
                         if f"{v[1]} ({v[3]})" == _primary_label), _export_mols[0])
        st.session_state["_mid_structure_img"] = _primary[5]
elif "_mid_struct_img_upload" in st.session_state:
    pass  # keep previously uploaded image

# ── Fallback: upload image directly ──────────────────────────────────────────
with st.expander("📁 Or upload a structure image directly (PNG/JPG/SVG)"):
    _struct_img = st.file_uploader("Structure image", type=["png", "jpg", "jpeg", "svg"],
                                    key="_mid_struct_img_upload")
    if _struct_img:
        st.image(_struct_img, caption="Uploaded structure", width=300)
        st.session_state["_mid_structure_img"] = _struct_img.getvalue()


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
    """Default output directory for mass assignments."""
    return r"/Users/danialmoh/Library/CloudStorage/OneDrive-StockholmUniversity(2)/OneDrive-StockholmUniversity/FELIX Data/Mass_Assignment"

_default_out = _get_default_output_dir()
_output_path = st.text_input(
    "Output folder", value=_default_out, key="_mid_output_path",
    help="Where assignment files will be saved. Change to any folder you like.",
)

def _get_output_dir():
    if not _output_path or not _output_path.strip():
        return None
    return Path(_output_path.strip())

_save_bm_plot = st.checkbox("Include band-matched references plot in save",
                            value=True, key="_mid_save_bm",
                            help="Save a PNG of the band-matched overlay from Section 4.")

_save_opt_c1, _save_opt_c2 = st.columns(2)
with _save_opt_c1:
    _rempi_ma_win = int(st.number_input(
        "REMPI moving-average window (pts)", value=5, min_value=1, max_value=51,
        step=2, key="_mid_rempi_ma",
        help="Window (in points) for the moving-average smoothed REMPI PNG. "
             "Set to 1 for no smoothing.",
    ))
with _save_opt_c2:
    _save_a4 = st.checkbox(
        "Save combined A4 page (structure + band-match + REMPI + mass spectrum)",
        value=True, key="_mid_save_a4",
        help="Assemble a single vertical A4 PDF + PNG with all four panels.",
    )
    _a4_struct_grid = st.checkbox(
        "Use structure grid in A4 (all selected structures)",
        value=False, key="_mid_a4_struct_grid",
        help="Show every selected structure as a grid in the A4 page instead of "
             "just the single active structure.",
    )

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

        # Panel data collected for the combined A4 page (populated below).
        _rempi_panel = None   # {"wl", "raw", "smooth", "win"}
        _ms_panel = None      # {"x", "y", "lo", "hi", "y_top", "y_at_sel"}

        def _moving_avg(_y, _w):
            """Simple centred moving average; returns input unchanged if _w<=1."""
            _y = np.asarray(_y, dtype=float)
            if _w is None or _w <= 1 or len(_y) < _w:
                return _y
            _k = np.ones(int(_w), dtype=float) / float(_w)
            return np.convolve(_y, _k, mode="same")

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

        # 1b) Integrated signal CSV + PNG for the selected mass
        _int_wn = st.session_state.get("_mid_your_wn")
        _int_without = st.session_state.get("_mid_int_without")
        _int_with = st.session_state.get("_mid_int_with")
        _int_sg_on = st.session_state.get("_mid_sg", False)
        _int_sg_w = st.session_state.get("_mid_sg_w", 5)
        if _int_wn is not None and _int_without is not None and _int_with is not None:
            try:
                def _int_smooth(y):
                    y = np.asarray(y, dtype=float)
                    if not _int_sg_on or len(y) < _int_sg_w:
                        return y
                    w = _int_sg_w if _int_sg_w % 2 == 1 else _int_sg_w + 1
                    return savgol_filter(y, window_length=w, polyorder=min(3, w - 1))

                _int_df = pd.DataFrame({
                    "wavenumber_cm-1": _int_wn,
                    "int_without_IR": _int_without,
                    "int_with_IR": _int_with,
                })
                if _int_sg_on:
                    _int_df["int_without_IR_smoothed"] = _int_smooth(_int_without)
                    _int_df["int_with_IR_smoothed"] = _int_smooth(_int_with)
                _int_csv_fname = f"integrated_signal_mz{_safe_mz}.csv"
                _int_df.to_csv(_assign_dir / _int_csv_fname, index=False)
                _saved_files.append(_int_csv_fname)

                _fig_int, _ax_int = plt.subplots(figsize=(10, 4.5))
                if _int_sg_on:
                    _ax_int.plot(_int_wn, _int_without, color="#c9c9c9", lw=1,
                                label="Without IR (raw)")
                    _ax_int.plot(_int_wn, _int_with, color="#dcdcdc", lw=1,
                                label="With IR (raw)")
                _ax_int.plot(_int_wn, _int_smooth(_int_without), color="#1f77b4", lw=2,
                            label="Without IR" + (" (smoothed)" if _int_sg_on else ""))
                _ax_int.plot(_int_wn, _int_smooth(_int_with), color="#ff7f0e", lw=2, ls="--",
                            label="With IR" + (" (smoothed)" if _int_sg_on else ""))
                _ax_int.set_xlim(float(np.nanmin(_int_wn)), float(np.nanmax(_int_wn)))
                _apply_wn_mpl(_ax_int, spacing=50.0, fontsize=7)
                _ax_int.set_ylabel("Ion intensity (a.u.)", fontsize=11)
                _ax_int.set_title(
                    f"Integrated signal — m/z {selected_mz:.1f} — "
                    f"{_chosen_formula or '?'} ({_verdict})",
                    fontsize=12, fontweight="bold")
                _ax_int.legend(fontsize=8, loc="upper right")
                _ax_int.grid(True, alpha=0.3)
                _fig_int.tight_layout()
                _int_png_path = _assign_dir / "integrated_signal.png"
                _fig_int.savefig(_int_png_path, dpi=300, bbox_inches="tight")
                plt.close(_fig_int)
                _saved_files.append("integrated_signal.png")
            except Exception as _e:
                st.warning(f"⚠️ Could not save integrated signal: {_e}")

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

        # 3) Comparison plot PNG — mirrors Section 4 settings exactly
        if _exp_wn is not None:
            try:
                # ── Read Section 4 settings (with safe fallbacks) ────────────
                _s4_smooth    = locals().get("_smooth_ref", _sg_on)
                _s4_smooth_w  = locals().get("_smooth_ref_w", _sg_w)
                _s4_norm      = locals().get("_normalize", True)
                _s4_ridge     = locals().get("_ridge", False)
                _s4_ridge_gap = locals().get("_ridge_gap", 1.0)
                _s4_peaks     = locals().get("_show_peaks", True)
                _s4_peak_prom = locals().get("_peak_prom_pct", 0.05)

                # ── Helpers matching Section 4 ────────────────────────────────
                def _s4_norm_fn(y):
                    if not _s4_norm:
                        return np.asarray(y, dtype=float)
                    a = np.asarray(y, dtype=float)
                    f = a[np.isfinite(a)]
                    if len(f) == 0 or np.ptp(f) == 0:
                        return a
                    return (a - np.nanmin(a)) / np.ptp(f)

                def _s4_smooth_fn(y):
                    y = np.asarray(y, dtype=float)
                    if not _s4_smooth or len(y) < _s4_smooth_w:
                        return y
                    _wo = _s4_smooth_w if _s4_smooth_w % 2 == 1 else _s4_smooth_w + 1
                    return savgol_filter(y, window_length=_wo, polyorder=min(3, _wo - 1))

                # ── x-range from experimental ─────────────────────────────────
                _save_xmin = float(np.nanmin(_exp_wn))
                _save_xmax = float(np.nanmax(_exp_wn))

                # ── Figure height: taller for ridge so traces don't squash ────
                _n_traces = 1 + len(ref_spectra)
                _fig_h_save = max(5, 2.5 + _n_traces * 1.2) if _s4_ridge else 5
                _fig_s, _ax_s = plt.subplots(figsize=(10, _fig_h_save))

                # ── Build traces list: (label, wn, y_raw, color) ─────────────
                _save_traces = []
                _exp_smoothed = _s4_smooth_fn(_exp_int)
                _save_traces.append(("Experimental", _exp_wn, _exp_smoothed, "#d62728"))

                for _ri, _rs in enumerate(ref_spectra):
                    _c = _palette[(_ri + 1) % len(_palette)]
                    _rwn = np.asarray(_rs["wn"])
                    _rint = np.asarray(_rs["intensity"], dtype=float)
                    _rmask = (_rwn >= _save_xmin) & (_rwn <= _save_xmax)
                    _rwn = _rwn[_rmask]
                    _rint = _rint[_rmask]
                    if len(_rint) == 0:
                        continue
                    _save_traces.append((_rs["name"][:40], _rwn, _rint, _c))

                # Raw (non-smoothed) experimental drawn first in grey (offset 0)
                if _s4_smooth:
                    _ax_s.plot(_exp_wn, _s4_norm_fn(_exp_int), color="#c9c9c9",
                               lw=1, label="Experimental (raw)")

                # ── Plot each trace with ridge offset ─────────────────────────
                for _ti, (_tlabel, _twn, _ty_raw, _tc) in enumerate(_save_traces):
                    _ty = _s4_norm_fn(_ty_raw)
                    _off = _ti * _s4_ridge_gap if _s4_ridge else 0
                    _ax_s.plot(_twn, _ty + _off, color=_tc, lw=2 if _ti == 0 else 1.5,
                               label=_tlabel)
                    # Ridge: draw a zero-baseline per trace
                    if _s4_ridge:
                        _ax_s.axhline(_off, color=_tc, lw=0.5, ls="--", alpha=0.3)
                    # Peak tick marks on reference traces
                    if _ti > 0 and _s4_peaks and len(_ty) > 3:
                        _pv = _s4_peak_prom * float(np.nanmax(_ty) - np.nanmin(_ty))
                        _rp, _ = find_peaks(np.nan_to_num(_ty, nan=0),
                                            prominence=max(_pv, 1e-6))
                        for _pi in _rp:
                            _ax_s.axvline(_twn[_pi], color=_tc, ls=":", lw=0.8, alpha=0.5)
                            _ax_s.annotate(f"{_twn[_pi]:.0f}",
                                           xy=(_twn[_pi], _ty[_pi] + _off),
                                           fontsize=6, color=_tc,
                                           ha="center", va="bottom", rotation=90)

                _ylabel = "Intensity"
                if _s4_norm:
                    _ylabel += " (norm.)"
                if _s4_ridge:
                    _ylabel += " [stacked]"

                _ax_s.set_xlim(_save_xmin, _save_xmax)
                _apply_wn_mpl(_ax_s, spacing=50.0, fontsize=7)
                _ax_s.set_ylabel(_ylabel, fontsize=11)
                _ax_s.set_title(f"m/z {selected_mz:.1f} — {_chosen_formula or '?'} ({_verdict})",
                                fontsize=12, fontweight="bold")
                _ax_s.legend(fontsize=8, loc="upper right")
                _ax_s.grid(True, alpha=0.3)
                _fig_s.tight_layout()
                _plot_path = _assign_dir / "comparison_plot.png"
                _fig_s.savefig(_plot_path, dpi=300, bbox_inches="tight")
                plt.close(_fig_s)
                _saved_files.append("comparison_plot.png")
            except Exception as _pe:
                st.warning(f"⚠️ Could not save plot: {_pe}")

        # 3b) Band-matched references plot PNG
        if _save_bm_plot and "_mid_bm_data" in st.session_state:
            _bmd = st.session_state["_mid_bm_data"]
            if abs(_bmd.get("mz", 0) - float(selected_mz)) < 1e-6 and _bmd.get("matched"):
                try:
                    _bm_matched = _bmd["matched"]
                    _bm_sel_bands = _bmd["sel_bands"]
                    _bm_ridge_save = _bmd.get("ridge", True)
                    _bm_xrange = _bmd.get("xrange")
                    _bm_pal_s = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
                                 "#e377c2", "#bcbd22", "#17becf", "#d62728", "#7f7f7f"]

                    _n_bm = 1 + len(_bm_matched)
                    _bm_fh = max(5, 2.5 + _n_bm * 1.2) if _bm_ridge_save else 5
                    _fig_bm, _ax_bm = plt.subplots(figsize=(10, _bm_fh))

                    # Shade band windows
                    for _b in _bm_sel_bands:
                        _ax_bm.axvspan(_b["lo"], _b["hi"], color="green", alpha=0.10)
                        _ax_bm.annotate(f"{_b['center']:.0f}", xy=((_b["lo"] + _b["hi"]) / 2, 0),
                                        fontsize=7, ha="center", va="bottom", color="green")

                    # Experimental trace
                    _bm_exp_wn = st.session_state.get("_mid_your_wn")
                    _bm_exp_int = st.session_state.get("_mid_your_intensity")
                    if _bm_exp_wn is not None:
                        _ax_bm.plot(_bm_exp_wn, _s4_norm_fn(_bm_exp_int), color="#d62728",
                                    lw=2, label="Experimental (−ln depl)")

                    for _mi, _s in enumerate(_bm_matched):
                        _mwn = np.asarray(_s["wn"], dtype=float)
                        _mit = np.asarray(_s["intensity"], dtype=float)
                        if _bm_exp_wn is not None:
                            _mm = (_mwn >= float(np.nanmin(_bm_exp_wn))) & (_mwn <= float(np.nanmax(_bm_exp_wn)))
                            _mwn, _mit = _mwn[_mm], _mit[_mm]
                        _off = (_mi + 1) * 1.0 if _bm_ridge_save else 0
                        _ax_bm.plot(_mwn, _s4_norm_fn(_mit) + _off,
                                    color=_bm_pal_s[_mi % len(_bm_pal_s)], lw=1.5,
                                    label=_s["name"][:40])
                        if _bm_ridge_save:
                            _ax_bm.axhline(_off, color=_bm_pal_s[_mi % len(_bm_pal_s)],
                                           lw=0.5, ls="--", alpha=0.3)

                    if _bm_xrange:
                        _ax_bm.set_xlim(_bm_xrange)
                    _apply_wn_mpl(_ax_bm, spacing=50.0, fontsize=7)
                    _ax_bm.set_ylabel("Intensity" + (" (stacked)" if _bm_ridge_save else ""),
                                      fontsize=11)
                    _ax_bm.set_title(
                        f"Band-matched references — m/z {selected_mz:.1f} — "
                        f"{_chosen_formula or '?'} ({_verdict})",
                        fontsize=12, fontweight="bold")
                    _ax_bm.legend(fontsize=8, loc="upper right")
                    _ax_bm.grid(True, alpha=0.3)
                    _fig_bm.tight_layout()
                    _bm_plot_path = _assign_dir / "band_matched_plot.png"
                    _fig_bm.savefig(_bm_plot_path, dpi=300, bbox_inches="tight")
                    plt.close(_fig_bm)
                    _saved_files.append("band_matched_plot.png")
                except Exception as _bme:
                    st.warning(f"⚠️ Could not save band-matched plot: {_bme}")

        # 4) Structure images — save every valid SMILES entry + a grid PNG
        if "_mid_structure_img" in st.session_state:
            _struct_path = _assign_dir / "structure_active.png"
            _struct_path.write_bytes(st.session_state["_mid_structure_img"])
            _saved_files.append("structure_active.png")

        if HAS_RDKIT:
            _smiles_entries = st.session_state.get(_mz_key, [])
            _struct_dir = _assign_dir / "structures"
            _struct_dir.mkdir(exist_ok=True)
            _grid_mols, _grid_labels = [], []
            for _si, _se in enumerate(_smiles_entries):
                _se_smi = _se.get("smiles", "").strip()
                _se_lbl = (_se.get("name", "").strip() or f"candidate_{_si + 1}")
                if not _se_smi:
                    continue
                try:
                    _se_mol = Chem.MolFromSmiles(_se_smi)
                    if _se_mol is None:
                        continue
                    _se_img = Draw.MolToImage(_se_mol, size=(400, 300))
                    # Sanitise label for filename
                    import re as _re2
                    _safe_lbl = _re2.sub(r"[^\w\-]", "_", _se_lbl)[:60]
                    _se_path = _struct_dir / f"{_safe_lbl}.png"
                    _se_img.save(_se_path)
                    _saved_files.append(f"structures/{_safe_lbl}.png")
                    _grid_mols.append(_se_mol)
                    _grid_labels.append(
                        f"{_se_lbl}\n{Chem.rdMolDescriptors.CalcMolFormula(_se_mol)}"
                    )
                except Exception:
                    pass

            # Save a single grid image with all structures
            if _grid_mols:
                try:
                    _ncols_g = min(len(_grid_mols), 4)
                    _grid_img = Draw.MolsToGridImage(
                        _grid_mols, molsPerRow=_ncols_g,
                        subImgSize=(400, 300), legends=_grid_labels,
                    )
                    _grid_path = _assign_dir / "structures_grid.png"
                    _grid_img.save(_grid_path)
                    _saved_files.append("structures_grid.png")
                except Exception:
                    pass

        # 5) REMPI action spectrum CSV for the assigned mass (if REMPI loaded)
        _rempi_ready = st.session_state.get("_mid_rempi_ready")
        if _rempi_ready is not None:
            try:
                _rx = np.asarray(_rempi_ready["x"], dtype=float)
                _rZ = np.asarray(_rempi_ready["Z"])
                _rwl_half = np.asarray(_rempi_ready["wl_half"], dtype=float)
                _r_tol = float(st.session_state.get("_mid_rempi_as_tol", 0.5))
                _r_idx = np.where(np.abs(_rx - float(selected_mz)) <= _r_tol)[0]
                if len(_r_idx) > 0 and _rZ.ndim == 2 and _rZ.shape[1] == len(_rwl_half):
                    _r_int = _rZ[_r_idx, :].mean(axis=0)
                    _r_order = np.argsort(_rwl_half)
                    _r_wl_s = _rwl_half[_r_order]
                    _r_raw_s = np.asarray(_r_int, dtype=float)[_r_order]
                    _r_smooth_s = _moving_avg(_r_raw_s, _rempi_ma_win)

                    _rempi_out_df = pd.DataFrame({
                        "wavelength_half_nm": _r_wl_s,
                        "intensity_au": _r_raw_s,
                        f"intensity_ma{_rempi_ma_win}": _r_smooth_s,
                    })
                    _rempi_fname = f"REMPI_action_mz{_safe_mz}_tol{_r_tol:g}.csv"
                    _rempi_out_df.to_csv(_assign_dir / _rempi_fname, index=False)
                    _saved_files.append(_rempi_fname)

                    # Store for the combined A4 page
                    _rempi_panel = {
                        "wl": _r_wl_s, "raw": _r_raw_s,
                        "smooth": _r_smooth_s, "win": _rempi_ma_win,
                    }

                    # REMPI PNG (raw faint + moving-average smoothed)
                    _fig_rp, _ax_rp = plt.subplots(figsize=(10, 4))
                    if _rempi_ma_win > 1:
                        _ax_rp.plot(_r_wl_s, _r_raw_s, color="#c9c9c9", lw=1,
                                    label="Raw")
                    _ax_rp.plot(_r_wl_s, _r_smooth_s, color="#7b2cbf", lw=2,
                                label=(f"MA smoothed (w={_rempi_ma_win})"
                                       if _rempi_ma_win > 1 else "REMPI"))
                    _ax_rp.axhline(0, color="gray", ls="--", lw=1)
                    _ax_rp.set_ylabel("Ion intensity (a.u.)", fontsize=11)
                    _ax_rp.set_title(
                        f"REMPI action spectrum — m/z {selected_mz:.1f} "
                        f"(±{_r_tol:g} amu)", fontsize=12, fontweight="bold")
                    _ax_rp.legend(fontsize=8, loc="upper right")
                    _ax_rp.grid(True, alpha=0.3)
                    _apply_wl_mpl(_ax_rp, fontsize=7)
                    _fig_rp.tight_layout()
                    _rempi_png = f"REMPI_action_mz{_safe_mz}_ma{_rempi_ma_win}.png"
                    _fig_rp.savefig(_assign_dir / _rempi_png, dpi=300,
                                    bbox_inches="tight")
                    plt.close(_fig_rp)
                    _saved_files.append(_rempi_png)
                else:
                    st.info(
                        f"ℹ️ No REMPI mass bins within ±{_r_tol:g} amu of "
                        f"{selected_mz:.1f} — REMPI action spectrum not saved."
                    )
            except Exception as _re_e:
                st.warning(f"⚠️ Could not save REMPI action spectrum: {_re_e}")

        # 6) + 7) Mass-spectrum PNG + CSV around the assigned mass, showing the
        #         integration window(s) and pointing at the selected mass. The
        #         view is zoomed enough to resolve neighbouring masses but wide
        #         enough to place the peak in context.
        try:
            _x_ms_all = np.asarray(x_mass, dtype=float)
            _y_ms_all = np.asarray(ms_y, dtype=float)
            _ms_center = float(np.mean(iso_centers))
            # Span covers all isotope centres + a ~±5 amu margin (min 10 amu wide).
            _ms_span = max(10.0, (max(iso_centers) - min(iso_centers)) + 10.0)
            _ms_zoom_lo = max(float(np.min(_x_ms_all)), _ms_center - _ms_span / 2.0)
            _ms_zoom_hi = min(float(np.max(_x_ms_all)), _ms_center + _ms_span / 2.0)
            _ms_zoom_mask = (_x_ms_all >= _ms_zoom_lo) & (_x_ms_all <= _ms_zoom_hi)
            _win_mask = _build_iso_mask(iso_centers, half_width)

            # ── PNG ───────────────────────────────────────────────────────────
            _fig_ms_s, _ax_ms_s = plt.subplots(figsize=(10, 4))
            _ax_ms_s.plot(_x_ms_all[_ms_zoom_mask], _y_ms_all[_ms_zoom_mask],
                          color="#1f77b4", lw=1.2, label="Mass spectrum")
            _y_zoom = _y_ms_all[_ms_zoom_mask]
            _y_top = float(np.nanmax(_y_zoom)) if _y_zoom.size and np.isfinite(np.nanmax(_y_zoom)) else 1.0
            for _c in iso_centers:
                _ax_ms_s.axvspan(_c - half_width, _c + half_width,
                                 color="#d62728", alpha=0.20)
                _ax_ms_s.axvline(_c, color="#d62728", ls="--", lw=1)
            # Arrow pointing at the primary selected mass
            _y_at_sel = float(np.interp(selected_mz, _x_ms_all, _y_ms_all))
            _ax_ms_s.annotate(
                f"m/z {selected_mz:.2f}\n±{half_width:g} amu",
                xy=(float(selected_mz), _y_at_sel),
                xytext=(float(selected_mz), _y_top * 1.08 + 1e-9),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                ha="center", va="bottom", fontsize=9, color="#d62728",
            )
            _ax_ms_s.set_xlim(_ms_zoom_lo, _ms_zoom_hi)
            _ax_ms_s.set_xlabel("m/z", fontsize=11)
            _ax_ms_s.set_ylabel("Intensity (a.u.)", fontsize=11)
            _ax_ms_s.set_title(
                f"Mass spectrum — m/z {selected_mz:.1f} "
                f"({_chosen_formula or '?'}, ±{half_width:g} amu window)",
                fontsize=12, fontweight="bold",
            )
            _ax_ms_s.grid(True, alpha=0.3)
            _fig_ms_s.tight_layout()
            _ms_png = f"mass_spectrum_mz{_safe_mz}_hw{half_width:g}.png"
            _fig_ms_s.savefig(_assign_dir / _ms_png, dpi=300, bbox_inches="tight")
            plt.close(_fig_ms_s)
            _saved_files.append(_ms_png)

            # ── CSV (zoomed region; flags which bins are inside the window) ────
            _ms_csv_df = pd.DataFrame({
                "m/z": _x_ms_all[_ms_zoom_mask],
                "intensity": _y_ms_all[_ms_zoom_mask],
                "in_integration_window": _win_mask[_ms_zoom_mask],
            })
            _ms_csv = f"mass_spectrum_mz{_safe_mz}_hw{half_width:g}.csv"
            _ms_csv_df.to_csv(_assign_dir / _ms_csv, index=False)
            _saved_files.append(_ms_csv)

            # Store for the combined A4 page
            _ms_panel = {
                "x": _x_ms_all[_ms_zoom_mask], "y": _y_zoom,
                "lo": _ms_zoom_lo, "hi": _ms_zoom_hi,
                "y_top": _y_top, "y_at_sel": _y_at_sel,
            }
        except Exception as _ms_e:
            st.warning(f"⚠️ Could not save mass spectrum: {_ms_e}")

        # 8) Combined vertical A4 page: structure + band-matched + REMPI + mass
        if _save_a4:
            try:
                _a4_fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait (inches)
                _gs = _a4_fig.add_gridspec(4, 1, height_ratios=[1.1, 1.2, 1.0, 1.0],
                                           hspace=0.5)
                _a4_fig.suptitle(
                    f"m/z {selected_mz:.2f} — {_chosen_formula or '?'} ({_verdict})",
                    fontsize=14, fontweight="bold", y=0.995,
                )

                def _a4_norm(_y):
                    _a = np.asarray(_y, dtype=float)
                    _f = _a[np.isfinite(_a)]
                    if len(_f) == 0 or np.ptp(_f) == 0:
                        return _a
                    return (_a - np.nanmin(_f)) / np.ptp(_f)

                # ── Panel 1: structure (single active, or grid of all) ───
                _ax1 = _a4_fig.add_subplot(_gs[0])
                _ax1.axis("off")
                _struct_bytes = st.session_state.get("_mid_structure_img")
                _export_mols_a4 = st.session_state.get("_mid_export_mols", [])
                _drew_struct = False
                from PIL import Image as _PILImage

                # Grid mode: assemble every selected structure into one image.
                if _a4_struct_grid and HAS_RDKIT and len(_export_mols_a4) >= 1:
                    try:
                        _g_mols = [v[2] for v in _export_mols_a4]
                        _g_legends = [f"{v[1]} ({v[3]})" for v in _export_mols_a4]
                        _ncols_a4 = min(len(_g_mols), 3)
                        _grid_img = Draw.MolsToGridImage(
                            _g_mols, molsPerRow=_ncols_a4,
                            subImgSize=(320, 240), legends=_g_legends,
                        )
                        _buf_g = io.BytesIO()
                        _grid_img.save(_buf_g, format="PNG")
                        _ax1.imshow(_PILImage.open(io.BytesIO(_buf_g.getvalue())))
                        _ax1.set_title(
                            f"Structures ({len(_g_mols)})", fontsize=11,
                            fontweight="bold")
                        _drew_struct = True
                    except Exception:
                        _drew_struct = False

                # Single-image mode (or grid fallback): show the active structure.
                if not _drew_struct and _struct_bytes:
                    try:
                        _ax1.imshow(_PILImage.open(io.BytesIO(_struct_bytes)))
                        _ax1.set_title("Structure", fontsize=11, fontweight="bold")
                        _drew_struct = True
                    except Exception:
                        _drew_struct = False

                if not _drew_struct:
                    _ax1.set_title("Structure", fontsize=11, fontweight="bold")
                    _ax1.text(0.5, 0.5, "No structure image", ha="center",
                              va="center", fontsize=10, color="gray",
                              transform=_ax1.transAxes)

                # ── Panel 2: band-matched references ─────────────────────
                _ax2 = _a4_fig.add_subplot(_gs[1])
                _ax2.set_title("Band-matched references", fontsize=11,
                               fontweight="bold")
                _bmd = st.session_state.get("_mid_bm_data")
                _bm_exp_wn = st.session_state.get("_mid_your_wn")
                _bm_exp_int = st.session_state.get("_mid_your_intensity")
                _drew_bm = False
                if (_bmd is not None
                        and abs(_bmd.get("mz", -1) - float(selected_mz)) < 1e-6
                        and _bmd.get("matched")):
                    _bm_matched = _bmd["matched"]
                    _bm_sel_bands = _bmd.get("sel_bands", [])
                    _bm_ridge = _bmd.get("ridge", True)
                    _bm_xr = _bmd.get("xrange")
                    _bm_pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
                               "#8c564b", "#e377c2", "#bcbd22", "#17becf",
                               "#d62728", "#7f7f7f"]
                    for _b in _bm_sel_bands:
                        _ax2.axvspan(_b["lo"], _b["hi"], color="green", alpha=0.10)
                    if _bm_exp_wn is not None:
                        _ax2.plot(_bm_exp_wn, _a4_norm(_bm_exp_int),
                                  color="#d62728", lw=1.8, label="Experimental")
                    for _mi, _s in enumerate(_bm_matched):
                        _mwn = np.asarray(_s["wn"], dtype=float)
                        _mit = np.asarray(_s["intensity"], dtype=float)
                        if _bm_exp_wn is not None:
                            _mm = ((_mwn >= float(np.nanmin(_bm_exp_wn)))
                                   & (_mwn <= float(np.nanmax(_bm_exp_wn))))
                            _mwn, _mit = _mwn[_mm], _mit[_mm]
                        _off = (_mi + 1) * 1.0 if _bm_ridge else 0
                        _ax2.plot(_mwn, _a4_norm(_mit) + _off,
                                  color=_bm_pal[_mi % len(_bm_pal)], lw=1.2,
                                  label=_s["name"][:30])
                    if _bm_xr:
                        _ax2.set_xlim(_bm_xr)
                    _apply_wn_mpl(_ax2, spacing=50.0, fontsize=6)
                    _ax2.set_ylabel("Intensity" + (" [stacked]" if _bm_ridge else ""),
                                    fontsize=10)
                    _ax2.legend(fontsize=6, loc="upper right", ncol=2)
                    _ax2.grid(True, alpha=0.3)
                    _drew_bm = True
                if not _drew_bm:
                    _ax2.text(0.5, 0.5, "No band-matched data", ha="center",
                              va="center", fontsize=10, color="gray",
                              transform=_ax2.transAxes)
                    _ax2.axis("off")

                # ── Panel 3: REMPI (moving-average smoothed) ─────────────
                _ax3 = _a4_fig.add_subplot(_gs[2])
                _ax3.set_title("REMPI action spectrum", fontsize=11,
                               fontweight="bold")
                if _rempi_panel is not None:
                    if _rempi_panel["win"] > 1:
                        _ax3.plot(_rempi_panel["wl"], _rempi_panel["raw"],
                                  color="#c9c9c9", lw=1, label="Raw")
                    _ax3.plot(_rempi_panel["wl"], _rempi_panel["smooth"],
                              color="#7b2cbf", lw=2,
                              label=(f"MA (w={_rempi_panel['win']})"
                                     if _rempi_panel["win"] > 1 else "REMPI"))
                    _ax3.axhline(0, color="gray", ls="--", lw=1)
                    _ax3.set_ylabel("Ion intensity (a.u.)", fontsize=10)
                    _ax3.legend(fontsize=7, loc="upper right")
                    _ax3.grid(True, alpha=0.3)
                    _apply_wl_mpl(_ax3, fontsize=6)
                else:
                    _ax3.text(0.5, 0.5, "No REMPI data", ha="center", va="center",
                              fontsize=10, color="gray", transform=_ax3.transAxes)
                    _ax3.axis("off")

                # ── Panel 4: mass spectrum ───────────────────────────────
                _ax4 = _a4_fig.add_subplot(_gs[3])
                _ax4.set_title("Mass spectrum", fontsize=11, fontweight="bold")
                if _ms_panel is not None:
                    _ax4.plot(_ms_panel["x"], _ms_panel["y"], color="#1f77b4", lw=1.2)
                    for _c in iso_centers:
                        _ax4.axvspan(_c - half_width, _c + half_width,
                                     color="#d62728", alpha=0.20)
                        _ax4.axvline(_c, color="#d62728", ls="--", lw=1)
                    _ax4.annotate(
                        f"m/z {selected_mz:.2f}\n±{half_width:g} amu",
                        xy=(float(selected_mz), _ms_panel["y_at_sel"]),
                        xytext=(float(selected_mz), _ms_panel["y_top"] * 1.08 + 1e-9),
                        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
                        ha="center", va="bottom", fontsize=8, color="#d62728",
                    )
                    _ax4.set_xlim(_ms_panel["lo"], _ms_panel["hi"])
                    _ax4.set_xlabel("m/z", fontsize=10)
                    _ax4.set_ylabel("Intensity (a.u.)", fontsize=10)
                    _ax4.grid(True, alpha=0.3)
                else:
                    _ax4.text(0.5, 0.5, "No mass spectrum data", ha="center",
                              va="center", fontsize=10, color="gray",
                              transform=_ax4.transAxes)
                    _ax4.axis("off")

                _a4_pdf = f"summary_A4_mz{_safe_mz}.pdf"
                _a4_png = f"summary_A4_mz{_safe_mz}.png"
                _a4_fig.savefig(_assign_dir / _a4_pdf, bbox_inches="tight")
                _a4_fig.savefig(_assign_dir / _a4_png, dpi=200, bbox_inches="tight")
                plt.close(_a4_fig)
                _saved_files.append(_a4_pdf)
                _saved_files.append(_a4_png)
            except Exception as _a4_e:
                st.warning(f"⚠️ Could not save combined A4 page: {_a4_e}")

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
        # Raw (non-smoothed) experimental drawn first in grey (offset 0)
        if _smooth_ref:
            _yr_raw = _norm(your_intensity) if _normalize else np.asarray(your_intensity, dtype=float)
            ax_ir.plot(your_wn, _yr_raw, color="#c9c9c9", lw=1, label="Experimental (raw)")
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

        # Panel 3: Structure image(s) — grid if multiple selected
        if ax_struct is not None and has_struct:
            from PIL import Image
            _pub_export_mols = st.session_state.get("_mid_export_mols", [])
            if HAS_RDKIT and len(_pub_export_mols) > 1:
                # Draw a grid of all selected structures into the panel
                _pg_mols   = [v[2] for v in _pub_export_mols]
                _pg_labels = [f"{v[1]}\n{v[3]}" for v in _pub_export_mols]
                _pg_ncols  = min(len(_pg_mols), 4)
                _pg_img = Draw.MolsToGridImage(
                    _pg_mols, molsPerRow=_pg_ncols,
                    subImgSize=(300, 220), legends=_pg_labels,
                )
                ax_struct.imshow(_pg_img)
            else:
                _img_data = st.session_state["_mid_structure_img"]
                _pil = Image.open(io.BytesIO(_img_data))
                ax_struct.imshow(_pil)
            _struct_title = "Structures" if len(_pub_export_mols) > 1 else "Structure"
            ax_struct.axis("off")
            ax_struct.set_title(_struct_title, fontsize=12, fontweight="bold")

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


# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — AI MECHANISM PROMPT (data summary for Claude Opus etc.)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 7. AI Mechanism Prompt")
st.caption(
    "Auto-builds a copy-paste prompt summarising your *actual* mass spectrum and "
    "assignments so an LLM can propose reachable product structures for each m/z."
)

with st.expander("🤖 Build LLM prompt from this dataset", expanded=False):
    _pp_c1, _pp_c2 = st.columns(2)
    with _pp_c1:
        _prec_text = st.text_area(
            "Precursor SMILES (one per line, `name: SMILES`)",
            value="Bromobenzene: c1ccccc1Br\n1,2-Dibromopropane: BrCC(Br)C",
            height=90, key="_mid_prec_smiles",
        )
    with _pp_c2:
        _constraints = st.text_area(
            "Detection constraints",
            value=(
                "Detection is 1+1' REMPI via ArF excimer — only UV-active closed-shell "
                "aromatic molecules are observed.\n"
                "Products form in a supersonic free-jet expansion after discharge — "
                "single-collision regime, no solution chemistry, no high-pressure rearrangements."
            ),
            height=90, key="_mid_prompt_constraints",
        )

    # ── Pull detected peaks + relative intensities from the actual spectrum ──
    if len(_peaks) > 0:
        _mz_vals  = _x_disp[_peaks]
        _int_vals = _y_disp[_peaks]
        _imax = float(np.nanmax(_int_vals)) if np.nanmax(_int_vals) > 0 else 1.0

        def _intensity_label(_frac):
            if _frac >= 0.66:
                return "strong"
            if _frac >= 0.33:
                return "medium"
            if _frac >= 0.10:
                return "weak"
            return "trace"

        # Map saved assignments by rounded m/z for annotation
        _assign_by_mz = {}
        for _a in st.session_state.get("_mass_assignments", {}).values():
            try:
                _assign_by_mz[round(float(_a.get("mz", -1)))] = _a
            except (TypeError, ValueError):
                pass

        _order = np.argsort(_mz_vals)
        _mass_lines, _anchor_lines = [], []
        for _idx in _order:
            _m = float(_mz_vals[_idx])
            _frac = float(_int_vals[_idx]) / _imax
            _lab = _intensity_label(_frac)
            _a = _assign_by_mz.get(round(_m))
            _tag = ""
            if _a and _a.get("formula"):
                _verd = _a.get("verdict", "")
                _tag = f"  ← assigned: {_a['formula']}" + (f" ({_verd})" if _verd else "")
                _anchor_lines.append(
                    f"m/z {_m:.0f} = {_a['formula']}"
                    + (f" [{_verd}]" if _verd else "")
                    + (f" — {_a['notes']}" if _a.get("notes") else "")
                )
            _mass_lines.append(f"m/z {_m:6.1f}  ({_lab}, {_frac*100:4.1f}% of base peak){_tag}")

        _mass_block = "\n".join(_mass_lines)
        _anchor_block = "\n".join(_anchor_lines) if _anchor_lines else "(none assigned yet)"

        # ── Assemble the full prompt ──
        _prompt = f"""I am analysing products from an electric-discharge / supersonic-jet experiment and need help proposing reachable product structures for each observed mass.

## 1. Precursor SMILES
{_prec_text.strip()}

## 2. Complete observed mass list (with relative intensities)
{_mass_block}

## 3. Known anchor assignments
{_anchor_block}

## 4. Detection constraints
{_constraints.strip()}

## What I need from you
For each observed m/z above:
- Extract the primary radical fragments from each precursor.
- Apply known gas-phase radical-combination rules (recombination, HACA, propargyl addition, ring closure).
- Walk up the mass ladder step by step, keeping consistency with the anchor assignments.
- Output only the 3–6 structures reachable from MY specific precursor fragments by simple bond-forming steps.
- Flag which candidates are UV-active (observable here) and which are not.
- Rank by mechanistic plausibility, and use the relative intensities as a clue to reaction efficiency.
- Give each candidate as a SMILES string and a one-line mechanistic rationale.
"""
        st.markdown("**Generated prompt** — copy or download and paste into Claude Opus:")
        st.code(_prompt, language="markdown")
        st.download_button(
            "📥 Download prompt (.md)", data=_prompt,
            file_name=f"mechanism_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown", key="_mid_dl_prompt",
        )
        st.caption(
            f"Summarised {len(_mass_lines)} detected peaks and "
            f"{len(_anchor_lines)} anchor assignment(s). Intensities are relative to the "
            f"base (most intense) detected peak. Adjust peak detection in Section 1 to "
            f"change which masses are included."
        )
    else:
        st.info("No peaks detected in Section 1 — adjust the peak prominence/distance there first.")
