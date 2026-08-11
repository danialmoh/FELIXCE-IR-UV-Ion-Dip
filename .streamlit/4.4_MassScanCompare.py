"""
4.4 Quick Mass-Scan Comparison
==============================

Standalone overlay tool: drop in one (or more) raw FELIX `.h5` mass scans and
compare them against the mass spectrum you are currently working on, without
running the Section 1 -> 2 -> 3 pipeline.

IMPORTANT: this page never writes to any pipeline session-state key
(`x_mass`, `compilation_baseline_corrected_data`, `MegaSum`, ...).
Everything it stores is namespaced with the `_msc_` prefix, so your ongoing
workflow cannot be disturbed.
"""

import configparser
import gzip
import io
import os
import pickle
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from scipy.signal import find_peaks

from packages.BaselineCorrection_REMPI import baseline_REMPI
from packages.FELIX_HDF5_ReadData import ReadData_FELIX_HDF5

st.title("⚖️ 4.4 Quick Mass-Scan Comparison")
st.caption(
    "Overlay extra raw `.h5` mass scans on your current mass spectrum. "
    "Read-only with respect to the main pipeline — nothing here changes your workflow state."
)

DEFAULTS_FILE = r"./.streamlit/defaults.ini"


# ========================================================================================
# HELPERS
# ========================================================================================
def _load_defaults():
    """Read calibration / baseline defaults from defaults.ini (best effort)."""
    out = {
        "t_off": 58.0,
        "alpha": 7.6987e-7,
        "baseline_reference": 172.0,
        "baseline_width": 4.0,
        "mass_xmin": 0.0,
        "mass_xmax": 1300.0,
        "file_directory": "",
    }
    cfg = configparser.ConfigParser()
    if os.path.exists(DEFAULTS_FILE):
        try:
            cfg.read(DEFAULTS_FILE)
            out["t_off"] = cfg.getfloat("Experiment Parameters", "t_off", fallback=out["t_off"])
            out["alpha"] = cfg.getfloat("Experiment Parameters", "alpha", fallback=out["alpha"])
            out["baseline_reference"] = cfg.getfloat(
                "Baseline Parameters", "baseline_reference", fallback=out["baseline_reference"]
            )
            out["baseline_width"] = cfg.getfloat(
                "Baseline Parameters", "baseline_width", fallback=out["baseline_width"]
            )
            out["mass_xmin"] = cfg.getfloat("Plot Parameters", "mass_xmin", fallback=out["mass_xmin"])
            out["mass_xmax"] = cfg.getfloat("Plot Parameters", "mass_xmax", fallback=out["mass_xmax"])
            out["file_directory"] = cfg.get("Import Data", "file_directory", fallback="")
        except (configparser.Error, ValueError) as exc:
            st.warning(f"Could not fully read `defaults.ini`: {exc}")
    return out


def _mass_axis(n_points, alpha, t_off):
    """Same calibration as Section 1.4: m/z = alpha * (counts - t_off)^2."""
    counts = np.linspace(1, n_points, n_points)
    return alpha * (counts - t_off) ** 2


def _read_h5_mass_spectrum(file_like, ir_mode):
    """
    Sum every wavenumber trace of a FELIX .h5 file into a single mass spectrum.

    Returns (y, n_wavenumbers, n_points).
    `ir_mode` selects which detector gate to use:
      'without IR' -> column 0, 'with IR' -> column 1, 'both (sum)' -> 0 + 1.
    """
    with h5py.File(file_like, "r") as handle:
        reader = ReadData_FELIX_HDF5(handle)
        reader.extract_wavenumbers()
        signal = reader.extract_signal()

    if signal is None or len(signal) == 0:
        raise ValueError("No traces found under 'Rawdat'.")

    traces = []
    for trace in signal:
        arr = np.asarray(trace, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if ir_mode == "without IR" or arr.shape[1] == 1:
            traces.append(-arr[:, 0])
        elif ir_mode == "with IR":
            traces.append(-arr[:, 1])
        else:
            traces.append(-(arr[:, 0] + arr[:, 1]))

    n_points = min(len(t) for t in traces)
    stacked = np.vstack([t[:n_points] for t in traces])
    return stacked.sum(axis=0), len(traces), n_points


def _load_bundle(path):
    """Load a baseline_corrected_full_dataset.pkl.gz, tolerating NumPy 1/2 pickles."""
    with gzip.open(path, "rb") as fh:
        try:
            return pickle.load(fh)
        except (ModuleNotFoundError, AttributeError):
            fh.seek(0)
            from packages.load_dataset import _NumpyCompatUnpickler

            return _NumpyCompatUnpickler(fh).load()


def _megasum_from_compilation(compilation, unique_wavenumbers, ir_mode):
    """Sum a baseline-corrected compilation dict into one mass spectrum."""
    frames = [compilation[wn] for wn in unique_wavenumbers if wn in compilation]
    if not frames:
        return None
    mega = pd.concat(frames, axis=1)
    without_ir = mega.iloc[:, 0::2].sum(axis=1)
    with_ir = mega.iloc[:, 1::2].sum(axis=1)
    if ir_mode == "without IR":
        return without_ir.to_numpy(dtype=float)
    if ir_mode == "with IR":
        return with_ir.to_numpy(dtype=float)
    return (without_ir + with_ir).to_numpy(dtype=float)


BASELINE_METHODS = ["None", "Mean Subtraction", "iarpls", "aspls", "fabc", "als"]


@st.cache_data(show_spinner="Fitting baseline…")
def _estimate_baseline_cached(x, y, method, param_items):
    """Cached wrapper — the pybaselines/ALS solvers are slow on ~60k-point spectra."""
    return _estimate_baseline(x, y, method, dict(param_items))


def _estimate_baseline(x, y, method, params):
    """
    Estimate a baseline for one mass spectrum.

    Mirrors the options of Section 2.0 (`packages.BaselineCorrection_v2.baseline`):
    a flat mean over a signal-free window, or the pybaselines fitters
    (`iarpls` / `aspls` / `fabc`), plus the ALS fitter from
    `packages.BaselineCorrection_REMPI.baseline_REMPI.als_baseline`.

    Returns a baseline array the same length as `y`.
    """
    y = np.asarray(y, dtype=float)
    m = (method or "None").lower()

    if m == "none":
        return np.zeros_like(y)

    if m == "mean subtraction":
        lo = params["bl_start"]
        hi = params["bl_start"] + params["bl_width"]
        mask = (x >= lo) & (x <= hi)
        if not mask.any():
            return np.zeros_like(y)
        return np.full_like(y, float(np.nanmean(y[mask])))

    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if m == "als":
        return baseline_REMPI.als_baseline(
            y_clean, lam=params["als_lam"], p=params["als_p"], niter=int(params["als_niter"])
        )

    from pybaselines import Baseline

    fitter = Baseline(x_data=np.asarray(x, dtype=float))
    if m == "iarpls":
        bl, _ = fitter.iarpls(y_clean, lam=params["iarpls_lam"])
    elif m == "aspls":
        bl, _ = fitter.aspls(y_clean, lam=params["aspls_lam"])
    elif m == "fabc":
        bl, _ = fitter.fabc(y_clean, lam=params["fabc_lam"], scale=params["fabc_scale"])
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    return np.asarray(bl, dtype=float)


def _normalize(x, y, mode, lo, hi, target_mz, target_width):
    """Scale a spectrum. Returns (y_scaled, factor)."""
    if mode == "None (raw counts)":
        return y, 1.0

    if mode == "Max in plotted range":
        mask = (x >= lo) & (x <= hi)
    elif mode == "Max at reference m/z":
        mask = (x >= target_mz - target_width) & (x <= target_mz + target_width)
    else:  # Area in plotted range
        mask = (x >= lo) & (x <= hi)

    if not mask.any():
        return y, 1.0

    if mode == "Area in plotted range":
        ref = float(np.nansum(np.abs(y[mask])))
    else:
        ref = float(np.nanmax(y[mask]))

    if not np.isfinite(ref) or ref == 0:
        return y, 1.0
    return y / ref, 1.0 / ref


defaults = _load_defaults()


# ========================================================================================
# 1. REFERENCE ("current") SPECTRUM
# ========================================================================================
st.markdown("## 1. Your current mass spectrum (reference)")

has_session_data = (
    st.session_state.get("x_mass") is not None
    and (
        st.session_state.get("MegaSum") is not None
        or st.session_state.get("compilation_baseline_corrected_data") is not None
    )
)

ref_source = st.radio(
    "Reference source",
    options=["Current session (read-only)", "Exported dataset (.pkl.gz)", "None — only plot the new scans"],
    index=0 if has_session_data else 1,
    horizontal=True,
    key="_msc_ref_source",
)

ir_mode = st.selectbox(
    "Which gate to plot",
    options=["without IR", "with IR", "both (sum)"],
    index=0,
    key="_msc_ir_mode",
    help="Applied to both the reference and the newly loaded scans so the comparison stays fair.",
)

ref_x = None
ref_y = None
ref_label = "Current spectrum"

if ref_source == "Current session (read-only)":
    if not has_session_data:
        st.warning("No pipeline data in session. Pick another reference source, or run Sections 1–2 first.")
    else:
        ref_x = np.asarray(st.session_state["x_mass"], dtype=float)
        mega = st.session_state.get("MegaSum")
        if mega is not None and "baseline_corrected_signal_withoutIR" in mega.columns:
            if ir_mode == "without IR":
                ref_y = mega["baseline_corrected_signal_withoutIR"].to_numpy(dtype=float)
            elif ir_mode == "with IR":
                ref_y = mega["baseline_corrected_signal_withIR"].to_numpy(dtype=float)
            else:
                ref_y = (
                    mega["baseline_corrected_signal_withoutIR"].to_numpy(dtype=float)
                    + mega["baseline_corrected_signal_withIR"].to_numpy(dtype=float)
                )
        else:
            ref_y = _megasum_from_compilation(
                st.session_state["compilation_baseline_corrected_data"],
                st.session_state.get("unique_wavenumbers", []),
                ir_mode,
            )
        if ref_y is not None:
            st.success(f"✅ Reference taken from session ({len(ref_x)} m/z bins).")

elif ref_source == "Exported dataset (.pkl.gz)":
    bundle_path = st.text_input(
        "Path to `baseline_corrected_full_dataset.pkl.gz`",
        value=st.session_state.get("_msc_bundle_path", defaults["file_directory"]),
        key="_msc_bundle_path_input",
        help="Exported by Section 2.1. Loaded into this page only — pipeline keys are untouched.",
    )
    if st.button("📥 Load reference dataset", key="_msc_load_bundle"):
        if not bundle_path or not os.path.exists(bundle_path):
            st.error(f"❌ File not found: `{bundle_path}`")
        else:
            try:
                bundle = _load_bundle(bundle_path)
                st.session_state["_msc_ref_x"] = np.asarray(bundle["x_mass"], dtype=float)
                st.session_state["_msc_ref_compilation"] = bundle["compilation_baseline_corrected_data"]
                st.session_state["_msc_ref_wavenumbers"] = bundle["unique_wavenumbers"]
                st.session_state["_msc_bundle_path"] = bundle_path
                st.success(f"✅ Loaded `{os.path.basename(bundle_path)}`")
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                st.error(f"❌ Failed to load: {exc}")

    if st.session_state.get("_msc_ref_x") is not None:
        ref_x = st.session_state["_msc_ref_x"]
        ref_y = _megasum_from_compilation(
            st.session_state["_msc_ref_compilation"],
            st.session_state["_msc_ref_wavenumbers"],
            ir_mode,
        )
        ref_label = os.path.basename(st.session_state.get("_msc_bundle_path", "reference"))

if ref_y is not None:
    ref_label = st.text_input("Reference legend label", value=ref_label, key="_msc_ref_label")


# ========================================================================================
# 2. EXTRA RAW SCANS
# ========================================================================================
st.markdown("---")
st.markdown("## 2. Extra mass scan(s) to compare")

uploads = st.file_uploader(
    "FELIX `.h5` mass scan(s)",
    type=["h5", "hdf5"],
    accept_multiple_files=True,
    key="_msc_uploads",
)

cal1, cal2 = st.columns(2)
with cal1:
    alpha = float(
        st.text_input(
            "alpha (calibration)",
            value=str(st.session_state.get("alpha", defaults["alpha"])),
            key="_msc_alpha",
            help="Defaults to the session/ini value so the new scan lands on the same mass axis.",
        )
    )
with cal2:
    t_off = st.number_input(
        "t_off (calibration)",
        value=float(st.session_state.get("t_off", defaults["t_off"])),
        format="%.4f",
        key="_msc_t_off",
    )

extras = []
if uploads:
    for upload in uploads:
        try:
            y_raw, n_wn, n_pts = _read_h5_mass_spectrum(io.BytesIO(upload.getvalue()), ir_mode)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"❌ `{upload.name}`: {exc}")
            continue
        extras.append(
            {
                "label": upload.name[:-3] if upload.name.endswith(".h5") else upload.name,
                "x": _mass_axis(n_pts, alpha, t_off),
                "y": y_raw,
                "n_wavenumbers": n_wn,
            }
        )

    if extras:
        st.success(
            "✅ Loaded: "
            + ", ".join(f"`{e['label']}` ({e['n_wavenumbers']} wn steps)" for e in extras)
        )

if ref_y is None and not extras:
    st.info("Load a reference and/or at least one `.h5` scan to see the overlay.")
    st.stop()


# ========================================================================================
# 3. PROCESSING OPTIONS
# ========================================================================================
st.markdown("---")
st.markdown("## 3. Baseline, normalization & range")

all_x = [e["x"] for e in extras] + ([ref_x] if ref_x is not None else [])
x_lo_data = float(min(np.nanmin(x) for x in all_x))
x_hi_data = float(max(np.nanmax(x) for x in all_x))

rng1, rng2 = st.columns(2)
with rng1:
    x_lo = st.number_input(
        "m/z min", value=max(x_lo_data, defaults["mass_xmin"]), step=5.0, key="_msc_x_lo"
    )
with rng2:
    x_hi = st.number_input(
        "m/z max", value=min(x_hi_data, defaults["mass_xmax"]), step=5.0, key="_msc_x_hi"
    )

st.markdown("### Baseline")
_pipeline_method = st.session_state.get("baseline_method", "iarpls")
bl_method = st.selectbox(
    "Baseline method",
    options=BASELINE_METHODS,
    index=BASELINE_METHODS.index(_pipeline_method) if _pipeline_method in BASELINE_METHODS else 2,
    key="_msc_bl_method",
    help=(
        "**Mean Subtraction**: flat offset from one signal-free window — only valid if the spectrum truly "
        "returns to a constant level. **iarpls / aspls / fabc**: the same pybaselines fitters as Section 2.0, "
        "they follow a curved/sloping baseline across a crowded mass spectrum. **als**: the ALS fitter from "
        "`BaselineCorrection_REMPI`."
    ),
)
do_baseline = bl_method != "None"

bl_params = {
    "bl_start": float(defaults["baseline_reference"]),
    "bl_width": float(defaults["baseline_width"]),
    "iarpls_lam": float(st.session_state.get("iarpls_lam", 1e6)),
    "aspls_lam": float(st.session_state.get("aspls_lam", 1e6)),
    "fabc_lam": float(st.session_state.get("fabc_lam", 1e6)),
    "fabc_scale": None,
    "als_lam": 1e6,
    "als_p": 0.01,
    "als_niter": 10,
}

if bl_method == "Mean Subtraction":
    st.info(
        "⚠️ With many mass channels a single flat offset rarely fits — if the subtraction looks wrong, "
        "switch to `iarpls`, `aspls`, `fabc` or `als`."
    )
    bl1, bl2 = st.columns(2)
    with bl1:
        bl_params["bl_start"] = st.number_input(
            "Baseline window start (m/z)",
            value=float(defaults["baseline_reference"]),
            step=1.0,
            key="_msc_bl_start",
        )
    with bl2:
        bl_params["bl_width"] = st.number_input(
            "Baseline window width (m/z)",
            value=float(defaults["baseline_width"]),
            min_value=0.1,
            step=0.5,
            key="_msc_bl_width",
        )
elif bl_method == "iarpls":
    bl_params["iarpls_lam"] = st.number_input(
        "iarpls λ (smoothness)",
        value=float(st.session_state.get("iarpls_lam", 1e6)),
        min_value=1e2,
        step=1e5,
        format="%.0e",
        key="_msc_iarpls_lam",
        help="Higher = smoother baseline that ignores narrow peaks. Increase if the baseline cuts into peaks.",
    )
elif bl_method == "aspls":
    bl_params["aspls_lam"] = st.number_input(
        "aspls λ (smoothness)",
        value=float(st.session_state.get("aspls_lam", 1e6)),
        min_value=1e2,
        step=1e5,
        format="%.0e",
        key="_msc_aspls_lam",
    )
elif bl_method == "fabc":
    fc1, fc2 = st.columns(2)
    with fc1:
        bl_params["fabc_lam"] = st.number_input(
            "fabc λ",
            value=float(st.session_state.get("fabc_lam", 1e6)),
            min_value=1e2,
            step=1e5,
            format="%.0e",
            key="_msc_fabc_lam",
        )
    with fc2:
        _scale = st.number_input(
            "fabc scale (0 = automatic)", value=0, min_value=0, step=1, key="_msc_fabc_scale"
        )
        bl_params["fabc_scale"] = int(_scale) if _scale > 0 else None
elif bl_method == "als":
    al1, al2, al3 = st.columns(3)
    with al1:
        bl_params["als_lam"] = st.number_input(
            "ALS λ", value=1e6, min_value=1e2, step=1e5, format="%.0e", key="_msc_als_lam"
        )
    with al2:
        bl_params["als_p"] = st.number_input(
            "ALS p (asymmetry)", value=0.01, min_value=0.0001, max_value=0.5,
            step=0.005, format="%.4f", key="_msc_als_p"
        )
    with al3:
        bl_params["als_niter"] = st.number_input(
            "ALS iterations", value=10, min_value=1, step=1, key="_msc_als_niter"
        )

show_baseline = st.checkbox(
    "Show fitted baselines on the overlay (pre-normalization)", value=False, key="_msc_show_bl"
)

norm_mode = st.selectbox(
    "Normalization",
    options=["Max in plotted range", "Max at reference m/z", "Area in plotted range", "None (raw counts)"],
    index=0,
    key="_msc_norm_mode",
    help="Absolute counts are rarely comparable between scans — normalize unless you know they are.",
)
nm1, nm2 = st.columns(2)
with nm1:
    norm_mz = st.number_input(
        "Reference m/z for normalization",
        value=140.0,
        step=0.1,
        format="%.3f",
        key="_msc_norm_mz",
        disabled=norm_mode != "Max at reference m/z",
    )
with nm2:
    norm_width = st.number_input(
        "± window (amu)",
        value=1.0,
        min_value=0.01,
        step=0.1,
        key="_msc_norm_width",
        disabled=norm_mode != "Max at reference m/z",
    )

marker_mz = st.number_input(
    "Mark this m/z (0 = off)", value=0.0, step=0.1, format="%.3f", key="_msc_marker_mz"
)
offset_step = st.number_input(
    "Vertical offset between traces (stacking)", value=0.0, step=0.1, key="_msc_offset"
)


def _process(label, x, y):
    """Baseline-correct then normalize one spectrum, returning a trace dict."""
    y = np.asarray(y, dtype=float)
    if do_baseline:
        try:
            bl = _estimate_baseline_cached(x, y, bl_method, tuple(sorted(bl_params.items())))
        except ImportError:
            st.error(
                "❌ `pybaselines` is not installed — install it or pick `als` / `Mean Subtraction`."
            )
            st.stop()
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            st.error(f"❌ Baseline fit failed for `{label}` ({bl_method}): {exc}")
            st.stop()
    else:
        bl = np.zeros_like(y)

    y_bl = y - bl
    y_norm, factor = _normalize(x, y_bl, norm_mode, x_lo, x_hi, norm_mz, norm_width)
    in_range = (x >= x_lo) & (x <= x_hi)
    return {
        "label": label,
        "x": x,
        "y": y_norm,
        "raw": y,
        "baseline_curve": bl,
        "baseline": float(np.nanmean(bl[in_range])) if in_range.any() else 0.0,
        "scale": factor,
    }


traces = []
if ref_y is not None:
    traces.append(_process(ref_label, ref_x, ref_y))
for extra in extras:
    traces.append(_process(extra["label"], extra["x"], extra["y"]))


# ========================================================================================
# 4. OVERLAY
# ========================================================================================
st.markdown("---")
st.markdown("## 4. Overlay")

palette = ["#000000", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

fig = go.Figure()
for idx, tr in enumerate(traces):
    mask = (tr["x"] >= x_lo) & (tr["x"] <= x_hi)
    fig.add_trace(
        go.Scatter(
            x=tr["x"][mask],
            y=tr["y"][mask] + idx * offset_step,
            mode="lines",
            name=tr["label"],
            line=dict(color=palette[idx % len(palette)], width=1.4),
        )
    )
    if show_baseline and do_baseline:
        fig.add_trace(
            go.Scatter(
                x=tr["x"][mask],
                y=tr["baseline_curve"][mask],
                mode="lines",
                name=f"{tr['label']} — fitted baseline",
                line=dict(color=palette[idx % len(palette)], width=1.0, dash="dot"),
                yaxis="y2",
            )
        )
if bl_method == "Mean Subtraction":
    fig.add_vrect(
        x0=bl_params["bl_start"],
        x1=bl_params["bl_start"] + bl_params["bl_width"],
        fillcolor="lightgrey",
        opacity=0.35,
        layer="below",
        line_width=0,
        annotation_text="baseline",
        annotation_position="top left",
    )
if marker_mz > 0:
    fig.add_vline(x=marker_mz, line=dict(color="green", dash="dash"), annotation_text=f"{marker_mz:g}")
fig.update_layout(
    xaxis_title="m/z",
    yaxis_title="Intensity (normalized)" if norm_mode != "None (raw counts)" else "Intensity (counts)",
    yaxis2=dict(title="Raw counts (baseline)", overlaying="y", side="right", showgrid=False)
    if (show_baseline and do_baseline)
    else None,
    xaxis_range=[x_lo, x_hi],
    height=520,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Applied corrections")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Spectrum": tr["label"],
                "Baseline method": bl_method,
                "Mean baseline in range": tr["baseline"],
                "Scale factor": tr["scale"],
                "m/z bins": len(tr["x"]),
            }
            for tr in traces
        ]
    ),
    hide_index=True,
    use_container_width=True,
)


# ========================================================================================
# 5. MASS DETECTION (prominence, as in 10.0) + PRESENCE / ABSENCE COMPARISON
# ========================================================================================
st.markdown("---")
st.markdown("## 5. Detected masses — what is there and what is not")
st.caption(
    "Peaks are detected on each mass spectrum independently with the same criterion as "
    "Section 10.0 (`find_peaks(prominence=…, distance=…)`), so a weak-but-real mass is found "
    "on its own merit instead of having to survive a Δ threshold."
)

st.info(
    "Detection is done **per spectrum** using prominence relative to that spectrum's own maximum, "
    "so a dim scan is not penalised by a bright one."
)

pk1, pk2, pk3, pk4 = st.columns(4)
with pk1:
    peak_prom_pct = st.number_input(
        "Prominence (% of each spectrum's max)",
        value=5.0,
        min_value=0.0,
        step=0.5,
        format="%.2f",
        key="_msc_pk_prom_pct",
        help="Same meaning as in Section 10.0, but scaled per spectrum. A weak mass (e.g. 230) in a "
        "low-intensity scan is found as long as it rises 5% above its local background there.",
    )
with pk2:
    peak_dist = st.number_input(
        "Min peak distance (m/z)", value=2.0, min_value=0.1, step=0.5, key="_msc_pk_dist"
    )
with pk3:
    match_tol = st.number_input(
        "Match tolerance (amu)",
        value=0.5,
        min_value=0.01,
        step=0.1,
        key="_msc_pk_tol",
        help="Peaks from different scans within this distance are treated as the same mass.",
    )
with pk4:
    min_intensity_pct = st.number_input(
        "Presence floor (% of each spectrum's max)",
        value=1.0,
        min_value=0.0,
        step=0.1,
        format="%.2f",
        key="_msc_pk_min_pct",
        help="A mass is also called 'present' if its local intensity reaches this % of that spectrum's "
        "own maximum, even if the peak finder does not return it (e.g. because of min-distance rules).",
    )


def _trace_max(tr):
    """Maximum absolute intensity of a trace inside the plotted range."""
    mask = (tr["x"] >= x_lo) & (tr["x"] <= x_hi)
    if not mask.any():
        return 1.0
    return float(np.nanmax(np.abs(tr["y"][mask])))


def _detect_peaks(tr):
    """Prominence-based peak detection, prominence scaled to the trace's own maximum."""
    mask = (tr["x"] >= x_lo) & (tr["x"] <= x_hi)
    xs, ys = tr["x"][mask], tr["y"][mask]
    if len(xs) < 3:
        return np.array([]), np.array([])
    tmax = _trace_max(tr)
    prom_val = (peak_prom_pct / 100.0) * tmax
    avg_sp = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    dist_idx = max(1, int(peak_dist / avg_sp)) if avg_sp > 0 else 1
    idx, _ = find_peaks(
        np.nan_to_num(ys, nan=0.0),
        prominence=prom_val if prom_val > 0 else None,
        distance=dist_idx,
    )
    return xs[idx], ys[idx]


def _local_max(tr, mz, tol):
    """Peak intensity of a spectrum in the window mz ± tol."""
    mask = (tr["x"] >= mz - tol) & (tr["x"] <= mz + tol)
    if not mask.any():
        return np.nan
    return float(np.nanmax(tr["y"][mask]))


def _is_present(tr, inten):
    """A mass is present in a trace if its local intensity reaches the user floor relative to that trace's max."""
    tmax = _trace_max(tr)
    if not np.isfinite(tmax) or tmax == 0:
        return False
    return np.isfinite(inten) and inten >= (min_intensity_pct / 100.0) * tmax


for tr in traces:
    tr["peak_x"], tr["peak_y"] = _detect_peaks(tr)

st.write(
    " · ".join(f"**{tr['label']}**: {len(tr['peak_x'])} peaks" for tr in traces)
)

# Cluster every detected mass across all spectra into one common mass list
_all_peaks = np.concatenate([tr["peak_x"] for tr in traces]) if traces else np.array([])
mass_list = []
if _all_peaks.size:
    _sorted = np.sort(_all_peaks)
    _group = [_sorted[0]]
    for _v in _sorted[1:]:
        if _v - _group[-1] <= match_tol:
            _group.append(_v)
        else:
            mass_list.append(float(np.mean(_group)))
            _group = [_v]
    mass_list.append(float(np.mean(_group)))

rows = []
for mz in mass_list:
    row = {"m/z": round(mz, 2)}
    present_in = []
    for tr in traces:
        inten = _local_max(tr, mz, match_tol)
        detected = bool(np.any(np.abs(tr["peak_x"] - mz) <= match_tol))
        is_present = detected or _is_present(tr, inten)
        row[f"{tr['label']} — I"] = round(inten, 5) if np.isfinite(inten) else np.nan
        row[f"{tr['label']} — present"] = is_present
        if is_present:
            present_in.append(tr["label"])
    if len(present_in) == len(traces):
        row["Status"] = "in all"
    elif not present_in:
        row["Status"] = "below floor"
    else:
        row["Status"] = "only in " + ", ".join(present_in)
    row["_present_in"] = present_in
    rows.append(row)

mass_table = pd.DataFrame(rows)

if mass_table.empty:
    st.warning("No peaks detected — lower the prominence.")
else:
    _view = mass_table.drop(columns=["_present_in"])
    _filter = st.multiselect(
        "Filter by status",
        options=sorted(_view["Status"].unique().tolist()),
        default=[],
        key="_msc_pk_status",
    )
    if _filter:
        _view = _view[_view["Status"].isin(_filter)]
    st.dataframe(_view, hide_index=True, use_container_width=True)
    st.download_button(
        "📥 Download detected-mass table (CSV)",
        data=_view.to_csv(index=False).encode("utf-8"),
        file_name=f"detected_masses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="_msc_dl_masses",
    )


# ========================================================================================
# 6. THREE-PANEL DIFFERENCE FIGURE (A / B / B - A with change markers)
# ========================================================================================
st.markdown("---")
st.markdown("## 6. Difference figure (a / b / Δ with change arrows)")

if len(traces) < 2:
    st.info("Load at least two spectra (reference + one scan) to build the difference figure.")
else:
    labels = [tr["label"] for tr in traces]
    dsel1, dsel2 = st.columns(2)
    with dsel1:
        idx_a = st.selectbox(
            "Panel a — reference ('OFF')",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=0,
            key="_msc_diff_a",
        )
    with dsel2:
        idx_b = st.selectbox(
            "Panel b — comparison ('ON')",
            options=list(range(len(labels))),
            format_func=lambda i: labels[i],
            index=1,
            key="_msc_diff_b",
        )

    if idx_a == idx_b:
        st.warning("Pick two different spectra for panels a and b.")
    else:
        tr_a, tr_b = traces[idx_a], traces[idx_b]

        # Interpolate both onto a shared grid so the difference is well defined
        n_grid = st.slider("Difference grid points", 1000, 20000, 8000, step=1000, key="_msc_diff_grid")
        dgrid = np.linspace(x_lo, x_hi, int(n_grid))

        def _on_grid(tr):
            order = np.argsort(tr["x"])
            return np.interp(dgrid, tr["x"][order], tr["y"][order], left=np.nan, right=np.nan)

        y_a = _on_grid(tr_a)
        y_b = _on_grid(tr_b)
        y_diff = y_b - y_a

        ref_max = float(np.nanmax(y_a)) if np.isfinite(np.nanmax(y_a)) else 0.0
        pct = (y_diff / ref_max) * 100.0 if ref_max else np.zeros_like(y_diff)

        thr1, thr2, thr3 = st.columns(3)
        with thr1:
            auto_thr = float(np.nanmax(np.abs(y_diff))) * 0.10 if np.isfinite(np.nanmax(np.abs(y_diff))) else 0.05
            thr = st.number_input(
                "|Δ| threshold for markers",
                value=round(auto_thr, 6),
                min_value=0.0,
                step=max(auto_thr / 10, 1e-6),
                format="%.6f",
                key="_msc_diff_thr",
            )
        with thr2:
            min_sep = st.number_input(
                "Min separation between markers (amu)", value=1.0, min_value=0.05, step=0.1, key="_msc_diff_sep"
            )
        with thr3:
            max_labels = st.number_input(
                "Max labels per direction", value=15, min_value=1, step=1, key="_msc_diff_maxlab"
            )

        diff_prom = st.number_input(
            "Δ prominence (0 = off)",
            value=0.0,
            min_value=0.0,
            step=max(auto_thr / 10, 1e-6),
            format="%.6f",
            key="_msc_diff_prom",
            help=(
                "Extra criterion on top of the |Δ| height, same idea as the prominence used for mass-spectrum "
                "peaks in Section 10.0. Use it when a broad Δ hump produces spurious markers — a peak then has "
                "to stand out from its local surroundings, not just exceed the threshold."
            ),
        )

        marker_source = st.radio(
            "Which masses to mark",
            options=["Detected masses (Section 5)", "Δ peaks (height threshold)"],
            index=0,
            horizontal=True,
            key="_msc_marker_src",
            help=(
                "**Detected masses**: every mass found by prominence detection on the spectra themselves, "
                "coloured by whether it is present in both scans, only in b (new) or only in a (lost). "
                "**Δ peaks**: only masses whose change exceeds the |Δ| threshold."
            ),
        )
        annotate_common = st.checkbox(
            "Also mark masses present in both scans", value=False, key="_msc_diff_common"
        )

        show_labels = st.checkbox("Show m/z labels on markers", value=True, key="_msc_diff_labels")
        show_table = st.checkbox("Draw the Δ table inside the figure", value=False, key="_msc_diff_intable")

        grid_step = (x_hi - x_lo) / max(int(n_grid) - 1, 1)
        dist_pts = max(1, int(min_sep / grid_step)) if grid_step > 0 else 1
        diff_clean = np.nan_to_num(y_diff, nan=0.0)

        marker_status = {}

        if marker_source.startswith("Detected"):
            _pos, _posh, _neg, _negh = [], [], [], []
            if mass_table.empty:
                st.warning("No masses detected in Section 5 — lower the prominence there.")
            else:
                _la, _lb = tr_a["label"], tr_b["label"]
                for _, _r in mass_table.iterrows():
                    _mz = float(_r["m/z"])
                    if not (x_lo <= _mz <= x_hi):
                        continue
                    _in_a = bool(_r.get(f"{_la} — present", False))
                    _in_b = bool(_r.get(f"{_lb} — present", False))
                    if not _in_a and not _in_b:
                        continue
                    if _in_a and _in_b and not annotate_common:
                        continue
                    _w = np.where(np.abs(dgrid - _mz) <= match_tol)[0]
                    if _w.size == 0:
                        continue
                    _i = int(_w[np.argmax(np.abs(diff_clean[_w]))])
                    if _in_b and not _in_a:
                        marker_status[_i] = "new in b"
                    elif _in_a and not _in_b:
                        marker_status[_i] = "only in a"
                    else:
                        marker_status[_i] = "in both"
                    if diff_clean[_i] >= 0:
                        _pos.append(_i)
                        _posh.append(abs(float(diff_clean[_i])))
                    else:
                        _neg.append(_i)
                        _negh.append(abs(float(diff_clean[_i])))
            pos_idx, pos_h = np.array(_pos, dtype=int), np.array(_posh, dtype=float)
            neg_idx, neg_h = np.array(_neg, dtype=int), np.array(_negh, dtype=float)
        else:
            _prom_arg = diff_prom if diff_prom > 0 else None
            pos_idx, pos_props = find_peaks(
                diff_clean, height=thr, distance=dist_pts, prominence=_prom_arg
            )
            neg_idx, neg_props = find_peaks(
                -diff_clean, height=thr, distance=dist_pts, prominence=_prom_arg
            )

            # Keep only the strongest N in each direction for a readable figure
            if len(pos_idx) > max_labels:
                keep = np.argsort(pos_props["peak_heights"])[::-1][: int(max_labels)]
                pos_idx, pos_h = pos_idx[keep], pos_props["peak_heights"][keep]
            else:
                pos_h = pos_props["peak_heights"]
            if len(neg_idx) > max_labels:
                keep = np.argsort(neg_props["peak_heights"])[::-1][: int(max_labels)]
                neg_idx, neg_h = neg_idx[keep], neg_props["peak_heights"][keep]
            else:
                neg_h = neg_props["peak_heights"]

        def _marker_color(i, default):
            """Green = new in b, red = lost from a, grey = present in both."""
            return {"new in b": "green", "only in a": "red", "in both": "dimgray"}.get(
                marker_status.get(i), default
            )

        def _alt_offsets(n, base, step):
            """Alternate label offsets so neighbouring annotations do not collide."""
            return np.array([base + (step if i % 2 else 0.0) for i in range(n)])

        span = float(np.nanmax(np.abs(diff_clean))) or 1.0

        if show_table:
            fig_d = plt.figure(figsize=(14.0, 8.0))
            gs = fig_d.add_gridspec(3, 2, width_ratios=[3, 1], wspace=0.35, hspace=0.35)
            ax_a = fig_d.add_subplot(gs[0, 0])
            ax_b = fig_d.add_subplot(gs[1, 0], sharex=ax_a)
            ax_d = fig_d.add_subplot(gs[2, 0], sharex=ax_a)
            ax_t = fig_d.add_subplot(gs[:, 1])
            ax_t.axis("off")
        else:
            fig_d, (ax_a, ax_b, ax_d) = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True)
            ax_t = None

        for ax, ydata, title, color in (
            (ax_a, y_a, f"a) {tr_a['label']}", "tab:blue"),
            (ax_b, y_b, f"b) {tr_b['label']}", "tab:orange"),
            (ax_d, y_diff, f"c) Difference (b − a)", "tab:purple"),
        ):
            ax.plot(dgrid, ydata, lw=1.0, color=color)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_title(title, fontsize=10, loc="left")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylabel("ΔIntensity" if ax is ax_d else "Intensity")
            ax.grid(alpha=0.25)
        ax_d.set_xlabel("m/z (u)")
        if not marker_source.startswith("Detected"):
            ax_d.axhspan(-thr, thr, color="grey", alpha=0.12)

        # Guide lines on panels a and b for masses that are missing from one of them
        for _i, _status in marker_status.items():
            if _status == "in both":
                continue
            _c = "green" if _status == "new in b" else "red"
            for _ax in (ax_a, ax_b):
                _ax.axvline(dgrid[_i], color=_c, ls=":", lw=0.8, alpha=0.45)

        # Increases: up-arrows
        pos_off = _alt_offsets(len(pos_idx), span * 0.06, span * 0.06)
        for k, i in enumerate(pos_idx):
            _c = _marker_color(i, "green")
            ax_d.annotate(
                "",
                xy=(dgrid[i], diff_clean[i]),
                xytext=(dgrid[i], diff_clean[i] + pos_off[k]),
                arrowprops=dict(arrowstyle="-|>", color=_c, lw=1.1),
            )
            if show_labels:
                ax_d.text(
                    dgrid[i],
                    diff_clean[i] + pos_off[k],
                    f"{dgrid[i]:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=_c,
                )

        # Decreases: down-arrows
        neg_off = _alt_offsets(len(neg_idx), span * 0.06, span * 0.06)
        for k, i in enumerate(neg_idx):
            _c = _marker_color(i, "red")
            ax_d.annotate(
                "",
                xy=(dgrid[i], diff_clean[i]),
                xytext=(dgrid[i], diff_clean[i] - neg_off[k]),
                arrowprops=dict(arrowstyle="-|>", color=_c, lw=1.1),
            )
            if show_labels:
                ax_d.text(
                    dgrid[i],
                    diff_clean[i] - neg_off[k],
                    f"{dgrid[i]:.1f}",
                    ha="center",
                    va="top",
                    fontsize=7,
                    color=_c,
                )

        if marker_source.startswith("Detected"):
            ax_d.plot([], [], "^", color="green", label=f"only in b ({tr_b['label'][:18]})")
            ax_d.plot([], [], "v", color="red", label=f"only in a ({tr_a['label'][:18]})")
            if annotate_common:
                ax_d.plot([], [], "s", color="dimgray", label="in both")
        else:
            ax_d.plot([], [], "^", color="green", label="increase (b > a)")
            ax_d.plot([], [], "v", color="red", label="decrease (b < a)")
        ax_d.legend(fontsize=7, loc="upper right", ncol=2)
        ax_d.margins(y=0.25)

        # Δ peak table
        rows = []
        for i, h in zip(pos_idx, pos_h):
            rows.append(
                {
                    "m/z (u)": round(float(dgrid[i]), 2),
                    "ΔIntensity": round(float(h), 5),
                    "Δ (% of a max)": round(float(pct[i]), 1),
                    "Change": "increase",
                    "Presence": marker_status.get(i, "—"),
                }
            )
        for i, h in zip(neg_idx, neg_h):
            rows.append(
                {
                    "m/z (u)": round(float(dgrid[i]), 2),
                    "ΔIntensity": round(-float(h), 5),
                    "Δ (% of a max)": round(float(pct[i]), 1),
                    "Change": "decrease",
                    "Presence": marker_status.get(i, "—"),
                }
            )
        df_delta = pd.DataFrame(
            rows, columns=["m/z (u)", "ΔIntensity", "Δ (% of a max)", "Change", "Presence"]
        )
        if not df_delta.empty:
            df_delta = df_delta.sort_values("m/z (u)").reset_index(drop=True)

        if ax_t is not None and not df_delta.empty:
            tbl = ax_t.table(
                cellText=df_delta.astype(str).values.tolist(),
                colLabels=df_delta.columns.tolist(),
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.3)
            ax_t.set_title(f"|Δ| ≥ {thr:g}", fontsize=9)

        fig_d.tight_layout()
        st.pyplot(fig_d)

        dbuf = io.BytesIO()
        fig_d.savefig(dbuf, format="png", dpi=300, bbox_inches="tight")
        plt.close(fig_d)
        dbuf.seek(0)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "📥 Download difference figure (PNG)",
                data=dbuf,
                file_name=f"difference_figure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png",
                key="_msc_dl_diff_png",
            )
        with dl2:
            if not df_delta.empty:
                st.download_button(
                    "📥 Download Δ peak table (CSV)",
                    data=df_delta.to_csv(index=False).encode("utf-8"),
                    file_name=f"delta_peaks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="_msc_dl_diff_csv",
                )

        if df_delta.empty:
            st.info(
                "No masses to mark — tick 'Also mark masses present in both scans', or lower the "
                "prominence in Section 5."
                if marker_source.startswith("Detected")
                else "No |Δ| peaks above the threshold — lower it to mark changes."
            )
        else:
            st.dataframe(df_delta, hide_index=True, use_container_width=True)


# ========================================================================================
# 7. EXPORT
# ========================================================================================
st.markdown("---")
st.markdown("## 7. Export")

exp1, exp2, exp3 = st.columns(3)
with exp1:
    fig_w = st.number_input("Width (in)", value=9.0, step=0.5, key="_msc_fig_w")
with exp2:
    fig_h = st.number_input("Height (in)", value=5.0, step=0.5, key="_msc_fig_h")
with exp3:
    fig_dpi = st.number_input("DPI", value=300, step=50, key="_msc_fig_dpi")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if st.button("🖼️ Render static figure (PNG)", key="_msc_render"):
    fig_static, ax = plt.subplots(figsize=(fig_w, fig_h))
    for idx, tr in enumerate(traces):
        mask = (tr["x"] >= x_lo) & (tr["x"] <= x_hi)
        ax.plot(
            tr["x"][mask],
            tr["y"][mask] + idx * offset_step,
            color=palette[idx % len(palette)],
            lw=1.0,
            label=tr["label"],
        )
    if bl_method == "Mean Subtraction":
        ax.axvspan(
            bl_params["bl_start"],
            bl_params["bl_start"] + bl_params["bl_width"],
            color="lightgrey",
            alpha=0.4,
        )
    if marker_mz > 0:
        ax.axvline(marker_mz, color="green", ls="--", lw=1)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity (normalized)" if norm_mode != "None (raw counts)" else "Intensity (counts)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig_static.tight_layout()
    st.pyplot(fig_static)

    buf = io.BytesIO()
    fig_static.savefig(buf, format="png", dpi=int(fig_dpi), bbox_inches="tight")
    plt.close(fig_static)
    buf.seek(0)
    st.download_button(
        "📥 Download PNG",
        data=buf,
        file_name=f"mass_scan_comparison_{stamp}.png",
        mime="image/png",
        key="_msc_dl_png",
    )

# CSV export on a shared grid so the traces line up column-wise
grid = np.linspace(x_lo, x_hi, 4000)
csv_df = pd.DataFrame({"mz": grid})
for tr in traces:
    order = np.argsort(tr["x"])
    csv_df[tr["label"]] = np.interp(grid, tr["x"][order], tr["y"][order], left=np.nan, right=np.nan)
st.download_button(
    "📥 Download overlay CSV",
    data=csv_df.to_csv(index=False).encode("utf-8"),
    file_name=f"mass_scan_comparison_{stamp}.csv",
    mime="text/csv",
    key="_msc_dl_csv",
)
