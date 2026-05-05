import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import io
import os
import pickle
import gzip
import configparser
from pathlib import Path
from functools import reduce
from plotly.subplots import make_subplots
from packages.ReportManager import add_plot_to_report_button, init_report_session
from packages.load_dataset import ensure_dataset_loaded

init_report_session()

st.title("🔬 Mass-Resolved IR Analysis")
st.caption(
    "Identify which m/z channels show real IR-induced depletion vs noise. "
    "Three analysis tools: noise-masked ΔI heatmap, on/off-resonance difference mass spectrum, "
    "and mass-channel IR spectra."
)

# ========================================================================================
# DATA LOADING & RANGE SELECTION
# ========================================================================================
ensure_dataset_loaded(
    require_keys=["x_mass", "compilation_baseline_corrected_data", "unique_wavenumbers"],
    compute_megasum=False,
    page_key_prefix="_misc",
)

# Re-read after potential load
x_mass = st.session_state.get("x_mass")
compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data")
unique_wavenumbers = st.session_state.get("unique_wavenumbers")
plot_col_without = st.session_state.get("plot_columnIndex_withoutIR")
plot_col_with = st.session_state.get("plot_columnIndex_withIR")

st.success(f"✅ Data loaded: {len(unique_wavenumbers)} wavenumber steps, {len(x_mass)} m/z bins")

@st.fragment
def _misc_controls_section():
    x_mass = st.session_state.get("x_mass")
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data")
    unique_wavenumbers = st.session_state.get("unique_wavenumbers")
    plot_col_without = st.session_state.get("plot_columnIndex_withoutIR")
    plot_col_with = st.session_state.get("plot_columnIndex_withIR")

    # Range controls
    st.markdown("### 📐 Analysis Range")
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    with rcol1:
        wn_min = st.number_input(
            "Wavenumber min (cm⁻¹)", value=float(min(unique_wavenumbers)),
            step=10.0, key="_misc_wn_min"
        )
    with rcol2:
        wn_max = st.number_input(
            "Wavenumber max (cm⁻¹)", value=float(max(unique_wavenumbers)),
            step=10.0, key="_misc_wn_max"
        )
    with rcol3:
        mz_min = st.number_input(
            "m/z min", value=float(x_mass.min()), step=1.0, key="_misc_mz_min"
        )
    with rcol4:
        mz_max = st.number_input(
            "m/z max", value=float(x_mass.max()), step=1.0, key="_misc_mz_max"
        )

    _nf_col1, _nf_col2 = st.columns([3, 1])
    with _nf_col1:
        noise_floor = st.number_input(
            "Noise floor (baseline signal threshold)",
            value=float(st.session_state.get("_misc_noise_floor_val", 0.001)),
            min_value=0.0, step=0.0005, format="%.5f",
            help=(
                "m/z bins whose mean absolute without-IR signal across all wavenumbers "
                "falls below this value are masked out as empty/noise channels. "
                "Lower = keep more channels. Higher = stricter masking. "
                "Use **Auto-detect** to find this automatically."
            ),
        )
        st.session_state["_misc_noise_floor_val"] = noise_floor
        if st.session_state.get("_misc_processed", False):
            _bpm_preview = st.session_state["_misc_baseline_per_mz"]
            _n_keep = int((_bpm_preview >= noise_floor).sum())
            _n_total = len(_bpm_preview)
            st.caption(f"→ {_n_keep}/{_n_total} m/z bins will pass this threshold")
    with _nf_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Auto-detect", key="_auto_noise_floor",
                     help=(
                         "Finds the optimal noise floor automatically using Otsu's method: "
                         "sweeps 512 candidate thresholds and picks the one that maximally "
                         "separates the noise cluster from real signal channels. "
                         "Requires **Process Data** to have been run at least once. "
                         "Instantly updates the noise floor value and applies the new mask — "
                         "no need to press Process Data again."
                     )):
            if st.session_state.get("_misc_processed", False):
                _bpm = st.session_state["_misc_baseline_per_mz"]
                # Log-space Otsu: data spans orders of magnitude,
                # so we work in log10 where noise/signal are actually bimodal
                _bpm_pos = _bpm[_bpm > 0]
                _log_bpm = np.log10(_bpm_pos)
                _thresholds = np.linspace(_log_bpm.min(), _log_bpm.max(), 1024)
                _best_log_t, _best_var = _log_bpm.min(), -1.0
                for _lt in _thresholds:
                    _below = _log_bpm[_log_bpm < _lt]
                    _above = _log_bpm[_log_bpm >= _lt]
                    if len(_below) == 0 or len(_above) == 0:
                        continue
                    _var = len(_below) * len(_above) * (_below.mean() - _above.mean()) ** 2
                    if _var > _best_var:
                        _best_var, _best_log_t = _var, _lt
                _best_t = round(float(10 ** _best_log_t), 6)
                st.session_state["_misc_noise_floor_val"] = _best_t
                noise_floor = _best_t
                _n_wn = len(st.session_state["_misc_wn_list"])
                noise_mask_1d = _bpm >= _best_t
                st.session_state["_misc_noise_mask_1d"] = noise_mask_1d
                st.session_state["_misc_noise_mask_2d"] = np.tile(noise_mask_1d, (_n_wn, 1))
                n_masked = int((~noise_mask_1d).sum())
                n_total = len(noise_mask_1d)
                st.success(f"✅ Noise floor auto-set to **{_best_t:.6f}** — {n_total - n_masked}/{n_total} m/z bins kept. Press **📊 Plot / Refresh** to update the plots.")
            else:
                st.warning("Run **Process Data** once first, then auto-detect.")

    # ========================================================================================
    # DATA PROCESSING
    # ========================================================================================
    _btn_col1, _btn_col2 = st.columns(2)
    with _btn_col1:
        _do_process = st.button("✨ Process Data", type="primary",
                                help="Rebuild all matrices from scratch using the range & noise floor above. Use after changing wavenumber/m/z ranges.")
    with _btn_col2:
        _do_plot = st.button("📊 Plot / Refresh", type="secondary",
                             disabled=not st.session_state.get("_misc_processed", False),
                             help="Refresh the plots below with the current session data. Use after Auto-detect or to simply re-render the tabs.")

    if _do_process:
        with st.spinner("Building m/z × wavenumber matrices…"):
            # Filter wavenumbers
            wn_list = sorted([wn for wn in unique_wavenumbers if wn_min <= float(wn) <= wn_max])
            mz_mask = (x_mass >= mz_min) & (x_mass <= mz_max)
            mz_vals = x_mass[mz_mask]

            n_wn = len(wn_list)
            n_mz = int(mz_mask.sum())

            mat_without = np.zeros((n_wn, n_mz))
            mat_with = np.zeros((n_wn, n_mz))
            mat_raw_without = np.zeros((n_wn, n_mz))
            mat_raw_with = np.zeros((n_wn, n_mz))

            for i, wn in enumerate(wn_list):
                data_wn = compilation_baseline_corrected_data[wn]
                mat_without[i, :] = data_wn.iloc[mz_mask, plot_col_without].values
                mat_with[i, :] = data_wn.iloc[mz_mask, plot_col_with].values
                # Raw sums (pre-baseline-correction) for Poisson error estimation
                mat_raw_without[i, :] = data_wn.iloc[mz_mask, plot_col_without - 2].values
                mat_raw_with[i, :] = data_wn.iloc[mz_mask, plot_col_with - 2].values

            # ΔI = without_IR - with_IR  (positive = depletion = parent ion lost signal)
            mat_delta = mat_without - mat_with

            # Noise mask: mask m/z bins where baseline (without IR) is below noise floor
            # Average across wavenumbers to get a robust per-m/z baseline estimate
            baseline_per_mz = np.mean(np.abs(mat_without), axis=0)
            noise_mask_1d = baseline_per_mz >= noise_floor  # True = real signal
            noise_mask_2d = np.tile(noise_mask_1d, (n_wn, 1))

            # Store everything
            st.session_state["_misc_wn_list"] = wn_list
            st.session_state["_misc_mz_vals"] = mz_vals
            st.session_state["_misc_mat_without"] = mat_without
            st.session_state["_misc_mat_with"] = mat_with
            st.session_state["_misc_mat_raw_without"] = mat_raw_without
            st.session_state["_misc_mat_raw_with"] = mat_raw_with
            st.session_state["_misc_mat_delta"] = mat_delta
            st.session_state["_misc_noise_mask_1d"] = noise_mask_1d
            st.session_state["_misc_noise_mask_2d"] = noise_mask_2d
            st.session_state["_misc_baseline_per_mz"] = baseline_per_mz
            st.session_state["_misc_processed"] = True

        n_masked = int((~noise_mask_1d).sum())
        st.success(
            f"✅ Processed {n_wn} wavenumbers × {n_mz} m/z bins. "
            f"Masked {n_masked}/{n_mz} m/z bins below noise floor ({noise_floor:.4f}). "
            f"Press **📊 Plot / Refresh** to update the plots."
        )

    if _do_plot:
        st.rerun(scope="app")
    if st.session_state.get("_misc_processed", False):
        _wl = st.session_state["_misc_wn_list"]
        _mv = st.session_state["_misc_mz_vals"]
        _nm = st.session_state["_misc_noise_mask_1d"]
        st.caption(
            f"ℹ️ Last processed: {len(_wl)} wavenumbers × {len(_mv)} m/z bins, "
            f"{int((~_nm).sum())} masked. Change settings above and press **Process Data** to update."
        )


@st.fragment
def _misc_tabs_section():
    # ========================================================================================
    # ANALYSIS TABS
    # ========================================================================================
    if not st.session_state.get("_misc_processed", False):
        st.info("👆 Press **Process Data** to build the analysis matrices.")
        return

    wn_list = st.session_state["_misc_wn_list"]
    mz_vals = st.session_state["_misc_mz_vals"]
    mat_without = st.session_state["_misc_mat_without"]
    mat_with = st.session_state["_misc_mat_with"]
    mat_raw_without = st.session_state.get("_misc_mat_raw_without")
    mat_raw_with = st.session_state.get("_misc_mat_raw_with")
    mat_delta = st.session_state["_misc_mat_delta"]
    noise_mask_1d = st.session_state["_misc_noise_mask_1d"]
    noise_mask_2d = st.session_state["_misc_noise_mask_2d"]
    baseline_per_mz = st.session_state["_misc_baseline_per_mz"]

    # --- Filename helpers ---
    _wn_lo = int(round(float(wn_list[0])))
    _wn_hi = int(round(float(wn_list[-1])))
    _n_mz = len(mz_vals)
    _wn_tag = f"wn{_wn_lo}-{_wn_hi}"
    _mz_tag = f"{_n_mz}mz"

    def _save_to_output(buf_or_df, fname, is_csv=False):
        """Save file to the output folder next to the loaded dataset."""
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
            st.warning("No output directory set. Configure 'file_directory' in defaults.ini.")
            return
        out_dir = Path(_file_dir) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        fpath = out_dir / fname
        if is_csv:
            buf_or_df.to_csv(fpath)
        else:
            buf_or_df.seek(0)
            fpath.write_bytes(buf_or_df.read())
        st.success(f"✅ Saved to `{fpath}`")

    tab_heatmap, tab_diff_ms, tab_fragment_ir, tab_quality = st.tabs([
        "🗺️ Noise-Masked ΔI Heatmap",
        "📊 On/Off-Resonance Difference MS",
        "📈 Mass-Channel IR Spectra",
        "🔍 Signal Quality & Uncertainty",
    ])

    # ========================================================================================
    # TAB 1: NOISE-MASKED ΔI HEATMAP
    # ========================================================================================
    with tab_heatmap:
        st.markdown("### Noise-Masked ΔI Heatmap (without IR − with IR)")
        st.caption(
            "Positive values (warm colors) = parent ion depletion. "
            "Negative values (cool colors) = fragment appearance or enhancement. "
            "m/z bins below the noise floor are masked out (grey)."
        )

        sigma_options = [0.0, 0.5, 1.0, 2.0, 5.0]
        hm_sigma = st.selectbox(
            "Gaussian smoothing σ", sigma_options,
            index=sigma_options.index(1.0), key="_hm_sigma",
        )

        # Apply noise mask
        mat_display = mat_delta.copy()
        mat_display[:, ~noise_mask_1d] = np.nan

        # Optional smoothing (NaN-aware)
        if hm_sigma > 0:
            mask_finite = np.isfinite(mat_display).astype(float)
            filled = np.where(np.isfinite(mat_display), mat_display, 0)
            smooth_data = gaussian_filter(filled, sigma=hm_sigma)
            smooth_mask = gaussian_filter(mask_finite, sigma=hm_sigma)
            with np.errstate(divide='ignore', invalid='ignore'):
                mat_display = np.where(smooth_mask > 0, smooth_data / smooth_mask, np.nan)

        fig_hm = go.Figure(data=go.Heatmap(
            z=mat_display,
            x=mz_vals,
            y=wn_list,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="ΔI (a.u.)"),
        ))
        fig_hm.update_layout(
            xaxis_title="m/z",
            yaxis_title="Wavenumber (cm⁻¹)",
            title="ΔI = I(no IR) − I(with IR)",
            height=600,
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        # Download buttons for heatmap
        _hm_dl1, _hm_dl2 = st.columns(2)
        with _hm_dl1:
            _buf_hm = io.BytesIO()
            fig_hm_static, ax_hm_s = plt.subplots(figsize=(12, 6))
            _disp = mat_display.copy()
            im = ax_hm_s.pcolormesh(
                mz_vals, wn_list, _disp, cmap="RdBu_r", shading="auto",
                vmin=-np.nanmax(np.abs(_disp)), vmax=np.nanmax(np.abs(_disp)),
            )
            plt.colorbar(im, ax=ax_hm_s, label="ΔI (a.u.)")
            ax_hm_s.set_xlabel("m/z", fontsize=12)
            ax_hm_s.set_ylabel("Wavenumber (cm⁻¹)", fontsize=12)
            ax_hm_s.set_title("ΔI = I(no IR) − I(with IR)", fontsize=13, fontweight="bold")
            fig_hm_static.tight_layout()
            fig_hm_static.savefig(_buf_hm, format="png", dpi=300, bbox_inches="tight")
            _buf_hm.seek(0)
            _hm_fname = f"DeltaI_heatmap_{_wn_tag}_{_mz_tag}.png"
            st.download_button(
                "⬇️ Download Heatmap (PNG)", data=_buf_hm, file_name=_hm_fname,
                mime="image/png", key="_dl_hm_png",
            )
            if st.button("💾 Save Heatmap PNG to output", key="_sv_hm_png"):
                _save_to_output(_buf_hm, _hm_fname)
        with _hm_dl2:
            _csv_hm = pd.DataFrame(mat_display, index=wn_list, columns=mz_vals)
            _csv_hm.index.name = "Wavenumber"
            _hm_csv_fname = f"DeltaI_heatmap_{_wn_tag}_{_mz_tag}.csv"
            st.download_button(
                "⬇️ Download Heatmap Data (CSV)", data=_csv_hm.to_csv(),
                file_name=_hm_csv_fname, mime="text/csv", key="_dl_hm_csv",
            )
            if st.button("💾 Save Heatmap CSV to output", key="_sv_hm_csv"):
                _save_to_output(_csv_hm, _hm_csv_fname, is_csv=True)
        add_plot_to_report_button(
            fig_hm_static, "ΔI Heatmap", key_suffix="heatmap",
            description="Noise-masked ΔI heatmap (without IR − with IR)",
        )
        plt.close(fig_hm_static)

        # 1D IR response profile
        st.markdown("#### IR Response Profile (mean |ΔI| per m/z)")
        st.caption(
            "Shows which m/z channels have the largest average response to IR. "
            "Helps identify the parent ion and fragment channels."
        )
        mean_abs_delta = np.nanmean(np.abs(mat_delta[:, noise_mask_1d]), axis=0)
        mz_active = mz_vals[noise_mask_1d]

        fig_profile = go.Figure()
        fig_profile.add_trace(go.Bar(
            x=mz_active, y=mean_abs_delta,
            marker_color="steelblue",
        ))
        fig_profile.update_layout(
            xaxis_title="m/z",
            yaxis_title="Mean |ΔI|",
            title="IR Response Profile",
            height=350,
        )
        st.plotly_chart(fig_profile, use_container_width=True)

        # Download buttons for IR response profile
        _pr_dl1, _pr_dl2 = st.columns(2)
        with _pr_dl1:
            _buf_pr = io.BytesIO()
            fig_pr_static, ax_pr = plt.subplots(figsize=(12, 4))
            ax_pr.bar(mz_active, mean_abs_delta, color="steelblue", width=np.mean(np.diff(mz_active)) * 0.8 if len(mz_active) > 1 else 1.0)
            ax_pr.set_xlabel("m/z", fontsize=12)
            ax_pr.set_ylabel("Mean |ΔI|", fontsize=12)
            ax_pr.set_title("IR Response Profile", fontsize=13, fontweight="bold")
            ax_pr.grid(True, alpha=0.3)
            fig_pr_static.tight_layout()
            fig_pr_static.savefig(_buf_pr, format="png", dpi=300, bbox_inches="tight")
            _buf_pr.seek(0)
            _pr_fname = f"IR_response_profile_{_wn_tag}_{_mz_tag}.png"
            st.download_button(
                "⬇️ Download IR Profile (PNG)", data=_buf_pr, file_name=_pr_fname,
                mime="image/png", key="_dl_pr_png",
            )
            if st.button("💾 Save IR Profile PNG to output", key="_sv_pr_png"):
                _save_to_output(_buf_pr, _pr_fname)
        with _pr_dl2:
            _csv_pr = pd.DataFrame({"m/z": mz_active, "mean_abs_DeltaI": mean_abs_delta})
            _pr_csv_fname = f"IR_response_profile_{_wn_tag}_{_mz_tag}.csv"
            st.download_button(
                "⬇️ Download IR Profile (CSV)", data=_csv_pr.to_csv(index=False),
                file_name=_pr_csv_fname, mime="text/csv", key="_dl_pr_csv",
            )
            if st.button("💾 Save IR Profile CSV to output", key="_sv_pr_csv"):
                _save_to_output(_csv_pr, _pr_csv_fname, is_csv=True)
        add_plot_to_report_button(
            fig_pr_static, "IR Response Profile", key_suffix="ir_profile",
            description="Mean |ΔI| per m/z showing IR-active channels",
        )
        plt.close(fig_pr_static)

        # Peak detection for IR-active masses
        st.markdown("#### Detect IR-Active Masses")
        st.caption(
            "Automatically identify m/z channels with significant IR response. "
            "Use these as starting points for Tab 3 (Mass-Channel IR Spectra)."
        )

        with st.form("peak_detection_form"):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                peak_prominence = st.number_input(
                    "Minimum prominence (a.u.)",
                    value=np.nanmax(mean_abs_delta) * 0.1 if len(mean_abs_delta) > 0 else 0.001,
                    min_value=0.0,
                    step=0.001,
                    format="%.4f",
                    help="Peaks must stand out by at least this amount above the surrounding baseline.",
                    key="_peak_prominence",
                )
            with pcol2:
                peak_distance = st.number_input(
                    "Minimum distance between peaks (m/z)",
                    value=5.0,
                    min_value=0.1,
                    step=0.5,
                    help="Minimum separation between detected peaks in m/z units.",
                    key="_peak_distance",
                )

            detect_submitted = st.form_submit_button("🔍 Detect IR-Active Masses")

        if detect_submitted:
            # Convert distance from m/z to index spacing
            if len(mz_active) > 1:
                avg_spacing = np.mean(np.diff(mz_active))
                min_distance_idx = max(1, int(peak_distance / avg_spacing))
            else:
                min_distance_idx = 1

            # Find peaks
            peak_indices, peak_properties = find_peaks(
                mean_abs_delta,
                prominence=peak_prominence,
                distance=min_distance_idx,
            )

            if len(peak_indices) > 0:
                detected_peaks_df = pd.DataFrame({
                    "m/z": mz_active[peak_indices],
                    "Mean |ΔI|": mean_abs_delta[peak_indices],
                    "Prominence": peak_properties["prominences"],
                })
                detected_peaks_df = detected_peaks_df.sort_values("Mean |ΔI|", ascending=False)
                detected_peaks_df.reset_index(drop=True, inplace=True)

                st.session_state["_detected_peaks"] = detected_peaks_df
                st.session_state["_peaks_detected"] = True
                st.success(f"✅ Detected {len(peak_indices)} IR-active masses.")
            else:
                st.warning("⚠️ No peaks detected. Try lowering the prominence threshold.")
                st.session_state["_peaks_detected"] = False

        if st.session_state.get("_peaks_detected", False):
            detected_peaks_df = st.session_state["_detected_peaks"]
            st.caption(f"Debug: DataFrame has {len(detected_peaks_df)} rows, columns: {list(detected_peaks_df.columns)}")

            st.dataframe(detected_peaks_df, use_container_width=True, height=300)

            # Download button for detected peaks
            _peak_dl1, _peak_dl2 = st.columns(2)
            with _peak_dl1:
                _peaks_fname = f"IR_active_masses_{_wn_tag}_{_mz_tag}.csv"
                st.download_button(
                    "⬇️ Download Detected Masses (CSV)",
                    data=detected_peaks_df.to_csv(index=False),
                    file_name=_peaks_fname,
                    mime="text/csv",
                    key="_dl_peaks_csv",
                )
                if st.button("💾 Save Detected Masses to output", key="_sv_peaks_csv"):
                    _save_to_output(detected_peaks_df, _peaks_fname, is_csv=True)
            with _peak_dl2:
                # Create a comma-separated list for easy copy-paste (sorted ascending)
                _all_masses = sorted(detected_peaks_df["m/z"].values)
                mass_list_str = ", ".join([f"{m:.1f}" for m in _all_masses])
                st.markdown(f"**📋 Copy-paste into Tab 3 ({len(_all_masses)} masses)**")
                # Use code block for horizontal scrolling (no wrapping)
                st.code(mass_list_str, language=None)
                st.caption("Click the copy button (top-right) or select all to copy the full list")

    # ========================================================================================
    # TAB 2: ON/OFF-RESONANCE DIFFERENCE MASS SPECTRUM
    # ========================================================================================
    with tab_diff_ms:
        st.markdown("### On/Off-Resonance Difference Mass Spectrum")
        st.caption(
            "Select 'resonant' wavenumbers (where you expect IR absorption) and 'off-resonance' "
            "wavenumbers (baseline). The difference shows which m/z channels gain or lose signal "
            "specifically due to IR absorption."
        )

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            on_res_min = st.number_input(
                "Resonant range min (cm⁻¹)", value=float(wn_list[len(wn_list)//4]),
                step=5.0, key="_on_res_min",
            )
            on_res_max = st.number_input(
                "Resonant range max (cm⁻¹)", value=float(wn_list[len(wn_list)//2]),
                step=5.0, key="_on_res_max",
            )
        with dcol2:
            off_res_min = st.number_input(
                "Off-resonance range min (cm⁻¹)", value=float(wn_list[0]),
                step=5.0, key="_off_res_min",
            )
            off_res_max = st.number_input(
                "Off-resonance range max (cm⁻¹)", value=float(wn_list[max(0, len(wn_list)//8)]),
                step=5.0, key="_off_res_max",
            )

        if st.button("📊 Compute Difference MS", key="_run_diff_ms"):
            wn_arr = np.array(wn_list, dtype=float)
            on_mask = (wn_arr >= on_res_min) & (wn_arr <= on_res_max)
            off_mask = (wn_arr >= off_res_min) & (wn_arr <= off_res_max)

            if on_mask.sum() == 0 or off_mask.sum() == 0:
                st.error("❌ No wavenumber steps found in one of the ranges. Adjust the limits.")
            else:
                # Average mass spectrum in each range
                ms_on = np.mean(mat_without[on_mask, :], axis=0)
                ms_off = np.mean(mat_without[off_mask, :], axis=0)
                ms_on_ir = np.mean(mat_with[on_mask, :], axis=0)
                ms_off_ir = np.mean(mat_with[off_mask, :], axis=0)

                # Difference: how does the with-IR spectrum change at resonance vs off-resonance?
                # (with_IR at resonance) - (with_IR off-resonance)
                # Negative = depletion of parent, Positive = fragment appearance
                diff_with_ir = ms_on_ir - ms_off_ir
                diff_without_ir = ms_on - ms_off  # should be ~0 if no IR effect on control

                # Also compute: depletion difference
                # ΔI_on = without - with at resonance; ΔI_off = without - with off-resonance
                depl_on = ms_on - ms_on_ir
                depl_off = ms_off - ms_off_ir
                diff_depletion = depl_on - depl_off  # net IR-induced depletion

                st.session_state["_diff_ms"] = {
                    "diff_with_ir": diff_with_ir,
                    "diff_depletion": diff_depletion,
                    "ms_on": ms_on, "ms_off": ms_off,
                    "ms_on_ir": ms_on_ir, "ms_off_ir": ms_off_ir,
                    "n_on": int(on_mask.sum()), "n_off": int(off_mask.sum()),
                }
                st.session_state["_diff_ms_done"] = True

        if st.session_state.get("_diff_ms_done", False):
            d = st.session_state["_diff_ms"]
            st.info(f"Using {d['n_on']} resonant and {d['n_off']} off-resonance wavenumber steps.")

            fig_diff = go.Figure()

            # Net IR-induced depletion (positive = parent lost, negative = fragment gained)
            fig_diff.add_trace(go.Scatter(
                x=mz_vals, y=d["diff_depletion"],
                mode="lines", name="Net IR depletion (on − off)",
                line=dict(color="crimson", width=2),
            ))

            # Reference: with-IR change
            fig_diff.add_trace(go.Scatter(
                x=mz_vals, y=d["diff_with_ir"],
                mode="lines", name="ΔI(with IR): on − off",
                line=dict(color="steelblue", width=1.5, dash="dash"),
            ))

            fig_diff.add_hline(y=0, line_dash="dot", line_color="grey")
            fig_diff.update_layout(
                xaxis_title="m/z",
                yaxis_title="Intensity difference (a.u.)",
                title="On/Off-Resonance Difference Mass Spectrum",
                height=450,
                legend=dict(x=0.6, y=0.95),
            )
            st.plotly_chart(fig_diff, use_container_width=True)

            st.caption(
                "**Red line** (Net IR depletion): positive peaks = parent ion depletion at resonance, "
                "negative dips = fragment appearance. "
                "**Blue dashed** (with-IR change): direct change in the with-IR mass spectrum between "
                "on- and off-resonance — fragments should appear as positive bumps."
            )

            # Static version for report
            fig_diff_static, ax_diff = plt.subplots(figsize=(12, 5))
            ax_diff.plot(mz_vals, d["diff_depletion"], "r-", lw=2, label="Net IR depletion (on − off)")
            ax_diff.plot(mz_vals, d["diff_with_ir"], "b--", lw=1.5, label="ΔI(with IR): on − off")
            ax_diff.axhline(0, color="grey", ls=":", lw=0.8)
            ax_diff.set_xlabel("m/z", fontsize=12)
            ax_diff.set_ylabel("Intensity difference (a.u.)", fontsize=12)
            ax_diff.set_title("On/Off-Resonance Difference Mass Spectrum", fontsize=13, fontweight="bold")
            ax_diff.legend(fontsize=9)
            ax_diff.grid(True, alpha=0.3)
            fig_diff_static.tight_layout()
            add_plot_to_report_button(
                fig_diff_static, "Difference Mass Spectrum",
                key_suffix="diff_ms",
                description="On/off-resonance difference mass spectrum showing IR-active channels",
            )

            # Download buttons for difference MS
            _dm_dl1, _dm_dl2 = st.columns(2)
            with _dm_dl1:
                _buf_dm = io.BytesIO()
                fig_diff_static.savefig(_buf_dm, format="png", dpi=300, bbox_inches="tight")
                _buf_dm.seek(0)
                _dm_fname = f"diff_mass_spectrum_{_wn_tag}_{_mz_tag}.png"
                st.download_button(
                    "⬇️ Download Diff MS (PNG)", data=_buf_dm, file_name=_dm_fname,
                    mime="image/png", key="_dl_dm_png",
                )
                if st.button("💾 Save Diff MS PNG to output", key="_sv_dm_png"):
                    _save_to_output(_buf_dm, _dm_fname)
            with _dm_dl2:
                _csv_dm = pd.DataFrame({
                    "m/z": mz_vals,
                    "net_IR_depletion": d["diff_depletion"],
                    "delta_with_IR": d["diff_with_ir"],
                })
                _dm_csv_fname = f"diff_mass_spectrum_{_wn_tag}_{_mz_tag}.csv"
                st.download_button(
                    "⬇️ Download Diff MS (CSV)", data=_csv_dm.to_csv(index=False),
                    file_name=_dm_csv_fname, mime="text/csv", key="_dl_dm_csv",
                )
                if st.button("💾 Save Diff MS CSV to output", key="_sv_dm_csv"):
                    _save_to_output(_csv_dm, _dm_csv_fname, is_csv=True)

    # ========================================================================================
    # TAB 3: MASS-CHANNEL IR SPECTRA
    # ========================================================================================
    with tab_fragment_ir:
        st.markdown("### Mass-Channel IR Spectra")
        st.caption(
            "Select up to 25 m/z windows. For each, the integrated signal is tracked across "
            "wavenumbers — giving you an IR spectrum for each mass channel. "
            "This reveals which species respond to which vibrational modes."
        )

        # Quick import and compute
        with st.expander("📋 Quick Import from Tab 1 Peak Detection", expanded=False):
            st.caption("Paste masses and compute instantly - no need to configure each channel individually.")

            with st.form("bulk_import_form"):
                bulk_input = st.text_area(
                    "Paste m/z values (comma-separated)",
                    placeholder="e.g., 178.0, 200.5, 225.3",
                    key="_bulk_masses_input",
                    height=80,
                )

                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    quick_width = st.number_input(
                        "Half-width for all channels (m/z)", value=1.0, min_value=0.1, step=0.5,
                        key="_quick_width",
                    )
                with bcol2:
                    quick_compute = st.form_submit_button("⚡ Quick Compute All")

            if quick_compute and bulk_input.strip():
                try:
                    # Parse and compute directly
                    mass_values = [float(x.strip()) for x in bulk_input.split(",") if x.strip()]

                    # Build channels list directly without widgets
                    channels = []
                    for i, mass in enumerate(mass_values):
                        channels.append({
                            "center": mass,
                            "width": quick_width,
                            "label": f"m/z {mass:.1f}"
                        })

                    # Compute immediately (copy computation code here)
                    wn_arr = np.array(wn_list, dtype=float)
                    frag_results = {}

                    for ch in channels:
                        ch_mask = (mz_vals >= ch["center"] - ch["width"]) & (mz_vals <= ch["center"] + ch["width"])
                        if ch_mask.sum() == 0:
                            continue

                        int_without = mat_without[:, ch_mask].sum(axis=1)
                        int_with = mat_with[:, ch_mask].sum(axis=1)
                        raw_without_sum = mat_raw_without[:, ch_mask].sum(axis=1)
                        raw_with_sum = mat_raw_with[:, ch_mask].sum(axis=1)

                        with np.errstate(divide='ignore', invalid='ignore'):
                            depl = int_with / int_without
                            ln_depl = -np.log(depl)

                        delta_i = int_without - int_with

                        frag_results[ch["label"]] = {
                            "wn": wn_arr,
                            "int_without": int_without,
                            "int_with": int_with,
                            "raw_without": raw_without_sum,
                            "raw_with": raw_with_sum,
                            "ln_depletion": ln_depl,
                            "delta_i": delta_i,
                            "center": ch["center"],
                            "width": ch["width"],
                        }

                    st.session_state["_frag_results"] = frag_results
                    st.session_state["_frag_done"] = True
                    st.success(f"✅ Computed spectra for {len(frag_results)} channels!")

                except ValueError:
                    st.error("❌ Invalid input. Use comma-separated numbers.")

        # Manual configuration (optional)
        with st.expander("⚙️ Manual Channel Configuration (Advanced)", expanded=False):
            st.caption("Configure each channel individually with custom labels and widths.")

            with st.form("channel_config_form"):
                n_channels = st.number_input(
                    "Number of m/z channels", min_value=1, max_value=25, value=2,
                    step=1, key="_n_frag_channels",
                )

                channels = []
                cols = st.columns(min(int(n_channels), 3))
                for i in range(int(n_channels)):
                    with cols[i % len(cols)]:
                        default_center = float(mz_vals[len(mz_vals)//2]) if i == 0 else float(mz_vals[len(mz_vals)//3 * (i % 3)])

                        center = st.number_input(
                            f"Channel {i+1} center (m/z)",
                            value=default_center,
                            step=1.0, key=f"_frag_center_{i}",
                        )
                        width = st.number_input(
                            f"Channel {i+1} half-width (m/z)",
                            value=1.0, min_value=0.1, step=0.5, key=f"_frag_width_{i}",
                        )
                        label = st.text_input(
                            f"Channel {i+1} label",
                            value=f"m/z {center:.0f}", key=f"_frag_label_{i}",
                        )
                        channels.append({"center": center, "width": width, "label": label})

                submitted = st.form_submit_button("📈 Compute Mass-Channel IR Spectra")

            if submitted:
                wn_arr = np.array(wn_list, dtype=float)
                frag_results = {}

                for ch in channels:
                    ch_mask = (mz_vals >= ch["center"] - ch["width"]) & (mz_vals <= ch["center"] + ch["width"])
                    if ch_mask.sum() == 0:
                        st.warning(f"⚠️ No m/z bins in range for {ch['label']}")
                        continue

                    # Integrate within the m/z window for each wavenumber
                    int_without = mat_without[:, ch_mask].sum(axis=1)
                    int_with = mat_with[:, ch_mask].sum(axis=1)
                    raw_without_sum = mat_raw_without[:, ch_mask].sum(axis=1)
                    raw_with_sum = mat_raw_with[:, ch_mask].sum(axis=1)

                    # Depletion
                    with np.errstate(divide='ignore', invalid='ignore'):
                        depl = int_with / int_without
                        ln_depl = -np.log(depl)

                    # Also compute ΔI (useful for fragments where depletion doesn't make sense)
                    delta_i = int_without - int_with

                    frag_results[ch["label"]] = {
                        "wn": wn_arr,
                        "int_without": int_without,
                        "int_with": int_with,
                        "raw_without": raw_without_sum,
                        "raw_with": raw_with_sum,
                        "ln_depletion": ln_depl,
                        "delta_i": delta_i,
                        "center": ch["center"],
                        "width": ch["width"],
                    }

                st.session_state["_frag_results"] = frag_results
                st.session_state["_frag_done"] = True

        if st.session_state.get("_frag_done", False):
            frag_results = st.session_state["_frag_results"]

            if not frag_results:
                st.warning("No valid channels found.")
            else:
                with st.form("plot_options_form"):
                    st.markdown("#### Visualization Options")
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    with scol1:
                        plot_mode = st.radio(
                            "Y-axis quantity",
                            options=["-ln(depletion)", "ΔI (without − with)", "Integrated signal (both)"],
                            horizontal=True, key="_frag_plot_mode",
                        )
                    with scol2:
                        smooth_window = st.selectbox(
                            "Savitzky-Golay smoothing window (0 = off)",
                            options=[0, 5, 7, 9, 11, 15, 21, 31],
                            index=0, key="_frag_smooth_window",
                        )
                    with scol3:
                        clip_negative = st.checkbox(
                            "Clip negative values to zero",
                            value=False,
                            help="Removes negative baseline dips from IR spectra.",
                            key="_frag_clip_negative",
                        )
                    with scol4:
                        view_mode = st.radio(
                            "View mode",
                            options=["Overlay", "Ridge Plot", "3D Waterfall"],
                            index=0, key="_frag_view_mode",
                            help="Ridge Plot: stacked 2D traces. 3D Waterfall: interactive 3D view with lines and markers.",
                        )

                    plot_submitted = st.form_submit_button("📊 Update Plot")

                # Store plot settings in session state when form is submitted
                if plot_submitted:
                    st.session_state["_plot_mode"] = plot_mode
                    st.session_state["_smooth_window"] = smooth_window
                    st.session_state["_clip_negative"] = clip_negative
                    st.session_state["_view_mode"] = view_mode

                # Use stored settings if available, otherwise use form defaults
                plot_mode = st.session_state.get("_plot_mode", plot_mode)
                smooth_window = st.session_state.get("_smooth_window", smooth_window)
                clip_negative = st.session_state.get("_clip_negative", clip_negative)
                view_mode = st.session_state.get("_view_mode", view_mode)

                # Expanded color palette for up to 25 channels
                colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
                    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
                    "#dbdb8d", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39",
                    "#7b4173"
                ]

                # Helper function for smoothing and clipping
                def apply_smooth(y_data, window, clip=False):
                    y_processed = y_data.copy()
                    if window > 0 and len(y_data) > window:
                        y_processed = savgol_filter(y_processed, window_length=window, polyorder=2)
                    if clip:
                        y_processed = np.clip(y_processed, 0, None)
                    return y_processed

                # Helper: get y data for a channel based on plot_mode
                # Returns (y,) for single-trace modes, (y_without, y_with) for "Integrated signal (both)"
                def get_y_data(data, mode):
                    if mode == "-ln(depletion)":
                        return (apply_smooth(data["ln_depletion"], smooth_window, clip_negative),)
                    elif mode == "ΔI (without − with)":
                        return (apply_smooth(data["delta_i"], smooth_window, clip_negative),)
                    else:
                        return (
                            apply_smooth(data["int_without"], smooth_window, clip_negative),
                            apply_smooth(data["int_with"], smooth_window, clip_negative),
                        )
                is_dual = plot_mode == "Integrated signal (both)"

                if plot_mode == "-ln(depletion)":
                    y_title = "-ln(depletion)"
                elif plot_mode == "ΔI (without − with)":
                    y_title = "ΔI (a.u.)"
                else:
                    y_title = "Integrated signal (a.u.)"

                if view_mode == "Overlay":
                    # ---- OVERLAY MODE (original) ----
                    fig_frag = go.Figure()
                    for idx, (label, data) in enumerate(frag_results.items()):
                        c = colors[idx % len(colors)]
                        if plot_mode == "-ln(depletion)":
                            y_smooth = apply_smooth(data["ln_depletion"], smooth_window, clip_negative)
                            fig_frag.add_trace(go.Scatter(
                                x=data["wn"], y=y_smooth,
                                mode="lines", name=label, line=dict(color=c, width=2),
                            ))
                        elif plot_mode == "ΔI (without − with)":
                            y_smooth = apply_smooth(data["delta_i"], smooth_window, clip_negative)
                            fig_frag.add_trace(go.Scatter(
                                x=data["wn"], y=y_smooth,
                                mode="lines", name=label, line=dict(color=c, width=2),
                            ))
                        else:
                            y_smooth_without = apply_smooth(data["int_without"], smooth_window, clip_negative)
                            y_smooth_with = apply_smooth(data["int_with"], smooth_window, clip_negative)
                            fig_frag.add_trace(go.Scatter(
                                x=data["wn"], y=y_smooth_without,
                                mode="lines", name=f"{label} (no IR)",
                                line=dict(color=c, width=2),
                            ))
                            fig_frag.add_trace(go.Scatter(
                                x=data["wn"], y=y_smooth_with,
                                mode="lines", name=f"{label} (with IR)",
                                line=dict(color=c, width=2, dash="dash"),
                            ))

                    fig_frag.update_layout(
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title=y_title,
                        title="Mass-Channel IR Spectra",
                        height=500,
                        legend=dict(x=0.7, y=0.95),
                    )
                    st.plotly_chart(fig_frag, use_container_width=True)

                    # Static overlay for report
                    fig_frag_static, ax_frag = plt.subplots(figsize=(12, 5))
                    for idx, (label, data) in enumerate(frag_results.items()):
                        c = colors[idx % len(colors)]
                        if plot_mode == "-ln(depletion)":
                            y_smooth = apply_smooth(data["ln_depletion"], smooth_window, clip_negative)
                            ax_frag.plot(data["wn"], y_smooth, color=c, lw=2, label=label)
                        elif plot_mode == "ΔI (without − with)":
                            y_smooth = apply_smooth(data["delta_i"], smooth_window, clip_negative)
                            ax_frag.plot(data["wn"], y_smooth, color=c, lw=2, label=label)
                        else:
                            y_smooth_without = apply_smooth(data["int_without"], smooth_window, clip_negative)
                            y_smooth_with = apply_smooth(data["int_with"], smooth_window, clip_negative)
                            ax_frag.plot(data["wn"], y_smooth_without, color=c, lw=2, label=f"{label} (no IR)")
                            ax_frag.plot(data["wn"], y_smooth_with, color=c, lw=2, ls="--", label=f"{label} (with IR)")
                    ax_frag.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
                    ax_frag.set_ylabel(y_title, fontsize=12)
                    ax_frag.set_title("Mass-Channel IR Spectra", fontsize=13, fontweight="bold")
                    ax_frag.legend(fontsize=9)
                    ax_frag.grid(True, alpha=0.3)
                    fig_frag_static.tight_layout()

                elif view_mode == "Ridge Plot":
                    # ---- RIDGE PLOT MODE ----
                    labels_list = list(frag_results.keys())
                    n_traces = len(labels_list)
                    wn_data = list(frag_results.values())[0]["wn"]

                    # Compute all y-data
                    all_y_tuples = []
                    for label, data in frag_results.items():
                        all_y_tuples.append(get_y_data(data, plot_mode))

                    # Compute peak-to-peak intensity per channel for relative scaling
                    raw_intensities = []
                    for yt in all_y_tuples:
                        all_vals = np.concatenate(yt)
                        raw_intensities.append(np.nanmax(all_vals) - np.nanmin(all_vals))
                    max_intensity = max(raw_intensities) if max(raw_intensities) > 0 else 1.0
                    rel_intensities = [ri / max_intensity for ri in raw_intensities]

                    # Annotated labels with relative intensity
                    annotated_labels = [
                        f"{lbl}  (×{rel:.2f})" for lbl, rel in zip(labels_list, rel_intensities)
                    ]

                    # Normalize: use the max across both traces (if dual) for consistent scaling
                    norm_y_tuples = []
                    for yt in all_y_tuples:
                        all_vals = np.concatenate(yt)
                        y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
                        if y_max - y_min > 0:
                            norm_y_tuples.append(tuple((y - y_min) / (y_max - y_min) for y in yt))
                        else:
                            norm_y_tuples.append(tuple(np.zeros_like(y) for y in yt))

                    # Ridge spacing
                    spacing = 1.0 if is_dual else 0.7

                    # Plotly ridge plot
                    fig_frag = go.Figure()
                    for idx in range(n_traces - 1, -1, -1):
                        c = colors[idx % len(colors)]
                        offset = idx * spacing
                        y_without_shifted = norm_y_tuples[idx][0] + offset

                        # Filled area (no-IR trace)
                        fig_frag.add_trace(go.Scatter(
                            x=np.concatenate([wn_data, wn_data[::-1]]),
                            y=np.concatenate([y_without_shifted, np.full(len(wn_data), offset)]),
                            fill="toself",
                            fillcolor=c, opacity=0.2,
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                        ))
                        # Solid line (no-IR or single trace)
                        fig_frag.add_trace(go.Scatter(
                            x=wn_data, y=y_without_shifted,
                            mode="lines",
                            name=f"{labels_list[idx]} (no IR)" if is_dual else labels_list[idx],
                            line=dict(color=c, width=1.5),
                        ))
                        # Dashed line (with-IR) if dual mode
                        if is_dual:
                            y_with_shifted = norm_y_tuples[idx][1] + offset
                            fig_frag.add_trace(go.Scatter(
                                x=wn_data, y=y_with_shifted,
                                mode="lines",
                                name=f"{labels_list[idx]} (with IR)",
                                line=dict(color=c, width=1.5, dash="dash"),
                            ))

                    fig_frag.update_layout(
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis=dict(
                            tickvals=[i * spacing + 0.5 for i in range(n_traces)],
                            ticktext=annotated_labels,
                            title="",
                        ),
                        title=f"Ridge Plot — {y_title}",
                        height=max(400, n_traces * 60 + 100),
                        showlegend=is_dual,
                        margin=dict(l=140),
                    )
                    st.plotly_chart(fig_frag, use_container_width=True)

                    # Static ridge plot for report
                    fig_height = max(6, n_traces * 0.8 + 1)
                    fig_frag_static, ax_frag = plt.subplots(figsize=(12, fig_height))
                    for idx in range(n_traces):
                        c = colors[idx % len(colors)]
                        offset = idx * spacing
                        y_without_shifted = norm_y_tuples[idx][0] + offset

                        ax_frag.fill_between(wn_data, offset, y_without_shifted, color=c, alpha=0.2)
                        ax_frag.plot(wn_data, y_without_shifted, color=c, lw=1.5,
                                     label=f"{labels_list[idx]} (no IR)" if is_dual else None)
                        if is_dual:
                            y_with_shifted = norm_y_tuples[idx][1] + offset
                            ax_frag.plot(wn_data, y_with_shifted, color=c, lw=1.5, ls="--",
                                         label=f"{labels_list[idx]} (with IR)")

                    ax_frag.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
                    ax_frag.set_yticks([i * spacing + 0.5 for i in range(n_traces)])
                    ax_frag.set_yticklabels(annotated_labels, fontsize=9)
                    ax_frag.set_title(f"Ridge Plot — {y_title}", fontsize=13, fontweight="bold")
                    ax_frag.grid(True, axis="x", alpha=0.3)
                    if is_dual:
                        ax_frag.legend(fontsize=7, ncol=2, loc="upper right")
                    fig_frag_static.tight_layout()

                else:
                    # ---- 3D WATERFALL MODE ----
                    labels_list = list(frag_results.keys())
                    n_traces = len(labels_list)
                    wn_data = list(frag_results.values())[0]["wn"]

                    # Extract m/z center values for the y-axis
                    mz_centers = [frag_results[lbl]["center"] for lbl in labels_list]

                    fig_frag = go.Figure()
                    for idx, (label, data) in enumerate(frag_results.items()):
                        c = colors[idx % len(colors)]
                        y_tuple = get_y_data(data, plot_mode)
                        mz_y = np.full_like(wn_data, mz_centers[idx])

                        # No-IR / single trace (solid line)
                        y_without = y_tuple[0]
                        fig_frag.add_trace(go.Scatter3d(
                            x=wn_data, y=mz_y, z=y_without,
                            mode="lines",
                            name=f"{label} (no IR)" if is_dual else label,
                            line=dict(color=c, width=3),
                        ))

                        # With-IR trace (dashed) if dual mode
                        if is_dual:
                            y_with = y_tuple[1]
                            # Slight y-offset so lines don't perfectly overlap
                            mz_offset = np.mean(np.diff(sorted(set(mz_centers)))) * 0.05 if len(set(mz_centers)) > 1 else 0.3
                            fig_frag.add_trace(go.Scatter3d(
                                x=wn_data,
                                y=np.full_like(wn_data, mz_centers[idx] + mz_offset),
                                z=y_with,
                                mode="lines",
                                name=f"{label} (with IR)",
                                line=dict(color=c, width=2, dash="dash"),
                            ))

                        # Scatter markers at peaks (local maxima above median)
                        median_val = np.nanmedian(y_without)
                        peak_mask = np.zeros(len(y_without), dtype=bool)
                        for j in range(1, len(y_without) - 1):
                            if y_without[j] > y_without[j-1] and y_without[j] > y_without[j+1] and y_without[j] > median_val:
                                peak_mask[j] = True
                        if peak_mask.sum() > 0:
                            fig_frag.add_trace(go.Scatter3d(
                                x=wn_data[peak_mask], y=mz_y[peak_mask], z=y_without[peak_mask],
                                mode="markers",
                                name=f"{label} peaks",
                                marker=dict(color=c, size=3, symbol="diamond"),
                                showlegend=False,
                            ))

                        # "Curtain" drop-lines from trace to z=0 for waterfall effect
                        z_min = 0 if clip_negative else float(np.nanmin(y_without))
                        for step in range(0, len(wn_data), max(1, len(wn_data) // 30)):
                            fig_frag.add_trace(go.Scatter3d(
                                x=[wn_data[step], wn_data[step]],
                                y=[mz_centers[idx], mz_centers[idx]],
                                z=[z_min, y_without[step]],
                                mode="lines",
                                line=dict(color=c, width=1),
                                opacity=0.15,
                                showlegend=False,
                                hoverinfo="skip",
                            ))

                    fig_frag.update_layout(
                        scene=dict(
                            xaxis_title="Wavenumber (cm⁻¹)",
                            yaxis_title="m/z",
                            zaxis_title=y_title,
                            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8)),
                        ),
                        title=f"3D Waterfall — {y_title}",
                        height=700,
                        showlegend=True,
                        legend=dict(x=0.85, y=0.95, font=dict(size=9)),
                    )
                    st.plotly_chart(fig_frag, use_container_width=True)

                    # Static fallback (ridge-style) for report/PNG
                    fig_height = max(6, n_traces * 0.8 + 1)
                    fig_frag_static, ax_frag = plt.subplots(figsize=(12, fig_height))
                    spacing = 1.0 if is_dual else 0.7
                    for idx, (label, data) in enumerate(frag_results.items()):
                        c = colors[idx % len(colors)]
                        y_tuple = get_y_data(data, plot_mode)
                        all_vals = np.concatenate(y_tuple)
                        y_min, y_max = np.nanmin(all_vals), np.nanmax(all_vals)
                        if y_max - y_min > 0:
                            y_norm_without = (y_tuple[0] - y_min) / (y_max - y_min)
                        else:
                            y_norm_without = np.zeros_like(y_tuple[0])
                        offset = idx * spacing
                        ax_frag.fill_between(wn_data, offset, y_norm_without + offset, color=c, alpha=0.2)
                        ax_frag.plot(wn_data, y_norm_without + offset, color=c, lw=1.5,
                                     label=f"{labels_list[idx]} (no IR)" if is_dual else None)
                        if is_dual:
                            y_norm_with = (y_tuple[1] - y_min) / (y_max - y_min)
                            ax_frag.plot(wn_data, y_norm_with + offset, color=c, lw=1.5, ls="--",
                                         label=f"{labels_list[idx]} (with IR)")

                    ax_frag.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
                    ax_frag.set_yticks([i * spacing + 0.5 for i in range(n_traces)])
                    ax_frag.set_yticklabels(labels_list, fontsize=9)
                    ax_frag.set_title(f"3D Waterfall (static) — {y_title}", fontsize=13, fontweight="bold")
                    ax_frag.grid(True, axis="x", alpha=0.3)
                    if is_dual:
                        ax_frag.legend(fontsize=7, ncol=2, loc="upper right")
                    fig_frag_static.tight_layout()

                add_plot_to_report_button(
                    fig_frag_static, "Mass-Channel IR Spectra",
                    key_suffix="frag_ir",
                    description="IR spectra for individual m/z channels",
                )

                # Download buttons for mass-channel IR spectra
                _fr_dl1, _fr_dl2 = st.columns(2)
                with _fr_dl1:
                    _buf_fr = io.BytesIO()
                    fig_frag_static.savefig(_buf_fr, format="png", dpi=300, bbox_inches="tight")
                    _buf_fr.seek(0)
                    _n_ch = len(frag_results)
                    _fr_fname = f"mass_channel_IR_spectra_{_wn_tag}_{_n_ch}ch.png"
                    st.download_button(
                        "⬇️ Download IR Spectra (PNG)", data=_buf_fr, file_name=_fr_fname,
                        mime="image/png", key="_dl_fr_png",
                    )
                    if st.button("💾 Save IR Spectra PNG to output", key="_sv_fr_png"):
                        _save_to_output(_buf_fr, _fr_fname)
                with _fr_dl2:
                    _csv_frames = []
                    for lbl, dat in frag_results.items():
                        _ch_wn = dat["wn"]
                        if plot_mode == "-ln(depletion)":
                            y_smooth = apply_smooth(dat["ln_depletion"], smooth_window, clip_negative)
                            _csv_frames.append(pd.DataFrame({"Wavenumber": _ch_wn, f"{lbl} -ln(depl)": y_smooth}))
                        elif plot_mode == "ΔI (without − with)":
                            y_smooth = apply_smooth(dat["delta_i"], smooth_window, clip_negative)
                            _csv_frames.append(pd.DataFrame({"Wavenumber": _ch_wn, f"{lbl} DeltaI": y_smooth}))
                        else:
                            y_smooth_without = apply_smooth(dat["int_without"], smooth_window, clip_negative)
                            y_smooth_with = apply_smooth(dat["int_with"], smooth_window, clip_negative)
                            _csv_frames.append(pd.DataFrame({"Wavenumber": _ch_wn, f"{lbl} no_IR": y_smooth_without, f"{lbl} with_IR": y_smooth_with}))
                    if _csv_frames:
                        _csv_fr = reduce(lambda left, right: pd.merge(left, right, on="Wavenumber", how="outer"), _csv_frames)
                    else:
                        _csv_fr = pd.DataFrame()
                    _fr_csv_fname = f"mass_channel_IR_spectra_{_wn_tag}_{_n_ch}ch.csv"
                    st.download_button(
                        "⬇️ Download IR Spectra (CSV)", data=_csv_fr.to_csv(index=False),
                        file_name=_fr_csv_fname, mime="text/csv", key="_dl_fr_csv",
                    )
                    if st.button("💾 Save IR Spectra CSV to output", key="_sv_fr_csv"):
                        _save_to_output(_csv_fr, _fr_csv_fname, is_csv=True)

    # ========================================================================================
    # TAB 4: SIGNAL QUALITY & UNCERTAINTY
    # ========================================================================================
    with tab_quality:
        st.markdown("### Signal Quality & Uncertainty")
        st.caption(
            "Evaluate which IR peaks are statistically significant using Poisson counting statistics. "
            "The error on −ln(I_with/I_without) is propagated from √N ion-counting noise."
        )

        if mat_raw_without is None or mat_raw_with is None:
            st.warning("⚠️ Raw sum matrices not found. Please re-run **✨ Process Data** (top of page) to enable uncertainty analysis.")
            st.stop()

        with st.expander("📊 Ion Count Statistics (raw data summary)", expanded=False):
            _rw = mat_raw_without
            _rw2 = mat_raw_with
            _stats = {
                "": ["I_without (raw)", "I_with (raw)"],
                "Min": [f"{np.nanmin(_rw):.1f}", f"{np.nanmin(_rw2):.1f}"],
                "1st %ile": [f"{np.nanpercentile(_rw, 1):.1f}", f"{np.nanpercentile(_rw2, 1):.1f}"],
                "Median": [f"{np.nanmedian(_rw):.1f}", f"{np.nanmedian(_rw2):.1f}"],
                "Mean": [f"{np.nanmean(_rw):.1f}", f"{np.nanmean(_rw2):.1f}"],
                "99th %ile": [f"{np.nanpercentile(_rw, 99):.1f}", f"{np.nanpercentile(_rw2, 99):.1f}"],
                "Max": [f"{np.nanmax(_rw):.1f}", f"{np.nanmax(_rw2):.1f}"],
            }
            st.dataframe(pd.DataFrame(_stats).set_index(""), use_container_width=True)
            _offset = np.nanmedian(_rw) - np.nanmedian(mat_without)
            st.caption(
                f"Shape: {_rw.shape[0]} wavenumbers × {_rw.shape[1]} m/z bins. "
                f"Typical baseline offset (median raw − median corrected): **{_offset:.1f}**"
            )

        if not st.session_state.get("_frag_done", False):
            st.info("👆 Compute mass channels in **Tab 3** first, then return here.")
            st.stop()

        frag_results_q = st.session_state["_frag_results"]
        if not frag_results_q:
            st.warning("No valid channels found.")
        else:
            labels_q = list(frag_results_q.keys())

            # --- Helper: compute sigma using delta-method standard ---
            def compute_sigma_ln(data_ch):
                """
                Delta-method variance for A = −ln(I_w/I_0):
                    σ_A = √(1/I_w + 1/I_0)
                Uses raw (pre-baseline) counts as the Poisson rate estimates.
                Valid for counts ≳ 20–25 (Bevington & Robinson Ch. 3).
                """
                i_w = np.abs(data_ch["raw_with"])
                i_0 = np.abs(data_ch["raw_without"])
                with np.errstate(divide='ignore', invalid='ignore'):
                    sigma = np.where(
                        (i_w > 0) & (i_0 > 0),
                        np.sqrt(1.0 / i_w + 1.0 / i_0),
                        np.nan,
                    )
                return sigma

            # ================================================================
            # Per-Channel Diagnostic Plot (2 panels)
            # ================================================================
            st.markdown("---")
            st.markdown("#### Per-Channel Diagnostic Plot")
            st.caption(
                "Two panels: (1) integrated signals with and without IR, "
                "(2) −ln(depletion) with ±1σ error band (delta-method: σ = √(1/I_w + 1/I_0))."
            )

            with st.form(key="_q_diag_form"):
                _qc1, _qc2 = st.columns([2, 1])
                with _qc1:
                    selected_channel = st.selectbox(
                        "Select channel", options=labels_q,
                    )
                with _qc2:
                    smooth_win = st.select_slider(
                        "Savitzky-Golay window",
                        options=[0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
                        value=0,
                        help="Window length for Savitzky-Golay smoothing (0 = off, odd values only).",
                    )
                _generate_plot = st.form_submit_button("📊 Generate Plot", use_container_width=True)

            if not _generate_plot:
                st.info("Select a channel and smoothing, then click **Generate Plot**.")
            else:
                ch_data = frag_results_q[selected_channel]
                wn_q = ch_data["wn"]
                ln_depl = ch_data["ln_depletion"]
                sigma_ln = compute_sigma_ln(ch_data)

                ln_clean = np.where(np.isfinite(ln_depl), ln_depl, np.nan)
                sigma_clean = np.where(np.isfinite(sigma_ln), sigma_ln, np.nan)

                if smooth_win >= 3 and len(ln_clean) >= smooth_win:
                    poly_order = min(3, smooth_win - 1)
                    mask_valid = np.isfinite(ln_clean)
                    ln_filled = np.where(mask_valid, ln_clean, 0.0)
                    ln_clean = np.where(mask_valid, savgol_filter(ln_filled, smooth_win, poly_order), np.nan)
                    sig_filled = np.where(mask_valid, sigma_clean, 0.0)
                    sigma_clean = np.where(mask_valid, savgol_filter(sig_filled, smooth_win, poly_order), np.nan)

                upper_band = ln_clean + sigma_clean
                lower_band = ln_clean - sigma_clean

                # --- Interactive Plotly (2 stacked subplots) ---
                fig_diag = make_subplots(
                    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                    row_heights=[0.45, 0.55],
                    subplot_titles=("Integrated Signals", "−ln(depletion) ± 1σ"),
                )

                fig_diag.add_trace(
                    go.Scatter(x=wn_q, y=ch_data["int_without"], mode="lines",
                               name="I (no IR)", line=dict(color="steelblue", width=1.5)),
                    row=1, col=1,
                )
                fig_diag.add_trace(
                    go.Scatter(x=wn_q, y=ch_data["int_with"], mode="lines",
                               name="I (with IR)", line=dict(color="darkorange", width=1.5, dash="dash")),
                    row=1, col=1,
                )

                fig_diag.add_trace(
                    go.Scatter(x=wn_q, y=upper_band, mode="lines",
                               line=dict(width=0), showlegend=False),
                    row=2, col=1,
                )
                fig_diag.add_trace(
                    go.Scatter(x=wn_q, y=lower_band, mode="lines", name="±1σ band",
                               line=dict(width=0), fill="tonexty",
                               fillcolor="rgba(220, 60, 60, 0.18)"),
                    row=2, col=1,
                )
                fig_diag.add_trace(
                    go.Scatter(x=wn_q, y=ln_clean, mode="lines", name="−ln(depl)",
                               line=dict(color="crimson", width=2)),
                    row=2, col=1,
                )
                fig_diag.add_hline(y=0, line_dash="dot", line_color="grey", row=2, col=1)

                fig_diag.add_annotation(
                    text="σ = √(1/I<sub>w</sub> + 1/I<sub>0</sub>)  (Poisson, delta-method)",
                    xref="x2 domain", yref="y2 domain", x=0.01, y=0.97,
                    showarrow=False, font=dict(size=11, color="grey"),
                    bgcolor="rgba(255,255,255,0.7)", borderpad=3,
                )
                fig_diag.update_layout(
                    height=600, title=f"Diagnostic: {selected_channel}",
                    legend=dict(x=0.01, y=1.0, bgcolor="rgba(255,255,255,0.7)"),
                )
                fig_diag.update_xaxes(title_text="Wavenumber (cm⁻¹)", row=2, col=1)
                fig_diag.update_yaxes(title_text="Signal (a.u.)", row=1, col=1)
                fig_diag.update_yaxes(title_text="−ln(depl)", row=2, col=1)

                st.plotly_chart(fig_diag, use_container_width=True)
                st.caption(
                    "**Top:** Blue = without IR, Orange = with IR. Where they diverge → depletion. "
                    "**Bottom:** Red line = −ln(depletion), shaded band = ±1σ uncertainty."
                )

                # --- Static matplotlib version (2 panels) ---
                fig_diag_static, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                                             sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
                fig_diag_static.subplots_adjust(hspace=0.10)

                ax1.plot(wn_q, ch_data["int_without"], "b-", lw=1.5, label="I (no IR)")
                ax1.plot(wn_q, ch_data["int_with"], color="darkorange", ls="--", lw=1.5, label="I (with IR)")
                ax1.set_ylabel("Signal (a.u.)", fontsize=11)
                ax1.legend(fontsize=9, loc="upper right")
                ax1.set_title(f"Diagnostic: {selected_channel}", fontsize=13, fontweight="bold")
                ax1.grid(True, alpha=0.3)

                ax2.fill_between(wn_q, lower_band, upper_band, color="red", alpha=0.15, label="±1σ")
                ax2.plot(wn_q, ln_clean, "r-", lw=2, label="−ln(depl)")
                ax2.axhline(0, color="grey", ls=":", lw=0.8)
                ax2.set_ylabel("−ln(depletion)", fontsize=11)
                ax2.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
                ax2.legend(fontsize=9, loc="upper right")
                ax2.text(0.01, 0.95, r"$\sigma = \sqrt{1/I_w + 1/I_0}$  (Poisson, delta-method)",
                         transform=ax2.transAxes, fontsize=10, color="grey",
                         va="top", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))
                ax2.grid(True, alpha=0.3)
                fig_diag_static.tight_layout()

                _diag_dl1, _diag_dl2, _diag_dl3 = st.columns(3)
                with _diag_dl1:
                    _buf_diag = io.BytesIO()
                    fig_diag_static.savefig(_buf_diag, format="png", dpi=300, bbox_inches="tight")
                    _buf_diag.seek(0)
                    st.download_button(
                        "⬇️ Static Plot (PNG)", data=_buf_diag,
                        file_name=f"diagnostic_{selected_channel.replace('/', '_')}.png",
                        mime="image/png", key="_dl_diag_png",
                    )
                with _diag_dl2:
                    _html_diag = fig_diag.to_html(include_plotlyjs="cdn")
                    st.download_button(
                        "⬇️ Interactive Plot (HTML)", data=_html_diag,
                        file_name=f"diagnostic_{selected_channel.replace('/', '_')}.html",
                        mime="text/html", key="_dl_diag_html",
                    )
                with _diag_dl3:
                    _csv_diag = pd.DataFrame({
                        "Wavenumber": wn_q,
                        "I_without": ch_data["int_without"],
                        "I_with": ch_data["int_with"],
                        "neg_ln_depletion": ln_clean,
                        "sigma_ln": sigma_clean,
                    })
                    st.download_button(
                        "⬇️ Data (CSV)", data=_csv_diag.to_csv(index=False),
                        file_name=f"diagnostic_{selected_channel.replace('/', '_')}.csv",
                        mime="text/csv", key="_dl_diag_csv",
                    )
                add_plot_to_report_button(
                    fig_diag_static, f"Diagnostic: {selected_channel}",
                    key_suffix="diag_plot",
                    description=f"Signal quality diagnostic for {selected_channel} with delta-method error band",
                )
                plt.close(fig_diag_static)

            # ================================================================
            # Channel Summary Table
            # ================================================================
            st.markdown("---")
            st.markdown("#### Channel Summary")
            st.caption(
                "Peak depletion value and its uncertainty (σ = √(1/I_w + 1/I_0)) at the peak position for each channel."
            )

            summary_rows = []
            for label in labels_q:
                ch_d = frag_results_q[label]
                ln_d = ch_d["ln_depletion"]
                sig_d = compute_sigma_ln(ch_d)

                ln_finite = np.where(np.isfinite(ln_d), ln_d, np.nan)
                sig_finite = np.where(np.isfinite(sig_d), sig_d, np.nan)

                if not np.any(np.isfinite(ln_finite)):
                    summary_rows.append({
                        "Channel": label, "m/z center": ch_d["center"],
                        "Peak −ln(depl)": 0.0, "Peak at (cm⁻¹)": 0.0,
                        "σ at peak": 0.0, "Mean σ": 0.0,
                    })
                    continue

                peak_wn_idx = int(np.nanargmax(np.abs(ln_finite)))
                peak_val = ln_finite[peak_wn_idx]
                peak_wn = ch_d["wn"][peak_wn_idx]
                sigma_at_peak = sig_finite[peak_wn_idx] if np.isfinite(sig_finite[peak_wn_idx]) else np.nan
                mean_sigma = np.nanmean(sig_finite)

                summary_rows.append({
                    "Channel": label,
                    "m/z center": ch_d["center"],
                    "Peak −ln(depl)": round(float(peak_val), 4),
                    "Peak at (cm⁻¹)": round(float(peak_wn), 1),
                    "σ at peak": round(float(sigma_at_peak), 4) if np.isfinite(sigma_at_peak) else "—",
                    "Mean σ": round(float(mean_sigma), 4),
                })

            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, height=min(400, len(summary_df) * 40 + 60))

            _n_ch_sum = len(summary_df)
            _sum_fname = f"channel_summary_{_wn_tag}_{_n_ch_sum}ch.csv"
            st.download_button(
                "⬇️ Download Summary Table (CSV)", data=summary_df.to_csv(index=False),
                file_name=_sum_fname, mime="text/csv", key="_dl_summary_csv",
            )
            if st.button("💾 Save Summary to output", key="_sv_summary_csv"):
                _save_to_output(summary_df, _sum_fname, is_csv=True)

_misc_controls_section()
_misc_tabs_section()
