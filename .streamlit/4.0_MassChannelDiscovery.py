import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import io
from packages.ReportManager import add_plot_to_report_button, init_report_session

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
x_mass = st.session_state.get("x_mass")
compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data")
unique_wavenumbers = st.session_state.get("unique_wavenumbers")
plot_col_without = st.session_state.get("plot_columnIndex_withoutIR")
plot_col_with = st.session_state.get("plot_columnIndex_withIR")

if x_mass is None or compilation_baseline_corrected_data is None or unique_wavenumbers is None:
    st.error("❌ Required data not found. Run Sections 1–2 (import → baseline correction) first.")
    st.stop()

st.success(f"✅ Data loaded: {len(unique_wavenumbers)} wavenumber steps, {len(x_mass)} m/z bins")

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

noise_floor = st.number_input(
    "Noise floor (baseline signal threshold)",
    value=0.001, min_value=0.0, step=0.0005, format="%.4f",
    help="m/z bins where the without-IR signal is below this value are masked out as empty/noise.",
    key="_misc_noise_floor",
)

# ========================================================================================
# DATA PROCESSING — build matrices once
# ========================================================================================
if st.button("✨ Process Data", type="primary"):
    with st.spinner("Building m/z × wavenumber matrices…"):
        # Filter wavenumbers
        wn_list = sorted([wn for wn in unique_wavenumbers if wn_min <= float(wn) <= wn_max])
        mz_mask = (x_mass >= mz_min) & (x_mass <= mz_max)
        mz_vals = x_mass[mz_mask]

        n_wn = len(wn_list)
        n_mz = int(mz_mask.sum())

        mat_without = np.zeros((n_wn, n_mz))
        mat_with = np.zeros((n_wn, n_mz))

        for i, wn in enumerate(wn_list):
            data_wn = compilation_baseline_corrected_data[wn]
            mat_without[i, :] = data_wn.iloc[mz_mask, plot_col_without].values
            mat_with[i, :] = data_wn.iloc[mz_mask, plot_col_with].values

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
        st.session_state["_misc_mat_delta"] = mat_delta
        st.session_state["_misc_noise_mask_1d"] = noise_mask_1d
        st.session_state["_misc_noise_mask_2d"] = noise_mask_2d
        st.session_state["_misc_baseline_per_mz"] = baseline_per_mz
        st.session_state["_misc_processed"] = True

    n_masked = int((~noise_mask_1d).sum())
    st.success(
        f"✅ Processed {n_wn} wavenumbers × {n_mz} m/z bins. "
        f"Masked {n_masked}/{n_mz} m/z bins below noise floor ({noise_floor:.4f})."
    )

# ========================================================================================
# ANALYSIS TABS
# ========================================================================================
if not st.session_state.get("_misc_processed", False):
    st.info("👆 Press **Process Data** to build the analysis matrices.")
    st.stop()

wn_list = st.session_state["_misc_wn_list"]
mz_vals = st.session_state["_misc_mz_vals"]
mat_without = st.session_state["_misc_mat_without"]
mat_with = st.session_state["_misc_mat_with"]
mat_delta = st.session_state["_misc_mat_delta"]
noise_mask_1d = st.session_state["_misc_noise_mask_1d"]
noise_mask_2d = st.session_state["_misc_noise_mask_2d"]
baseline_per_mz = st.session_state["_misc_baseline_per_mz"]

tab_heatmap, tab_diff_ms, tab_fragment_ir = st.tabs([
    "🗺️ Noise-Masked ΔI Heatmap",
    "📊 On/Off-Resonance Difference MS",
    "📈 Mass-Channel IR Spectra",
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
        st.download_button(
            "⬇️ Download Heatmap (PNG)", data=_buf_hm, file_name="DeltaI_heatmap.png",
            mime="image/png", key="_dl_hm_png",
        )
    with _hm_dl2:
        _csv_hm = pd.DataFrame(mat_display, index=wn_list, columns=mz_vals)
        _csv_hm.index.name = "Wavenumber"
        st.download_button(
            "⬇️ Download Heatmap Data (CSV)", data=_csv_hm.to_csv(),
            file_name="DeltaI_heatmap.csv", mime="text/csv", key="_dl_hm_csv",
        )
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
        st.download_button(
            "⬇️ Download IR Profile (PNG)", data=_buf_pr, file_name="IR_response_profile.png",
            mime="image/png", key="_dl_pr_png",
        )
    with _pr_dl2:
        _csv_pr = pd.DataFrame({"m/z": mz_active, "mean_abs_DeltaI": mean_abs_delta})
        st.download_button(
            "⬇️ Download IR Profile (CSV)", data=_csv_pr.to_csv(index=False),
            file_name="IR_response_profile.csv", mime="text/csv", key="_dl_pr_csv",
        )
    add_plot_to_report_button(
        fig_pr_static, "IR Response Profile", key_suffix="ir_profile",
        description="Mean |ΔI| per m/z showing IR-active channels",
    )
    plt.close(fig_pr_static)

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
            st.download_button(
                "⬇️ Download Diff MS (PNG)", data=_buf_dm, file_name="diff_mass_spectrum.png",
                mime="image/png", key="_dl_dm_png",
            )
        with _dm_dl2:
            _csv_dm = pd.DataFrame({
                "m/z": mz_vals,
                "net_IR_depletion": d["diff_depletion"],
                "delta_with_IR": d["diff_with_ir"],
            })
            st.download_button(
                "⬇️ Download Diff MS (CSV)", data=_csv_dm.to_csv(index=False),
                file_name="diff_mass_spectrum.csv", mime="text/csv", key="_dl_dm_csv",
            )

# ========================================================================================
# TAB 3: MASS-CHANNEL IR SPECTRA
# ========================================================================================
with tab_fragment_ir:
    st.markdown("### Mass-Channel IR Spectra")
    st.caption(
        "Select up to 6 m/z windows. For each, the integrated signal is tracked across "
        "wavenumbers — giving you an IR spectrum for each mass channel. "
        "This reveals which species respond to which vibrational modes."
    )

    n_channels = st.number_input(
        "Number of m/z channels", min_value=1, max_value=6, value=2,
        step=1, key="_n_frag_channels",
    )

    channels = []
    cols = st.columns(min(int(n_channels), 3))
    for i in range(int(n_channels)):
        with cols[i % len(cols)]:
            center = st.number_input(
                f"Channel {i+1} center (m/z)",
                value=float(mz_vals[len(mz_vals)//2]) if i == 0 else float(mz_vals[len(mz_vals)//3 * (i % 3)]),
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

    if st.button("📈 Compute Mass-Channel IR Spectra", key="_run_frag_ir"):
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
            plot_mode = st.radio(
                "Y-axis quantity",
                options=["-ln(depletion)", "ΔI (without − with)", "Integrated signal (both)"],
                horizontal=True, key="_frag_plot_mode",
            )

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

            fig_frag = go.Figure()
            for idx, (label, data) in enumerate(frag_results.items()):
                c = colors[idx % len(colors)]
                if plot_mode == "-ln(depletion)":
                    fig_frag.add_trace(go.Scatter(
                        x=data["wn"], y=data["ln_depletion"],
                        mode="lines", name=label, line=dict(color=c, width=2),
                    ))
                    y_title = "-ln(depletion)"
                elif plot_mode == "ΔI (without − with)":
                    fig_frag.add_trace(go.Scatter(
                        x=data["wn"], y=data["delta_i"],
                        mode="lines", name=label, line=dict(color=c, width=2),
                    ))
                    y_title = "ΔI (a.u.)"
                else:
                    fig_frag.add_trace(go.Scatter(
                        x=data["wn"], y=data["int_without"],
                        mode="lines", name=f"{label} (no IR)",
                        line=dict(color=c, width=2),
                    ))
                    fig_frag.add_trace(go.Scatter(
                        x=data["wn"], y=data["int_with"],
                        mode="lines", name=f"{label} (with IR)",
                        line=dict(color=c, width=2, dash="dash"),
                    ))
                    y_title = "Integrated signal (a.u.)"

            fig_frag.update_layout(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title=y_title,
                title="Mass-Channel IR Spectra",
                height=500,
                legend=dict(x=0.7, y=0.95),
            )
            st.plotly_chart(fig_frag, use_container_width=True)

            # Static version for report
            fig_frag_static, ax_frag = plt.subplots(figsize=(12, 5))
            for idx, (label, data) in enumerate(frag_results.items()):
                c = colors[idx % len(colors)]
                if plot_mode == "-ln(depletion)":
                    ax_frag.plot(data["wn"], data["ln_depletion"], color=c, lw=2, label=label)
                elif plot_mode == "ΔI (without − with)":
                    ax_frag.plot(data["wn"], data["delta_i"], color=c, lw=2, label=label)
                else:
                    ax_frag.plot(data["wn"], data["int_without"], color=c, lw=2, label=f"{label} (no IR)")
                    ax_frag.plot(data["wn"], data["int_with"], color=c, lw=2, ls="--", label=f"{label} (with IR)")
            ax_frag.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
            ax_frag.set_ylabel(y_title, fontsize=12)
            ax_frag.set_title("Mass-Channel IR Spectra", fontsize=13, fontweight="bold")
            ax_frag.legend(fontsize=9)
            ax_frag.grid(True, alpha=0.3)
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
                st.download_button(
                    "⬇️ Download IR Spectra (PNG)", data=_buf_fr, file_name="mass_channel_IR_spectra.png",
                    mime="image/png", key="_dl_fr_png",
                )
            with _fr_dl2:
                _csv_rows = {"Wavenumber": wn_list}
                for lbl, dat in frag_results.items():
                    if plot_mode == "-ln(depletion)":
                        _csv_rows[f"{lbl} -ln(depl)"] = dat["ln_depletion"]
                    elif plot_mode == "ΔI (without − with)":
                        _csv_rows[f"{lbl} DeltaI"] = dat["delta_i"]
                    else:
                        _csv_rows[f"{lbl} no_IR"] = dat["int_without"]
                        _csv_rows[f"{lbl} with_IR"] = dat["int_with"]
                _csv_fr = pd.DataFrame(_csv_rows)
                st.download_button(
                    "⬇️ Download IR Spectra (CSV)", data=_csv_fr.to_csv(index=False),
                    file_name="mass_channel_IR_spectra.csv", mime="text/csv", key="_dl_fr_csv",
                )
