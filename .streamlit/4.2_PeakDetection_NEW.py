import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
import re
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

st.markdown("# 🔬 Peak Detection and Candidate Formula Matching")
st.markdown(
    "Automated workflow for detecting peaks in mass spectra and matching candidate molecular formulas."
)

# Progress indicator
def show_progress():
    """Display workflow progress"""
    steps = {
        "Data Loaded": st.session_state.get("signal") is not None,
        "Peaks Detected": st.session_state.get("detected_mz") is not None,
        "Formulas Matched": st.session_state.get("candidates_df") is not None,
    }
    
    cols = st.columns(len(steps))
    for i, (step, completed) in enumerate(steps.items()):
        with cols[i]:
            if completed:
                st.success(f"✅ {step}")
            else:
                st.info(f"⏳ {step}")

show_progress()
st.divider()

#####################################
# STEP 0: DATA SOURCE & SIGNAL SELECTION
#####################################
with st.expander("📂 Step 0: Data Source & Signal Selection", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        data_source = st.radio(
            "Select Data Source",
            options=["Use data from session", "Upload CSV file"],
            key="data_source",
            horizontal=True
        )
        
        if data_source == "Upload CSV file":
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                if "x_mass" in df.columns:
                    x_mass = df["x_mass"].values
                    MegaSum = df
                    st.session_state["MegaSum"] = MegaSum
                    st.session_state["x_mass"] = x_mass
                    st.success("✅ CSV file uploaded and data loaded!")
                else:
                    st.error("❌ CSV file must contain a column named 'x_mass'.")
        else:
            MegaSum = st.session_state.get("MegaSum", None)
            x_mass = st.session_state.get("x_mass", None)
            if MegaSum is None or x_mass is None:
                st.error("❌ No data found in session_state. Please upload a CSV file or run previous sections.")
                st.stop()
            else:
                st.success(f"✅ Data loaded from session ({len(x_mass)} data points)")
    
    with col2:
        if st.session_state.get("MegaSum") is not None:
            MegaSum = st.session_state["MegaSum"]
            x_mass = st.session_state["x_mass"]
            
            signal_choice = st.radio(
                "Select Signal",
                options=["With IR", "Without IR"],
                key="signal_choice"
            )
            
            if signal_choice == "With IR":
                if "baseline_corrected_signal_withIR" in MegaSum.columns:
                    signal = MegaSum["baseline_corrected_signal_withIR"].values
                    signal_label = "Baseline Corrected Signal With IR"
                else:
                    st.warning("Falling back to 'without IR'")
                    signal = MegaSum["baseline_corrected_signal_withoutIR"].values
                    signal_label = "Baseline Corrected Signal Without IR"
            else:
                if "baseline_corrected_signal_withoutIR" in MegaSum.columns:
                    signal = MegaSum["baseline_corrected_signal_withoutIR"].values
                    signal_label = "Baseline Corrected Signal Without IR"
                else:
                    st.warning("Falling back to 'with IR'")
                    signal = MegaSum["baseline_corrected_signal_withIR"].values
                    signal_label = "Baseline Corrected Signal With IR"
            
            st.session_state["signal"] = signal
            st.session_state["signal_label"] = signal_label
            st.info(f"📊 Signal: {signal_label}")

#####################################
# STEP 1: PEAK DETECTION
#####################################
with st.expander("🔍 Step 1: Peak Detection Parameters", expanded=False):
    st.markdown("### Detection Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.get("signal") is not None:
            max_intensity = float(np.max(st.session_state["signal"]))
            default_prominence = max_intensity * 0.05
        else:
            max_intensity = 1.0
            default_prominence = 0.05
        
        min_prominence = st.number_input(
            "Minimum peak prominence",
            value=st.session_state.get("min_prominence", default_prominence),
            min_value=0.0,
            max_value=max_intensity,
            step=0.01,
            key="min_prominence",
            help="How much a peak must stand out from surrounding baseline (higher = more selective)"
        )
    
    with col2:
        min_distance = st.number_input(
            "Minimum distance between peaks",
            value=st.session_state.get("min_distance", 3),
            step=1,
            key="min_distance",
            help="Minimum separation in index units"
        )
    
    with col3:
        formula_tol = st.number_input(
            "Formula matching tolerance (Da)",
            value=st.session_state.get("formula_tol", 1.00),
            step=0.01,
            key="formula_tol",
            help="Mass tolerance for candidate matching"
        )
    
    st.divider()
    
    if st.button("🚀 Run Peak Detection", use_container_width=True, type="primary"):
        if st.session_state.get("signal") is None:
            st.error("❌ No signal loaded. Please complete Step 0 first.")
        else:
            signal = st.session_state["signal"]
            x_mass = st.session_state["x_mass"]
            
            with st.spinner("Detecting peaks..."):
                peaks, properties = find_peaks(
                    signal, 
                    prominence=min_prominence,
                    distance=min_distance
                )
                detected_mz = np.array(x_mass)[peaks]
                
                # Get prominences from scipy output
                peak_prominences = properties.get('prominences', np.array([]))
                
                st.success(f"✅ Detected {len(detected_mz)} peaks!")
                
                peaks_df = pd.DataFrame({
                    "Peak Index": peaks,
                    "m/z": detected_mz,
                    "Intensity": signal[peaks],
                    "Prominence": peak_prominences if len(peak_prominences) > 0 else [np.nan] * len(peaks)
                })
                
                st.dataframe(peaks_df, use_container_width=True)
                
                # Plot
                fig_signal = go.Figure()
                fig_signal.add_trace(go.Scatter(
                    x=x_mass, y=signal, mode="lines",
                    name=st.session_state.get("signal_label", "Signal")
                ))
                fig_signal.add_trace(go.Scatter(
                    x=detected_mz, y=signal[peaks],
                    mode="markers", marker=dict(color="red", size=8),
                    name="Detected Peaks"
                ))
                fig_signal.update_layout(
                    title="Signal with Detected Peaks",
                    xaxis_title="Mass (amu)",
                    yaxis_title="Intensity",
                    height=400
                )
                st.plotly_chart(fig_signal, use_container_width=True)
                
                # Save to session state
                st.session_state["detected_peaks_df"] = peaks_df
                st.session_state["detected_mz"] = detected_mz

#####################################
# STEP 2: CANDIDATE FORMULA MATCHING
#####################################
with st.expander("🧪 Step 2: Candidate Formula Matching Options", expanded=False):
    st.markdown("### Formula Type Selection")
    
    # Define the naphthalene option
    NAPH_CH_OPTION = "Alkylated naphthalene core (C10H8 base + alkyl chains)"
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("**CH-Only Formulas**")
            candidate_options_CH = st.multiselect(
                "Select CH formula types:",
                options=[
                    "Alkane (CnH2n+2)",
                    "Alkene (CnH2n)",
                    "Alkyne (CnH2n-2)",
                    "Cyclic (CnH2n)",
                    "Highly Unsaturated (PAH: CnH2n+2-2u)",
                    NAPH_CH_OPTION,
                ],
                default=["Highly Unsaturated (PAH: CnH2n+2-2u)", "Alkane (CnH2n+2)"],
                key="candidate_options_CH"
            )
    
    with col2:
        with st.container(border=True):
            st.markdown("**CHBr Formulas**")
            candidate_options_CHBr = st.multiselect(
                "Select CHBr formula types:",
                options=[
                    "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)",
                    "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)",
                    "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)",
                    "Highly Unsaturated (PAH: CnH2n+2-2u)",
                    "Polybrominated Alkanes (CnH(2n+2-x)Brx)",
                    "Polybrominated Alkenes (CnH(2n-x)Brx)",
                    "Polybrominated Alkynes (CnH(2n-2-x)Brx)",
                ],
                default=["Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)"],
                key="candidate_options_CHBr"
            )
    
    st.divider()
    st.markdown("### Additional Parameters")
    
    # Parameters for specific formula types
    param_col1, param_col2, param_col3 = st.columns(3)
    
    with param_col1:
        if ("Highly Unsaturated (PAH: CnH2n+2-2u)" in candidate_options_CH or 
            "Highly Unsaturated (PAH: CnH2n+2-2u)" in candidate_options_CHBr):
            min_u = st.number_input(
                "Min degrees of unsaturation (u)",
                value=st.session_state.get("min_u", 1),
                min_value=0,
                step=1,
                key="min_u"
            )
            max_u = st.number_input(
                "Max degrees of unsaturation (u)",
                value=st.session_state.get("max_u", 8),
                min_value=0,
                step=1,
                key="max_u"
            )
        else:
            min_u = max_u = None
    
    with param_col2:
        if NAPH_CH_OPTION in candidate_options_CH:
            min_n_subs = int(st.number_input(
                "Min methyl substituents (n)",
                value=0, min_value=0, max_value=8, step=1,
                key="min_n_subs"
            ))
            max_n_subs = int(st.number_input(
                "Max methyl substituents (n)",
                value=4, min_value=min_n_subs, max_value=8, step=1,
                key="max_n_subs"
            ))
        else:
            min_n_subs = max_n_subs = None
    
    with param_col3:
        if NAPH_CH_OPTION in candidate_options_CH:
            min_x_total = int(st.number_input(
                "Min total alkyl carbons (x)",
                value=max(2, min_n_subs) if min_n_subs else 2,
                min_value=0, max_value=40, step=1,
                key="min_x_total"
            ))
            max_x_total = int(st.number_input(
                "Max total alkyl carbons (x)",
                value=max(6, min_x_total) if min_x_total else 6,
                min_value=max(min_x_total, min_n_subs) if min_n_subs else min_x_total,
                max_value=40, step=1,
                key="max_x_total"
            ))
        else:
            min_x_total = max_x_total = None
        
        max_Br = st.number_input(
            "Max Br atoms (CHBr)",
            value=st.session_state.get("max_Br", 3),
            min_value=1,
            step=1,
            key="max_Br"
        )
    
    # Carbon range
    min_C = 2
    max_C = 30
    
    st.divider()
    
    # Helper functions (same as before)
    def hydrogen_count(candidate_type, n_C, u=None):
        if candidate_type == "Alkane (CnH2n+2)":
            return 2 * n_C + 2
        elif candidate_type == "Alkene (CnH2n)":
            return 2 * n_C
        elif candidate_type == "Alkyne (CnH2n-2)":
            return 2 * n_C - 2
        elif candidate_type == "Cyclic (CnH2n)":
            return 2 * n_C
        elif candidate_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
            if u is None:
                u = min_u
            return 2 * n_C + 2 - 2 * u
        elif candidate_type == NAPH_CH_OPTION:
            return None
        else:
            return None
    
    def candidate_formulas_CH(target_mz, tol, candidate_types):
        candidates = []
        for cand_type in candidate_types:
            if cand_type == NAPH_CH_OPTION and min_n_subs is not None:
                base_c = 10
                base_h = 8
                for n_subs in range(min_n_subs, max_n_subs + 1):
                    for x_total in range(max(n_subs, min_x_total), max_x_total + 1):
                        total_c = base_c + x_total
                        total_h = base_h + 2 * x_total
                        calc_mass = total_c * 12.0000 + total_h * 1.007825
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(
                                f"{cand_type}: C{total_c}H{total_h} (n={n_subs}, x={x_total})"
                            )
            else:
                for n_C in range(min_C, max_C + 1):
                    if cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                        for u in range(int(min_u), int(max_u) + 1):
                            n_H = hydrogen_count(cand_type, n_C, u)
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825
                            if abs(calc_mass - target_mz) <= tol:
                                candidates.append(f"{cand_type}: C{n_C}H{n_H} (u={u})")
                    else:
                        n_H = hydrogen_count(cand_type, n_C)
                        if n_H is not None and n_H < 1:
                            continue
                        if n_H is not None:
                            calc_mass = n_C * 12.0000 + n_H * 1.007825
                            if abs(calc_mass - target_mz) <= tol:
                                candidates.append(f"{cand_type}: C{n_C}H{n_H}")
        return candidates
    
    def candidate_formulas_CHBr(target_mz, tol, candidate_types):
        Br_mass = 78.9183
        candidates = []
        for cand_type in candidate_types:
            if cand_type == "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)":
                for n_C in range(1, max_C + 1):
                    n_H = 2 * n_C + 1
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
            elif cand_type == "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)":
                for n_C in range(2, max_C + 1):
                    n_H = 2 * n_C - 1
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
            elif cand_type == "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)":
                for n_C in range(2, max_C + 1):
                    n_H = 2 * n_C - 3
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
            elif cand_type == "Polybrominated Alkanes (CnH(2n+2-x)Brx)":
                for k in range(2, max_Br + 1):
                    for n_C in range(1, max_C + 1):
                        n_H = 2 * n_C + 2 - k
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
            elif cand_type == "Polybrominated Alkenes (CnH(2n-x)Brx)":
                for k in range(2, max_Br + 1):
                    for n_C in range(2, max_C + 1):
                        n_H = 2 * n_C - k
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
            elif cand_type == "Polybrominated Alkynes (CnH(2n-2-x)Brx)":
                for k in range(2, max_Br + 1):
                    for n_C in range(2, max_C + 1):
                        n_H = 2 * n_C - 2 - k
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
            elif cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                for k in range(1, max_Br + 1):
                    for n_C in range(min_C, max_C + 1):
                        for u in range(int(min_u), int(max_u) + 1):
                            n_H = hydrogen_count(cand_type, n_C, u)
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                            if abs(calc_mass - target_mz) <= tol:
                                candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k} (u={u})")
        return candidates
    
    if st.button("🧬 Run Candidate Formula Matching", use_container_width=True, type="primary"):
        if "detected_mz" not in st.session_state:
            st.error("❌ No detected peaks found. Please run peak detection first.")
        else:
            detected_mz = st.session_state["detected_mz"]
            
            with st.spinner(f"Matching formulas for {len(detected_mz)} peaks..."):
                candidate_data = []
                for mz_val in detected_mz:
                    ch_candidates = candidate_formulas_CH(mz_val, tol=formula_tol, candidate_types=candidate_options_CH)
                    chbr_candidates = candidate_formulas_CHBr(mz_val, tol=formula_tol, candidate_types=candidate_options_CHBr)
                    
                    if ch_candidates or chbr_candidates:
                        candidate_data.append({
                            "m/z": mz_val,
                            "CH candidates": "; ".join(ch_candidates) if ch_candidates else "",
                            "CHBr candidates": "; ".join(chbr_candidates) if chbr_candidates else ""
                        })
                
                if candidate_data:
                    candidates_df = pd.DataFrame(candidate_data)
                    st.success(f"✅ Found candidates for {len(candidates_df)} peaks!")
                    st.dataframe(candidates_df, use_container_width=True)
                    st.session_state["candidates_df"] = candidates_df
                    
                    csv_candidates = candidates_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Candidate Formulas CSV",
                        data=csv_candidates,
                        file_name="candidate_formulas.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No candidates found matching the criteria.")

#####################################
# STEP 3: VISUALIZATION
#####################################
with st.expander("📊 Step 3: Visualization & Plotting", expanded=False):
    st.markdown("### Plot Configuration")
    
    plot_col1, plot_col2, plot_col3 = st.columns(3)
    
    with plot_col1:
        if st.session_state.get("x_mass") is not None:
            x_mass = st.session_state["x_mass"]
            x_min = st.number_input("X-axis min (amu)", value=float(x_mass[0]), key="x_min")
            x_max = st.number_input("X-axis max (amu)", value=float(x_mass[-1]), key="x_max")
        else:
            x_min = st.number_input("X-axis min (amu)", value=0.0, key="x_min")
            x_max = st.number_input("X-axis max (amu)", value=1000.0, key="x_max")
    
    with plot_col2:
        if st.session_state.get('signal') is not None:
            y_max_default = float(np.max(st.session_state.get('signal', [1.0])))
        else:
            y_max_default = 1.0
        y_max = st.number_input("Y-axis max (Intensity)", value=y_max_default, step=1.0, key="y_max")
        
        plot_intensity_threshold = st.number_input(
            "Min intensity for annotations",
            value=st.session_state.get("plot_intensity_threshold", 0.01),
            step=0.01,
            key="plot_intensity_threshold"
        )
    
    with plot_col3:
        annotation_style = st.radio(
            "Annotation Style",
            options=["Mass Value", "Numeric Label"],
            key="annotation_style"
        )
    
    st.markdown("### Optional: Add Custom Peaks")
    custom_peaks_input = st.text_area(
        "Enter custom peaks (format: m/z, intensity, annotation)",
        value="",
        height=100,
        key="custom_peaks_input",
        help="One peak per line, e.g., 150.5, 0.8, MyPeak"
    )
    
    custom_peaks = []
    if custom_peaks_input.strip():
        for line in custom_peaks_input.splitlines():
            parts = line.split(',')
            try:
                mz_val = float(parts[0].strip())
                intensity_val = float(parts[1].strip()) if len(parts) >= 2 else None
                annotation = parts[2].strip() if len(parts) >= 3 else "Custom"
                if intensity_val is not None:
                    custom_peaks.append({"m/z": mz_val, "Intensity": intensity_val, "Annotation": annotation})
            except:
                pass
    
    st.divider()
    
    # Tabs for different plot types
    tab1, tab2 = st.tabs(["📈 Static Plot with Annotations", "🔄 Interactive Plot"])
    
    with tab1:
        if st.button("Generate Static Plot", use_container_width=True, key="static_plot_btn"):
            if ("detected_peaks_df" not in st.session_state or
                "candidates_df" not in st.session_state or
                "signal" not in st.session_state):
                st.error("❌ Please run both peak detection and candidate matching steps first.")
            else:
                peaks_df = st.session_state["detected_peaks_df"]
                candidates_df = st.session_state["candidates_df"]
                signal = st.session_state["signal"]
                x_mass = st.session_state["x_mass"]
                
                matched_peaks = peaks_df[
                    peaks_df["m/z"].isin(candidates_df["m/z"]) &
                    (peaks_df["Intensity"] >= plot_intensity_threshold)
                ]
                
                if custom_peaks:
                    custom_df = pd.DataFrame(custom_peaks)
                    matched_peaks = pd.concat([matched_peaks, custom_df], ignore_index=True)
                
                matched_peaks = matched_peaks[(matched_peaks["m/z"] >= x_min) & (matched_peaks["m/z"] <= x_max)]
                
                fig, (ax_spectrum, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(14, 6))
                
                ax_spectrum.plot(x_mass, signal, label=st.session_state.get("signal_label", "Signal"), color="blue")
                ax_spectrum.scatter(matched_peaks["m/z"], matched_peaks["Intensity"], color="red", zorder=5)
                ax_spectrum.set_xlabel("Mass (amu)")
                ax_spectrum.set_ylabel("Intensity")
                ax_spectrum.set_title("Spectrum with Detected Peaks")
                ax_spectrum.set_xlim(x_min, x_max)
                ax_spectrum.set_ylim(0, y_max)
                
                offset = 0.05 * y_max
                legend_entries = []
                
                for i, (_, row) in enumerate(matched_peaks.iterrows(), start=1):
                    mz_val = row["m/z"]
                    intensity_val = row["Intensity"]
                    cand_row = candidates_df[candidates_df["m/z"] == mz_val]
                    
                    if not cand_row.empty:
                        ch_cand = cand_row["CH candidates"].values[0]
                        chbr_cand = cand_row["CHBr candidates"].values[0]
                        details = ""
                        if ch_cand:
                            details += f"CH: {ch_cand}"
                        if chbr_cand:
                            details += f"\nCHBr: {chbr_cand}"
                    else:
                        details = row.get("Annotation", "Custom")
                    
                    if annotation_style == "Numeric Label":
                        annotation_text = str(i)
                        legend_entry = f"Peak {i}: m/z = {mz_val:.4f}, intensity = {intensity_val:.2f}\n{details}"
                    else:
                        annotation_text = f"{mz_val:.2f}"
                        legend_entry = f"m/z = {mz_val:.2f}, intensity = {intensity_val:.2f}\n{details}"
                    
                    ax_spectrum.annotate(
                        annotation_text,
                        xy=(mz_val, intensity_val),
                        xytext=(mz_val, intensity_val + offset),
                        ha='center',
                        fontsize=10,
                        fontweight='bold',
                        color="black",
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
                    )
                    legend_entries.append(legend_entry)
                
                ax_legend.axis('off')
                legend_text = "\n\n".join(legend_entries)
                ax_legend.text(0, 1, legend_text, va='top', ha='left', fontsize=10, wrap=True)
                ax_legend.set_title("Peak Assignments", fontsize=12)
                
                plt.tight_layout()
                
                directory = st.session_state.get("file_directory", "./output")
                filename = f"{directory}/spectrum_{x_min:.2f}_{x_max:.2f}_{y_max:.2f}_{annotation_style.replace(' ', '_')}.png"
                plt.savefig(filename, dpi=300)
                st.success(f"✅ Figure saved as '{filename}'")
                
                st.pyplot(fig)
                
                # Add to Report button
                add_plot_to_report_button(
                    fig,
                    f"Peak Detection - {x_min:.0f}-{x_max:.0f} amu",
                    key_suffix="peak_detection",
                    description=f"Peak detection spectrum from {x_min:.0f} to {x_max:.0f} amu"
                )
    
    with tab2:
        if st.button("Generate Interactive Plot", use_container_width=True, key="interactive_plot_btn"):
            if ("detected_peaks_df" not in st.session_state or
                "candidates_df" not in st.session_state or
                "signal" not in st.session_state):
                st.error("❌ Please run peak detection and candidate matching steps first.")
            else:
                peaks_df = st.session_state["detected_peaks_df"]
                candidates_df = st.session_state["candidates_df"]
                signal = st.session_state["signal"]
                x_mass = st.session_state["x_mass"]
                
                matched_peaks = peaks_df[peaks_df["m/z"].isin(candidates_df["m/z"])]
                
                hover_text = []
                for _, row in matched_peaks.iterrows():
                    mz_val = row["m/z"]
                    intensity_val = row["Intensity"]
                    cand_row = candidates_df[candidates_df["m/z"] == mz_val]
                    
                    if not cand_row.empty:
                        details = ""
                        if cand_row.iloc[0]["CH candidates"]:
                            details += f"CH: {cand_row.iloc[0]['CH candidates']}"
                        if cand_row.iloc[0]["CHBr candidates"]:
                            details += f"<br>CHBr: {cand_row.iloc[0]['CHBr candidates']}"
                    else:
                        details = "No candidate details"
                    hover_text.append(f"m/z: {mz_val:.4f}<br>Intensity: {intensity_val:.2f}<br>{details}")
                
                fig_int = go.Figure()
                
                fig_int.add_trace(go.Scatter(
                    x=x_mass,
                    y=signal,
                    mode='lines',
                    name=st.session_state.get("signal_label", "Signal"),
                    line=dict(color='blue')
                ))
                
                fig_int.add_trace(go.Scatter(
                    x=matched_peaks["m/z"],
                    y=matched_peaks["Intensity"],
                    mode='markers',
                    name='Detected Peaks with Candidates',
                    marker=dict(color='red', size=10),
                    text=hover_text,
                    hoverinfo='text'
                ))
                
                fig_int.update_layout(
                    title="Interactive Spectrum with Candidate Annotations",
                    xaxis_title="Mass (amu)",
                    yaxis_title="Intensity",
                    height=600,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_int, use_container_width=True)

st.divider()
st.markdown("### 📌 Workflow Complete")
show_progress()
