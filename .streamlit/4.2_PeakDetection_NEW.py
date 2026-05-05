import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
import re
from packages.ReportManager import add_plot_to_report_button, init_report_session
from packages.load_dataset import ensure_dataset_loaded

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
                st.warning("No MegaSum in session. Loading from .pkl.gz file...")
                ensure_dataset_loaded(
                    require_keys=["x_mass", "compilation_baseline_corrected_data", "unique_wavenumbers"],
                    compute_megasum=True,
                    page_key_prefix="_peakdet",
                )
                # After loading, refresh local refs
                MegaSum = st.session_state.get("MegaSum")
                x_mass = st.session_state.get("x_mass")
            if MegaSum is not None and x_mass is not None:
                st.success(f"✅ Data loaded from session ({len(x_mass)} data points)")
            else:
                st.error("❌ Could not load data.")
                st.stop()
    
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
            value=st.session_state.get("formula_tol", 0.50),
            step=0.01,
            key="formula_tol",
            help="Mass tolerance for candidate matching (0.5 Da recommended for unit-mass resolution)"
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
    ETHYNYL_OPTION = "Ethynyl-bearing species (core + m×C≡CH)"
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown("**CH-Only Formulas**")
            candidate_options_CH = st.multiselect(
                "Select CH formula types:",
                options=[
                    "Alkane (CnH2n+2)",
                    "Alkene/Cyclic (CnH2n)",
                    "Alkyne (CnH2n-2)",
                    "Highly Unsaturated (PAH: CnH2n+2-2u)",
                    NAPH_CH_OPTION,
                    ETHYNYL_OPTION,
                ],
                default=["Highly Unsaturated (PAH: CnH2n+2-2u)", "Alkane (CnH2n+2)"],
                key="candidate_options_CH"
            )
            st.caption("ℹ️ Alkene & Cyclic merged (same mass formula CₙH₂ₙ)")
    
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
                default=[],
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
    
    # Ethynyl-bearing species parameters
    if ETHYNYL_OPTION in candidate_options_CH:
        st.markdown("**Ethynyl-bearing species parameters**")
        eth_col1, eth_col2, eth_col3, eth_col4 = st.columns(4)
        with eth_col1:
            min_ethynyl = st.number_input(
                "Min ethynyl groups (m)", value=1, min_value=1, max_value=6, step=1,
                key="min_ethynyl",
            )
        with eth_col2:
            max_ethynyl = st.number_input(
                "Max ethynyl groups (m)", value=3, min_value=int(min_ethynyl), max_value=6, step=1,
                key="max_ethynyl",
            )
        with eth_col3:
            min_core_u = st.number_input(
                "Min core DoU (u_core)", value=0, min_value=0, max_value=15, step=1,
                key="min_core_u",
                help="Degrees of unsaturation of the hydrocarbon core (0=alkane, 4=benzene, 7=naphthalene)",
            )
        with eth_col4:
            max_core_u = st.number_input(
                "Max core DoU (u_core)", value=8, min_value=int(min_core_u), max_value=15, step=1,
                key="max_core_u",
            )
    else:
        min_ethynyl = max_ethynyl = min_core_u = max_core_u = None

    param_col_br = st.columns(1)[0]
    with param_col_br:
        max_Br = st.number_input(
            "Max Br atoms (CHBr)",
            value=st.session_state.get("max_Br", 3),
            min_value=1,
            step=1,
            key="max_Br"
        )
    
    st.divider()
    st.markdown("### Carbon Range & Validation Filters")
    
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    with filt_col1:
        min_C = int(st.number_input("Min carbons", value=2, min_value=1, max_value=50, step=1, key="min_C"))
        max_C = int(st.number_input("Max carbons", value=30, min_value=min_C, max_value=100, step=1, key="max_C"))
    with filt_col2:
        hc_min = st.number_input("Min H/C ratio", value=0.3, min_value=0.0, max_value=5.0, step=0.1, key="hc_min",
                                  help="Reject candidates with H/C below this (typical hydrocarbons: 0.5–2.5)")
        hc_max = st.number_input("Max H/C ratio", value=2.5, min_value=hc_min, max_value=5.0, step=0.1, key="hc_max")
    with filt_col3:
        rdb_max = st.number_input("Max RDB (Ring+Double Bond)", value=15, min_value=0, max_value=50, step=1, key="rdb_max",
                                   help="Flag/reject formulas with RDB above this value")
        check_br_isotope = st.checkbox("Check ⁸¹Br isotopologue (+2 Da)", value=True, key="check_br_isotope",
                                        help="Also match peaks that could be ⁸¹Br isotopologues")
        require_br_pair = st.checkbox("Require Br isotope pair", value=True, key="require_br_pair",
                                       help="Reject CHBr candidates unless both ⁷⁹Br and ⁸¹Br peaks "
                                            "(±2 Da per Br) are detected. Br has ~50/50 abundance, "
                                            "so a genuine brominated species must show both peaks.")
    with filt_col4:
        rank_by_error = st.checkbox("Rank by mass error", value=True, key="rank_by_error",
                                     help="Sort candidates by closeness to detected m/z")
        show_mass_error = st.checkbox("Show mass error (Da)", value=True, key="show_mass_error")
    
    st.divider()
    
    # ==========================================
    # IMPROVED FORMULA MATCHING ENGINE
    # ==========================================
    Br79_mass = 78.9183
    Br81_mass = 80.9163

    def calc_rdb(n_C, n_H, n_Br=0):
        """Ring + Double Bond equivalents: (2C + 2 - H - X) / 2"""
        return (2 * n_C + 2 - n_H - n_Br) / 2.0

    def is_valid_candidate(n_C, n_H, n_Br=0):
        """Check chemical validity: RDB >= 0, H/C in range, RDB <= max."""
        if n_H < 1:
            return False, "H<1"
        rdb = calc_rdb(n_C, n_H, n_Br)
        if rdb < 0:
            return False, "RDB<0"
        if rdb > rdb_max:
            return False, f"RDB={rdb:.1f}>{rdb_max}"
        hc_ratio = n_H / n_C if n_C > 0 else 0
        if hc_ratio < hc_min or hc_ratio > hc_max:
            return False, f"H/C={hc_ratio:.2f} out of range"
        return True, ""

    def make_candidate_entry(formula_str, calc_mass, target_mz, n_C, n_H, n_Br=0, cand_type="", extra=""):
        """Create a standardized candidate dict with mass error and RDB."""
        rdb = calc_rdb(n_C, n_H, n_Br)
        mass_err = calc_mass - target_mz
        return {
            "formula": formula_str,
            "type": cand_type,
            "calc_mass": calc_mass,
            "mass_error_Da": mass_err,
            "RDB": rdb,
            "H/C": n_H / n_C if n_C > 0 else 0,
            "n_C": n_C,
            "n_H": n_H,
            "n_Br": n_Br,
            "extra": extra,
        }

    def generate_candidates_CH(target_mz, tol, candidate_types):
        """Generate CH-only candidates with validation."""
        results = []
        for cand_type in candidate_types:
            if cand_type == NAPH_CH_OPTION and min_n_subs is not None:
                # Alkylated naphthalene: C10H8 base
                # For n substituents: lose n ring-H, gain sum of (2*chain_i + 1) chain-H
                # Simplification: n methyl groups of total x carbons
                # total_C = 10 + x_total, total_H = 8 - n_subs + n_subs*(2*1+1) for methyls
                # More general: H = 8 - n_subs + (2*x_total + n_subs) = 8 + 2*x_total
                # CORRECTED: for n_subs substituents with total x_total carbons:
                # Each substitution: -1 ring H, chain contributes (2*chain_len + 1) H
                # If all chains are methyl (chain_len=1): H_chain = n_subs * 3
                # General: H = 8 - n_subs + (2*x_total + n_subs) = 8 + 2*x_total
                # This is actually correct for linear alkyl chains summing to x_total carbons
                base_c = 10
                base_h = 8
                for n_subs in range(min_n_subs, max_n_subs + 1):
                    for x_total in range(max(n_subs, min_x_total), max_x_total + 1):
                        total_c = base_c + x_total
                        # Corrected H count: base_h - n_subs (ring H lost) + (2*x_total + n_subs) (alkyl H)
                        total_h = base_h - n_subs + (2 * x_total + n_subs)  # = base_h + 2*x_total
                        calc_mass = total_c * 12.0000 + total_h * 1.007825
                        if abs(calc_mass - target_mz) <= tol:
                            valid, reason = is_valid_candidate(total_c, total_h)
                            if valid:
                                results.append(make_candidate_entry(
                                    f"C{total_c}H{total_h}",
                                    calc_mass, target_mz, total_c, total_h,
                                    cand_type=cand_type,
                                    extra=f"n={n_subs}, x={x_total}"
                                ))

            elif cand_type == ETHYNYL_OPTION and min_ethynyl is not None:
                for m in range(min_ethynyl, max_ethynyl + 1):
                    for u_core in range(min_core_u, max_core_u + 1):
                        for n_core in range(min_C, max_C + 1):
                            total_c = n_core + 2 * m
                            if total_c > max_C:
                                break
                            # Core H + ethynyl terminal H (each C≡CH contributes 1 H)
                            core_h = 2 * n_core + 2 - 2 * u_core
                            # Each ethynyl replaces 1 H on core and adds terminal H → net: -1 + 1 = 0
                            # But actually ethynyl = -C≡CH: removes 1 H from core, adds 1 terminal H
                            total_h = core_h - m + m  # net zero change... simplify:
                            total_h = core_h  # ethynyl substitution: -1 core H + 1 terminal H = net 0
                            if total_h < 1:
                                continue
                            calc_mass = total_c * 12.0000 + total_h * 1.007825
                            if abs(calc_mass - target_mz) <= tol:
                                valid, reason = is_valid_candidate(total_c, total_h)
                                if valid:
                                    results.append(make_candidate_entry(
                                        f"C{total_c}H{total_h}",
                                        calc_mass, target_mz, total_c, total_h,
                                        cand_type=cand_type,
                                        extra=f"core=C{n_core}, u_core={u_core}, {m}×C≡CH"
                                    ))

            elif cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                for n_C in range(min_C, max_C + 1):
                    for u in range(int(min_u), int(max_u) + 1):
                        n_H = 2 * n_C + 2 - 2 * u
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825
                        if abs(calc_mass - target_mz) <= tol:
                            valid, reason = is_valid_candidate(n_C, n_H)
                            if valid:
                                results.append(make_candidate_entry(
                                    f"C{n_C}H{n_H}",
                                    calc_mass, target_mz, n_C, n_H,
                                    cand_type=cand_type,
                                    extra=f"u={u}"
                                ))

            elif cand_type == "Alkane (CnH2n+2)":
                for n_C in range(min_C, max_C + 1):
                    n_H = 2 * n_C + 2
                    calc_mass = n_C * 12.0000 + n_H * 1.007825
                    if abs(calc_mass - target_mz) <= tol:
                        valid, reason = is_valid_candidate(n_C, n_H)
                        if valid:
                            results.append(make_candidate_entry(
                                f"C{n_C}H{n_H}", calc_mass, target_mz, n_C, n_H,
                                cand_type=cand_type
                            ))

            elif cand_type == "Alkene/Cyclic (CnH2n)":
                for n_C in range(min_C, max_C + 1):
                    n_H = 2 * n_C
                    calc_mass = n_C * 12.0000 + n_H * 1.007825
                    if abs(calc_mass - target_mz) <= tol:
                        valid, reason = is_valid_candidate(n_C, n_H)
                        if valid:
                            results.append(make_candidate_entry(
                                f"C{n_C}H{n_H}", calc_mass, target_mz, n_C, n_H,
                                cand_type=cand_type
                            ))

            elif cand_type == "Alkyne (CnH2n-2)":
                for n_C in range(min_C, max_C + 1):
                    n_H = 2 * n_C - 2
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825
                    if abs(calc_mass - target_mz) <= tol:
                        valid, reason = is_valid_candidate(n_C, n_H)
                        if valid:
                            results.append(make_candidate_entry(
                                f"C{n_C}H{n_H}", calc_mass, target_mz, n_C, n_H,
                                cand_type=cand_type
                            ))

        return results

    def generate_candidates_CHBr(target_mz, tol, candidate_types):
        """Generate CHBr candidates checking both ⁷⁹Br and ⁸¹Br isotopologues."""
        results = []

        # Determine which Br masses to check
        br_masses = [(Br79_mass, "⁷⁹Br")]
        if check_br_isotope:
            br_masses.append((Br81_mass, "⁸¹Br"))

        for cand_type in candidate_types:
            for br_m, br_label in br_masses:
                if cand_type == "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)":
                    for n_C in range(1, max_C + 1):
                        n_H = 2 * n_C + 1
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + br_m
                        if abs(calc_mass - target_mz) <= tol:
                            valid, reason = is_valid_candidate(n_C, n_H, 1)
                            if valid:
                                results.append(make_candidate_entry(
                                    f"C{n_C}H{n_H}Br", calc_mass, target_mz, n_C, n_H, 1,
                                    cand_type=cand_type, extra=br_label
                                ))

                elif cand_type == "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)":
                    for n_C in range(2, max_C + 1):
                        n_H = 2 * n_C - 1
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + br_m
                        if abs(calc_mass - target_mz) <= tol:
                            valid, reason = is_valid_candidate(n_C, n_H, 1)
                            if valid:
                                results.append(make_candidate_entry(
                                    f"C{n_C}H{n_H}Br", calc_mass, target_mz, n_C, n_H, 1,
                                    cand_type=cand_type, extra=br_label
                                ))

                elif cand_type == "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)":
                    for n_C in range(2, max_C + 1):
                        n_H = 2 * n_C - 3
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + br_m
                        if abs(calc_mass - target_mz) <= tol:
                            valid, reason = is_valid_candidate(n_C, n_H, 1)
                            if valid:
                                results.append(make_candidate_entry(
                                    f"C{n_C}H{n_H}Br", calc_mass, target_mz, n_C, n_H, 1,
                                    cand_type=cand_type, extra=br_label
                                ))

                elif cand_type == "Polybrominated Alkanes (CnH(2n+2-x)Brx)":
                    for k in range(2, max_Br + 1):
                        for n_C in range(1, max_C + 1):
                            n_H = 2 * n_C + 2 - k
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * br_m
                            if abs(calc_mass - target_mz) <= tol:
                                valid, reason = is_valid_candidate(n_C, n_H, k)
                                if valid:
                                    results.append(make_candidate_entry(
                                        f"C{n_C}H{n_H}Br{k}", calc_mass, target_mz, n_C, n_H, k,
                                        cand_type=cand_type, extra=f"Br{k} {br_label}"
                                    ))

                elif cand_type == "Polybrominated Alkenes (CnH(2n-x)Brx)":
                    for k in range(2, max_Br + 1):
                        for n_C in range(2, max_C + 1):
                            n_H = 2 * n_C - k
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * br_m
                            if abs(calc_mass - target_mz) <= tol:
                                valid, reason = is_valid_candidate(n_C, n_H, k)
                                if valid:
                                    results.append(make_candidate_entry(
                                        f"C{n_C}H{n_H}Br{k}", calc_mass, target_mz, n_C, n_H, k,
                                        cand_type=cand_type, extra=f"Br{k} {br_label}"
                                    ))

                elif cand_type == "Polybrominated Alkynes (CnH(2n-2-x)Brx)":
                    for k in range(2, max_Br + 1):
                        for n_C in range(2, max_C + 1):
                            n_H = 2 * n_C - 2 - k
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * br_m
                            if abs(calc_mass - target_mz) <= tol:
                                valid, reason = is_valid_candidate(n_C, n_H, k)
                                if valid:
                                    results.append(make_candidate_entry(
                                        f"C{n_C}H{n_H}Br{k}", calc_mass, target_mz, n_C, n_H, k,
                                        cand_type=cand_type, extra=f"Br{k} {br_label}"
                                    ))

                elif cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                    for k in range(1, max_Br + 1):
                        for n_C in range(min_C, max_C + 1):
                            for u in range(int(min_u), int(max_u) + 1):
                                n_H = 2 * n_C + 2 - 2 * u - k
                                if n_H < 1:
                                    continue
                                calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * br_m
                                if abs(calc_mass - target_mz) <= tol:
                                    valid, reason = is_valid_candidate(n_C, n_H, k)
                                    if valid:
                                        results.append(make_candidate_entry(
                                            f"C{n_C}H{n_H}Br{k}", calc_mass, target_mz, n_C, n_H, k,
                                            cand_type=cand_type, extra=f"u={u}, Br{k} {br_label}"
                                        ))
        return results

    def deduplicate_candidates(candidates):
        """Remove duplicate formulas, keeping the one with smallest mass error."""
        seen = {}
        for c in candidates:
            key = c["formula"]
            if key not in seen or abs(c["mass_error_Da"]) < abs(seen[key]["mass_error_Da"]):
                seen[key] = c
        return list(seen.values())

    if st.button("🧬 Run Candidate Formula Matching", use_container_width=True, type="primary"):
        if "detected_mz" not in st.session_state:
            st.error("❌ No detected peaks found. Please run peak detection first.")
        else:
            detected_mz = st.session_state["detected_mz"]
            
            with st.spinner(f"Matching formulas for {len(detected_mz)} peaks..."):
                all_rows = []
                for mz_val in detected_mz:
                    ch_raw = generate_candidates_CH(mz_val, tol=formula_tol, candidate_types=candidate_options_CH)
                    chbr_raw = generate_candidates_CHBr(mz_val, tol=formula_tol, candidate_types=candidate_options_CHBr)

                    # Deduplicate each group
                    ch_cands = deduplicate_candidates(ch_raw)
                    chbr_cands = deduplicate_candidates(chbr_raw)

                    # Br isotope pair validation: for each CHBr candidate,
                    # check that the complementary isotopologue peak exists
                    # in the detected peak list (±2*n_Br Da).
                    if require_br_pair and chbr_cands:
                        _validated_chbr = []
                        _detected_set = np.array(detected_mz)
                        for c in chbr_cands:
                            n_br = c["n_Br"]
                            _shift = 2.0 * n_br  # mass diff between all-⁷⁹Br and all-⁸¹Br
                            # Check: does a detected peak exist at mz ± shift?
                            _has_partner = False
                            for _delta in [+_shift, -_shift]:
                                _partner_mz = mz_val + _delta
                                if np.any(np.abs(_detected_set - _partner_mz) <= formula_tol):
                                    _has_partner = True
                                    break
                            if _has_partner:
                                _validated_chbr.append(c)
                            else:
                                c["extra"] = (c["extra"] + ", " if c["extra"] else "") + "⚠️ no isotope pair"
                                _validated_chbr.append(c)  # keep but flag
                        # Separate validated from flagged
                        _clean = [c for c in _validated_chbr if "⚠️ no isotope pair" not in c.get("extra", "")]
                        _flagged = [c for c in _validated_chbr if "⚠️ no isotope pair" in c.get("extra", "")]
                        chbr_cands = _clean  # only keep validated ones
                        if _flagged:
                            # Store flagged for optional display
                            chbr_cands_flagged = _flagged
                        else:
                            chbr_cands_flagged = []
                    else:
                        chbr_cands_flagged = []

                    # Rank by mass error if requested
                    if rank_by_error:
                        ch_cands.sort(key=lambda x: abs(x["mass_error_Da"]))
                        chbr_cands.sort(key=lambda x: abs(x["mass_error_Da"]))

                    # Format for display
                    def _fmt(c):
                        s = f"{c['formula']}"
                        if c["extra"]:
                            s += f" ({c['extra']})"
                        if show_mass_error:
                            s += f" [Δ={c['mass_error_Da']:+.3f}Da, RDB={c['RDB']:.1f}]"
                        return s

                    ch_str = "; ".join(_fmt(c) for c in ch_cands)
                    chbr_str = "; ".join(_fmt(c) for c in chbr_cands)
                    chbr_flagged_str = "; ".join(_fmt(c) for c in chbr_cands_flagged) if chbr_cands_flagged else ""

                    if ch_cands or chbr_cands:
                        row = {"m/z": mz_val}
                        if show_mass_error:
                            # Best match info
                            best = min(ch_cands + chbr_cands, key=lambda x: abs(x["mass_error_Da"]))
                            row["Best Match"] = best["formula"]
                            row["Error (Da)"] = f"{best['mass_error_Da']:+.4f}"
                            row["RDB"] = f"{best['RDB']:.1f}"
                            row["H/C"] = f"{best['H/C']:.2f}"
                        row["CH candidates"] = ch_str
                        row["CHBr candidates"] = chbr_str
                        if chbr_flagged_str:
                            row["CHBr rejected (no isotope pair)"] = chbr_flagged_str
                        row["# candidates"] = len(ch_cands) + len(chbr_cands)
                        all_rows.append(row)
                
                if all_rows:
                    candidates_df = pd.DataFrame(all_rows)
                    st.success(f"✅ Found candidates for {len(candidates_df)} peaks "
                               f"({candidates_df['# candidates'].sum()} total formulas)")
                    st.dataframe(candidates_df, use_container_width=True)
                    st.session_state["candidates_df"] = candidates_df
                    
                    # Build metadata header with search parameters
                    _meta_lines = []
                    _meta_lines.append(f"# Peak Detection & Formula Matching Results")
                    _meta_lines.append(f"# Signal: {st.session_state.get('signal_label', 'N/A')}")
                    _meta_lines.append(f"# Peak prominence: {min_prominence}")
                    _meta_lines.append(f"# Peak distance: {min_distance}")
                    _meta_lines.append(f"# Formula tolerance: {formula_tol} Da")
                    _meta_lines.append(f"# CH types: {', '.join(candidate_options_CH) if candidate_options_CH else 'None'}")
                    _meta_lines.append(f"# CHBr types: {', '.join(candidate_options_CHBr) if candidate_options_CHBr else 'None'}")
                    if min_u is not None:
                        _meta_lines.append(f"# Unsaturation range: u={min_u}-{max_u}")
                    _meta_lines.append(f"# Max Br: {max_Br}")
                    _meta_lines.append(f"# C range: {min_C}-{max_C}")
                    _meta_lines.append(f"# H/C filter: {hc_min}-{hc_max}")
                    _meta_lines.append(f"# Max RDB: {rdb_max}")
                    _meta_lines.append(f"# Br isotope check: {check_br_isotope}")
                    _meta_lines.append(f"# Peaks with candidates: {len(candidates_df)}")
                    _meta_lines.append(f"# Total candidate formulas: {candidates_df['# candidates'].sum()}")
                    _meta_lines.append(f"#")

                    _header = "\n".join(_meta_lines) + "\n"
                    csv_candidates = (_header + candidates_df.to_csv(index=False)).encode("utf-8")

                    # Informative filename
                    _fname = f"candidate_formulas_prom{min_prominence:.3f}_tol{formula_tol:.2f}_dist{min_distance}.csv"

                    st.download_button(
                        label="📥 Download Candidate Formulas CSV",
                        data=csv_candidates,
                        file_name=_fname,
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
