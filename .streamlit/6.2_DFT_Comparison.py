import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime
import os
import io

from packages.ReportManager import add_plot_to_report_button, init_report_session
from packages.DFT_Parsers import parse_dft_file, broaden_spectrum_felix
from packages.PCC_Scoring import (
    DEFAULT_DIAGNOSTIC_REGIONS,
    DEFAULT_PCC_THRESHOLDS,
    AVAILABLE_METRICS,
    compute_pcc,
    compute_similarity,
    preprocess_spectrum,
    score_label,
    compute_batch_pcc,
    rank_batch_results,
    find_optimal_scaling_factor,
)

init_report_session()

st.title("🔬 DFT Spectrum Comparison & PCC Scoring")
st.caption("Compare experimental IR-UV ion-dip spectra with DFT calculations using region-specific Pearson Correlation.")

# ========================================================================================
# THIN UI HELPERS (read from session state, delegate to package functions)
# ========================================================================================

def get_diagnostic_regions():
    """Get diagnostic regions from session state or use defaults"""
    if 'custom_regions_enabled' in st.session_state and st.session_state['custom_regions_enabled']:
        return st.session_state.get('diagnostic_regions', DEFAULT_DIAGNOSTIC_REGIONS)
    return DEFAULT_DIAGNOSTIC_REGIONS

def get_pcc_thresholds():
    """Get PCC thresholds from session state or use defaults"""
    return {
        "excellent": st.session_state.get("pcc_threshold_excellent", DEFAULT_PCC_THRESHOLDS["excellent"]),
        "good": st.session_state.get("pcc_threshold_good", DEFAULT_PCC_THRESHOLDS["good"]),
        "weak": st.session_state.get("pcc_threshold_weak", DEFAULT_PCC_THRESHOLDS["weak"]),
    }

def ui_score_label(r):
    """score_label wrapper that reads thresholds from session state"""
    return score_label(r, thresholds=get_pcc_thresholds())

# Main UI
st.markdown("---")
st.markdown("## Step 1 — Upload DFT Output Files")

# Initialize session state for multiple structures
if 'dft_structures' not in st.session_state:
    st.session_state['dft_structures'] = []

uploaded_files = st.file_uploader(
    "Upload DFT calculation outputs (single or multiple for batch comparison)",
    type=['out', 'dat', 'log', 'txt'],
    accept_multiple_files=True,
    help="Supported formats: Gaussian .out/.log, ORCA .out/.dat, custom parsed reports. Upload multiple files to compare different candidate structures."
)

if uploaded_files:
    try:
        structures = []
        
        with st.spinner(f"Parsing {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode('utf-8', errors='ignore')
                frequencies, intensities, metadata = parse_dft_file(content, uploaded_file.name)
                
                if len(frequencies) == 0:
                    st.warning(f"⚠️ No IR spectrum data found in {uploaded_file.name}")
                    continue
                
                structures.append({
                    'filename': uploaded_file.name,
                    'frequencies': frequencies,
                    'intensities': intensities,
                    'metadata': metadata
                })
        
        if structures:
            st.success(f"✅ Successfully parsed {len(structures)} structure(s)")
            
            # Store all structures in session state
            st.session_state['dft_structures'] = structures
            
            # Display summary
            st.markdown("### 📋 Loaded Structures")
            summary_data = []
            for i, struct in enumerate(structures):
                row = {
                    '#': i + 1,
                    'File': struct['filename'],
                    'Modes': len(struct['frequencies']),
                    'Freq Range (cm⁻¹)': f"{struct['frequencies'].min():.1f} - {struct['frequencies'].max():.1f}"
                }
                if struct['metadata'].get('type') == 'anharmonic':
                    nf = struct['metadata'].get('n_fundamentals', 0)
                    no = struct['metadata'].get('n_overtones', 0)
                    nc = struct['metadata'].get('n_combinations', 0)
                    row['Breakdown'] = f"{nf} fund + {no} over + {nc} comb"
                summary_data.append(row)
            st.dataframe(pd.DataFrame(summary_data), width='stretch', hide_index=True)
            
            # Select active structure for detailed view
            if len(structures) > 1:
                st.markdown("### 🔍 Select Structure for Detailed View")
                struct_idx = st.selectbox(
                    "Choose structure:",
                    options=range(len(structures)),
                    format_func=lambda x: structures[x]['filename']
                )
            else:
                struct_idx = 0
            
            selected_struct = structures[struct_idx]
            
            # Display metadata if available
            if selected_struct['metadata']:
                st.markdown(f"#### 📋 {selected_struct['filename']} - Calculation Details")
                cols = st.columns(3)
                idx = 0
                for key, value in selected_struct['metadata'].items():
                    if key == 'band_types':  # skip large list
                        continue
                    with cols[idx % 3]:
                        st.metric(key.replace('_', ' ').title(), value)
                    idx += 1
            
            # Display raw stick spectrum data
            with st.expander(f"📊 View {selected_struct['filename']} Raw Spectrum Data"):
                spec_dict = {
                    'Mode': range(1, len(selected_struct['frequencies']) + 1),
                    'Frequency (cm⁻¹)': selected_struct['frequencies'],
                    'Intensity (km/mol)': selected_struct['intensities'],
                }
                if 'band_types' in selected_struct['metadata']:
                    spec_dict['Band Type'] = selected_struct['metadata']['band_types']
                df_spectrum = pd.DataFrame(spec_dict)
                st.dataframe(df_spectrum, height=300)
            
            # Store primary structure (for single file workflow compatibility)
            st.session_state['dft_frequencies'] = selected_struct['frequencies']
            st.session_state['dft_intensities'] = selected_struct['intensities']
            st.session_state['dft_metadata'] = selected_struct['metadata']
            st.session_state['dft_band_types'] = selected_struct['metadata'].get('band_types', None)
            st.session_state['selected_struct_idx'] = struct_idx
        else:
            st.error("❌ No valid DFT data found in uploaded files.")
            
    except Exception as e:
        st.error(f"Error parsing files: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# Broadening and Plotting Section
if 'dft_frequencies' in st.session_state:
    st.markdown("---")
    st.markdown("## Step 2 — Broadening & Visualization")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        freq_scale_factor = st.number_input(
            "Scale Factor",
            value=0.967, min_value=0.8, max_value=1.1, step=0.001, format="%.3f",
            help="Scale DFT frequencies. Common: B3LYP/6-31G(d) = 0.967, B3LYP/cc-pVTZ = 0.989"
        )
    with col_s2:
        bw_percent = st.number_input(
            "FELIX Bandwidth (%)", value=0.7, min_value=0.1, max_value=5.0, step=0.1, format="%.2f",
            help="FWHM = bandwidth% × frequency. Default 0.7% for FELIX."
        )
        bw_frac = bw_percent / 100.0
    with col_s3:
        x_min = st.number_input("Wavenum. Min (cm⁻¹)", value=500.0, step=50.0)
    with col_s4:
        x_max = st.number_input("Wavenum. Max (cm⁻¹)", value=2200.0, step=50.0)
    
    # Apply frequency scaling
    scaled_frequencies = st.session_state['dft_frequencies'] * freq_scale_factor
    
    # Apply broadening with scaled frequencies
    x_broad, y_broad = broaden_spectrum_felix(
        scaled_frequencies,
        st.session_state['dft_intensities'],
        x_range=(x_min, x_max),
        bw_frac=bw_frac,
        npoints=4000
    )
    
    # Store broadened spectrum, scaled frequencies, and broadening parameters
    st.session_state['dft_x_broad'] = x_broad
    st.session_state['dft_y_broad'] = y_broad
    st.session_state['dft_frequencies_scaled'] = scaled_frequencies
    st.session_state['freq_scale_factor'] = freq_scale_factor
    st.session_state['bw_frac'] = bw_frac
    st.session_state['x_min'] = x_min
    st.session_state['x_max'] = x_max
    
    # Show scaling info
    if abs(freq_scale_factor - 1.0) > 0.001:
        st.caption(f"📐 Scaled by {freq_scale_factor:.3f} — e.g. {st.session_state['dft_frequencies'][0]:.1f} → {scaled_frequencies[0]:.1f} cm⁻¹")
    
    # DFT spectrum in tabs
    tab_plotly_dft, tab_mpl_dft, tab_info_dft = st.tabs(["📈 Interactive Plot", "🖼️ Static Plot (Report)", "ℹ️ About Broadening"])
    
    with tab_plotly_dft:
        fig_dft = go.Figure()
        fig_dft.add_trace(go.Scatter(
            x=scaled_frequencies, y=st.session_state['dft_intensities'],
            mode='markers', marker=dict(size=8, color='red', symbol='line-ns-open'),
            name='Stick Spectrum (Scaled)'
        ))
        fig_dft.add_trace(go.Scatter(
            x=x_broad, y=y_broad, mode='lines', line=dict(color='blue', width=2),
            name=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)'
        ))
        fig_dft.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity (km/mol)",
            title="DFT IR Spectrum", hovermode='closest', legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig_dft, use_container_width=True)
    
    with tab_mpl_dft:
        fig_static, ax = plt.subplots(figsize=(12, 5))
        ax.vlines(scaled_frequencies, 0, st.session_state['dft_intensities'], 
                  colors='red', alpha=0.6, linewidths=1.5, label='Stick Spectrum (Scaled)')
        ax.plot(x_broad, y_broad, 'b-', linewidth=2, label=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)')
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (km/mol)", fontsize=12)
        ax.set_title("DFT IR Spectrum", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(x_min, x_max)
        fig_static.tight_layout()
        st.pyplot(fig_static)
        add_plot_to_report_button(fig_static, "DFT IR Spectrum", key_suffix="dft_spectrum",
                                  description="DFT-calculated IR spectrum with FELIX-style broadening")
    
    with tab_info_dft:
        st.markdown("""
        **Frequency-Proportional Broadening:** FWHM scales linearly with frequency — `FWHM(ν) = bw_frac × ν`.
        
        | Wavenumber | FWHM (0.7%) |
        |---|---|
        | 500 cm⁻¹ | 3.5 cm⁻¹ |
        | 1000 cm⁻¹ | 7.0 cm⁻¹ |
        | 1500 cm⁻¹ | 10.5 cm⁻¹ |
        | 3000 cm⁻¹ | 21.0 cm⁻¹ |
        
        This is characteristic of FEL instruments and provides more physically accurate comparison than constant FWHM broadening.
        """)

# ========================================================================================
# OPTIMAL SCALING FACTOR SEARCH
# (Von der Esch et al., J. Chem. Theory Comput. 2021, 17, 985–995)
# ========================================================================================
fullrange_depletion_data_for_opt = st.session_state.get("fullrange_depletion_data", None)
if 'dft_frequencies' in st.session_state and fullrange_depletion_data_for_opt is not None:
    with st.expander("🔎 Optimal Scaling Factor Search (Von der Esch et al., JCTC 2021)", expanded=False):
        st.caption(
            "Sweep scaling factors and pick the one that maximises mean PCC. "
            "Based on: Von der Esch, B. et al., *J. Chem. Theory Comput.* **2021**, 17, 985–995. "
            "[DOI: 10.1021/acs.jctc.0c01279](https://doi.org/10.1021/acs.jctc.0c01279)"
        )
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        with opt_col1:
            opt_fmin = st.number_input("Factor min", value=0.82, step=0.01, format="%.2f", key="_opt_fmin")
        with opt_col2:
            opt_fmax = st.number_input("Factor max", value=1.05, step=0.01, format="%.2f", key="_opt_fmax")
        with opt_col3:
            opt_nsteps = st.number_input("Steps", value=230, min_value=20, max_value=500, step=10, key="_opt_nsteps")

        if st.button("🔍 Find Optimal Scaling Factor", type="primary", key="_run_opt_scale"):
            exp_x_opt = fullrange_depletion_data_for_opt.iloc[:, 0].values
            exp_y_opt = fullrange_depletion_data_for_opt.iloc[:, 4].values
            bw_opt = st.session_state.get('bw_frac', 0.007)
            xmin_opt = st.session_state.get('x_min', 500.0)
            xmax_opt = st.session_state.get('x_max', 2200.0)
            shift_opt = st.session_state.get('shift_theory', 0.0)
            regions_opt = get_diagnostic_regions()

            with st.spinner("Sweeping scaling factors…"):
                opt_result = find_optimal_scaling_factor(
                    exp_x_opt, exp_y_opt,
                    st.session_state['dft_frequencies'],
                    st.session_state['dft_intensities'],
                    factor_range=(opt_fmin, opt_fmax),
                    n_steps=opt_nsteps,
                    broaden_func=broaden_spectrum_felix,
                    bw_frac=bw_opt, x_range=(xmin_opt, xmax_opt),
                    regions=regions_opt, shift=shift_opt,
                )
            st.session_state['_opt_scale_result'] = opt_result

        # Display persisted results
        if '_opt_scale_result' in st.session_state:
            opt_result = st.session_state['_opt_scale_result']
            best = opt_result['best_factor']
            best_pcc = opt_result['best_mean_pcc']

            st.success(f"**Best scaling factor: {best:.4f}**  |  Mean PCC = {best_pcc:.4f}")

            # Plot the sweep curve
            fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
            ax_opt.plot(opt_result['factors'], opt_result['mean_pcc'],
                        'k-', linewidth=2, label='Mean PCC')
            for rname, rvals in opt_result.get('per_region_pcc', {}).items():
                ax_opt.plot(opt_result['factors'], rvals, '--', alpha=0.5, label=rname)
            ax_opt.axvline(best, color='red', linestyle=':', linewidth=1.5, label=f'Optimum = {best:.4f}')
            ax_opt.set_xlabel("Scaling Factor", fontsize=12)
            ax_opt.set_ylabel("PCC (r)", fontsize=12)
            ax_opt.set_title("Scaling Factor Optimisation (Von der Esch et al., JCTC 2021)", fontsize=13, fontweight='bold')
            ax_opt.legend(fontsize=8, loc='lower left')
            ax_opt.grid(True, alpha=0.3)
            st.pyplot(fig_opt)
            plt.close(fig_opt)

            st.caption(
                "Reference scaling factors from Von der Esch et al.: "
                "B3LYP = 0.966, PBE = 0.985, PBEh-3c = 0.938, HF-3c = 0.832, GFN2-xTB = 0.999"
            )

# Experimental vs Theoretical Comparison
st.markdown("---")
st.markdown("## Step 3 — Compare with Experimental Data")

# ---- Upload experimental data directly (bypass pipeline) ----
_pipeline_data = st.session_state.get("fullrange_depletion_data", None)
_upload_source = "pipeline" if _pipeline_data is not None else None

with st.expander(
    "📂 Upload experimental spectrum (skip pipeline)"
    if _pipeline_data is not None
    else "📂 Upload experimental spectrum",
    expanded=(_pipeline_data is None),
):
    st.caption(
        "Upload a CSV or TXT file from a previous analysis. "
        "Expected format: columns separated by comma, tab, or whitespace. "
        "The file should contain at least a **wavenumber** column and an **intensity** column "
        "(e.g. ``-ln(depletion)``)."
    )
    exp_upload = st.file_uploader(
        "Experimental spectrum file",
        type=["csv", "txt"],
        key="_exp_upload",
        help="Accepts the CSV exported by Section 3.0 or any 2-column (wavenumber, intensity) file.",
    )
    if exp_upload is not None:
        try:
            raw = exp_upload.read().decode("utf-8", errors="ignore")
            # Auto-detect separator
            if "\t" in raw[:500]:
                _sep = "\t"
            elif "," in raw[:500]:
                _sep = ","
            else:
                _sep = r"\s+"
            uploaded_exp_df = pd.read_csv(
                io.StringIO(raw), sep=_sep, engine="python"
            )

            st.markdown(f"**Loaded {len(uploaded_exp_df)} rows × {len(uploaded_exp_df.columns)} columns**")

            col_names = list(uploaded_exp_df.columns)

            # Auto-pick sensible defaults
            _default_x = 0
            _default_y = min(4, len(col_names) - 1)
            for ci, c in enumerate(col_names):
                cl = str(c).lower()
                if "wavenum" in cl or cl == "x":
                    _default_x = ci
                if "ln" in cl and "depletion" in cl:
                    _default_y = ci
                elif "intensity" in cl or cl == "y":
                    _default_y = ci

            ucol1, ucol2 = st.columns(2)
            with ucol1:
                x_col = st.selectbox(
                    "Wavenumber column",
                    options=range(len(col_names)),
                    format_func=lambda i: f"{i}: {col_names[i]}",
                    index=_default_x,
                    key="_exp_xcol",
                )
            with ucol2:
                y_col = st.selectbox(
                    "Intensity column (e.g. -ln(depletion))",
                    options=range(len(col_names)),
                    format_func=lambda i: f"{i}: {col_names[i]}",
                    index=_default_y,
                    key="_exp_ycol",
                )

            if st.button("✅ Use this spectrum", key="_use_uploaded_exp"):
                # Build a 5-column DataFrame matching the pipeline format so all
                # downstream code (iloc[:,0] and iloc[:,4]) works unchanged.
                x_vals = pd.to_numeric(uploaded_exp_df.iloc[:, x_col], errors="coerce")
                y_vals = pd.to_numeric(uploaded_exp_df.iloc[:, y_col], errors="coerce")
                mask = x_vals.notna() & y_vals.notna()
                x_vals = x_vals[mask].values
                y_vals = y_vals[mask].values

                compat_df = pd.DataFrame({
                    "wavenumber": x_vals,
                    "integrated_signal_withoutIR": np.zeros_like(x_vals),
                    "integrated_signal_withIR": np.zeros_like(x_vals),
                    "depletion": np.zeros_like(x_vals),
                    "-ln(depletion)": y_vals,
                })
                st.session_state["fullrange_depletion_data"] = compat_df
                _upload_source = "upload"
                st.success(
                    f"✅ Loaded {len(compat_df)} points "
                    f"({x_vals.min():.1f} – {x_vals.max():.1f} cm⁻¹). "
                    "You can now run the comparison below."
                )
                st.rerun()

        except Exception as exc:
            st.error(f"Failed to read file: {exc}")

# ---- Scoring Configuration: Regions, Thresholds & Info ----
if 'dft_frequencies' in st.session_state:
    with st.expander("⚙️ Scoring Configuration — Regions, Thresholds & Info", expanded=False):
        cfg_tab_regions, cfg_tab_thresholds, cfg_tab_about = st.tabs(["🎯 Diagnostic Regions", "📏 Score Thresholds", "ℹ️ About Metrics"])

        with cfg_tab_regions:
            use_custom = st.checkbox(
                "Enable Custom Diagnostic Regions",
                value=st.session_state.get('custom_regions_enabled', False),
                help="Override default C₁₁H₈ regions with your own spectral windows"
            )
            st.session_state['custom_regions_enabled'] = use_custom

            if use_custom:
                st.caption("'Full Overlap' is always included. Set regions to match your experimental coverage.")

                if 'diagnostic_regions' not in st.session_state:
                    st.session_state['diagnostic_regions'] = DEFAULT_DIAGNOSTIC_REGIONS.copy()

                custom_regions = {"Full Overlap": None}
                num_regions = st.number_input("Number of custom regions", min_value=1, max_value=8, value=4, step=1)

                for i in range(num_regions):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        region_name = st.text_input(
                            f"Region {i+1} Name",
                            value=list(DEFAULT_DIAGNOSTIC_REGIONS.keys())[i+1] if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS) else f"Custom_{i+1}",
                            key=f"region_name_{i}"
                        )
                    with col2:
                        default_min = 500.0
                        if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS):
                            default_vals = list(DEFAULT_DIAGNOSTIC_REGIONS.values())[i+1]
                            if default_vals: default_min = float(default_vals[0])
                        region_min = st.number_input(f"Min (cm⁻¹)", value=default_min, step=50.0, key=f"region_min_{i}", format="%.0f")
                    with col3:
                        default_max = 1500.0
                        if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS):
                            default_vals = list(DEFAULT_DIAGNOSTIC_REGIONS.values())[i+1]
                            if default_vals: default_max = float(default_vals[1])
                        region_max = st.number_input(f"Max (cm⁻¹)", value=default_max, step=50.0, key=f"region_max_{i}", format="%.0f")

                    if region_name and region_min < region_max:
                        custom_regions[region_name] = (region_min, region_max)

                st.session_state['diagnostic_regions'] = custom_regions
                preview_df = pd.DataFrame([
                    {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
                    for name, rng in custom_regions.items()
                ])
                st.dataframe(preview_df, width='stretch', hide_index=True)
            else:
                st.markdown("**Using default C₁₁H₈ isomer regions:**")
                default_df = pd.DataFrame([
                    {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
                    for name, rng in DEFAULT_DIAGNOSTIC_REGIONS.items()
                ])
                st.dataframe(default_df, width='stretch', hide_index=True)

        with cfg_tab_thresholds:
            st.caption("Set the score boundaries for each verdict category. These apply to whichever similarity metric you select.")
            thr_col1, thr_col2, thr_col3 = st.columns(3)
            with thr_col1:
                st.session_state["pcc_threshold_excellent"] = st.number_input(
                    "Excellent ✅ (score ≥)", value=st.session_state.get("pcc_threshold_excellent", DEFAULT_PCC_THRESHOLDS["excellent"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_exc"
                )
            with thr_col2:
                st.session_state["pcc_threshold_good"] = st.number_input(
                    "Good 🟡 (score ≥)", value=st.session_state.get("pcc_threshold_good", DEFAULT_PCC_THRESHOLDS["good"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_good"
                )
            with thr_col3:
                st.session_state["pcc_threshold_weak"] = st.number_input(
                    "Weak ⚠️ (score ≥)", value=st.session_state.get("pcc_threshold_weak", DEFAULT_PCC_THRESHOLDS["weak"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_weak"
                )
            st.caption("Below **Weak** → **Poor / Rule Out ❌**")

        with cfg_tab_about:
            thresholds_info = get_pcc_thresholds()
            st.markdown(f"""
            **Three similarity metrics** are computed in parallel. Choose the primary metric below.

            | Metric | Formula | Range | Best for |
            |--------|---------|-------|----------|
            | **PCC** | Pearson *r* of normalised curves | −1 … 1 | Clean absorption IR |
            | **SEC** | cos²(θ) between normalised vectors | 0 … 1 | Moderate noise |
            | **SFEC** | cos²(θ) of **first-derivative** vectors (SG smoothed) | 0 … 1 | **Noisy action spectra** (IR-UV, IRMPD) — removes baseline & slope artefacts |

            **Verdict thresholds** (applied to whichever metric is selected):

            | Verdict | Threshold |
            |---|---|
            | Excellent ✅ | score ≥ {thresholds_info['excellent']:.2f} |
            | Good 🟡 | score ≥ {thresholds_info['good']:.2f} |
            | Weak ⚠️ | score ≥ {thresholds_info['weak']:.2f} |
            | Poor ❌ | score < {thresholds_info['weak']:.2f} |

            *Default thresholds are tuned for action spectra (IR-UV ion dip, IRMPD), which are noisier than direct absorption.*

            **Tips:** Use SFEC for action spectra with noisy baselines. Use regional scores for isomer discrimination. Compare multiple structures — highest score wins. Visual inspection remains critical.

            ---
            **References:**
            - Von der Esch, B. et al. *J. Chem. Theory Comput.* **2021**, 17, 985–995. [DOI: 10.1021/acs.jctc.0c01279](https://doi.org/10.1021/acs.jctc.0c01279)
            - Samuel, A. Z. et al. *ACS Omega* **2021**, 6, 2060–2065. [DOI: 10.1021/acsomega.0c05041](https://doi.org/10.1021/acsomega.0c05041)
            """)

# Check if experimental data is available
fullrange_depletion_data = st.session_state.get("fullrange_depletion_data", None)

if fullrange_depletion_data is not None and 'dft_x_broad' in st.session_state:
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    with col1:
        shift_theory = st.number_input("Shift Theory (cm⁻¹)", value=st.session_state.get("shift_theory", 0.0), step=1.0, format="%.1f",
                                      help="Shift theoretical spectrum for alignment", key="shift_theory")
    with col2:
        primary_metric = st.selectbox(
            "Similarity Metric",
            options=list(AVAILABLE_METRICS.keys()),
            index=2,  # default SFEC
            help="**PCC** — Pearson r (sensitive to noise/baseline). "
                 "**SEC** — Cosine similarity² (intensity pattern). "
                 "**SFEC** — First-derivative cosine² (robust to baseline & noise, recommended for action spectra).",
            key="_metric_choice",
        )
    with col3:
        invert_theory = st.checkbox("Invert Theory", value=False, key="invert_theory",
                                   help="Invert theoretical spectrum if needed")
    with col4:
        run_comparison = st.button("📊 Compare", type="primary", width='stretch')

    # Warn if data was already smoothed upstream
    _already_smoothed = st.session_state.get("data_display_option") == "Smoothed"
    _prev_sw = st.session_state.get("smoothing_window", None)
    if _already_smoothed and _prev_sw:
        st.warning(
            f"⚠️ Your experimental spectrum was **already smoothed** in Section 3.0 "
            f"(Savitzky-Golay window = {_prev_sw}). Applying additional smoothing here "
            f"risks **over-smoothing**, which can wash out real peaks and artificially inflate scores. "
            f"Consider uploading **raw / unsmoothed** data instead, or leave preprocessing off below."
        )

    # Preprocessing options
    with st.expander("🔧 Experimental spectrum preprocessing", expanded=False):
        st.caption(
            "Optional preprocessing applied to the experimental spectrum before scoring. "
            "Smoothing reduces high-frequency noise; clipping removes negative baseline artefacts. "
            "**Note:** SFEC already smooths internally via its Savitzky-Golay derivative — "
            "extra smoothing is usually not needed when using SFEC."
        )
        pp_col1, pp_col2 = st.columns(2)
        with pp_col1:
            pp_smooth = st.number_input(
                "Savitzky-Golay smooth window", value=0, min_value=0, max_value=101, step=2,
                help="0 = off. Odd window size for SG smoothing (e.g. 11). Reduces noise.",
                key="_pp_smooth",
            )
        with pp_col2:
            pp_clip = st.checkbox(
                "Clip negative values to zero",
                value=False,
                help="Removes negative baseline dips from action spectra.",
                key="_pp_clip",
            )

    # Compute comparison when button pressed, store results in session state
    if run_comparison:
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values

        # Apply preprocessing if requested
        if pp_smooth > 0 or pp_clip:
            exp_x, exp_y = preprocess_spectrum(exp_x, exp_y,
                                               smooth_window=pp_smooth,
                                               clip_negative=pp_clip)

        theory_x_shifted = st.session_state['dft_x_broad'] + shift_theory
        theory_y = st.session_state['dft_y_broad'].copy()
        if invert_theory:
            theory_y = -theory_y

        # Compute ALL metrics for each diagnostic region
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        pcc_results = []
        for region_name, region_range in DIAGNOSTIC_REGIONS.items():
            row = {
                "Region": region_name,
                "Range (cm⁻¹)": f"{region_range[0]}–{region_range[1]}" if region_range else "Full",
            }
            for metric_name in AVAILABLE_METRICS:
                score, p_val, _, _, _ = compute_similarity(
                    exp_x, exp_y, theory_x_shifted, theory_y,
                    region=region_range, metric=metric_name,
                )
                row[metric_name] = round(score, 4) if score is not None else None
                if metric_name == 'PCC' and p_val is not None:
                    row["p-value"] = f"{p_val:.2e}"

            # Verdict uses the user-selected primary metric
            primary_score = row.get(primary_metric)
            label, color = ui_score_label(primary_score)
            row["Verdict"] = label
            pcc_results.append(row)

        # Persist all results in session state
        st.session_state['_comp_exp_x'] = exp_x
        st.session_state['_comp_exp_y'] = exp_y
        st.session_state['_comp_theory_x'] = theory_x_shifted
        st.session_state['_comp_theory_y'] = theory_y
        st.session_state['_comp_pcc_results'] = pcc_results
        st.session_state['_comp_primary_metric'] = primary_metric
        st.session_state['_comp_done'] = True
    
    # ---- Display persisted results (survives widget interactions) ----
    if st.session_state.get('_comp_done', False):
        exp_x = st.session_state['_comp_exp_x']
        exp_y = st.session_state['_comp_exp_y']
        theory_x_shifted = st.session_state['_comp_theory_x']
        theory_y = st.session_state['_comp_theory_y']
        pcc_results = st.session_state['_comp_pcc_results']
        
        # --- Comparison Plots in Tabs ---
        tab_comp_plotly, tab_comp_mpl = st.tabs(["📈 Interactive Comparison", "🖼️ Static Plot (Report)"])
        
        with tab_comp_plotly:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=exp_x, y=exp_y, mode='lines', line=dict(color='black', width=2),
                name='Experimental -ln(depletion)', yaxis='y1'
            ))
            fig_comp.add_trace(go.Scatter(
                x=theory_x_shifted, y=theory_y, mode='lines', line=dict(color='red', width=2),
                name='DFT Theory', yaxis='y2'
            ))
            fig_comp.update_layout(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis=dict(title="-ln(depletion)", side='left', showgrid=True),
                yaxis2=dict(title="Intensity (km/mol)", side='right', overlaying='y', showgrid=False),
                title="Experimental vs DFT Comparison", hovermode='x unified', legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with tab_comp_mpl:
            fig_comp_static, ax1 = plt.subplots(figsize=(14, 6))
            ax1.plot(exp_x, exp_y, 'k-', linewidth=2, label='Experimental -ln(depletion)', alpha=0.8)
            ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
            ax1.set_ylabel("-ln(depletion)", fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black'); ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax2 = ax1.twinx()
            ax2.plot(theory_x_shifted, theory_y, 'r-', linewidth=2, label='DFT Theory', alpha=0.8)
            ax2.set_ylabel("Intensity (km/mol)", fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
            ax1.set_title("Experimental vs DFT IR Spectrum", fontsize=14, fontweight='bold')
            fig_comp_static.tight_layout()
            st.pyplot(fig_comp_static)
            add_plot_to_report_button(fig_comp_static, "Experimental vs DFT Comparison",
                                      key_suffix="exp_vs_dft", description="Comparison of experimental and DFT-calculated IR spectra")
        
        # --- Scoring Results ---
        _pm = st.session_state.get('_comp_primary_metric', 'SFEC')
        st.markdown(f"### 📐 Spectral Similarity — all metrics  (verdict uses **{_pm}**)")
        df_pcc = pd.DataFrame(pcc_results)

        def highlight_verdict(row):
            ncols = len(row)
            base = [''] * (ncols - 1)
            verdict = str(row["Verdict"])
            if "Excellent" in verdict:
                return base + ["background-color: #d4edda; color: #155724; font-weight: bold"]
            elif "Good" in verdict or "Weak" in verdict:
                return base + ["background-color: #fff3cd; color: #856404"]
            else:
                return base + ["background-color: #f8d7da; color: #721c24"]

        st.dataframe(df_pcc.style.apply(highlight_verdict, axis=1), width='stretch', hide_index=True)

        # Bar chart — uses the primary metric
        fig_pcc, ax_pcc = plt.subplots(figsize=(10, 4))
        valid = df_pcc.dropna(subset=[_pm])
        thresholds = get_pcc_thresholds()
        colors_bar = [
            "#28a745" if r >= thresholds["excellent"] else "#ffc107" if r >= thresholds["good"] else "#dc3545"
            for r in valid[_pm]
        ]
        ax_pcc.barh(valid["Region"], valid[_pm], color=colors_bar, alpha=0.8)
        ax_pcc.axvline(thresholds["good"], color='orange', linestyle='--', linewidth=1.2, label=f'Good ({thresholds["good"]:.2f})', alpha=0.7)
        ax_pcc.axvline(thresholds["excellent"], color='green', linestyle='--', linewidth=1.2, label=f'Excellent ({thresholds["excellent"]:.2f})', alpha=0.7)
        ax_pcc.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        _x_label = "Pearson r" if _pm == "PCC" else f"{_pm} score"
        ax_pcc.set_xlabel(_x_label, fontsize=11)
        ax_pcc.set_title(f"Regional {_pm} Scores", fontsize=13, fontweight='bold')
        _xlim = (-1, 1) if _pm == "PCC" else (0, 1)
        ax_pcc.set_xlim(*_xlim)
        ax_pcc.legend(fontsize=9, loc='lower right'); ax_pcc.grid(True, axis='x', alpha=0.3)
        fig_pcc.tight_layout()
        st.pyplot(fig_pcc)
        add_plot_to_report_button(fig_pcc, f"{_pm} Region Scores", key_suffix="pcc_scores",
                                  description=f"{_pm} similarity scores per diagnostic spectral region")

        # --- Structure Assignment Decision ---
        st.markdown("#### 🧪 Structure Assignment Decision")
        def _get_score(region_name):
            vals = df_pcc[df_pcc["Region"] == region_name][_pm].values
            return vals[0] if len(vals) > 0 and not pd.isna(vals[0]) else None

        full_r = _get_score("Full Overlap")
        fp_r = _get_score("Fingerprint")
        cc_r = _get_score("C≡C Stretch")

        thresholds_dec = get_pcc_thresholds()
        if full_r is not None and full_r > thresholds_dec["excellent"] and fp_r is not None and fp_r > thresholds_dec["good"]:
            st.success("🟢 **Candidate Structure:** Strong overall and fingerprint agreement.")
        elif full_r is not None and full_r > thresholds_dec["good"]:
            st.warning("🟡 **Tentative Match:** Moderate agreement. Check diagnostic regions manually.")
        elif cc_r is not None and cc_r < thresholds_dec["weak"]:
            st.error("🔴 **Rule Out:** C≡C stretch shows very poor agreement.")
        elif full_r is not None and full_r < thresholds_dec["weak"]:
            st.error("🔴 **Rule Out:** Overall score too low for this structure.")
        else:
            st.info("ℹ️ **Inconclusive:** Mixed scores. Consider visual inspection and additional regions.")
        
        # --- Save ---
        if st.checkbox("💾 Save Comparison Data"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            exp_df = pd.DataFrame({'Wavenumber_Exp': exp_x, 'Ln_Depletion_Exp': exp_y})
            theory_df = pd.DataFrame({'Wavenumber_Theory': theory_x_shifted, 'Intensity_Theory_km_mol': theory_y})
            exp_filename = os.path.join(file_directory, f"exp_data_{timestamp}.csv")
            theory_filename = os.path.join(file_directory, f"theory_data_{timestamp}.csv")
            pcc_filename = os.path.join(file_directory, f"pcc_scores_{timestamp}.csv")
            exp_df.to_csv(exp_filename, index=False)
            theory_df.to_csv(theory_filename, index=False)
            df_pcc.to_csv(pcc_filename, index=False)
            st.success(f"✅ Saved: `{exp_filename}`, `{theory_filename}`, `{pcc_filename}`")

elif fullrange_depletion_data is None:
    st.info("⚠️ No experimental data found in session. Run the depletion calculation (Section 3.0) or **upload a spectrum file** above.")
elif 'dft_x_broad' not in st.session_state:
    st.info("⚠️ Please upload and process a DFT file first.")

# ========================================================================================
# BATCH MULTI-STRUCTURE COMPARISON SECTION
# ========================================================================================

# Batch Multi-Structure Comparison
if fullrange_depletion_data is not None and len(st.session_state.get('dft_structures', [])) > 1:
    st.markdown("---")
    st.markdown("## Step 4 — Multi-Structure Batch Comparison")
    st.caption("Rank all candidate structures using similarity scoring across diagnostic regions.")

    batch_metric = st.selectbox(
        "Batch Similarity Metric",
        options=list(AVAILABLE_METRICS.keys()),
        index=2,  # SFEC default
        help="Metric used for ranking structures. SFEC recommended for action spectra.",
        key="_batch_metric",
    )

    if st.button(f"🚀 Run Batch {batch_metric} Analysis", type="primary"):
        structures = st.session_state['dft_structures']
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values
        freq_scale = st.session_state.get('freq_scale_factor', 0.967)
        bw_frac = st.session_state.get('bw_frac', 0.007)
        x_min = st.session_state.get('x_min', 500.0)
        x_max = st.session_state.get('x_max', 2200.0)
        shift = st.session_state.get('shift_theory', 0.0)
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        
        progress_bar = st.progress(0)
        all_results = compute_batch_pcc(
            structures, exp_x, exp_y, DIAGNOSTIC_REGIONS,
            freq_scale=freq_scale, bw_frac=bw_frac,
            x_range=(x_min, x_max), shift=shift,
            broaden_func=broaden_spectrum_felix,
            metric=batch_metric,
        )
        progress_bar.progress(1.0)
        
        df_batch, scoring_regions, avg_col = rank_batch_results(
            all_results, DIAGNOSTIC_REGIONS, metric=batch_metric,
        )
        
        # Persist in session state
        st.session_state['_batch_df'] = df_batch
        st.session_state['_batch_regions'] = list(DIAGNOSTIC_REGIONS.keys())
        st.session_state['_batch_scoring_regions'] = scoring_regions
        st.session_state['_batch_avg_col'] = avg_col
        st.session_state['_batch_metric_used'] = batch_metric
        st.session_state['_batch_params'] = {
            'freq_scale': freq_scale, 'bw_frac': bw_frac, 'shift': shift,
            'n_structures': len(structures), 'metric': batch_metric,
        }
        st.session_state['_batch_exp_x'] = exp_x
        st.session_state['_batch_done'] = True
    
    # ---- Display persisted batch results ----
    if st.session_state.get('_batch_done', False):
        df_batch = st.session_state['_batch_df']
        regions_to_plot = st.session_state['_batch_regions']
        scoring_regions = st.session_state['_batch_scoring_regions']
        
        _bm = st.session_state.get('_batch_metric_used', 'PCC')
        _avg_col = st.session_state.get('_batch_avg_col', f'Average {_bm}')
        # Guard against stale session state from previous code versions
        if _avg_col not in df_batch.columns:
            _avg_candidates = [c for c in df_batch.columns if c.startswith('Average')]
            _avg_col = _avg_candidates[0] if _avg_candidates else None

        best_match = df_batch.iloc[0]
        if _avg_col and _avg_col in df_batch.columns:
            st.success(f"🥇 **Best Match:** {best_match['filename']} (Avg {_bm}: {best_match[_avg_col]:.3f}, {best_match['Valid Regions']:.0f} regions)")
        else:
            st.success(f"🥇 **Best Match:** {best_match['filename']}")
            st.warning("⚠️ Stale batch results from a previous session. Please re-run the batch analysis.")

        nan_counts = df_batch[scoring_regions].isna().sum()
        if nan_counts.sum() > 0:
            st.warning(f"⚠️ Missing coverage: {', '.join([f'{r} ({nan_counts[r]})' for r in nan_counts[nan_counts > 0].index])}")

        _vmin = -1 if _bm == 'PCC' else 0
        _grad_cols = [_avg_col] if (_avg_col and _avg_col in df_batch.columns) else []
        st.dataframe(
            df_batch.style.background_gradient(subset=_grad_cols, cmap='RdYlGn', vmin=_vmin, vmax=1) if _grad_cols else df_batch,
            width='stretch', hide_index=True
        )
        
        # Batch visualizations in tabs
        tab_batch_bar, tab_batch_heat = st.tabs(["📈 Bar Chart", "🗺️ Heatmap"])
        
        with tab_batch_bar:
            n_structs = len(df_batch)
            fig_batch, ax_batch = plt.subplots(figsize=(12, max(5, n_structs * 0.5)))
            x_pos = np.arange(n_structs)
            num_regions = len(regions_to_plot)
            width = 0.8 / num_regions
            colors_regions = plt.cm.tab10(np.linspace(0, 1, num_regions))
            for i, region in enumerate(regions_to_plot):
                if region in df_batch.columns:
                    ax_batch.barh(x_pos + i * width, df_batch[region].values, width,
                                  label=region, color=colors_regions[i], alpha=0.8)
            ax_batch.set_yticks(x_pos + width * (num_regions - 1) / 2)
            ax_batch.set_yticklabels(df_batch['filename'].values)
            _xlabel = 'Pearson r' if _bm == 'PCC' else f'{_bm} score'
            ax_batch.set_xlabel(_xlabel, fontsize=12)
            ax_batch.set_title(f'Multi-Structure {_bm} Comparison', fontsize=14, fontweight='bold')
            thresholds_batch = get_pcc_thresholds()
            ax_batch.axvline(thresholds_batch["good"], color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'Good ({thresholds_batch["good"]:.2f})')
            ax_batch.axvline(thresholds_batch["excellent"], color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Excellent ({thresholds_batch["excellent"]:.2f})')
            ax_batch.legend(loc='lower right', fontsize=8, ncol=2)
            _xlim = (-0.2, 1.0) if _bm == 'PCC' else (0, 1.0)
            ax_batch.grid(True, axis='x', alpha=0.3); ax_batch.set_xlim(*_xlim)
            fig_batch.tight_layout()
            st.pyplot(fig_batch)
            add_plot_to_report_button(fig_batch, f"Multi-Structure {_bm} Comparison", key_suffix="batch_pcc",
                                      description=f"Batch comparison of all candidate structures using {_bm} scoring")
        
        with tab_batch_heat:
            n_structs = len(df_batch)
            fig_heat, ax_heat = plt.subplots(figsize=(10, max(5, n_structs * 0.4)))
            heat_data = df_batch[regions_to_plot].values
            _hmin = -0.5 if _bm == 'PCC' else 0.0
            im = ax_heat.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=_hmin, vmax=1.0)
            ax_heat.set_xticks(np.arange(len(regions_to_plot)))
            ax_heat.set_yticks(np.arange(n_structs))
            ax_heat.set_xticklabels(regions_to_plot, rotation=45, ha='right')
            ax_heat.set_yticklabels(df_batch['filename'].values)
            cbar = plt.colorbar(im, ax=ax_heat)
            cbar.set_label(f'{_bm} Score', rotation=270, labelpad=20)
            for i in range(n_structs):
                for j in range(len(regions_to_plot)):
                    value = heat_data[i, j]
                    if not np.isnan(value):
                        ax_heat.text(j, i, f'{value:.2f}', ha="center", va="center",
                                     color="black" if value < 0.5 else "white", fontsize=9, fontweight='bold')
            ax_heat.set_title(f'{_bm} Heatmap: All Structures vs Experimental', fontsize=14, fontweight='bold')
            fig_heat.tight_layout()
            st.pyplot(fig_heat)
            add_plot_to_report_button(fig_heat, f"{_bm} Heatmap", key_suffix="pcc_heatmap",
                                      description=f"Heatmap showing {_bm} scores for all structures across regions")
        
        # Save batch results
        if st.checkbox("💾 Save Batch Results", key="save_batch"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            batch_filename = os.path.join(file_directory, f"batch_{_bm.lower()}_comparison_{timestamp}.csv")
            df_batch.to_csv(batch_filename, index=False)
            
            params = st.session_state['_batch_params']
            exp_x = st.session_state['_batch_exp_x']
            summary_text = f"""Multi-Structure {_bm} Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Metric: {_bm} | Structures: {params['n_structures']} | Scale: {params['freq_scale']:.3f} | BW: {params['bw_frac']*100:.2f}% | Shift: {params['shift']:.1f} cm⁻¹
Exp range: {exp_x.min():.1f} - {exp_x.max():.1f} cm⁻¹\n\nRankings:\n"""
            DIAGNOSTIC_REGIONS = get_diagnostic_regions()
            for _, row in df_batch.iterrows():
                summary_text += f"\n{row['Rank']:.0f}. {row['filename']} — Avg {_bm}: {row[_avg_col]:.3f}"
                for region_name in DIAGNOSTIC_REGIONS.keys():
                    val = row.get(region_name, np.nan)
                    summary_text += f"\n   {region_name}: {val:.3f}" if not np.isnan(val) else f"\n   {region_name}: N/A"
                summary_text += "\n"
            
            report_filename = os.path.join(file_directory, f"batch_{_bm.lower()}_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(summary_text)
            st.success(f"✅ Saved: `{batch_filename}` and `{report_filename}`")

elif len(st.session_state.get('dft_structures', [])) <= 1:
    st.info("💡 Upload multiple DFT files to enable batch comparison and rank candidate structures.")
