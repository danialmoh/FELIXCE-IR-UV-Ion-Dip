"""
REMPI Experiment Parameters & Baseline Correction Page

This page combines:
1. Experiment parameters (calibration, molecule info)
2. Baseline correction preview with interactive Plotly plots
3. Apply baseline correction to full dataset

Similar to 2.0_BaselineCorrection.py but for single-trace REMPI data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import configparser
import os
from pathlib import Path

import re

from packages.BaselineCorrection_REMPI import baseline_REMPI
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

# Import variables from defaults.ini
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'./.streamlit/defaults.ini'
    defaults = {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            defaults['t_off'] = config.getfloat('Experiment Parameters', 't_off', fallback=58)
            defaults['alpha'] = config.getfloat('Experiment Parameters', 'alpha', fallback=7.6987e-7)
            defaults['baseline_reference'] = config.getfloat('Baseline Parameters', 'baseline_reference', fallback=98)
            defaults['baseline_width'] = config.getfloat('Baseline Parameters', 'baseline_width', fallback=3)
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults

defaults = load_defaults()

# Check if data is loaded
if "rempi_compiled_dataframe" not in st.session_state:
    st.error("Please import REMPI data first (Step 8.0)", icon="🚫")
    st.stop()

if "rempi_compiled_data" not in st.session_state:
    st.error("Please import REMPI data first (Step 8.0)", icon="🚫")
    st.stop()

# Get data from session state
compiled_dataframe = st.session_state["rempi_compiled_dataframe"]
compiled_data = st.session_state["rempi_compiled_data"]
dataset_length = st.session_state.get("rempi_dataset_length", len(compiled_dataframe))

# Layout: parameters on left, plot on right
col1, col2, col3 = st.columns([1, 0.1, 2.5])

with col1:
    st.markdown("#### Molecule Parameters")
    st.session_state["rempi_molecule_name"] = st.text_input(
        "Molecule name", 
        value=st.session_state.get("rempi_molecule_name", "")
    )
    st.session_state["rempi_molecule_mass"] = float(st.text_input(
        "Molecule mass (amu)", 
        value=st.session_state.get("rempi_molecule_mass", 100)
    ))
    
    st.markdown("#### Calibration Parameters")
    st.session_state["rempi_t_off"] = st.number_input(
        "t_off (time offset)", 
        value=float(st.session_state.get("rempi_t_off", defaults.get("t_off", 58)))
    )
    st.session_state["rempi_alpha"] = float(st.text_input(
        "alpha (calibration constant)", 
        value=st.session_state.get("rempi_alpha", defaults.get("alpha", 7.6987e-7))
    ))
    
    st.markdown("#### Baseline Parameters")
    baseline_method = st.radio(
        "Method",
        options=["Manual region", "Automatic (ALS)"],
        horizontal=True,
        key="rempi_baseline_method",
        help="Manual: pick a flat region. ALS: automatic smooth baseline — no clean region needed.",
    )

    if baseline_method == "Manual region":
        st.session_state["rempi_baseline_reference"] = float(st.text_input(
            "Start of baseline (amu)", 
            value=st.session_state.get("rempi_baseline_reference", defaults.get("baseline_reference", 98))
        ))
        st.session_state["rempi_baseline_width"] = float(st.text_input(
            "Width of baseline (amu)", 
            value=st.session_state.get("rempi_baseline_width", defaults.get("baseline_width", 3))
        ))
    else:
        st.session_state["rempi_als_lam"] = st.select_slider(
            "Smoothness (λ)",
            options=[1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9],
            value=st.session_state.get("rempi_als_lam", 1e6),
            format_func=lambda x: f"{x:.0e}",
            key="_als_lam_slider",
            help="Larger = smoother baseline curve",
        )
        st.session_state["rempi_als_p"] = st.number_input(
            "Asymmetry (p)",
            value=float(st.session_state.get("rempi_als_p", 0.01)),
            min_value=0.001, max_value=0.5, step=0.005,
            format="%.3f",
            key="_als_p_input",
            help="Smaller = baseline stays further below peaks",
        )
    
    st.markdown("#### Plot Parameters")
    wavelengths = sorted(compiled_data.keys())
    default_idx = 0
    if "rempi_plot_wavelength" in st.session_state and st.session_state["rempi_plot_wavelength"] in wavelengths:
        default_idx = wavelengths.index(st.session_state["rempi_plot_wavelength"])
    st.session_state["rempi_plot_wavelength"] = st.selectbox(
        "Wavelength to preview (nm)",
        options=wavelengths,
        index=default_idx,
        format_func=lambda x: f"{x:.2f} nm"
    )
    
    st.session_state["rempi_ymax_top"] = float(st.text_input(
        "Y-max for top plot", 
        value=st.session_state.get("rempi_ymax_top", 0.1)
    ))
    st.session_state["rempi_ymax_bottom"] = float(st.text_input(
        "Y-max for bottom plot", 
        value=st.session_state.get("rempi_ymax_bottom", 0.1)
    ))

with col3:
    if st.button("✨ Register parameters and preview baseline!", width='stretch'):
        # Get parameters
        t_off = st.session_state.get("rempi_t_off", 58)
        alpha = st.session_state.get("rempi_alpha", 7.6987e-7)
        molecule_mass = st.session_state.get("rempi_molecule_mass", 100)
        molecule_name = st.session_state.get("rempi_molecule_name", "")
        plot_wavelength = st.session_state.get("rempi_plot_wavelength", wavelengths[0])
        ymax_top = st.session_state.get("rempi_ymax_top", 0.1)
        ymax_bottom = st.session_state.get("rempi_ymax_bottom", 0.1)
        _bl_method = st.session_state.get("rempi_baseline_method", "Manual region")
        
        # Generate mass axis
        x_counts = np.linspace(1, dataset_length, int(dataset_length))
        x_mass = alpha * (x_counts - t_off) ** 2
        st.session_state["rempi_x_mass"] = x_mass
        
        # Get signal for selected wavelength
        if plot_wavelength in compiled_dataframe.columns:
            signal = compiled_dataframe[plot_wavelength].values
        else:
            signal = compiled_data[plot_wavelength].iloc[:, 0].values
        
        # --- Compute baseline depending on method ---
        als_baseline_curve = None
        baseline_range_indices = np.array([], dtype=int)
        mean_value = None
        
        if _bl_method == "Manual region":
            baseline_reference = st.session_state.get("rempi_baseline_reference", 98)
            baseline_width = st.session_state.get("rempi_baseline_width", 3)
            baseline_corrector = baseline_REMPI(
                baseline_reference=baseline_reference,
                interval=baseline_width,
                mass_axis=x_mass
            )
            baseline_corrector.data = signal
            baseline_corrector.baseline_range()
            baseline_corrector.baseline_mean()
            
            baseline_range_indices = baseline_corrector.baseline_range_indices
            mean_value = baseline_corrector.mean_value
            corrected_signal = signal - mean_value
        else:
            als_lam = float(st.session_state.get("rempi_als_lam", 1e6))
            als_p = float(st.session_state.get("rempi_als_p", 0.01))
            als_baseline_curve = baseline_REMPI.als_baseline(signal, lam=als_lam, p=als_p)
            corrected_signal = signal - als_baseline_curve
        
        # Save to session state
        st.session_state["rempi_baseline_range_indices"] = baseline_range_indices
        st.session_state["rempi_baseline_mean"] = mean_value
        st.session_state["rempi_als_baseline_curve"] = als_baseline_curve
        st.session_state["rempi_baseline_method_used"] = _bl_method
        
        # Calculate mass range indices (around molecule mass)
        mass_range_min = molecule_mass - 10
        mass_range_max = molecule_mass + 10
        mass_range_indices = np.where((x_mass >= mass_range_min) & (x_mass <= mass_range_max))[0]
        
        # Tabs for interactive and static plots
        tab1, tab2 = st.tabs(["📈 Interactive plot (Plotly)", "📊 Static plot (Matplotlib)"])
        
        with tab1:
            # Create 2-layer Plotly subplot
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(f"Original - {plot_wavelength} nm", f"Baseline Corrected - {plot_wavelength} nm")
            )
            
            # Top subplot - Original data (zoomed to molecule mass region)
            fig.add_trace(go.Scatter(
                x=x_mass[mass_range_indices],
                y=signal[mass_range_indices],
                mode='lines',
                name='Original signal',
                line=dict(width=1, color='blue'),
                legendgroup="original"
            ), row=1, col=1)
            
            # Bottom subplot - Baseline corrected data
            fig.add_trace(go.Scatter(
                x=x_mass[mass_range_indices],
                y=corrected_signal[mass_range_indices],
                mode='lines',
                name='Baseline corrected',
                line=dict(width=1, color='black'),
                legendgroup="corrected"
            ), row=2, col=1)
            
            # Vertical line for molecule mass on both subplots
            fig.add_vline(
                x=molecule_mass,
                line_width=2,
                line_dash="solid",
                line_color="green",
                annotation_text=f"{molecule_name} ({molecule_mass} amu)",
                annotation_position="top",
                annotation_font_size=14,
                annotation_font_color="green",
                row=1
            )
            fig.add_vline(
                x=molecule_mass,
                line_width=2,
                line_dash="solid",
                line_color="green",
                row=2
            )
            
            # Baseline visualisation — depends on method
            if als_baseline_curve is not None:
                # ALS: show estimated baseline curve on top subplot
                fig.add_trace(go.Scatter(
                    x=x_mass[mass_range_indices],
                    y=als_baseline_curve[mass_range_indices],
                    mode='lines',
                    name='ALS baseline',
                    line=dict(width=2, color='red', dash='dash'),
                    legendgroup="baseline"
                ), row=1, col=1)
            else:
                # Manual: shaded region + horizontal mean line
                if len(baseline_range_indices) > 0:
                    fig.add_trace(go.Scatter(
                        x=[x_mass[baseline_range_indices[0]], x_mass[baseline_range_indices[-1]],
                           x_mass[baseline_range_indices[-1]], x_mass[baseline_range_indices[0]],
                           x_mass[baseline_range_indices[0]]],
                        y=[-0.001, -0.001, ymax_top, ymax_top, -0.001],
                        fill="toself",
                        fillcolor='rgba(211,211,211,0.3)',
                        line=dict(color='rgba(211,211,211,0.5)', width=1),
                        name='Baseline region',
                        showlegend=True,
                        legendgroup="baseline"
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=[x_mass[baseline_range_indices[0]], x_mass[baseline_range_indices[-1]],
                           x_mass[baseline_range_indices[-1]], x_mass[baseline_range_indices[0]],
                           x_mass[baseline_range_indices[0]]],
                        y=[-0.001, -0.001, ymax_bottom, ymax_bottom, -0.001],
                        fill="toself",
                        fillcolor='rgba(211,211,211,0.3)',
                        line=dict(color='rgba(211,211,211,0.5)', width=1),
                        name='Baseline region',
                        showlegend=False,
                        legendgroup="baseline"
                    ), row=2, col=1)
                
                if mean_value is not None:
                    fig.add_hline(
                        y=mean_value,
                        line_width=1,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Baseline mean: {mean_value:.4f}",
                        annotation_position="right",
                        row=1
                    )
            
            # Horizontal lines at y=0
            fig.add_hline(y=0, line_width=1, line_color="lime", row=1)
            fig.add_hline(y=0, line_width=1, line_color="lime", row=2)
            
            # Update layout
            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(
                    xanchor='right',
                    yanchor='top',
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='black',
                    borderwidth=1
                )
            )
            
            # Update x-axes - zoom to molecule mass region
            fig.update_xaxes(
                range=[molecule_mass - 5, molecule_mass + 5],
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor='black'
            )
            
            # Update y-axes
            fig.update_yaxes(
                title_text="Intensity (a.u.)",
                title_font=dict(size=14, color='black'),
                tickfont=dict(size=12, color='black'),
                showgrid=False,
                showline=True,
                linewidth=2,
                linecolor='black'
            )
            
            # Set y-axis ranges
            fig.update_yaxes(range=[-0.001, ymax_top], row=1, col=1)
            fig.update_yaxes(range=[-0.001, ymax_bottom], row=2, col=1)
            
            # Add x-axis title only to bottom subplot
            fig.update_xaxes(title_text="Mass (amu)", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Static matplotlib plot
            fig_mpl, axes = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)
            
            # Top: Original
            axes[0].plot(x_mass[mass_range_indices], signal[mass_range_indices], 'b-', linewidth=0.5, label='Original')
            axes[0].axvline(molecule_mass, color='green', linestyle='-', linewidth=1, alpha=0.7, label=f'{molecule_name} ({molecule_mass} amu)')
            if als_baseline_curve is not None:
                axes[0].plot(x_mass[mass_range_indices], als_baseline_curve[mass_range_indices], 'r--', linewidth=1.5, label='ALS baseline')
            else:
                if mean_value is not None:
                    axes[0].axhline(mean_value, color='red', linestyle='--', linewidth=1, label=f'Baseline mean: {mean_value:.4f}')
                if len(baseline_range_indices) > 0:
                    axes[0].fill_between(x_mass[baseline_range_indices], ymax_top, color='lightgray', alpha=0.5, label='Baseline region')
            axes[0].axhline(0, color='lime', linewidth=1)
            axes[0].set_ylim(-0.001, ymax_top)
            axes[0].set_xlim(molecule_mass - 5, molecule_mass + 5)
            axes[0].set_ylabel('Intensity (a.u.)')
            axes[0].set_title(f'Original - {plot_wavelength} nm')
            axes[0].legend(fontsize=8, loc='upper right')
            
            # Bottom: Baseline corrected
            axes[1].plot(x_mass[mass_range_indices], corrected_signal[mass_range_indices], 'k-', linewidth=0.5, label='Baseline corrected')
            axes[1].axvline(molecule_mass, color='green', linestyle='-', linewidth=1, alpha=0.7)
            if als_baseline_curve is None and len(baseline_range_indices) > 0:
                axes[1].fill_between(x_mass[baseline_range_indices], ymax_bottom, color='lightgray', alpha=0.5, label='Baseline region')
            axes[1].axhline(0, color='lime', linewidth=1)
            axes[1].set_ylim(-0.001, ymax_bottom)
            axes[1].set_xlim(molecule_mass - 5, molecule_mass + 5)
            axes[1].set_xlabel('Mass (amu)')
            axes[1].set_ylabel('Intensity (a.u.)')
            axes[1].set_title(f'Baseline Corrected - {plot_wavelength} nm')
            axes[1].legend(fontsize=8, loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig_mpl)
            
            # Add to Report button
            add_plot_to_report_button(
                fig_mpl, 
                f"REMPI Baseline Correction - {plot_wavelength} nm",
                key_suffix=f"rempi_bc_{plot_wavelength}",
                description=f"REMPI baseline correction for wavelength {plot_wavelength} nm"
            )
            
            plt.close(fig_mpl)
        
        if mean_value is not None:
            st.success(f"✅ Parameters registered! Baseline mean: {mean_value:.6f}")
        else:
            st.success("✅ Parameters registered! ALS baseline estimated.")

st.markdown("---")

# Apply baseline correction to full dataset
st.markdown("### Apply Baseline Correction to Full Dataset")

if st.button("📏 Apply baseline correction to all wavelengths", width='stretch'):
    
    if "rempi_x_mass" not in st.session_state:
        st.error("Please register parameters first by clicking the button above.", icon="🚫")
        st.stop()
    
    x_mass = st.session_state["rempi_x_mass"]
    _method_used = st.session_state.get("rempi_baseline_method_used", "Manual region")
    
    with st.spinner("Applying baseline correction to all wavelengths…"):
        baseline_corrector = baseline_REMPI(mass_axis=x_mass)
        
        if _method_used == "Automatic (ALS)":
            als_lam = float(st.session_state.get("rempi_als_lam", 1e6))
            als_p = float(st.session_state.get("rempi_als_p", 0.01))
            corrected_df = baseline_corrector.process_single_dataframe_als(
                compiled_dataframe, lam=als_lam, p=als_p
            )
        else:
            baseline_corrector.baseline_reference = st.session_state.get("rempi_baseline_reference", 98)
            baseline_corrector.interval = st.session_state.get("rempi_baseline_width", 3)
            corrected_df = baseline_corrector.process_single_dataframe(compiled_dataframe)
    
    # Save to session state
    st.session_state["rempi_baseline_corrected"] = corrected_df
    
    st.success(f"✅ Baseline correction applied to all wavelengths! (method: {_method_used})")

# Show results if available
if "rempi_baseline_corrected" in st.session_state:
    st.markdown("---")
    st.markdown("### Baseline-Corrected Data")
    
    corrected_df = st.session_state["rempi_baseline_corrected"]
    st.dataframe(corrected_df.head(50))
    st.caption(f"Showing first 50 rows of {len(corrected_df)} total rows")

    # ================================================================
    # Summed Spectrum (Baseline Corrected)
    # ================================================================
    st.markdown("### Summed Spectrum (Baseline Corrected)")

    if st.button("📊 Plot Summed Spectrum (Interactive)", width='stretch'):
        x_mass = st.session_state["rempi_x_mass"]
        summed_signal = corrected_df["Summed"].values
        molecule_mass = st.session_state.get("rempi_molecule_mass", 100)
        molecule_name = st.session_state.get("rempi_molecule_name", "")
        st.session_state["plot_summed_spectrum"] = {
            "x_mass": x_mass,
            "summed_signal": summed_signal,
            "molecule_mass": molecule_mass,
            "molecule_name": molecule_name,
        }

    if "plot_summed_spectrum" in st.session_state:
        plot_data = st.session_state["plot_summed_spectrum"]
        x_mass = plot_data["x_mass"]
        summed_signal = plot_data["summed_signal"]
        molecule_mass = plot_data["molecule_mass"]
        molecule_name = plot_data["molecule_name"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_mass, y=summed_signal,
            mode='lines', name='Summed spectrum',
            line=dict(width=1, color='blue'),
        ))
        fig.add_vline(
            x=molecule_mass, line_width=2, line_dash="solid", line_color="green",
            annotation_text=f"{molecule_name} ({molecule_mass} amu)",
            annotation_position="top",
        )
        fig.add_hline(y=0, line_width=1, line_color="lime")
        fig.update_layout(
            title="REMPI Summed Spectrum (Baseline Corrected)",
            xaxis_title="Mass (amu)", yaxis_title="Intensity (a.u.)",
            height=500, showlegend=True,
        )
        fig.update_xaxes(range=[molecule_mass - 20, molecule_mass + 20])
        st.plotly_chart(fig, use_container_width=True)

        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("💾 Save Plot as PNG", key="save_summed_spectrum"):
                import plotly.io as pio
                file_directory = st.session_state.get("rempi_file_directory", "")
                if file_directory:
                    output_path = Path(file_directory) / "output"
                    output_path.mkdir(parents=True, exist_ok=True)
                    filepath = output_path / "REMPI_summed_spectrum.png"
                    pio.write_image(fig, str(filepath), width=1200, height=600)
                    st.success(f"✅ Saved to `{filepath}`")
                else:
                    st.warning("No output directory set")
        with col_export2:
            if st.button("📊 Add to Report", key="report_summed_spectrum"):
                fig_mpl, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_mass, summed_signal, 'b-', linewidth=1)
                ax.axvline(molecule_mass, color='green', linestyle='--', linewidth=2,
                           label=f'{molecule_name} ({molecule_mass} amu)')
                ax.axhline(0, color='lime', linewidth=1)
                ax.set_xlabel('Mass (amu)', fontsize=12)
                ax.set_ylabel('Intensity (a.u.)', fontsize=12)
                ax.set_title('REMPI Summed Spectrum (Baseline Corrected)', fontsize=14)
                ax.set_xlim(molecule_mass - 20, molecule_mass + 20)
                ax.legend(); ax.grid(alpha=0.3)
                add_plot_to_report_button(
                    fig_mpl, "REMPI Summed Spectrum",
                    key_suffix="rempi_summed",
                    description="Baseline-corrected REMPI summed mass spectrum",
                )
                plt.close(fig_mpl)

    # ================================================================
    # 2D Action Spectrum (Wavelength vs Mass)
    # ================================================================
    st.markdown("---")
    st.markdown("### 2D Action Spectrum (Wavelength vs Mass)")
    st.caption("Shows which mass channels have transitions at which wavelengths")

    x_mass = st.session_state["rempi_x_mass"]
    st.markdown("#### Plot Range Settings")
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        mass_min_2d = st.number_input("Min mass (amu)", value=float(np.min(x_mass)), key="heatmap_mass_min_preset")
    with col_range2:
        mass_max_2d = st.number_input("Max mass (amu)", value=float(np.max(x_mass)), key="heatmap_mass_max_preset")

    col_heat_opt1, col_heat_opt2 = st.columns(2)
    with col_heat_opt1:
        heat_exclude_text = st.text_input(
            "Exclude masses from color scale (comma-separated)", value="",
            key="heat_exclude_masses",
            help="e.g. 18, 28, 32  —  these masses still appear but don't affect the colour range",
        )
        heat_exclude_tol = st.number_input(
            "Exclusion tolerance (±amu)", min_value=0.1, max_value=10.0,
            value=1.0, step=0.5, key="heat_exclude_tol",
        )
    with col_heat_opt2:
        heat_clip_pct = st.slider(
            "Percentile clipping", min_value=0.0, max_value=10.0, value=2.0, step=0.5,
            key="heat_clip_pct",
            help="Clip colour scale at this percentile from bottom and top (0 = no clipping)",
        )
        heat_log_scale = st.checkbox(
            "Log scale", value=False, key="heat_log_scale",
            help="Apply log₁₀ to intensities — useful when features span orders of magnitude",
        )

    if st.button("🗺️ Generate 2D Heatmap", width='stretch'):
        st.session_state["generate_2d_heatmap"] = True

    if st.session_state.get("generate_2d_heatmap", False):
        x_mass = st.session_state["rempi_x_mass"]
        wavelength_cols = [col for col in corrected_df.columns if col != 'Summed']

        if len(wavelength_cols) == 0:
            st.error("No wavelength data found")
        else:
            Z = corrected_df[wavelength_cols].T.values
            wavelengths = []
            for col in wavelength_cols:
                col_str = str(col)
                if col_str.startswith('bc_'):
                    col_str = col_str[3:]
                try:
                    wavelengths.append(float(col_str))
                except ValueError:
                    match = re.search(r'[\d.]+', col_str)
                    if match:
                        wavelengths.append(float(match.group()))
                    else:
                        st.warning(f"Could not parse wavelength from column: {col}")
            wavelengths = np.array(wavelengths)

            molecule_mass = st.session_state.get("rempi_molecule_mass", None)
            mass_indices = np.where((x_mass >= mass_min_2d) & (x_mass <= mass_max_2d))[0]

            if len(mass_indices) > 0:
                Z_filtered = Z[:, mass_indices]
                x_mass_filtered = x_mass[mass_indices]
                wl_min, wl_max = wavelengths.min(), wavelengths.max()

                _excl_tokens = re.split(r'[,\s]+', heat_exclude_text.strip()) if heat_exclude_text.strip() else []
                excluded_masses = []
                for tok in _excl_tokens:
                    try:
                        excluded_masses.append(float(tok))
                    except ValueError:
                        pass

                include_mask = np.ones(len(x_mass_filtered), dtype=bool)
                for em in excluded_masses:
                    include_mask &= np.abs(x_mass_filtered - em) > heat_exclude_tol

                Z_for_scale = Z_filtered[:, include_mask] if include_mask.any() else Z_filtered
                if heat_clip_pct > 0:
                    zmin_val = float(np.percentile(Z_for_scale, heat_clip_pct))
                    zmax_val = float(np.percentile(Z_for_scale, 100 - heat_clip_pct))
                else:
                    zmin_val = float(Z_for_scale.min())
                    zmax_val = float(Z_for_scale.max())

                Z_plot = Z_filtered.copy()
                cb_title = "Intensity (a.u.)"
                if heat_log_scale:
                    Z_plot = np.where(Z_plot > 0, np.log10(Z_plot), np.nan)
                    zmin_val = np.log10(zmin_val) if zmin_val > 0 else float(np.nanmin(Z_plot))
                    zmax_val = np.log10(zmax_val) if zmax_val > 0 else float(np.nanmax(Z_plot))
                    cb_title = "log₁₀ Intensity"

                fig_zoom = go.Figure(data=go.Heatmap(
                    z=Z_plot, x=x_mass_filtered, y=wavelengths,
                    colorscale='Hot', zmin=zmin_val, zmax=zmax_val,
                    colorbar=dict(title=cb_title),
                    hovertemplate='Wavelength: %{y:.2f} nm<br>Mass: %{x:.2f} amu<br>Intensity: %{z:.4f}<extra></extra>',
                ))
                if molecule_mass and mass_min_2d <= molecule_mass <= mass_max_2d:
                    fig_zoom.add_vline(x=molecule_mass, line_width=2, line_dash="dash", line_color="cyan")

                _title_suffix = ""
                if excluded_masses:
                    _title_suffix = f" | excl: {', '.join(f'{m:.0f}' for m in excluded_masses)} amu"
                fig_zoom.update_layout(
                    title=f"2D REMPI Action Spectrum (Mass: {mass_min_2d:.1f}-{mass_max_2d:.1f} amu, λ: {wl_min:.1f}-{wl_max:.1f} nm{_title_suffix})",
                    xaxis_title="Mass (amu)", yaxis_title="Wavelength (nm)",
                    height=600, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_zoom, use_container_width=True)

                col_exp_zoom1, col_exp_zoom2 = st.columns(2)
                with col_exp_zoom1:
                    if st.button("💾 Save Zoomed Heatmap as PNG", key="save_2d_zoom"):
                        import plotly.io as pio
                        file_directory = st.session_state.get("rempi_file_directory", "")
                        if file_directory:
                            output_path = Path(file_directory) / "output"
                            output_path.mkdir(parents=True, exist_ok=True)
                            filepath = output_path / f"REMPI_2D_heatmap_{mass_min_2d:.0f}-{mass_max_2d:.0f}amu_{wl_min:.0f}-{wl_max:.0f}nm.png"
                            pio.write_image(fig_zoom, str(filepath), width=1400, height=800)
                            st.success(f"✅ Saved to `{filepath}`")
                        else:
                            st.warning("No output directory set")
                with col_exp_zoom2:
                    if st.button("📊 Add Zoomed Heatmap to Report", key="report_2d_zoom"):
                        from matplotlib.colors import LogNorm
                        fig_mpl, ax = plt.subplots(figsize=(12, 6))
                        if heat_log_scale:
                            Z_mpl = np.where(Z_filtered > 0, Z_filtered, np.nan)
                            _vmin = 10**zmin_val if zmin_val != 0 else None
                            _vmax = 10**zmax_val if zmax_val != 0 else None
                            im = ax.pcolormesh(x_mass_filtered, wavelengths, Z_mpl,
                                               cmap='hot', shading='auto',
                                               norm=LogNorm(vmin=_vmin, vmax=_vmax))
                        else:
                            im = ax.pcolormesh(x_mass_filtered, wavelengths, Z_filtered,
                                               cmap='hot', shading='auto',
                                               vmin=zmin_val, vmax=zmax_val)
                        ax.set_xlabel('Mass (amu)', fontsize=12)
                        ax.set_ylabel('Wavelength (nm)', fontsize=12)
                        ax.set_title(f'2D REMPI Action Spectrum ({mass_min_2d:.1f}-{mass_max_2d:.1f} amu)', fontsize=14)
                        plt.colorbar(im, ax=ax, label=cb_title)
                        add_plot_to_report_button(
                            fig_mpl,
                            f"2D REMPI Action Spectrum ({mass_min_2d:.1f}-{mass_max_2d:.1f} amu)",
                            key_suffix="rempi_2d_zoom",
                            description=f"2D heatmap for mass range {mass_min_2d:.1f}-{mass_max_2d:.1f} amu",
                        )
                        plt.close(fig_mpl)

    # ================================================================
    # 1D Action Spectrum (Single Mass Channel)
    # ================================================================
    st.markdown("---")
    st.markdown("### 1D Action Spectrum (Single Mass Channel)")
    st.caption("Extract wavelength-dependent intensity for a specific mass")

    x_mass = st.session_state.get("rempi_x_mass")
    if x_mass is None:
        st.warning("Mass axis not found. Please register parameters first.")
    else:
        col_mass_select, col_tolerance = st.columns([3, 1])
        with col_mass_select:
            target_mass = st.number_input(
                "Select mass (amu)",
                min_value=float(x_mass.min()), max_value=float(x_mass.max()),
                value=float(st.session_state.get("rempi_molecule_mass", x_mass[len(x_mass)//2])),
                step=0.1, key="action_spectrum_mass",
            )
        with col_tolerance:
            mass_tolerance = st.number_input(
                "Mass tolerance (±amu)",
                min_value=0.1, max_value=10.0, value=0.5, step=0.1,
                key="action_spectrum_tolerance",
            )

        wavelength_cols = [col for col in corrected_df.columns if col != 'Summed' and str(col).startswith('bc_')]
        if len(wavelength_cols) == 0:
            wavelength_cols = [col for col in corrected_df.columns if col != 'Summed']

        if st.button("📊 Plot 1D Action Spectrum", width='stretch'):
            st.session_state["plot_1d_action"] = {
                "target_mass": target_mass,
                "mass_tolerance": mass_tolerance,
                "wavelength_cols": wavelength_cols,
            }

        if "plot_1d_action" in st.session_state:
            params = st.session_state["plot_1d_action"]
            target_mass = params["target_mass"]
            mass_tolerance = params["mass_tolerance"]
            wavelength_cols = params["wavelength_cols"]

            def _parse_wavelengths(cols):
                vals = []
                for c in cols:
                    s = str(c)
                    if s.startswith('bc_'):
                        s = s[3:]
                    try:
                        vals.append(float(s))
                    except ValueError:
                        m = re.search(r'[\d.]+', s)
                        if m:
                            vals.append(float(m.group()))
                return np.array(vals)

            mass_indices = np.where(np.abs(x_mass - target_mass) <= mass_tolerance)[0]

            if len(mass_indices) == 0:
                st.error(f"No mass points found within ±{mass_tolerance} amu of {target_mass} amu")
            else:
                avg_mass = np.mean(x_mass[mass_indices])
                present_cols = [c for c in wavelength_cols if c in corrected_df.columns]
                if len(present_cols) == 0:
                    st.error("No matching wavelength columns found in the corrected data.")
                    st.stop()
                Z = corrected_df[present_cols].to_numpy()
                intensities = Z[mass_indices, :].mean(axis=0).tolist()
                wavelengths = _parse_wavelengths(present_cols)

                fig_1d = go.Figure()
                fig_1d.add_trace(go.Scatter(
                    x=wavelengths, y=intensities,
                    mode='lines', name=f'Mass {target_mass:.1f} amu',
                    line=dict(width=2, color='blue'),
                ))
                fig_1d.update_layout(
                    title=f"Action Spectrum for m/z = {target_mass:.1f} ± {mass_tolerance} amu (actual: {avg_mass:.2f} amu)",
                    xaxis_title="Wavelength (nm)", yaxis_title="Ion Intensity (a.u.)",
                    height=400, showlegend=True, hovermode='x unified',
                )
                fig_1d.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_1d, use_container_width=True)
                st.info(f"**Statistics:** Peak intensity: {max(intensities):.4f} at {wavelengths[np.argmax(intensities)]:.2f} nm | Mean: {np.mean(intensities):.4f}")

                col_exp_1d1, col_exp_1d2 = st.columns(2)
                with col_exp_1d1:
                    if st.button("💾 Save 1D Action Spectrum as PNG", key="save_1d_action"):
                        import plotly.io as pio
                        file_directory = st.session_state.get("rempi_file_directory", "")
                        if file_directory:
                            output_path = Path(file_directory) / "output"
                            output_path.mkdir(parents=True, exist_ok=True)
                            filepath = output_path / f"REMPI_1D_action_spectrum_m{target_mass:.1f}.png"
                            pio.write_image(fig_1d, str(filepath), width=1200, height=500)
                            st.success(f"✅ Saved to `{filepath}`")
                        else:
                            st.warning("No output directory set")
                with col_exp_1d2:
                    if st.button("📊 Add 1D Action Spectrum to Report", key="report_1d_action"):
                        fig_mpl, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(wavelengths, intensities, 'b-', marker='o', linewidth=2, markersize=4)
                        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
                        ax.set_xlabel('Wavelength (nm)', fontsize=12)
                        ax.set_ylabel('Ion Intensity (a.u.)', fontsize=12)
                        ax.set_title(f'Action Spectrum for m/z = {target_mass:.1f} ± {mass_tolerance} amu', fontsize=14)
                        ax.grid(alpha=0.3)
                        add_plot_to_report_button(
                            fig_mpl,
                            f"1D Action Spectrum (m/z = {target_mass:.1f})",
                            key_suffix=f"rempi_1d_{target_mass:.0f}",
                            description=f"Wavelength-dependent ion yield for mass {target_mass:.1f} ± {mass_tolerance} amu",
                        )
                        plt.close(fig_mpl)

    # ================================================================
    # Ridge Plot (Multiple Mass Channels)
    # ================================================================
    st.markdown("---")
    st.markdown("### Ridge Plot (Multiple Mass Channels)")
    st.caption("Paste several masses to compare their action spectra in a stacked ridge plot.")

    x_mass = st.session_state.get("rempi_x_mass")
    if x_mass is None:
        st.warning("Mass axis not found. Please register parameters first.")
    else:
        col_rp1, col_rp2 = st.columns([3, 1])
        with col_rp1:
            masses_text = st.text_area(
                "Masses (one per line or comma-separated)",
                value=st.session_state.get("rempi_ridge_masses_text", ""),
                height=120, key="rempi_ridge_masses_text",
                help="e.g.  78, 102, 128  or one value per line",
            )
        with col_rp2:
            ridge_tol = st.number_input("Mass tolerance (±amu)", min_value=0.1, max_value=10.0, value=0.5, step=0.1, key="rempi_ridge_tol")
            ridge_spacing = st.number_input("Vertical spacing", min_value=0.1, max_value=5.0, value=0.7, step=0.1, key="rempi_ridge_spacing")

        col_wl1, col_wl2 = st.columns(2)
        with col_wl1:
            ridge_wl_min = st.number_input("Wavelength min (nm)", value=0.0, step=0.5, key="rempi_ridge_wl_min", help="Leave at 0 to use full range")
        with col_wl2:
            ridge_wl_max = st.number_input("Wavelength max (nm)", value=0.0, step=0.5, key="rempi_ridge_wl_max", help="Leave at 0 to use full range")

        ridge_normalize = st.checkbox("Normalize each trace to [0, 1]", value=True, key="rempi_ridge_normalize",
                                      help="Off = plot raw intensities (shared y-scale). On = each trace scaled independently.")

        if st.button("📈 Generate Ridge Plot", width='stretch', key="btn_ridge_plot"):
            raw_tokens = re.split(r'[,\s]+', masses_text.strip())
            parsed_masses = []
            for tok in raw_tokens:
                try:
                    parsed_masses.append(float(tok))
                except ValueError:
                    pass
            if len(parsed_masses) == 0:
                st.error("No valid masses found. Enter numbers separated by commas or newlines.")
            else:
                st.session_state["rempi_ridge_data"] = {
                    "masses": parsed_masses, "tolerance": ridge_tol, "spacing": ridge_spacing,
                    "wl_min": ridge_wl_min, "wl_max": ridge_wl_max, "normalize": ridge_normalize,
                }

        if "rempi_ridge_data" in st.session_state:
            rd = st.session_state["rempi_ridge_data"]
            _masses = rd["masses"]; _tol = rd["tolerance"]; _spacing = rd["spacing"]
            _wl_min = rd.get("wl_min", 0.0); _wl_max = rd.get("wl_max", 0.0)

            wl_cols = [c for c in corrected_df.columns if c != 'Summed' and str(c).startswith('bc_')]
            if len(wl_cols) == 0:
                wl_cols = [c for c in corrected_df.columns if c != 'Summed']

            def _parse_wl(cols):
                out = []
                for c in cols:
                    s = str(c)
                    if s.startswith('bc_'):
                        s = s[3:]
                    try:
                        out.append(float(s))
                    except ValueError:
                        m = re.search(r'[\d.]+', s)
                        if m:
                            out.append(float(m.group()))
                return np.array(out)

            present_cols = [c for c in wl_cols if c in corrected_df.columns]
            wavelengths_ridge = _parse_wl(present_cols)
            Z_all = corrected_df[present_cols].to_numpy()

            if _wl_min > 0 or _wl_max > 0:
                wl_lo = _wl_min if _wl_min > 0 else wavelengths_ridge.min()
                wl_hi = _wl_max if _wl_max > 0 else wavelengths_ridge.max()
                wl_mask = (wavelengths_ridge >= wl_lo) & (wavelengths_ridge <= wl_hi)
                wavelengths_ridge = wavelengths_ridge[wl_mask]
                Z_all = Z_all[:, wl_mask]

            labels = []; traces_y = []
            for m_val in _masses:
                idxs = np.where(np.abs(x_mass - m_val) <= _tol)[0]
                if len(idxs) == 0:
                    st.warning(f"No data points for mass {m_val:.1f} ± {_tol} amu — skipped")
                    continue
                traces_y.append(Z_all[idxs, :].mean(axis=0))
                labels.append(f"{m_val:.1f} amu")

            if len(traces_y) == 0:
                st.error("No valid mass channels found.")
            else:
                n_traces = len(traces_y)
                _do_norm = rd.get("normalize", True)
                raw_intensities = [np.nanmax(y) - np.nanmin(y) for y in traces_y]
                max_int = max(raw_intensities) if max(raw_intensities) > 0 else 1.0
                rel_intensities = [ri / max_int for ri in raw_intensities]

                if _do_norm:
                    norm_traces = []
                    for y in traces_y:
                        ymin, ymax = np.nanmin(y), np.nanmax(y)
                        ptp = ymax - ymin
                        norm_traces.append((y - ymin) / ptp if ptp > 0 else np.zeros_like(y))
                    annotated_labels = [f"{lbl}  (×{rel:.2f})" for lbl, rel in zip(labels, rel_intensities)]
                else:
                    global_max = max(np.nanmax(np.abs(y)) for y in traces_y) or 1.0
                    norm_traces = [y / global_max for y in traces_y]
                    annotated_labels = list(labels)

                colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                fig_ridge = go.Figure()
                for idx in range(n_traces - 1, -1, -1):
                    c = colors[idx % len(colors)]
                    offset = idx * _spacing
                    y_shifted = norm_traces[idx] + offset
                    fig_ridge.add_trace(go.Scatter(
                        x=np.concatenate([wavelengths_ridge, wavelengths_ridge[::-1]]),
                        y=np.concatenate([y_shifted, np.full(len(wavelengths_ridge), offset)]),
                        fill="toself", fillcolor=c, opacity=0.2,
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                    ))
                    fig_ridge.add_trace(go.Scatter(
                        x=wavelengths_ridge, y=y_shifted,
                        mode="lines", name=labels[idx], line=dict(color=c, width=1.5),
                    ))

                fig_ridge.update_layout(
                    xaxis_title="Wavelength (nm)",
                    yaxis=dict(tickvals=[i * _spacing + 0.5 for i in range(n_traces)], ticktext=annotated_labels, title=""),
                    title=f"REMPI Ridge Plot — {n_traces} mass channels",
                    height=max(400, n_traces * 60 + 100), showlegend=False, margin=dict(l=140),
                )
                st.plotly_chart(fig_ridge, use_container_width=True)

                fig_height = max(6, n_traces * 0.8 + 1)
                fig_mpl_r, ax_r = plt.subplots(figsize=(12, fig_height))
                for idx in range(n_traces):
                    c = colors[idx % len(colors)]
                    offset = idx * _spacing
                    y_shifted = norm_traces[idx] + offset
                    ax_r.fill_between(wavelengths_ridge, offset, y_shifted, color=c, alpha=0.2)
                    ax_r.plot(wavelengths_ridge, y_shifted, color=c, lw=1.5, label=labels[idx])
                ax_r.set_xlabel("Wavelength (nm)", fontsize=12)
                ax_r.set_yticks([i * _spacing + 0.5 for i in range(n_traces)])
                ax_r.set_yticklabels(annotated_labels, fontsize=9)
                ax_r.set_title(f"REMPI Plot - {n_traces} mass channels", fontsize=13, fontweight="bold")
                ax_r.grid(True, axis="x", alpha=0.3)
                fig_mpl_r.tight_layout()

                col_rp_exp1, col_rp_exp2 = st.columns(2)
                with col_rp_exp1:
                    if st.button("💾 Save Ridge Plot as PNG", key="save_ridge_png"):
                        file_directory = st.session_state.get("rempi_file_directory", "")
                        if file_directory:
                            import datetime
                            output_path = Path(file_directory) / "output"
                            output_path.mkdir(parents=True, exist_ok=True)
                            _mass_str = "-".join(f"{m:.0f}" for m in _masses[:6])
                            _ts = datetime.datetime.now().strftime("%H%M%S")
                            filepath = output_path / f"REMPI_ridge_plot_m{_mass_str}_{_ts}.png"
                            fig_mpl_r.savefig(str(filepath), dpi=300, bbox_inches='tight')
                            st.success(f"✅ Saved to `{filepath}`")
                        else:
                            st.warning("No output directory set")
                with col_rp_exp2:
                    add_plot_to_report_button(
                        fig_mpl_r, f"REMPI Ridge Plot ({n_traces} masses)",
                        key_suffix="rempi_ridge",
                        description=f"Ridge plot for masses: {', '.join(labels)}",
                    )
                plt.close(fig_mpl_r)

    # ================================================================
    # Export Data
    # ================================================================
    st.markdown("---")
    st.markdown("### Export Data")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        if st.button("💾 Export Full Baseline-Corrected Data", width='stretch'):
            file_directory = st.session_state.get("rempi_file_directory", "")
            if file_directory:
                output_path = Path(file_directory) / "output"
                output_path.mkdir(parents=True, exist_ok=True)
                filepath = output_path / "REMPI_baseline_corrected.csv"
                corrected_df.to_csv(filepath, index=True)
                st.success(f"✅ Exported to `{filepath}`")
            else:
                st.warning("No output directory set")

    with col_exp2:
        if st.button("🔬 Export for Peak Detection (4.2)", width='stretch'):
            x_mass = st.session_state.get("rempi_x_mass")
            if x_mass is None:
                st.error("Mass axis not found. Please register parameters first.")
            else:
                peak_detection_df = pd.DataFrame({
                    'x_mass': x_mass,
                    'baseline_corrected_signal_withoutIR': corrected_df["Summed"].values,
                })
                file_directory = st.session_state.get("rempi_file_directory", "")
                if file_directory:
                    output_path = Path(file_directory) / "output"
                    output_path.mkdir(parents=True, exist_ok=True)
                    filepath = output_path / "REMPI_for_peak_detection.csv"
                    peak_detection_df.to_csv(filepath, index=False)
                    st.success(f"✅ Exported to `{filepath}`")
                    st.info("📌 Upload this CSV in **Section 4.2 Peak Detection** → Upload CSV file")
                else:
                    st.warning("No output directory set")

