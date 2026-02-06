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

from packages.BaselineCorrection_REMPI import baseline_REMPI

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
    st.session_state["rempi_baseline_reference"] = float(st.text_input(
        "Start of baseline (amu)", 
        value=st.session_state.get("rempi_baseline_reference", defaults.get("baseline_reference", 98))
    ))
    st.session_state["rempi_baseline_width"] = float(st.text_input(
        "Width of baseline (amu)", 
        value=st.session_state.get("rempi_baseline_width", defaults.get("baseline_width", 3))
    ))
    
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
    if st.button("✨ Register parameters and preview baseline!", use_container_width=True):
        # Get parameters
        t_off = st.session_state.get("rempi_t_off", 58)
        alpha = st.session_state.get("rempi_alpha", 7.6987e-7)
        molecule_mass = st.session_state.get("rempi_molecule_mass", 100)
        molecule_name = st.session_state.get("rempi_molecule_name", "")
        baseline_reference = st.session_state.get("rempi_baseline_reference", 98)
        baseline_width = st.session_state.get("rempi_baseline_width", 3)
        plot_wavelength = st.session_state.get("rempi_plot_wavelength", wavelengths[0])
        ymax_top = st.session_state.get("rempi_ymax_top", 0.1)
        ymax_bottom = st.session_state.get("rempi_ymax_bottom", 0.1)
        
        # Generate mass axis
        x_counts = np.linspace(1, dataset_length, int(dataset_length))
        x_mass = alpha * (x_counts - t_off) ** 2
        st.session_state["rempi_x_mass"] = x_mass
        
        # Get signal for selected wavelength
        if plot_wavelength in compiled_dataframe.columns:
            signal = compiled_dataframe[plot_wavelength].values
        else:
            signal = compiled_data[plot_wavelength].iloc[:, 0].values
        
        # Create baseline corrector
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
        
        # Apply correction
        corrected_signal = signal - mean_value
        
        # Save to session state
        st.session_state["rempi_baseline_range_indices"] = baseline_range_indices
        st.session_state["rempi_baseline_mean"] = mean_value
        
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
            
            # Baseline range (filled area) for top subplot
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
                
                # Baseline range for bottom subplot
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
            
            # Horizontal baseline mean line on top plot
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
            axes[0].axhline(mean_value, color='red', linestyle='--', linewidth=1, label=f'Baseline mean: {mean_value:.4f}')
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
            plt.close(fig_mpl)
        
        st.success(f"✅ Parameters registered! Baseline mean: {mean_value:.6f}")

st.markdown("---")

# Apply baseline correction to full dataset
st.markdown("### Apply Baseline Correction to Full Dataset")

if st.button("📏 Apply baseline correction to all wavelengths", use_container_width=True):
    
    if "rempi_x_mass" not in st.session_state:
        st.error("Please register parameters first by clicking the button above.", icon="🚫")
        st.stop()
    
    x_mass = st.session_state["rempi_x_mass"]
    baseline_reference = st.session_state.get("rempi_baseline_reference", 98)
    baseline_width = st.session_state.get("rempi_baseline_width", 3)
    
    # Create baseline corrector
    baseline_corrector = baseline_REMPI(
        baseline_reference=baseline_reference,
        interval=baseline_width,
        mass_axis=x_mass
    )
    
    # Apply to the compiled DataFrame
    corrected_df = baseline_corrector.process_single_dataframe(compiled_dataframe)
    
    # Save to session state
    st.session_state["rempi_baseline_corrected"] = corrected_df
    
    st.success("✅ Baseline correction applied to all wavelengths!")

# Show results if available
if "rempi_baseline_corrected" in st.session_state:
    st.markdown("---")
    st.markdown("### Baseline-Corrected Data")
    
    corrected_df = st.session_state["rempi_baseline_corrected"]
    st.dataframe(corrected_df.head(50))
    st.caption(f"Showing first 50 rows of {len(corrected_df)} total rows")
    
    # Plot summed spectrum with Plotly
    st.markdown("### Summed Spectrum (Baseline Corrected)")
    
    if st.button("📊 Plot Summed Spectrum (Interactive)", use_container_width=True):
        x_mass = st.session_state["rempi_x_mass"]
        summed_signal = corrected_df["Summed"].values
        molecule_mass = st.session_state.get("rempi_molecule_mass", 100)
        molecule_name = st.session_state.get("rempi_molecule_name", "")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_mass,
            y=summed_signal,
            mode='lines',
            name='Summed spectrum',
            line=dict(width=1, color='blue')
        ))
        
        fig.add_vline(
            x=molecule_mass,
            line_width=2,
            line_dash="solid",
            line_color="green",
            annotation_text=f"{molecule_name} ({molecule_mass} amu)",
            annotation_position="top"
        )
        
        fig.add_hline(y=0, line_width=1, line_color="lime")
        
        fig.update_layout(
            title="REMPI Summed Spectrum (Baseline Corrected)",
            xaxis_title="Mass (amu)",
            yaxis_title="Intensity (a.u.)",
            height=500,
            showlegend=True
        )
        
        # Zoom to molecule mass region by default
        fig.update_xaxes(range=[molecule_mass - 20, molecule_mass + 20])
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Export button
    st.markdown("### Export Data")
    if st.button("💾 Export Baseline-Corrected Data to CSV", use_container_width=True):
        file_directory = st.session_state.get("rempi_file_directory", "")
        if file_directory:
            output_path = Path(file_directory) / "output"
            output_path.mkdir(parents=True, exist_ok=True)
            filepath = output_path / "REMPI_baseline_corrected.csv"
            corrected_df.to_csv(filepath, index=True)
            st.success(f"✅ Exported to `{filepath}`")
        else:
            st.error("Please set an output directory first (Step 8.0).")
