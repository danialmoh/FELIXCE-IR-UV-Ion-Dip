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
            
            # Add to Report button
            add_plot_to_report_button(
                fig_mpl, 
                f"REMPI Baseline Correction - {plot_wavelength} nm",
                key_suffix=f"rempi_bc_{plot_wavelength}",
                description=f"REMPI baseline correction for wavelength {plot_wavelength} nm"
            )
            
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
        
        # Store in session state for persistent display
        st.session_state["plot_summed_spectrum"] = {
            "x_mass": x_mass,
            "summed_signal": summed_signal,
            "molecule_mass": molecule_mass,
            "molecule_name": molecule_name
        }
    
    # Display plot if data exists in session state
    if "plot_summed_spectrum" in st.session_state:
        plot_data = st.session_state["plot_summed_spectrum"]
        x_mass = plot_data["x_mass"]
        summed_signal = plot_data["summed_signal"]
        molecule_mass = plot_data["molecule_mass"]
        molecule_name = plot_data["molecule_name"]
        
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
        
        # Export options
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
                # Convert Plotly to matplotlib for report
                import matplotlib.pyplot as plt
                fig_mpl, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_mass, summed_signal, 'b-', linewidth=1)
                ax.axvline(molecule_mass, color='green', linestyle='--', linewidth=2, label=f'{molecule_name} ({molecule_mass} amu)')
                ax.axhline(0, color='lime', linewidth=1)
                ax.set_xlabel('Mass (amu)', fontsize=12)
                ax.set_ylabel('Intensity (a.u.)', fontsize=12)
                ax.set_title('REMPI Summed Spectrum (Baseline Corrected)', fontsize=14)
                ax.set_xlim(molecule_mass - 20, molecule_mass + 20)
                ax.legend()
                ax.grid(alpha=0.3)
                
                add_plot_to_report_button(
                    fig_mpl,
                    "REMPI Summed Spectrum",
                    key_suffix="rempi_summed",
                    description="Baseline-corrected REMPI summed mass spectrum"
                )
                plt.close(fig_mpl)
    
    # 2D Heatmap: Wavelength vs Mass
    st.markdown("### 2D Action Spectrum (Wavelength vs Mass)")
    st.caption("Shows which mass channels have transitions at which wavelengths")
    
    # Mass range selector (always visible)
    x_mass = st.session_state["rempi_x_mass"]
    st.markdown("#### Plot Range Settings")
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        mass_min_2d = st.number_input("Min mass (amu)", value=float(np.min(x_mass)), key="heatmap_mass_min_preset")
    with col_range2:
        mass_max_2d = st.number_input("Max mass (amu)", value=float(np.max(x_mass)), key="heatmap_mass_max_preset")
    
    if st.button("🗺️ Generate 2D Heatmap", use_container_width=True):
        x_mass = st.session_state["rempi_x_mass"]
        
        # Get only wavelength columns (exclude 'Summed')
        wavelength_cols = [col for col in corrected_df.columns if col != 'Summed']
        
        # Store in session state for persistent display
        st.session_state["generate_2d_heatmap"] = True
    
    # Display heatmap if flag is set
    if st.session_state.get("generate_2d_heatmap", False):
        x_mass = st.session_state["rempi_x_mass"]
        wavelength_cols = [col for col in corrected_df.columns if col != 'Summed']
        
        # Debug info
        st.info(f"**Debug Info:**")
        st.write(f"- Total columns: {len(corrected_df.columns)}")
        st.write(f"- Wavelength columns: {len(wavelength_cols)}")
        st.write(f"- First 5 column names: {list(corrected_df.columns[:5])}")
        st.write(f"- Data shape: {corrected_df.shape}")
        st.write(f"- Mass range: {x_mass.min():.2f} - {x_mass.max():.2f} amu")
        st.write(f"- Data value range: {corrected_df[wavelength_cols].min().min():.6f} to {corrected_df[wavelength_cols].max().max():.6f}")
        
        if len(wavelength_cols) == 0:
            st.error("No wavelength data found")
        else:
            # Create Z matrix (intensity values)
            Z = corrected_df[wavelength_cols].T.values  # Transpose: rows=wavelength, cols=mass
            
            # Create mesh for plotting - extract wavelength values from column names
            # Column names may be like 'bc_616.02' or just '616.02'
            wavelengths = []
            for col in wavelength_cols:
                col_str = str(col)
                # Remove 'bc_' prefix if present
                if col_str.startswith('bc_'):
                    col_str = col_str[3:]
                try:
                    wavelengths.append(float(col_str))
                except ValueError:
                    # If still can't convert, try to extract numeric part
                    import re
                    match = re.search(r'[\d.]+', col_str)
                    if match:
                        wavelengths.append(float(match.group()))
                    else:
                        st.warning(f"Could not parse wavelength from column: {col}")
            wavelengths = np.array(wavelengths)
            
            # Create heatmap
            fig_2d = go.Figure(data=go.Heatmap(
                z=Z,
                x=x_mass,
                y=wavelengths,
                colorscale='Hot',
                colorbar=dict(title="Intensity (a.u.)"),
                hoverongaps=False,
                hovertemplate='Wavelength: %{y:.2f} nm<br>Mass: %{x:.2f} amu<br>Intensity: %{z:.4f}<extra></extra>'
            ))
            
            # Add molecule mass line if available
            molecule_mass = st.session_state.get("rempi_molecule_mass", None)
            if molecule_mass:
                fig_2d.add_vline(
                    x=molecule_mass,
                    line_width=2,
                    line_dash="dash",
                    line_color="cyan",
                    annotation_text=f"Parent: {molecule_mass} amu",
                    annotation_position="top right"
                )
            
            fig_2d.update_layout(
                title="2D REMPI Action Spectrum",
                xaxis_title="Mass (amu)",
                yaxis_title="Wavelength (nm)",
                height=600,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
            )
            
            st.plotly_chart(fig_2d, use_container_width=True)
            
            # Export options for full heatmap
            col_exp_2d1, col_exp_2d2 = st.columns(2)
            with col_exp_2d1:
                if st.button("💾 Save Full Heatmap as PNG", key="save_2d_full"):
                    import plotly.io as pio
                    file_directory = st.session_state.get("rempi_file_directory", "")
                    if file_directory:
                        output_path = Path(file_directory) / "output"
                        output_path.mkdir(parents=True, exist_ok=True)
                        filepath = output_path / "REMPI_2D_heatmap_full.png"
                        pio.write_image(fig_2d, str(filepath), width=1400, height=800)
                        st.success(f"✅ Saved to `{filepath}`")
                    else:
                        st.warning("No output directory set")
            with col_exp_2d2:
                if st.button("📊 Add Full Heatmap to Report", key="report_2d_full"):
                    # Convert to matplotlib
                    import matplotlib.pyplot as plt
                    fig_mpl, ax = plt.subplots(figsize=(12, 6))
                    im = ax.pcolormesh(x_mass, wavelengths, Z, cmap='hot', shading='auto')
                    ax.set_xlabel('Mass (amu)', fontsize=12)
                    ax.set_ylabel('Wavelength (nm)', fontsize=12)
                    ax.set_title('2D REMPI Action Spectrum', fontsize=14)
                    plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
                    
                    add_plot_to_report_button(
                        fig_mpl,
                        "2D REMPI Action Spectrum (Full)",
                        key_suffix="rempi_2d_full",
                        description="2D wavelength vs mass heatmap showing resonant transitions"
                    )
                    plt.close(fig_mpl)
            
            # Use preset mass range for zoomed view
            st.markdown("#### Zoomed View (using range settings above)")
            
            # Filter by mass range using preset values
            mass_indices = np.where((x_mass >= mass_min_2d) & (x_mass <= mass_max_2d))[0]
            
            if len(mass_indices) > 0:
                Z_filtered = Z[:, mass_indices]
                x_mass_filtered = x_mass[mass_indices]
                
                fig_zoom = go.Figure(data=go.Heatmap(
                    z=Z_filtered,
                    x=x_mass_filtered,
                    y=wavelengths,
                    colorscale='Hot',
                    colorbar=dict(title="Intensity (a.u.)"),
                    hovertemplate='Wavelength: %{y:.2f} nm<br>Mass: %{x:.2f} amu<br>Intensity: %{z:.4f}<extra></extra>'
                ))
                
                if molecule_mass and mass_min_2d <= molecule_mass <= mass_max_2d:
                    fig_zoom.add_vline(
                        x=molecule_mass,
                        line_width=2,
                        line_dash="dash",
                        line_color="cyan"
                    )
                
                fig_zoom.update_layout(
                    title=f"2D REMPI Action Spectrum (Mass: {mass_min_2d:.1f}-{mass_max_2d:.1f} amu)",
                    xaxis_title="Mass (amu)",
                    yaxis_title="Wavelength (nm)",
                    height=600,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False)
                )
                
                st.plotly_chart(fig_zoom, use_container_width=True)
                
                # Export options for zoomed heatmap
                col_exp_zoom1, col_exp_zoom2 = st.columns(2)
                with col_exp_zoom1:
                    if st.button("💾 Save Zoomed Heatmap as PNG", key="save_2d_zoom"):
                        import plotly.io as pio
                        file_directory = st.session_state.get("rempi_file_directory", "")
                        if file_directory:
                            output_path = Path(file_directory) / "output"
                            output_path.mkdir(parents=True, exist_ok=True)
                            filepath = output_path / f"REMPI_2D_heatmap_zoom_{mass_min_2d:.0f}-{mass_max_2d:.0f}amu.png"
                            pio.write_image(fig_zoom, str(filepath), width=1400, height=800)
                            st.success(f"✅ Saved to `{filepath}`")
                        else:
                            st.warning("No output directory set")
                with col_exp_zoom2:
                    if st.button("📊 Add Zoomed Heatmap to Report", key="report_2d_zoom"):
                        # Convert to matplotlib
                        import matplotlib.pyplot as plt
                        fig_mpl, ax = plt.subplots(figsize=(12, 6))
                        im = ax.pcolormesh(x_mass_filtered, wavelengths, Z_filtered, cmap='hot', shading='auto')
                        ax.set_xlabel('Mass (amu)', fontsize=12)
                        ax.set_ylabel('Wavelength (nm)', fontsize=12)
                        ax.set_title(f'2D REMPI Action Spectrum ({mass_min_2d:.1f}-{mass_max_2d:.1f} amu)', fontsize=14)
                        plt.colorbar(im, ax=ax, label='Intensity (a.u.)')
                        
                        add_plot_to_report_button(
                            fig_mpl,
                            f"2D REMPI Action Spectrum (Zoom: {mass_min_2d:.1f}-{mass_max_2d:.1f} amu)",
                            key_suffix="rempi_2d_zoom",
                            description=f"Zoomed 2D heatmap for mass range {mass_min_2d:.1f}-{mass_max_2d:.1f} amu"
                        )
                        plt.close(fig_mpl)
    
    # 1D Action Spectrum for specific mass channel
    st.markdown("---")
    st.markdown("### 1D Action Spectrum (Single Mass Channel)")
    st.caption("Extract wavelength-dependent intensity for a specific mass")
    
    # Get mass axis from session state
    x_mass = st.session_state.get("rempi_x_mass")
    if x_mass is None:
        st.warning("Mass axis not found. Please register parameters first.")
    else:
        col_mass_select, col_tolerance = st.columns([3, 1])
        with col_mass_select:
            target_mass = st.number_input(
                "Select mass (amu)", 
                min_value=float(x_mass.min()),
                max_value=float(x_mass.max()),
                value=float(st.session_state.get("rempi_molecule_mass", x_mass[len(x_mass)//2])),
                step=0.1,
                key="action_spectrum_mass"
            )
        with col_tolerance:
            mass_tolerance = st.number_input(
                "Mass tolerance (±amu)",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                step=0.1,
                key="action_spectrum_tolerance"
            )
        
        # Get wavelength columns
        wavelength_cols = [col for col in corrected_df.columns if col != 'Summed']
        
        if st.button("📊 Plot 1D Action Spectrum", use_container_width=True):
            # Store parameters in session state
            st.session_state["plot_1d_action"] = {
                "target_mass": target_mass,
                "mass_tolerance": mass_tolerance,
                "wavelength_cols": wavelength_cols
            }
        
        # Display 1D action spectrum if parameters are set
        if "plot_1d_action" in st.session_state:
            params = st.session_state["plot_1d_action"]
            target_mass = params["target_mass"]
            mass_tolerance = params["mass_tolerance"]
            wavelength_cols = params["wavelength_cols"]
            
            # Parse wavelengths from column names
            wavelengths = []
            for col in wavelength_cols:
                col_str = str(col)
                if col_str.startswith('bc_'):
                    col_str = col_str[3:]
                try:
                    wavelengths.append(float(col_str))
                except ValueError:
                    import re
                    match = re.search(r'[\d.]+', col_str)
                    if match:
                        wavelengths.append(float(match.group()))
            wavelengths = np.array(wavelengths)
            
            # Find mass indices within tolerance
            mass_indices = np.where(np.abs(x_mass - target_mass) <= mass_tolerance)[0]
            
            if len(mass_indices) == 0:
                st.error(f"No mass points found within ±{mass_tolerance} amu of {target_mass} amu")
            else:
                # Average signal over mass tolerance window
                avg_mass = np.mean(x_mass[mass_indices])
                
                # Extract intensity at each wavelength for this mass range
                intensities = []
                for col in wavelength_cols:
                    intensity = np.mean(corrected_df[col].values[mass_indices])
                    intensities.append(intensity)
                
                # Create 1D plot
                fig_1d = go.Figure()
                
                fig_1d.add_trace(go.Scatter(
                    x=wavelengths,
                    y=intensities,
                    mode='lines+markers',
                    name=f'Mass {target_mass:.1f} amu',
                    line=dict(width=2, color='blue'),
                    marker=dict(size=6)
                ))
                
                fig_1d.update_layout(
                    title=f"Action Spectrum for m/z = {target_mass:.1f} ± {mass_tolerance} amu (actual: {avg_mass:.2f} amu)",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Ion Intensity (a.u.)",
                    height=400,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                # Add zero line
                fig_1d.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                
                st.plotly_chart(fig_1d, use_container_width=True)
                
                # Show stats
                st.info(f"**Statistics:** Peak intensity: {max(intensities):.4f} at {wavelengths[np.argmax(intensities)]:.2f} nm | Mean: {np.mean(intensities):.4f}")
                
                # Export options for 1D action spectrum
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
                        # Convert to matplotlib
                        import matplotlib.pyplot as plt
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
                            description=f"Wavelength-dependent ion yield for mass {target_mass:.1f} ± {mass_tolerance} amu"
                        )
                        plt.close(fig_mpl)
    
    # Export buttons
    st.markdown("### Export Data")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.button("💾 Export Full Baseline-Corrected Data", use_container_width=True):
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
        if st.button("🔬 Export for Peak Detection (4.2)", use_container_width=True):
            x_mass = st.session_state.get("rempi_x_mass")
            if x_mass is None:
                st.error("Mass axis not found. Please register parameters first.")
            else:
                # Create Peak Detection compatible format
                peak_detection_df = pd.DataFrame({
                    'x_mass': x_mass,
                    'baseline_corrected_signal_withoutIR': corrected_df["Summed"].values
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
