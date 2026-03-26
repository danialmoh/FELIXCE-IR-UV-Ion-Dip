import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from packages.BaselineCorrection_v2 import *
# from packages.BaselineCorrection import mass_range, baseline_new

from packages.utils import require_state
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

if not require_state(
    [
        "compiled_data",
        "unique_wavenumbers",
        "element1",
        "element2",
        "element3",
        "mass_element1",
        "mass_element2",
        "mass_element3",
        "charge_state",
        "x_mass",
    ],
    section="2.0 Baseline correction",
    hint="Import data (1.0) and register experiment parameters (1.3) before running this step.",
):
    st.stop()

# Import variables from defaults.ini
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'./.streamlit/defaults.ini'  # or provide full path
    defaults= {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            # Update defaults with values from file
            defaults["n_element1"] = config.getint('Complex Parameters', 'n_element1')
            defaults["n_element2"] = config.getint('Complex Parameters', 'n_element2')
            defaults["n_element3"] = config.getint('Complex Parameters', 'n_element3')
            defaults["baseline_reference"] = config.getfloat('Baseline Parameters', 'baseline_reference')
            defaults["baseline_width"] = config.getfloat('Baseline Parameters', 'baseline_width')
            defaults["baseline_method"] = config.get('Baseline Parameters', 'baseline_method')
            defaults["baseline_ymax_top"] = config.getfloat('Baseline Parameters', 'baseline_ymax_top')
            defaults["baseline_ymax_bottom"] = config.getfloat('Baseline Parameters', 'baseline_ymax_bottom')
            defaults["plot_columnIndex_withoutIR"] = config.getint('Plot Parameters', 'columnIndex_withoutIR')
            defaults["plot_columnIndex_withIR"] = config.getint('Plot Parameters', 'columnIndex_withIR')
            defaults["plot_wavenumber"] = config.getfloat('Plot Parameters', 'plot_wavenumber')            
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()

# Import variables from session_state
file_directory = st.session_state.get("file_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
element1 = st.session_state.get("element1", None)
element2 = st.session_state.get("element2", None)
element3 = st.session_state.get("element3", None)
mass_element1 = st.session_state.get("mass_element1", None)
mass_element2 = st.session_state.get("mass_element2", None)
mass_element3 = st.session_state.get("mass_element3", None)
charge_state = st.session_state.get("charge_state", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)

col1, col2, col3 = st.columns([1, 0.1, 3])  # col2 is spacing

with col1:
    st.markdown("#### Complex parameters")
    st.session_state["n_element1"] = st.number_input(f"Size of {element1}", value=st.session_state.get("n_element1", defaults.get("n_element1", None)))
    st.session_state["n_element2"] = st.number_input(f"Size of {element2}", value=st.session_state.get("n_element2", defaults.get("n_element2", None)))
    st.session_state["n_element3"] = st.number_input(f"Size of {element3}", value=st.session_state.get("n_element3", defaults.get("n_element3", None)))
    
    st.markdown("#### Baseline parameters")
    st.session_state["baseline_reference"] = float(
        st.text_input("Start of baseline in amu", value=st.session_state.get("baseline_reference", defaults.get("baseline_reference", None)))
    )
    st.session_state["baseline_width"] = float(
        st.text_input("Width of baseline in amu", value=st.session_state.get("baseline_width", defaults.get("baseline_width", None)))
    )
    baseline_method_options = ["Mean Subtraction", "iarpls", "aspls", "fabc"]
    st.session_state["baseline_method"] = st.selectbox(
        "Select baseline correction method",
        options=baseline_method_options,
        index= baseline_method_options.index(st.session_state.get("baseline_method", defaults.get("baseline_method", 0))) if st.session_state.get("baseline_method") in baseline_method_options else 0,
        help="**Mean Subtraction**: Simple - subtracts average of baseline region. **iarpls**: Improved arPLS - better for noisy data with close peaks. **aspls**: Adaptive smoothing - handles variable baseline shapes. **fabc**: Fully automatic - no parameter tuning needed."
    )
    # Show additional parameter boxes based on the selected method
    if st.session_state["baseline_method"].lower() == "iarpls":
         with st.expander("✨ iarpls parameters", expanded=True):
              st.info("💡 **iarpls** (Improved arPLS) - Upgraded algorithm that avoids overestimating baseline under small peaks in noisy data. Best for closely-spaced peaks.")
              st.session_state["iarpls_lam"] = st.number_input(
                  "Lambda (λ) for iarpls", 
                  value=1e6,
                  min_value=1e3,
                  max_value=1e9,
                  step=1e5,
                  format="%.0e",
                  help="**Smoothness parameter**: For ~60k data points, start with 1e6. Higher (1e7-1e8) = smoother, ignores narrow peaks. Lower (1e5) = more flexible, follows signal closely. If baseline cuts through peaks, INCREASE this."
              )
    elif st.session_state["baseline_method"].lower() == "aspls":
         with st.expander("✨ aspls parameters", expanded=True):
              st.info("💡 **aspls** (Adaptive Smoothing PLS) - Adapts smoothing based on local signal characteristics. Good for variable baseline shapes.")
              st.session_state["aspls_lam"] = st.number_input(
                  "Lambda (λ) for aspls", 
                  value=1e6,
                  min_value=1e3,
                  max_value=1e9,
                  step=1e5,
                  format="%.0e",
                  help="**Smoothness parameter**: Similar to iarpls. Start with 1e6 for your data size. Increase if baseline pulls into peaks."
              )
    elif st.session_state["baseline_method"].lower() == "fabc":
         with st.expander("✨ fabc parameters (mostly automatic)", expanded=True):
              st.info("💡 **fabc** (Fully Automatic Baseline Correction) - Uses wavelet transform to identify baseline points automatically. Minimal tuning needed!")
              st.session_state["fabc_lam"] = st.number_input(
                  "Lambda (λ) for fabc (optional override)", 
                  value=1e6,
                  min_value=1e3,
                  max_value=1e9,
                  step=1e5,
                  format="%.0e",
                  help="**Optional smoothness override**: fabc auto-selects this internally. Only change if automatic results are poor. Default 1e6 is usually fine."
              )
              st.session_state["fabc_scale"] = st.number_input(
                  "Scale parameter for wavelet detection",
                  value=None,
                  min_value=1,
                  max_value=100,
                  step=1,
                  help="**Wavelet scale**: Controls peak width detection. Leave as None for automatic. Only adjust if small peaks are missed (decrease) or noise is detected as peaks (increase)."
              )

with col3:
    st.markdown("#### Plot parameters")
    available_wavenumbers = sorted(compiled_data.keys())
    default_idx = 0
    if "plot_wavenumber" in st.session_state and st.session_state["plot_wavenumber"] in available_wavenumbers:
        default_idx = available_wavenumbers.index(st.session_state["plot_wavenumber"])
    st.session_state["plot_wavenumber"] = st.selectbox(
        "Wavenumber to check plots",
        options=available_wavenumbers,
        index=default_idx,
        format_func=lambda x: f"{x:.2f}",
    )
        
    st.session_state["baseline_ymax_top"] = float(st.text_input("Maximum y-value for top plot", value=st.session_state.get("baseline_ymax_top", defaults.get("baseline_ymax_top", None))))
    st.session_state["baseline_ymax_bottom"] = float(st.text_input("Maximum y-value for bottom plot", value=st.session_state.get("baseline_ymax_bottom", defaults.get("baseline_ymax_bottom", None))))
    st.session_state["plot_columnIndex_withoutIR"] = st.number_input("Column index for signal without IR", value=st.session_state.get("plot_columnIndex_withoutIR", defaults.get("plot_columnIndex_withoutIR", None)))
    st.session_state["plot_columnIndex_withIR"] = st.number_input("Column index for signal with IR", value=st.session_state.get("plot_columnIndex_withIR", defaults.get("plot_columnIndex_withIR", None)))

if st.button("✨ Register parameters and make plot!"):
    # Retrieve parameters
    element1 = st.session_state.get("element1", None)
    element2 = st.session_state.get("element2", None)
    element3 = st.session_state.get("element3", None)
    mass_element1 = st.session_state.get("mass_element1", None)
    mass_element2 = st.session_state.get("mass_element2", None)
    mass_element3 = st.session_state.get("mass_element3", None)
    n_element1 = st.session_state.get("n_element1", None)
    n_element2 = st.session_state.get("n_element2", None)
    n_element3 = st.session_state.get("n_element3", None)
    charge_state = st.session_state.get("charge_state", None)
    baseline_reference = st.session_state.get("baseline_reference", None)
    baseline_width = st.session_state.get("baseline_width", None)
    plot_wavenumber = st.session_state.get("plot_wavenumber", None)
    baseline_ymax_top = st.session_state.get("baseline_ymax_top", None)
    baseline_ymax_bottom = st.session_state.get("baseline_ymax_bottom", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    key = round(plot_wavenumber, 2)#this is for when no rounding is happening, remove when rounding

    
    # Get complex properties (for plotting purposes)
    complex_name, mass_complex, mass_range_indices = mass_range(
        n_element1, n_element2, n_element3, element1, element2, element3,
        mass_element1, mass_element2, mass_element3,
        charge_state, x_mass
    )
    
    # Instantiate the baseline correction object using the new class, passing extra parameters
    baseline_correction = baseline(
        baseline_reference=baseline_reference,
        interval=baseline_width,
        # wavenumber=check_wavenumber,##this is for when you rounded the wavumnumbers
        wavenumber=key,#this is for when no rounding is happening
        column_withoutIR=compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
        column_withIR=compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR],
        data_withoutIR=compiled_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
        data_withIR=compiled_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
        mass_range=x_mass,
        method=baseline_method,
        iarpls_lam=st.session_state.get("iarpls_lam", 1e6),
        aspls_lam=st.session_state.get("aspls_lam", 1e6),
        fabc_lam=st.session_state.get("fabc_lam", 1e6),
        fabc_scale=st.session_state.get("fabc_scale", None)
    )
    
    # For Mean Subtraction, compute baseline range & mean first
    if baseline_method == "Mean Subtraction":
        baseline_range_indices = baseline_correction.baseline_range()
        baseline_correction.baseline_mean()
    else:
        # For pybaselines methods, baseline range is not used (full range)
        baseline_range_indices = np.arange(len(x_mass))
    
    # Perform baseline correction (handles all methods internally)
    baseline_corrected_data = baseline_correction.baseline_correction()

    # Save variables into session_state for later use
    st.session_state["complex"] = complex_name
    st.session_state["mass_complex"] = mass_complex
    st.session_state["mass_range_indices"] = mass_range_indices
    st.session_state["baseline_range_indices"] = baseline_range_indices

    tab1, tab2 = st.tabs(["📈 Interactive plot with plotly", "📈 Static plot with matplotlib"])

    with tab1:
        # Create 2-layer Plotly subplot with synchronized axes
        fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        shared_yaxes=True,  # Synchronize y-axes for same intensity scale
                        vertical_spacing=0.08)

        # Top subplot - Raw data traces
        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=compiled_data[plot_wavenumber].iloc[mass_range_indices, plot_columnIndex_withoutIR],
            mode='lines',
            name=compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
            line=dict(width=3, color='blue'),
            legendgroup="raw"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=compiled_data[plot_wavenumber].iloc[mass_range_indices, plot_columnIndex_withIR],
            mode='lines',
            name=compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR],
            line=dict(width=3, color="orange"),
            legendgroup="raw"
        ), row=1, col=1)

        # Bottom subplot - Baseline corrected data
        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=baseline_corrected_data.iloc[mass_range_indices, 0],
            mode='lines',
            name="baseline corrected signal without IR",
            line=dict(width=3, color='black'),
            legendgroup="corrected"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=baseline_corrected_data.iloc[mass_range_indices, 1],
            mode='lines',
            name="baseline corrected signal with IR",
            line=dict(width=3, color='red'),
            legendgroup="corrected"
        ), row=2, col=1)

        # Vertical lines for mass complex on both subplots
        fig.add_vline(x=mass_complex, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color="green",
                    annotation_text=f"{complex_name}",
                    annotation_position="top",
                    annotation_font_size=20,
                    annotation_font_color="black",
                    row=1)

        fig.add_vline(x=mass_complex, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color="green",
                    row=2)

        # Baseline range (filled area) - only show for Mean Subtraction
        if baseline_method == "Mean Subtraction":
            # Top subplot baseline range
            fig.add_trace(go.Scatter(
                x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1], 
                x_mass[baseline_range_indices][-1], x_mass[baseline_range_indices][0], 
                x_mass[baseline_range_indices][0]],
                y=[-0.001, -0.001, baseline_ymax_top, baseline_ymax_top, -0.001],
                fill="toself",
                fillcolor='rgba(211,211,211,0.3)',
                line=dict(color='rgba(211,211,211,0.5)', width=1),
                name='baseline range',
                showlegend=True,
                legendgroup="baseline"
            ), row=1, col=1)

            # Bottom subplot baseline range
            fig.add_trace(go.Scatter(
                x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1], 
                x_mass[baseline_range_indices][-1], x_mass[baseline_range_indices][0], 
                x_mass[baseline_range_indices][0]],
                y=[-0.001, -0.001, baseline_ymax_bottom, baseline_ymax_bottom, -0.001],
                fill="toself",
                fillcolor='rgba(211,211,211,0.3)',
                line=dict(color='rgba(211,211,211,0.5)', width=1),
                name='baseline range',
                showlegend=False,  # Don't show duplicate legend
                legendgroup="baseline"
            ), row=2, col=1)

        # Horizontal lines at y=0 for both subplots
        fig.add_hline(y=0, 
                    line_width=1, 
                    line_color="lime",
                    row=1)
        
        fig.add_hline(y=0, 
                    line_width=1, 
                    line_color="lime",
                    row=2)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                # x=1.0,                     # Position at far right
                # y=0.5,                     # Position at top
                xanchor='right',           # Anchor point for x
                yanchor='top',             # Anchor point for y
                font=dict(size=14),        # Legend font size
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        # Update x-axes
        fig.update_xaxes(
            range=[mass_complex-5, mass_complex+5],
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=16, color='black'),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False,
            row=2, col=1  # Only show x-axis title on bottom plot
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Intensity (a.u.)",
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=16, color='black'),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )

        # Set synchronized y-axis range for both subplots
        # Use the maximum of both to ensure both plots fit
        max_ymax = max(baseline_ymax_top, baseline_ymax_bottom)
        fig.update_yaxes(range=[-0.001, max_ymax], row=1, col=1)
        fig.update_yaxes(range=[-0.001, max_ymax], row=2, col=1)
        
        # Enable synchronized zooming and panning
        fig.update_xaxes(matches='x', row=1, col=1)
        fig.update_xaxes(matches='x', row=2, col=1)
        fig.update_yaxes(matches='y', row=1, col=1)
        fig.update_yaxes(matches='y', row=2, col=1)

        # Add x-axis title only to bottom subplot
        fig.update_xaxes(title_text="Mass (amu)", row=2, col=1)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Plot raw data and baseline corrected data
        fig,ax = plt.subplots(2,1, figsize=(10, 6), dpi=300, sharex=True)

        ax[0].axvline(mass_complex, alpha=0.75,linestyle="solid",linewidth=1, color="green", label = f"{complex_name}")
        ax[0].plot(x_mass[mass_range_indices], compiled_data[plot_wavenumber].iloc[mass_range_indices,plot_columnIndex_withoutIR], label =compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR])
        ax[0].plot(x_mass[mass_range_indices], compiled_data[plot_wavenumber].iloc[mass_range_indices,plot_columnIndex_withIR], label = compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR])
        # ax[0].plot(x_mass[:], compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withoutIR])
        # ax[0].plot(x_mass[:], compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withIR])
        ax[0].fill_between(x_mass[baseline_range_indices],0.1, color = "lightgray", label = "baseline range")
        ax[0].hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
        ax[0].legend(fontsize = 8)
        
        # Plot baseline corrected data
        ax[1].axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=1, color="green")
        ax[1].plot(x_mass[mass_range_indices], baseline_corrected_data.iloc[mass_range_indices,0], label = "baseline corrected signal without IR")
        ax[1].plot(x_mass[mass_range_indices], baseline_corrected_data.iloc[mass_range_indices,1], label = "baseline corrected signal with IR")
        ax[1].fill_between(x_mass[baseline_range_indices],0.1, color = "lightgray", label = "baseline range")
        ax[1].hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
        ax[1].legend(fontsize = 8)
        
        # Plot scaling
        ax[0].set_ylim(-0.001, baseline_ymax_top)
        ax[0].set_xlim(mass_complex-5, mass_complex+5)
        ax[1].set_xlim(mass_complex-5, mass_complex+5)
        ax[1].set_ylim(-0.001, baseline_ymax_bottom)

        fig.supxlabel("Mass (amu)")
        fig.supylabel("Intensity (a.u.)")
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Add to Report button
        add_plot_to_report_button(
            fig, 
            f"Baseline Correction - {plot_wavenumber} cm-1",
            key_suffix=f"bc_{plot_wavenumber}",
            description=f"Baseline correction for wavenumber {plot_wavenumber} cm⁻¹"
        )
        
        plt.close(fig)