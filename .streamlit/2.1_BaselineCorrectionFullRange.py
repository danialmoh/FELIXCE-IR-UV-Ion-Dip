import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import configparser
import os
from packages.BaselineCorrection_v2 import *
from packages.utils import require_state
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

if not require_state(
    [
        "compiled_data",
        "unique_wavenumbers",
        "baseline_range_indices",
        "mass_range_indices",
        "x_mass",
        "complex",
        "mass_complex",
    ],
    section="2.1 Baseline correction - full range",
    hint="Run 2.0 Baseline correction first to register baseline parameters.",
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
            defaults["plot_wavenumber"] = config.getfloat('Plot Parameters', 'plot_wavenumber')
            defaults["mass_xmin"] = config.getfloat('Plot Parameters', 'mass_xmin')
            defaults["mass_xmax"] = config.getfloat('Plot Parameters', 'mass_xmax')
            defaults["mass_ymax"] = config.getfloat('Plot Parameters', 'mass_ymax')
            defaults["plot_columnIndex_withoutIR"] = config.getint('Plot Parameters', 'columnIndex_withoutIR')
            defaults["plot_columnIndex_withIR"] = config.getint('Plot Parameters', 'columnIndex_withIR')
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
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)
plot_wavenumber = st.session_state.get("plot_wavenumber", None)

if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):
    baseline_method = st.session_state.get("baseline_method", "Mean Subtraction")
    
    # Get baseline method parameters from session_state
    iarpls_lam = st.session_state.get("iarpls_lam", 1e6)
    aspls_lam = st.session_state.get("aspls_lam", 1e6)
    fabc_lam = st.session_state.get("fabc_lam", 1e6)
    fabc_scale = st.session_state.get("fabc_scale", None)
    
    compilation_baseline_corrected_data = {}
    
    for wavenumber in unique_wavenumbers:
        # Initialize variables, make sure they are cleared at each iteration
        sum_withoutIR = None
        sum_withIR = None
        corrected_without = None
        corrected_with = None
        mean_value_withIR = None
        mean_value_withoutIR = None
        baseline_without = None
        baseline_with = None
        baseline_fitter = None

        # Sum every other column to obtain total signals
        sum_withoutIR = compiled_data[wavenumber].iloc[:, 0::2].sum(axis=1)
        sum_withIR = compiled_data[wavenumber].iloc[:, 1::2].sum(axis=1)
        
        if baseline_method == "Mean Subtraction":
            mean_value_withoutIR = np.mean(sum_withoutIR[baseline_range_indices])
            mean_value_withIR = np.mean(sum_withIR[baseline_range_indices])
            corrected_without = sum_withoutIR - mean_value_withoutIR
            corrected_with = sum_withIR - mean_value_withIR
        elif baseline_method.lower() == "iarpls":
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            # Use UI parameter from session_state
            baseline_without, _ = baseline_fitter.iarpls(sum_withoutIR, lam=iarpls_lam)
            baseline_with, _ = baseline_fitter.iarpls(sum_withIR, lam=iarpls_lam)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        elif baseline_method.lower() == "aspls":
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            # Use UI parameter from session_state
            baseline_without, _ = baseline_fitter.aspls(sum_withoutIR, lam=aspls_lam)
            baseline_with, _ = baseline_fitter.aspls(sum_withIR, lam=aspls_lam)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        elif baseline_method.lower() == "fabc":
            from pybaselines import Baseline
            baseline_fitter = Baseline(x_data=x_mass)
            # Use UI parameters from session_state
            baseline_without, _ = baseline_fitter.fabc(sum_withoutIR, lam=fabc_lam, scale=fabc_scale)
            baseline_with, _ = baseline_fitter.fabc(sum_withIR, lam=fabc_lam, scale=fabc_scale)
            corrected_without = sum_withoutIR - baseline_without
            corrected_with = sum_withIR - baseline_with
        else:
            raise ValueError(f"Unsupported baseline method: {baseline_method}. Supported: Mean Subtraction, iarpls, aspls, fabc")
        
        new_table = pd.DataFrame({
            "sum_" + str(wavenumber) + "_withoutIR": sum_withoutIR,
            "sum_" + str(wavenumber) + "_withIR": sum_withIR,
            "baseline_corrected_sum_" + str(wavenumber) + "_withoutIR": corrected_without,
            "baseline_corrected_sum_" + str(wavenumber) + "_withIR": corrected_with
        })
        compilation_baseline_corrected_data[wavenumber] = pd.concat([compiled_data[wavenumber], new_table], axis=1)
    # export mass spectra
    export_MassSpectra = pd.DataFrame({
        "Mass (amu)": x_mass,
        "Mass per atom (amu)": x_mass_perAtom,
        compilation_baseline_corrected_data[plot_wavenumber].columns[-2]: compilation_baseline_corrected_data[plot_wavenumber].iloc[:, -2],
        compilation_baseline_corrected_data[plot_wavenumber].columns[-1]: compilation_baseline_corrected_data[plot_wavenumber].iloc[:, -1]
    })
    # export.to_csv(rf"{file_directory}/output/MassSpectra_{complex}_{plot_wavenumber}cm-1.csv", index=False)
    st.session_state["export_MassSpectra"] = export_MassSpectra

    st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
    st.success("Success! 😎")
    # st.table(compilation_baseline_corrected_data[plot_wavenumber].head())
    print(compilation_baseline_corrected_data[plot_wavenumber].head())

if st.button("**:blue[#2]** 🚢 Export baseline corrected data"):
    if "export_MassSpectra" not in st.session_state:
        st.error("⚠️ Please perform baseline correction first.")
        st.stop()
    else:
        output_directory = st.session_state.get("file_directory", None)
        export_MassSpectra = st.session_state.get("export_MassSpectra", None)
        export_baseline_corrected = st.session_state.get("compilation_baseline_corrected_data", None)[plot_wavenumber]
        output_name_MassSpectra = f"MassSpectra_{complex}_{plot_wavenumber}cm⁻¹.csv"
        output_name_BaselineCorrected = f"BaselineCorrected_{complex}_{plot_wavenumber}cm⁻¹.csv"
        output_fullpath_MassSpectra = os.path.join(output_directory, output_name_MassSpectra)
        output_fullpath_BaselineCorrected = os.path.join(output_directory, output_name_BaselineCorrected)
        export_MassSpectra.to_csv(output_fullpath_MassSpectra, index=False)
        export_baseline_corrected.to_csv(output_fullpath_BaselineCorrected, index=False)
        st.success(rf"Successfully exported:", icon="✅")
        st.success(rf"baseline corrected mass spectra @ '`{output_fullpath_MassSpectra}`'")
        st.success(rf"baseline corrected data @ '`{output_fullpath_BaselineCorrected}`'")

st.markdown("#### Plot parameters")

col1, col2, col3, col4, col5 = st.columns([0.4, 0.05, 0.7, 0.05, 1])
with col1:
    available_wavenumbers = sorted(compiled_data.keys())
    default_idx = 0
    if (
        "plot_wavenumber" in st.session_state
        and st.session_state["plot_wavenumber"] in available_wavenumbers
    ):
        default_idx = available_wavenumbers.index(st.session_state["plot_wavenumber"])
    st.session_state["plot_wavenumber"] = st.selectbox(
        "Wavenumber to check plots",
        options=available_wavenumbers,
        index=default_idx,
        format_func=lambda x: f"{x:.2f}",
    )
    st.session_state["mass_ymax"] = float(
        st.text_input("Maximum y-value", value=st.session_state.get("mass_ymax" , defaults.get("mass_ymax", None)))
    )
with col3:
    st.session_state["mass_xmin"] = float(
        st.text_input("Minimum x-value", value=st.session_state.get("mass_xmin", defaults.get("mass_xmin", None)))
    )
    st.session_state["mass_xmax"] = float(
        st.text_input("Maximum x-value", value=st.session_state.get("mass_xmax", defaults.get("mass_xmax", None)))
    )
with col5:
    st.session_state["plot_columnIndex_withoutIR"] = int(
        st.number_input("Column index for signal without IR irradiation", value=st.session_state.get("plot_columnIndex_withoutIR", defaults.get("plot_columnIndex_withoutIR", None)))
    )
    st.session_state["plot_columnIndex_withIR"] = int(
        st.number_input("Column index for signal with IR irradiation", value=st.session_state.get("plot_columnIndex_withIR", defaults.get("plot_columnIndex_withIR", None)))
    )

if st.button("**:blue[#3]** 🚀 Plot full range data"):
    # Retrieve plot_wavenumber from session state
    plot_wavenumber = st.session_state.get("plot_wavenumber", None)

    # If the value is None or not present in compiled_data, use the first available key
    if plot_wavenumber is None or plot_wavenumber not in compiled_data:
        st.write("Available compiled_data keys:", list(compiled_data.keys()))
        st.write("Provided plot_wavenumber:", plot_wavenumber)
        plot_wavenumber = list(compiled_data.keys())[0]  # Use first key as fallback

    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
    mass_ymax = st.session_state.get("mass_ymax", None)
    mass_xmin = st.session_state.get("mass_xmin", None)
    mass_xmax = st.session_state.get("mass_xmax", None)

    # Mass spectra plot (no tabs)
    st.markdown("### 📈 Mass Spectra")
    fig_mass = go.Figure()
    fig_mass.add_trace(
        go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[
                :, plot_columnIndex_withoutIR
            ],
            mode="lines",
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[
                plot_columnIndex_withoutIR
            ],
            line=dict(color="#1f77b4"),
            opacity=0.75,
        )
    )
    fig_mass.add_trace(
        go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[
                :, plot_columnIndex_withIR
            ],
            mode="lines",
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[
                plot_columnIndex_withIR
            ],
            line=dict(color="#ff7f0e"),
            opacity=0.75,
        )
    )
    fig_mass.add_trace(
        go.Scatter(
            x=[mass_complex, mass_complex],
            y=[-0.001, mass_ymax],
            mode="lines",
            line=dict(color="green", width=1, dash="solid"),
            name=str(complex),
        )
    )
    fig_mass.add_shape(
        type="rect",
        x0=min(x_mass[baseline_range_indices]),
        y0=0,
        x1=max(x_mass[baseline_range_indices]),
        y1=mass_ymax,
        fillcolor="lightsteelblue",
        line=dict(color="rgba(0,0,0,0)"),
        opacity=0.4,
        layer="below",
    )
    fig_mass.add_trace(
        go.Scatter(
            x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
            y=[0, 0],
            mode="lines",
            line=dict(color="lime", width=1),
            name="zero line",
        )
    )
    fig_mass.update_layout(
        yaxis=dict(range=[-0.001, mass_ymax]),
        xaxis_title="Mass (amu)",
        yaxis_title="Intensity",
        xaxis=dict(range=[mass_xmin, mass_xmax]),
        title=f"Summed and then baseline corrected mass spectra of {complex} at {plot_wavenumber} cm⁻¹",
        legend=dict(x=0.8, y=0.9),
    )
    st.plotly_chart(fig_mass, use_container_width=True)
