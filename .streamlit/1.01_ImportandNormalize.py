import streamlit as st
import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import time
from packages.FELIX_HDF5_ReadData import *
from packages.FELIX_HDF5_ProcessData import *

st.title("HDF5 Import and Normalization")

# Session state initialization for keeping track between page refreshes
if "normalized_data" not in st.session_state:
    st.session_state["normalized_data"] = None

# File upload and directory selection
with st.expander("Import HDF5 Files", expanded=True):
    uploaded_files = st.file_uploader("Select HDF5 files to read", accept_multiple_files=True, type=["h5"])
    st.session_state["file_directory"] = st.text_input("Enter file directory where data is saved. All outputs will be saved here.", 
                                                      value=st.session_state.get("file_directory", ""))
    
    # Power normalization file upload
    power_file = st.file_uploader(
        "Upload the FELIX Power Scan CSV (columns: undulator wavelength (µm), mean power (mJ))",
        type=["csv"]
    )
    
    # Add signal processing options
    st.subheader("Signal Processing Options")
    signal_processing = st.radio(
        "How to handle negative signal values:",
        options=["Use Absolute Values", "Add Offset to Make Positive", "No Processing (Requires Positive Values)"],
        index=0,
        help="The normalization requires positive signal values. Choose how to process negative values."
    )

# Initialize variables
file_directory = st.session_state.get("file_directory", None)
files = []
raw_data = None
data = None
compiled_data = {}

if uploaded_files:
    for file in uploaded_files:
        files.append(h5py.File(file, "r"))

# Import, read, process and normalize HDF5 files
if st.button("📖 Import, Process and Normalize HDF5 Files"):
    if not power_file:
        st.error("Please upload a FELIX Power Scan CSV file to perform normalization.")
        st.stop()
        
    # Show loading animation
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.write("Processing...")
        progress_bar = st.progress(0)
        
        # Total steps: reading files + processing steps + normalization
        total_steps = len(files) + 5  
        current_step = 0

    # Read each file
    for file in files:
        time.sleep(0.1)  # Small delay to show progress
        current_step += 1
        progress_bar.progress(current_step / total_steps)

    # Process data
    raw_data = ProcessData_FELIX_HDF5(files, directory=file_directory, streamlit_uploaded_files=uploaded_files)
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Extract data
    data = raw_data.extract_FELIX_data()
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Compile data
    compiled_data = raw_data.compile_FELIX_data()
    unique_wavenumbers = raw_data.get_wavenumbers()
    current_step += 1
    progress_bar.progress(current_step / total_steps)
    
    # Debug: Check compiled data
    st.write(f"Number of wavenumbers found: {len(compiled_data.keys())}")
    if len(compiled_data.keys()) > 0:
        sample_wavenumber = list(compiled_data.keys())[0]
        st.write(f"Sample wavenumber: {sample_wavenumber} cm⁻¹")
        st.write(f"Corresponding wavelength: {1e4/sample_wavenumber:.2f} µm")
        
    # Read power scan file
    try:
        power_df = pd.read_csv(power_file, sep=";", encoding="latin-1")
        # Rename columns for consistency
        rename_cols = {
            "undulator wavelength (µm)": "undulator_wavelength",
            "undulator wavelength (m)": "undulator_wavelength",
            "mean power (mJ)": "mean_power"
        }
        power_df.rename(columns=rename_cols, inplace=True)
        
        if "undulator_wavelength" not in power_df.columns or "mean_power" not in power_df.columns:
            st.error("FELIX Power Scan CSV must have 'undulator wavelength (µm)' and 'mean power (mJ)' columns.")
            st.stop()
        
        # Debug: Show power scan range
        st.write(f"Power scan wavelength range: {power_df['undulator_wavelength'].min()}-{power_df['undulator_wavelength'].max()} µm")
        
        # Check for range mismatch
        data_min_wl = 1e4/max(unique_wavenumbers) if unique_wavenumbers else 0
        data_max_wl = 1e4/min(unique_wavenumbers) if unique_wavenumbers else 0
        
        if (data_min_wl < power_df['undulator_wavelength'].min() or 
            data_max_wl > power_df['undulator_wavelength'].max()):
            st.warning(f"⚠️ Your data wavelength range ({data_min_wl:.2f}-{data_max_wl:.2f} µm) is " +
                      f"outside the power scan range ({power_df['undulator_wavelength'].min()}-{power_df['undulator_wavelength'].max()} µm). " +
                      "Normalization may fail. Please use a power scan that covers your data range.")
        
        power_df.sort_values("undulator_wavelength", inplace=True)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
    except Exception as e:
        st.error(f"Error reading FELIX Power Scan CSV: {e}")
        st.stop()
    
    # Normalize the data and create a processed dataframe
    normalized_data = {}
    valid_pairs_count = 0
    invalid_pairs_count = 0
    
    # For each wavenumber, calculate the normalized ion yield
    for wavenumber, df in compiled_data.items():
        # For each pair of columns (withoutIR and withIR)
        normalized_pairs = []
        
        # Extract all column names
        columns = df.columns.tolist()
        
        # Group columns by their base name (without _withIR or _withoutIR suffix)
        base_names = set()
        for col in columns:
            parts = col.split('_')
            if parts[-1] in ["withIR", "withoutIR"]:
                base_name = '_'.join(parts[:-1])
                base_names.add(base_name)
        
        # Process each pair
        for base_name in base_names:
            withoutIR_col = f"{base_name}_withoutIR"
            withIR_col = f"{base_name}_withIR"
            
            if withoutIR_col in columns and withIR_col in columns:
                # Calculate wavenumber in microns for interpolation
                wavelength_um = 1e4 / wavenumber  # Convert from cm⁻¹ to µm
                
                # Interpolate power
                mean_power = np.interp(
                    wavelength_um,
                    power_df["undulator_wavelength"],
                    power_df["mean_power"],
                    left=0,  # Return 0 for values below range
                    right=0  # Return 0 for values above range
                )
                
                # Calculate sum of signals
                sum_withoutIR_raw = df[withoutIR_col].sum()
                sum_withIR_raw = df[withIR_col].sum()
                
                # Process signals based on selected option
                if signal_processing == "Use Absolute Values":
                    sum_withoutIR = abs(sum_withoutIR_raw)
                    sum_withIR = abs(sum_withIR_raw)
                elif signal_processing == "Add Offset to Make Positive":
                    # Find the minimum value across both signals and add offset to make positive
                    min_value = min(sum_withoutIR_raw, sum_withIR_raw)
                    offset = abs(min_value) + 1 if min_value < 0 else 0
                    sum_withoutIR = sum_withoutIR_raw + offset
                    sum_withIR = sum_withIR_raw + offset
                else:  # No Processing
                    sum_withoutIR = sum_withoutIR_raw
                    sum_withIR = sum_withIR_raw
                
                # Debug values
                if np.isscalar(sum_withoutIR) and np.isscalar(sum_withIR) and np.isscalar(mean_power):
                    debug_str = f"WN: {wavenumber}, λ: {wavelength_um:.2f}µm, WithoutIR: {sum_withoutIR:.2f} (raw: {sum_withoutIR_raw:.2f}), WithIR: {sum_withIR:.2f} (raw: {sum_withIR_raw:.2f}), Power: {mean_power:.2f}"
                    if sum_withoutIR <= 0 or sum_withIR <= 0 or mean_power <= 0:
                        invalid_pairs_count += 1
                        if invalid_pairs_count <= 5:  # Limit debug output
                            st.write(f"❌ Invalid: {debug_str}")
                    else:
                        valid_pairs_count += 1
                        if valid_pairs_count <= 5:  # Limit debug output
                            st.write(f"✅ Valid: {debug_str}")
                
                # Calculate depletion and normalized ion yield
                # Fix: Use scalar comparisons to avoid pandas Series truth value ambiguity
                if np.isscalar(sum_withoutIR) and np.isscalar(sum_withIR) and np.isscalar(mean_power):
                    if sum_withoutIR > 0 and sum_withIR > 0 and mean_power > 0:
                        depletion = sum_withIR / sum_withoutIR
                        ln_depletion = -np.log(depletion)
                        ion_yield = ln_depletion / mean_power
                        
                        normalized_pairs.append({
                            "wavenumber": wavenumber,
                            "wavelength_um": wavelength_um,
                            "base_name": base_name,
                            "sum_withoutIR": sum_withoutIR,
                            "sum_withIR": sum_withIR,
                            "raw_withoutIR": sum_withoutIR_raw,
                            "raw_withIR": sum_withIR_raw,
                            "mean_power": mean_power,
                            "depletion": depletion,
                            "-ln(depletion)": ln_depletion,
                            "ion_yield": ion_yield
                        })
                else:
                    # If we have Series objects instead of scalars, process each element
                    for i in range(len(sum_withoutIR) if hasattr(sum_withoutIR, '__len__') else 1):
                        s_without = sum_withoutIR.iloc[i] if hasattr(sum_withoutIR, 'iloc') else sum_withoutIR[i] if hasattr(sum_withoutIR, '__len__') else sum_withoutIR
                        s_with = sum_withIR.iloc[i] if hasattr(sum_withIR, 'iloc') else sum_withIR[i] if hasattr(sum_withIR, '__len__') else sum_withIR
                        m_power = mean_power.iloc[i] if hasattr(mean_power, 'iloc') else mean_power[i] if hasattr(mean_power, '__len__') else mean_power
                        
                        if s_without > 0 and s_with > 0 and m_power > 0:
                            depletion = s_with / s_without
                            ln_depletion = -np.log(depletion)
                            ion_yield = ln_depletion / m_power
                            
                            normalized_pairs.append({
                                "wavenumber": wavenumber,
                                "wavelength_um": wavelength_um,
                                "base_name": f"{base_name}_{i}" if hasattr(sum_withoutIR, '__len__') else base_name,
                                "sum_withoutIR": s_without,
                                "sum_withIR": s_with,
                                "mean_power": m_power,
                                "depletion": depletion,
                                "-ln(depletion)": ln_depletion,
                                "ion_yield": ion_yield
                            })
        
        if normalized_pairs:
            normalized_data[wavenumber] = pd.DataFrame(normalized_pairs)
    
    # Create a combined dataframe with all normalized data
    all_normalized = []
    for wavenumber, df in normalized_data.items():
        all_normalized.append(df)
    
    # Summary of normalization
    st.write(f"Valid pairs: {valid_pairs_count}, Invalid pairs: {invalid_pairs_count}")
    
    if all_normalized:
        st.session_state["normalized_data"] = pd.concat(all_normalized)
        
        # Clear loading animation
        loading_placeholder.empty()
        st.success("Files imported, processed, and normalized successfully! 🤩")
    else:
        # More detailed error message
        st.error("No valid data pairs found for normalization. Possible causes:")
        st.markdown("""
        1. **Negative signal values**: Your signal values are negative. Try using the "Use Absolute Values" option.
        2. **Wavelength mismatch**: Your data wavelengths are outside the power scan range
        3. **Incorrect column names**: The script looks for column names ending with "_withIR" and "_withoutIR"
        """)

# Display and plot normalized data if available
if st.session_state["normalized_data"] is not None:
    with st.expander("Normalized Data", expanded=True):
        st.subheader("Normalized Data Preview")
        st.dataframe(st.session_state["normalized_data"].head(10))
        
        # Download button for the normalized data
        csv_data = st.session_state["normalized_data"].to_csv(index=False)
        st.download_button(
            label="Download Normalized Data as CSV",
            data=csv_data,
            file_name="normalized_data.csv",
            mime="text/csv"
        )
    
    # Plotting section
    with st.expander("Visualize Results", expanded=True):
        tab1, tab2 = st.tabs(["Static Plot", "Interactive Plot"])
        
        with tab1:
            st.subheader("Static Plot: Ion Yield vs. Wavenumber")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Group by base_name to color by file
            grouped = st.session_state["normalized_data"].groupby("base_name")
            for name, group in grouped:
                # Sort by wavenumber for connected lines
                group = group.sort_values("wavenumber")
                ax.plot(group["wavenumber"], group["ion_yield"], 
                       marker="o", linestyle="-", label=name)
            
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Ion Yield")
            ax.set_title("Normalized Ion Yield vs. Wavenumber")
            ax.grid(True)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Download button for static plot
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                label="Download Static Plot as PNG",
                data=buf,
                file_name="ion_yield_plot.png",
                mime="image/png"
            )
        
        with tab2:
            st.subheader("Interactive Plot: Ion Yield vs. Wavenumber")
            
            # Create interactive plot with Plotly
            fig = px.line(
                st.session_state["normalized_data"],
                x="wavenumber",
                y="ion_yield",
                color="base_name",
                title="Normalized Ion Yield vs. Wavenumber",
                markers=True,
                labels={
                    "ion_yield": "Ion Yield",
                    "wavenumber": "Wavenumber (cm⁻¹)",
                    "base_name": "File"
                }
            )
            
            # Improve layout
            fig.update_layout(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis_title="Ion Yield",
                legend_title="Files",
                hovermode="closest"
            )
            
            st.plotly_chart(fig)
            
            # Download button for interactive plot
            html_str = fig.to_html()
            st.download_button(
                label="Download Interactive Plot as HTML",
                data=html_str,
                file_name="interactive_ion_yield_plot.html",
                mime="text/html"
            )

st.markdown("<h3 style='color: red;'>Note: </h3>", unsafe_allow_html=True)
st.markdown("If you navigate between pages, your processed data will remain in memory. <br>To clear the memory, click the `kebab menu` icon on the top right and then `clear cache`.", unsafe_allow_html=True)
