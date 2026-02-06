"""
REMPI Data Import Page - Step 1

This page handles importing REMPI HDF5 files.
Similar to 1.0_ImportData.py but for REMPI (single-trace) data.
"""

from pathlib import Path
import time

import h5py
import streamlit as st
import configparser
import os

from packages.REMPI_HDF5_ProcessData import ProcessData_REMPI_HDF5

# Import variables from defaults.ini
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'.streamlit/defaults.ini'
    defaults = {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            defaults['file_directory'] = config.get('Import Data', 'file_directory', fallback='')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults

defaults = load_defaults()

# File uploader
uploaded_files = st.file_uploader(
    "Select REMPI HDF5 files to read (adds files to existing list, click `x` to remove)", 
    accept_multiple_files=True, 
    type=["h5"],
    key="rempi_uploader"
)

st.session_state["rempi_file_directory"] = st.text_input(
    "Enter file directory where outputs are saved.", 
    value=st.session_state.get("rempi_file_directory", defaults.get('file_directory', ''))
)
directory_input = st.session_state["rempi_file_directory"]

# Initialize variables
file_directory = st.session_state.get("rempi_file_directory", None)
files = []

if uploaded_files:
    for file in uploaded_files:
        files.append(h5py.File(file, "r"))

# Import button
if st.button("📖 Click to import, read, and process the REMPI H5 files"):
    
    if not file_directory:
        st.error("Please provide a valid directory before importing data.", icon="🚫")
        st.stop()
    
    if not uploaded_files:
        st.error("Please upload at least one HDF5 file.", icon="🚫")
        st.stop()

    # Show loading animation
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.write("Processing.....")
        progress_bar = st.progress(0)

        total_steps = len(files) + 4
        current_step = 0

    # Read each file
    for file in files:
        time.sleep(0.1)
        current_step += 1
        progress_bar.progress(current_step / total_steps)

    # Process data
    rempi_data = ProcessData_REMPI_HDF5(
        files, 
        streamlit_uploaded_files=uploaded_files,
        directory=file_directory
    )
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Extract data
    data = rempi_data.extract_REMPI_data()
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Check data extraction
    rempi_data.check_extract_REMPI_data()
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Compile data
    compiled_data = rempi_data.compile_REMPI_data_by_wavelength()
    compiled_dataframe = rempi_data.compile_REMPI_data_to_dataframe()
    
    # Get wavelengths
    wavelengths_table = rempi_data.check_wavelengths()
    unique_wavelengths, unique_wavelengths_df = rempi_data.get_wavelengths(min_count=0)
    
    # Get dataset length
    dataset_length = len(compiled_dataframe)

    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Save variables into memory
    st.session_state["rempi_data"] = rempi_data
    st.session_state["rempi_compiled_data"] = compiled_data
    st.session_state["rempi_compiled_dataframe"] = compiled_dataframe
    st.session_state["rempi_unique_wavelengths"] = unique_wavelengths
    st.session_state["rempi_unique_wavelengths_df"] = unique_wavelengths_df
    st.session_state["rempi_file_directory"] = Path(file_directory).resolve()
    st.session_state["rempi_dataset_length"] = dataset_length

    # Clear loading animation
    loading_placeholder.empty()
    st.success(f"Success! Files will be saved at '`{Path(file_directory).resolve()}`' 🤩", icon="✅")
    st.info(f"Found {len(unique_wavelengths)} unique wavelengths. Dataset length: {dataset_length} points.")

st.markdown("<h3 style='color: red;'>Note: </h3>", unsafe_allow_html=True)
st.markdown(
    "If you navigate between pages, it may seem like your previously imported files are gone. "
    "<br><span style='color:blue;'>Rest assured they remain loaded in memory.</span> "
    "<br> To clear the memory, click the `kebab menu` icon on the top right and then `clear cache`.", 
    unsafe_allow_html=True
)

# Show compiled data if available
if "rempi_compiled_data" in st.session_state:
    st.write("### Compiled Data Preview")
    st.write(st.session_state["rempi_compiled_dataframe"].head(50))
