from pathlib import Path
import time
import re

import h5py
import streamlit as st
import configparser
import os

from packages.FELIX_HDF5_Reader_v2 import *

# Helper function to extract step size from filename
def extract_step_size_from_filename(filename):
    """
    Extract step size from filename pattern like 'step2' or 'step5.5'.
    
    Examples:
        'BRBnz_DisON_FELIX_ArF.008_2200-2070cm-1_step2.h5' -> 2.0
        'scan_step0.5.h5' -> 0.5
        'data_step3_run1.h5' -> 3.0
        'no_step_info.h5' -> None
    
    Args:
        filename (str): The filename to parse
    
    Returns:
        float or None: Extracted step size, or None if not found
    """
    # Pattern: 'step' followed by a number (integer or decimal)
    # Case insensitive match
    pattern = r'step(\d+\.?\d*)'
    match = re.search(pattern, filename, re.IGNORECASE)
    
    if match:
        try:
            step_size = float(match.group(1))
            return step_size
        except ValueError:
            return None
    return None

# Import variables from defaults.ini
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'.streamlit/defaults.ini'  # or provide full path
    defaults= {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            # Update defaults with values from file
            defaults['file_directory'] = config.get('Import Data', 'file_directory')
            defaults['step_size'] = config.getfloat('Import Data', 'step_size')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()


uploaded_files = st.file_uploader("Select HDF5 files to read (adds files to existing list, click `x` to remove)", accept_multiple_files=True, type=["h5"])
st.session_state["file_directory"] = st.text_input("Enter file directory where outputs are saved.", value= st.session_state.get("file_directory", defaults['file_directory']))
directory_input = st.session_state["file_directory"]

# Choose step size input mode
st.markdown("### Step Size Configuration")
step_size_mode = st.radio(
    "How would you like to specify step sizes?",
    options=["Same step size for all files", "Different step size per file"],
    index=0,
    help="Choose 'Different step size per file' if your scans were taken with different step sizes"
)

if step_size_mode == "Same step size for all files":
    st.session_state["step_size"] = st.text_input("Enter the step size of your scans (cm⁻¹):", value=st.session_state.get("step_size", defaults.get("step_size", None)))
    st.session_state["use_per_file_step_size"] = False
else:
    st.info("You will specify step size for each file individually below.")
    st.session_state["use_per_file_step_size"] = True
    
    # Show per-file step size inputs if files are uploaded
    if uploaded_files:
        st.markdown("#### Specify step size for each file:")
        
        # Initialize step sizes dictionary if not exists
        if "file_step_sizes" not in st.session_state:
            st.session_state["file_step_sizes"] = {}
        
        for i, file in enumerate(uploaded_files):
            # Try to auto-detect step size from filename first
            auto_detected = extract_step_size_from_filename(file.name)
            
            # Priority: 1) already set in session state, 2) auto-detected, 3) default
            if file.name in st.session_state["file_step_sizes"]:
                default_value = st.session_state["file_step_sizes"][file.name]
            elif auto_detected is not None:
                default_value = auto_detected
            else:
                default_value = defaults.get("step_size", 0.5)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📁 {file.name}")
            with col2:
                step_size_value = st.number_input(
                    f"Step size",
                    min_value=0.1,
                    max_value=10.0,
                    value=float(default_value),
                    step=0.1,
                    format="%.1f",
                    key=f"step_size_{i}_{file.name}",
                    label_visibility="collapsed"
                )
                st.session_state["file_step_sizes"][file.name] = step_size_value
    else:
        st.warning("⚠️ Please upload files first to specify individual step sizes.")



# Initialize variables
file_directory =st.session_state.get("file_directory", None)
files = []
raw_data = []
data = []
compiled_data = {}


if uploaded_files:
    for file in uploaded_files:
        files.append(h5py.File(file, "r"))


# Import, read, process H5 files.
if st.button("📖 click this button to import, read, and process the H5 files. Wait for a prompt that files have been loaded to memory."):

    if not file_directory:
        st.error("Please provide a valid directory before importing data.", icon="🚫")
        st.stop()

    # Show loading animation
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.write("Processing.....")
        progress_bar = st.progress(0)

        # Actual file processing with progress updates
        total_steps = len(files) + 4  # Number of steps (reading files + processing steps)
        current_step = 0

    # Read each file (increment progress for each file)
    for file in files:
        # Simulate reading the file
        time.sleep(0.2)  # Add a delay to simulate file loading time if necessary
        current_step += 1
        progress_bar.progress(current_step / total_steps)


    # Actual file processing
    
    # Process data
    # Prepare step sizes based on mode
    if st.session_state.get("use_per_file_step_size", False):
        # Per-file step sizes
        step_sizes = [st.session_state["file_step_sizes"].get(uploaded_files[i].name, 0.5) for i in range(len(uploaded_files))]
    else:
        # Single step size mode - but still check for auto-detection from filenames
        auto_detected_step_sizes = []
        for i, file in enumerate(uploaded_files):
            detected = extract_step_size_from_filename(file.name)
            auto_detected_step_sizes.append(detected)
        
        # If all files have auto-detected step sizes, use them
        if all(s is not None for s in auto_detected_step_sizes):
            step_sizes = auto_detected_step_sizes
        # If some but not all have auto-detected, use mixed approach
        elif any(s is not None for s in auto_detected_step_sizes):
            single_step_size = float(st.session_state["step_size"])
            step_sizes = [detected if detected is not None else single_step_size for detected in auto_detected_step_sizes]
        # If none have auto-detected, use single step size for all
        else:
            single_step_size = float(st.session_state["step_size"])
            step_sizes = [single_step_size] * len(uploaded_files)
    
    # Simple Analogy; Hey Python, here's a worker (raw_data) from the FELIX_HDF5_Reader team. I want them to handle the task of processing our data files.
    raw_data = FELIX_HDF5_Reader(files, directory = file_directory, streamlit_uploaded_files = uploaded_files, step_size=step_sizes) # turn into class object
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Extract data # We tell the worker, "Now that you know where the files are, start extracting the important parts: the wavenumbers and signals."
    data = raw_data.import_files() # get the wavenumber and signal data
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Check data extraction #After extraction, we ask the worker, "Show me what you've done so far. I want to make sure everything looks good."
    raw_data.check_import() # check output
    st.session_state["dataset_length"] = raw_data.count_rows
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Compile data # We tell the worker, "Now take all the data you extracted and organize it. Group the signals by their wavenumbers."
    compiled_data = raw_data.compile_data() # compile data on a per wavenumber basis
    wavenumbers_raw = raw_data.visualize_imported_wavenumbers_raw()   #We tell the worker, "Lay out all the wavenumbers you've seen in the files. I want to make sure there are no missing or duplicate ones."
    wavenumbers = raw_data.visualize_imported_wavenumbers()
    unique_wavenumbers = raw_data.visualize_imported_unique_wavenumbers(min_count=None)
    # unique_wavenumbers, unique_wavenumbers_df = raw_data.get_wavenumbers(min_count=min_count)
    # min_count = st.session_state.get("min_wavenumber_count", 3)
    
    # print("\n")
    # print("List of unique wavenumbers should match the dataframe") 
    # print(unique_wavenumbers, unique_wavenumbers_df)


    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Save variables into memory
    st.session_state["raw_data"] = raw_data
    st.session_state["compiled_data"] = compiled_data
    st.session_state["unique_wavenumbers"] = unique_wavenumbers
    st.session_state["file_directory"] = Path(file_directory).resolve()

    # Clear loading animation after processing
    loading_placeholder.empty()
    st.success(f"Succes! Files will be saved at '`{Path(file_directory).resolve()}`' 🤩", icon="✅")

st.markdown("<h3 style='color: red;'>Note: </h3>", unsafe_allow_html=True)
st.markdown("If you navigate between pages, it may seem like your previously imported files are gone. <br><span style='color:blue;'>Rest assured they remain loaded in memory.</span> <br> To clear the memory, click the `kebab menu` icon on the top right and then `clear cache`.", unsafe_allow_html=True)

st.write(compiled_data)