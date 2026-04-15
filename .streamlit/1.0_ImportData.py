from pathlib import Path
import time
import re

import h5py
import streamlit as st
import configparser
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import interp1d

from packages.FELIX_HDF5_Reader_v2 import *

# Helper function to extract calibration key from filename for auto-matching
def extract_calibration_key(filename):
    """
    Extract a calibration matching key from a filename.
    
    Matches data files to calibration files based on the 'shift{N}' identifier
    and the associated number (e.g., FELIX setting number).
    
    Data file examples:
        'BRBnz_DisON_FELIX_ArF_Scan10_1125-1000_shift1_1910_step2.h5' -> 'shift1_1910'
        'Molecule_FELIX_shift2_1450_step3.h5' -> 'shift2_1450'
    
    Calibration file examples:
        '2026-03-18_FELIX_Shift1_WavenumberCalibration_1910.csv' -> 'shift1_1910'
        'FELIX_Shift2_WavenumberCalibration_1450.csv' -> 'shift2_1450'
    
    Args:
        filename (str): The filename to parse
    
    Returns:
        str or None: Normalized calibration key like 'shift1_1910', or None if not found
    """
    name = filename.lower()
    
    # Pattern 1 (data files): shift{N}_{M} where M is a 3-5 digit number
    # e.g., shift1_1910 in '..._shift1_1910_step2.h5'
    match = re.search(r'shift(\d+)_(\d{3,5})', name)
    if match:
        return f"shift{match.group(1)}_{match.group(2)}"
    
    # Pattern 2 (calibration files): shift{N}...calibration_{M}
    # e.g., Shift1_WavenumberCalibration_1910 in '..._Shift1_WavenumberCalibration_1910.csv'
    shift_match = re.search(r'shift(\d+)', name)
    cal_num_match = re.search(r'calibration_(\d{3,5})', name)
    if shift_match and cal_num_match:
        return f"shift{shift_match.group(1)}_{cal_num_match.group(1)}"
    
    return None

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

# ==================== WAVELENGTH CALIBRATION SECTION ====================
st.markdown("### Wavelength Calibration Configuration")
st.info("📏 Apply wavelength calibration to correct scanned wavenumbers before grouping data. Different experiments may require different calibrations.")

# ==================== CREATE CALIBRATION TABLE ====================
with st.expander("🛠️ Create Calibration Table (Optional)", expanded=False):
    st.markdown("**Create a calibration table if you don't have one yet**")
    
    # Initialize default calibration table if not exists
    if "calibration_table_creator" not in st.session_state:
        st.session_state["calibration_table_creator"] = pd.DataFrame({
            "Scanned_cm-1": [1400.0, 1450.0, 1500.0],
            "Real_cm-1": [1398.5, 1448.2, 1497.8]
        })
    
    st.markdown("**Edit Calibration Points** (click buttons below to preview or download):")
    edited_cal_table = st.data_editor(
        st.session_state["calibration_table_creator"],
        num_rows="dynamic",
        use_container_width=True,
        key="calibration_creator_editor",
        disabled=False
    )
    
    # Action buttons
    cal_col1, cal_col2 = st.columns(2)
    
    with cal_col1:
        if st.button("📊 Preview Calibration Curve", use_container_width=True, key="preview_cal_import"):
            st.session_state["calibration_table_creator"] = edited_cal_table
            try:
                scanned = edited_cal_table["Scanned_cm-1"].values
                real = edited_cal_table["Real_cm-1"].values
                
                if len(scanned) < 2:
                    st.error("Need at least 2 calibration points")
                else:
                    # Create preview plot
                    fig_cal = go.Figure()
                    
                    # Calibration points
                    fig_cal.add_trace(go.Scatter(
                        x=scanned, 
                        y=real,
                        mode='markers',
                        name='Calibration Points',
                        marker=dict(size=10, color='red', symbol='circle')
                    ))
                    
                    # Fitted curve - select interpolation method based on number of points
                    if len(scanned) >= 4:
                        # Cubic interpolation requires at least 4 points
                        cal_func = interp1d(scanned, real, kind='cubic', fill_value='extrapolate', bounds_error=False)
                        interp_type = 'Cubic spline'
                    elif len(scanned) == 3:
                        # Quadratic for 3 points
                        cal_func = interp1d(scanned, real, kind='quadratic', fill_value='extrapolate', bounds_error=False)
                        interp_type = 'Quadratic'
                    else:
                        # Linear for 2 points
                        cal_func = interp1d(scanned, real, kind='linear', fill_value='extrapolate', bounds_error=False)
                        interp_type = 'Linear'
                    
                    x_fit = np.linspace(scanned.min(), scanned.max(), 200)
                    y_fit = cal_func(x_fit)
                    
                    fig_cal.add_trace(go.Scatter(
                        x=x_fit, 
                        y=y_fit,
                        mode='lines',
                        name='Interpolation Curve',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # 1:1 reference line
                    ref_min = min(scanned.min(), real.min())
                    ref_max = max(scanned.max(), real.max())
                    fig_cal.add_trace(go.Scatter(
                        x=[ref_min, ref_max],
                        y=[ref_min, ref_max],
                        mode='lines',
                        name='1:1 Reference',
                        line=dict(dash='dash', color='gray', width=1)
                    ))
                    
                    fig_cal.update_layout(
                        xaxis_title="Scanned Wavenumber (cm⁻¹)",
                        yaxis_title="Real Wavenumber (cm⁻¹)",
                        title="Wavenumber Calibration Curve",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_cal, use_container_width=True)
                    
                    # Show offset statistics
                    offset = real - scanned
                    st.markdown(f"""
                    **Calibration Statistics:**
                    - Points: {len(scanned)}
                    - Offset range: {offset.min():.2f} to {offset.max():.2f} cm⁻¹
                    - Mean offset: {offset.mean():.2f} cm⁻¹
                    - Interpolation: {interp_type}
                    """)
                    
            except Exception as e:
                st.error(f"Error creating preview: {e}")
    
    with cal_col2:
        if st.button("💾 Download Calibration CSV", use_container_width=True, key="download_cal_import"):
            st.session_state["calibration_table_creator"] = edited_cal_table
            csv_data = edited_cal_table.to_csv(index=False)
            st.download_button(
                label="📥 Download",
                data=csv_data,
                file_name="wavenumber_calibration.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_cal_csv_button"
            )
    
    st.info("💡 **Tip**: After downloading, upload the CSV file below and assign it to your data files.")

st.divider()

# Upload calibration CSV files
calibration_files = st.file_uploader(
    "Upload Calibration CSV Files (columns: Scanned_cm-1, Real_cm-1)",
    accept_multiple_files=True,
    type=['csv'],
    help="Upload one or more calibration CSV files. Each file should have 'Scanned_cm-1' and 'Real_cm-1' columns.",
    key="calibration_uploader"
)

# Parse and store calibration tables
if "calibration_tables" not in st.session_state:
    st.session_state["calibration_tables"] = {}

if calibration_files:
    for cal_file in calibration_files:
        try:
            cal_df = pd.read_csv(cal_file)
            if "Scanned_cm-1" in cal_df.columns and "Real_cm-1" in cal_df.columns:
                # Store calibration table with filename as key
                st.session_state["calibration_tables"][cal_file.name] = cal_df
            else:
                st.error(f"❌ {cal_file.name}: CSV must have columns 'Scanned_cm-1' and 'Real_cm-1'")
        except Exception as e:
            st.error(f"❌ Error reading {cal_file.name}: {e}")

# Show loaded calibration tables
if st.session_state["calibration_tables"]:
    st.success(f"✅ Loaded {len(st.session_state['calibration_tables'])} calibration table(s): {', '.join(st.session_state['calibration_tables'].keys())}")
    
    # Option to preview calibration tables
    with st.expander("📊 Preview Calibration Tables", expanded=False):
        for cal_name, cal_df in st.session_state["calibration_tables"].items():
            st.markdown(f"**{cal_name}**")
            st.dataframe(cal_df, use_container_width=True)

# Per-file calibration assignment with auto-matching
if uploaded_files:
    st.markdown("#### Assign Calibration to Each File")
    
    # Initialize file calibration assignments if not exists
    if "file_calibrations" not in st.session_state:
        st.session_state["file_calibrations"] = {}
    
    # Create list of calibration options
    cal_options = ["None (no calibration)"] + list(st.session_state["calibration_tables"].keys())
    
    # Build calibration key → filename lookup from loaded calibration files
    cal_key_lookup = {}
    for cal_name in st.session_state["calibration_tables"].keys():
        cal_key = extract_calibration_key(cal_name)
        if cal_key:
            cal_key_lookup[cal_key] = cal_name
    
    # Auto-match and display
    auto_matched_count = 0
    for i, file in enumerate(uploaded_files):
        # Try auto-matching by calibration key
        data_key = extract_calibration_key(file.name)
        auto_match = cal_key_lookup.get(data_key) if data_key else None
        
        # Priority: 1) explicit user selection in session state (not "None"), 2) auto-matched, 3) session state "None", 4) default None
        existing_cal = st.session_state["file_calibrations"].get(file.name)
        if existing_cal and existing_cal != "None (no calibration)" and existing_cal in cal_options:
            current_cal = existing_cal
        elif auto_match and auto_match in cal_options:
            current_cal = auto_match
            st.session_state["file_calibrations"][file.name] = current_cal
            auto_matched_count += 1
        else:
            current_cal = "None (no calibration)"
        
        if current_cal not in cal_options:
            current_cal = "None (no calibration)"
        
        # Set the selectbox key before rendering (only if not already set by user)
        widget_key = f"cal_select_{i}_{file.name}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_cal
        
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.text(f"📁 {file.name}")
        with col2:
            selected_cal = st.selectbox(
                "Calibration",
                options=cal_options,
                key=widget_key,
                label_visibility="collapsed"
            )
            st.session_state["file_calibrations"][file.name] = selected_cal
        with col3:
            if selected_cal != "None (no calibration)":
                if auto_match and selected_cal == auto_match:
                    st.markdown("🔗 *auto*")
                else:
                    st.markdown("✅ *manual*")
            elif data_key and not auto_match:
                st.markdown("⚠️ *no match*")
    
    if auto_matched_count > 0 and cal_key_lookup:
        st.success(f"🔗 Auto-matched {auto_matched_count} file(s) to calibration tables by shift identifier")
else:
    st.warning("⚠️ Upload files first to assign calibrations.")



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
    
    # Prepare calibration functions for each file
    calibration_functions = []
    for file in uploaded_files:
        cal_name = st.session_state.get("file_calibrations", {}).get(file.name, "None (no calibration)")
        
        if cal_name != "None (no calibration)" and cal_name in st.session_state.get("calibration_tables", {}):
            # Create interpolation function for this file
            cal_table = st.session_state["calibration_tables"][cal_name]
            scanned = cal_table["Scanned_cm-1"].values
            real = cal_table["Real_cm-1"].values
            
            # Create interpolation function with extrapolation
            cal_func = interp1d(scanned, real, kind='linear', fill_value='extrapolate', bounds_error=False)
            calibration_functions.append(cal_func)
        else:
            # No calibration for this file
            calibration_functions.append(None)
    
    # Simple Analogy; Hey Python, here's a worker (raw_data) from the FELIX_HDF5_Reader team. I want them to handle the task of processing our data files.
    raw_data = FELIX_HDF5_Reader(files, directory = file_directory, streamlit_uploaded_files = uploaded_files, step_size=step_sizes, calibration_functions=calibration_functions) # turn into class object
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