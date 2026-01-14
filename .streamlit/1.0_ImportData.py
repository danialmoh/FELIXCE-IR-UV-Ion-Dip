from pathlib import Path
import time

import h5py
import streamlit as st

from packages.FELIX_HDF5_ProcessData import *
from packages.FELIX_HDF5_ReadData import *


uploaded_files = st.file_uploader("Select HDF5 files to read (adds files to existing list, click `x` to remove)", accept_multiple_files=True, type=["h5"])
directory_input = st.text_input(
    "Enter file directory where data is saved. All outputs will be saved here.",
    value=st.session_state.get("file_directory", "/Users/danialmoh/Library/CloudStorage/OneDrive-KULeuven/Thesis/All scans"),
).strip()
st.session_state["file_directory"] = directory_input
st.session_state["step_size"] = st.text_input("Enter the step size of your scans. (not yet working)")


# Initialize variables
file_directory = Path(directory_input).expanduser() if directory_input else None
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

    output_dir = file_directory / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    #Simple Analogy; Hey Python, here's a worker (raw_data) from the ProcessData_FELIX_HDF5 team. I want them to handle the task of processing our data files.
    raw_data = ProcessData_FELIX_HDF5(files, directory = file_directory, streamlit_uploaded_files = uploaded_files) # turn into class object
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Extract data # We tell the worker, "Now that you know where the files are, start extracting the important parts: the wavenumbers and signals."
    data = raw_data.extract_FELIX_data() # get the wavenumber and signal data
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Check data extraction #After extraction, we ask the worker, "Show me what you've done so far. I want to make sure everything looks good."
    raw_data.check_extract_FELIX_data() # check output
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Compile data # We tell the worker, "Now take all the data you extracted and organize it. Group the signals by their wavenumbers."
    compiled_data = raw_data.compile_FELIX_data() # compile data on a per wavenumber basis
    x = raw_data.check_wavenumbers() #We tell the worker, "Lay out all the wavenumbers you've seen in the files. I want to make sure there are no missing or duplicate ones."
    unique_wavenumbers, unique_wavenumbers_df = raw_data.get_wavenumbers()
    
    print("\n")
    print("List of unique wavenumbers should match the dataframe")
    print(unique_wavenumbers, unique_wavenumbers_df)


    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Save variables into memory
    st.session_state["compiled_data"] = compiled_data
    st.session_state["unique_wavenumbers"] = unique_wavenumbers
    st.session_state["unique_wavenumbers_df"] = unique_wavenumbers_df

    # Clear loading animation after processing
    loading_placeholder.empty()
    st.write("Succes! 🤩")

st.markdown("<h3 style='color: red;'>Note: </h3>", unsafe_allow_html=True)
st.markdown("If you navigate between pages, it may seem like your previously imported files are gone. <br><span style='color:blue;'>Rest assured they remain loaded in memory.</span> <br> To clear the memory, click the `kebab menu` icon on the top right and then `clear cache`.", unsafe_allow_html=True)

st.write(compiled_data)