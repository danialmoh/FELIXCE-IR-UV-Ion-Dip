import sys
import streamlit as st

st.write("Python executable:", sys.executable)
st.write("Python path:", sys.path)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from amespahdbpythonsuite.amespahdb import AmesPAHdb
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


st.title("PAH Comparison: Experimental vs. Theoretical IR Spectrum")

# Check if experimental data exists in session state - first check for normalized data from 3.01
if "normalized_data" in st.session_state and not st.session_state.normalized_data.empty:
    st.success("Using normalized data from Section 3.01 (with power normalization)")
    # Convert the normalized data to the format needed for this section
    norm_df = st.session_state.normalized_data
    
    # Create a numpy array similar to what was used with fullrange_depletion_data
    # This allows the rest of the code to work without major changes
    # The structure should be: [wavenumber, sum_withoutIR, sum_withIR, depletion, -ln(depletion), ion_yield]
    if "ion_yield" in norm_df.columns:
        # Create a temporary dataframe with the columns we need
        temp_df = pd.DataFrame({
            "wavenumber": norm_df["wavenumber"],
            "sum_withoutIR": norm_df["sum_withoutIR"],
            "sum_withIR": norm_df["sum_withIR"],
            "depletion": norm_df["depletion"] if "depletion" in norm_df.columns else norm_df["sum_withIR"] / norm_df["sum_withoutIR"],
            "-ln(depletion)": norm_df["-ln(depletion)"] if "-ln(depletion)" in norm_df.columns else -np.log(norm_df["sum_withIR"] / norm_df["sum_withoutIR"]),
            "ion_yield": norm_df["ion_yield"]
        })
        
        # Convert to numpy array
        data = temp_df.values
        
        # Display info about using normalized ion yield instead of just depletion
        st.info("Using power-normalized ion yield for spectral comparison")
        
        # Option to use either ion_yield or -ln(depletion) for comparison
        use_ion_yield = st.checkbox("Use power-normalized ion yield (recommended)", value=True)
        
        if use_ion_yield:
            # Use the ion_yield column (power-normalized) which is at index 5
            data_column_index = 5
            column_name = "ion_yield"
        else:
            # Use the -ln(depletion) column which is at index 4
            data_column_index = 4
            column_name = "-ln(depletion)"
    else:
        st.warning("The normalized data does not contain an ion_yield column. Using -ln(depletion) instead.")
        data = norm_df.values
        data_column_index = 4
        column_name = "-ln(depletion)"
        
elif "fullrange_depletion_data" in st.session_state:
    # Use the original depletion data if normalized data is not available
    st.info("Using depletion data from Section 3.0 (without power normalization)")
    
    # Convert the stored depletion data (assumed to be in a format convertible to a NumPy array) 
    if isinstance(st.session_state.fullrange_depletion_data, pd.DataFrame):
        data = st.session_state.fullrange_depletion_data.values
    else:
        data = np.array(st.session_state.fullrange_depletion_data)
    
    # Default to using the -ln(depletion) column which is at index 4
    data_column_index = 4
    column_name = "-ln(depletion)"
else:
    st.error("No experimental data found. Please run Section 3.0 for depletion calculation or Section 3.01 for power normalization first.")
    st.stop()

# Add smoothing option for experimental data
st.sidebar.markdown("### Experimental Data Options")
if st.session_state.get("smoothed_experimental", False):
    st.sidebar.info("Data already smoothed in Section 3 – skipping re-smoothing")
    apply_smoothing = False
else:
    apply_smoothing = st.sidebar.checkbox("Apply Savitzky–Golay smoothing", value=False)


if apply_smoothing:
    # Add smoothing parameters
    window_size = st.sidebar.slider("Window size (must be odd)", 
                                   min_value=5, max_value=51, value=9, step=2)
    poly_order = st.sidebar.slider("Polynomial order", 
                                  min_value=1, max_value=5, value=2)
    
    # Apply Savitzky-Golay smoothing to the intensity data
    smoothed_intensity = savgol_filter(data[:, data_column_index], window_size, poly_order)
    
    # Create DataFrame with smoothed data
    exp_df = pd.DataFrame({
        "wavenumber": data[:, 0],
        "norm_intensity": smoothed_intensity / np.max(smoothed_intensity)
    })
    st.subheader(f"Experimental Spectrum (with smoothing) - using {column_name}")
else:
    # Use original unsmoothed data
    exp_df = pd.DataFrame({
        "wavenumber": data[:, 0],
        "norm_intensity": data[:, data_column_index] / np.max(data[:, data_column_index])
    })
    st.subheader(f"Experimental Spectrum - using {column_name}")

# Add theoretical spectrum options in sidebar
st.sidebar.markdown("### Theoretical Spectrum Options")
# Initialize shift_val in session state if it doesn't exist
if 'shift_val' not in st.session_state:
    st.session_state.shift_val = 0

# Add the shift slider to the sidebar
shift_val = st.sidebar.slider(
    "Shift Theoretical Spectrum (cm⁻¹)", 
    min_value=-50, 
    max_value=50, 
    value=st.session_state.shift_val,
    key="spectrum_shift_slider"
)
# right after shift_val…
show_stick = st.sidebar.checkbox("Show stick spectrum (discrete lines)", value=False)

# Store the current slider value in session state
st.session_state.shift_val = shift_val

st.dataframe(exp_df.head())

# --- Theoretical Spectrum Loader ---
st.markdown("##### Load Theoretical Spectrum from AmesPAHdb")
with st.form("theory_form"):
    xml_path = st.text_input(
        "Enter path to PAH XML file",
        value="/Users/danialmoh/Documents/Thesis/pahdb-complete-theoretical-v4.00-alpha.xml"  # Update with your XML file path
    )
    uid_input = st.text_input("Enter PAH UID (e.g., 18 for coronene)", value="495")
    conv_type = st.selectbox("Convolution Type", options=["Gaussian", "Lorentzian"], index=0)
    fwhm = st.number_input("FWHM for convolution (cm⁻¹)", value=15.0)
    submitted = st.form_submit_button("Load Theoretical Spectrum")

theory_df = None  # Initialize theory_df to None
if submitted:
    try:
        # Create the database instance (with caching and without online check)
        pahdb = AmesPAHdb(filename=xml_path, check=False, cache=True)
        uid = int(uid_input)
        transitions = pahdb.gettransitionsbyuid([uid])
        # … after you’ve got `transitions = pahdb.gettransitionsbyuid([uid])` …
        sticks      = transitions.get()

        # 1) Pull out the list of transition dicts for this UID
        raw_list    = sticks["data"][uid]  

        # 2) Build arrays of frequencies & intensities
        raw_freqs   = np.array([ entry["frequency"] for entry in raw_list ])
        raw_ints    = np.array([ entry["intensity"] for entry in raw_list  ])
        # … your existing raw_freqs/raw_ints extraction …

        


        # 3) Normalize so the tallest stick is 1.0
        if raw_ints.max() != 0:
            raw_ints = raw_ints / raw_ints.max()
        # store for later
        st.session_state["raw_freqs"] = raw_freqs
        st.session_state["raw_ints"]  = raw_ints
        # 4) Plot as a stem (stick) chart
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.stem(
            raw_freqs,
            raw_ints,
            linefmt="C1-",    # red sticks
            markerfmt=" ",    # no dot at top
            basefmt="k-"      # black baseline
        )
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Relative intensity")
        ax.invert_xaxis()    # IR convention: high → low
        ax.set_title(f"Raw stick spectrum for UID {uid}")

        # 5) Show in Streamlit
        st.pyplot(fig)







        # Convolve the stick spectrum; use Gaussian if conv_type equals "Gaussian"
        spectrum = transitions.convolve(fwhm=fwhm, gaussian=(conv_type == "Gaussian"), multiprocessing=False)
        
        # Debug: Print information about what's returned
        try:
            result = spectrum.get()  # Get arrays for frequency and intensity
            st.write(f"Debug - Result type: {type(result)}")
            st.write(f"Debug - Result shape or length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            # Create a more robust handling of the result
            if isinstance(result, tuple):
                # If it's a tuple, take the first two elements
                if len(result) >= 2:
                    freq = result[0]
                    conv_intensity = result[1]
                else:
                    raise ValueError(f"Expected at least 2 elements in tuple, got {len(result)}")
            elif isinstance(result, dict):
                # If it's a dictionary, look for the data in the 'data' key
                st.write("Processing result as dictionary with keys:", list(result.keys()))
                
                # The result has a specific structure from AmesPAHdb:
                # - 'grid' key contains frequency values
                # - 'data' key contains a dictionary with UIDs as keys and intensity arrays as values
                if 'data' in result and 'grid' in result:
                    st.write("Found 'data' and 'grid' keys in the result dictionary")
                    data = result['data']
                    grid = result['grid']
                    
                    st.write(f"Data type: {type(data)}, Shape or Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    st.write(f"Grid type: {type(grid)}, Shape or Length: {len(grid) if hasattr(grid, '__len__') else 'N/A'}")
                    
                    # Get the UID we're working with
                    if 'uids' in result:
                        st.write(f"UIDs in result: {result['uids']}")
                    
                    # The data dictionary has UIDs as keys and intensity arrays as values
                    if isinstance(data, dict) and len(data) > 0:
                        # Get the first (or specified) UID
                        uid = int(uid_input)  # We already have this from user input
                        if uid in data:
                            # The grid (frequency) is the same for all UIDs
                            freq = grid
                            # Get the intensity for this specific UID
                            conv_intensity = data[uid]
                            
                            st.success(f"Successfully extracted spectral data for UID {uid}")
                        else:
                            available_uids = list(data.keys())
                            st.warning(f"UID {uid} not found in data. Available UIDs: {available_uids}")
                            
                            # Use the first available UID
                            first_uid = available_uids[0]
                            st.info(f"Using first available UID: {first_uid}")
                            
                            freq = grid
                            conv_intensity = data[first_uid]
                    else:
                        # Display the data structure for debugging
                        st.error(f"Unexpected structure in 'data' key: {type(data)}")
                        st.write("Data content (first few items):", str(data)[:1000] + "..." if len(str(data)) > 1000 else str(data))
                        raise ValueError(f"Unsupported data structure in 'data' key: {type(data)}")
                else:
                    # Display all keys and their types/shapes for debugging
                    st.warning("Required 'data' and/or 'grid' keys not found in result")
                    st.write("Keys and their content types:")
                    for key in result:
                        value = result[key]
                        shape_info = getattr(value, 'shape', len(value) if hasattr(value, '__len__') else 'N/A')
                        st.write(f"  {key}: {type(value)}, Shape/Length: {shape_info}")
                        
                        # If this is a numpy array or list with numeric data, it might be our spectrum
                        if isinstance(value, (np.ndarray, list)) and hasattr(value, '__len__') and len(value) > 0:
                            st.write(f"  Sample of {key}: {str(value)[:100]}...")
                    
                    raise ValueError("Could not find spectral data in the dictionary. Please check the structure of the result.")
            elif isinstance(result, np.ndarray) and result.ndim == 2:
                # If it's a 2D numpy array, the first column is freq, second is intensity
                freq = result[:, 0]
                conv_intensity = result[:, 1]
            else:
                # Otherwise, try the original unpacking (this will likely fail with a dictionary)
                st.write("Attempting original unpacking...")
                try:
                    freq, conv_intensity = result
                except ValueError as e:
                    st.error(f"Could not unpack result: {e}")
                    st.write("Full result data:", result)
                    raise
            
            # Print more information about freq and conv_intensity
            st.write(f"Debug - freq type: {type(freq)}, shape: {getattr(freq, 'shape', 'N/A')}")
            st.write(f"Debug - conv_intensity type: {type(conv_intensity)}, shape: {getattr(conv_intensity, 'shape', 'N/A')}")
            
            # Ensure both are numpy arrays
            if not isinstance(freq, np.ndarray):
                st.warning("Converting frequency to numpy array")
                freq = np.array(freq)
            if not isinstance(conv_intensity, np.ndarray):
                st.warning("Converting intensity to numpy array")
                conv_intensity = np.array(conv_intensity)
            
            # Normalize the intensity
            try:
                max_intensity = np.max(conv_intensity)
                st.write(f"Maximum intensity: {max_intensity}")
                norm_conv_intensity = conv_intensity / max_intensity if max_intensity != 0 else conv_intensity
            except Exception as e:
                st.error(f"Error normalizing intensity: {str(e)}")
                st.write("This might indicate that the intensity data is not numeric.")
                raise
            
            # Create DataFrame
            theory_df = pd.DataFrame({
                "wavenumber": freq,
                "norm_intensity": norm_conv_intensity
            })
            
            st.subheader("Theoretical Spectrum")
            st.dataframe(theory_df.head())
            
        except Exception as e:
            st.error(f"Error loading theoretical spectrum: {str(e)}")
            st.write("Exception details:", e)
            import traceback
            st.code(traceback.format_exc())
            theory_df = None
    except Exception as e:
        st.error(f"Error with AmesPAHdb: {str(e)}")
        st.write("Exception details:", e)
        import traceback
        st.code(traceback.format_exc())
        theory_df = None

# --- Generate the Comparison Plot ---
if theory_df is not None:
    # Get the shift value from session state (set in the sidebar)
    shift_val = st.session_state.shift_val
    
    # Apply the shift to the theoretical spectrum
    theory_df['wavenumber_shifted'] = theory_df['wavenumber'] + shift_val

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=exp_df['wavenumber'],
        y=exp_df['norm_intensity'],
        mode='lines',
        name="Experimental",
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=theory_df['wavenumber_shifted'],
        y=theory_df['norm_intensity'],
        mode='lines',
        name="Theoretical",
        line=dict(color='red', width=2)
    ))
         # stick overlay?
    if show_stick and "raw_freqs" in st.session_state:
        rf = st.session_state["raw_freqs"] + shift_val
        ri = st.session_state["raw_ints"]   # these are already normalized to 1.0
        fig.add_trace(go.Bar(
            x=rf,
            y=ri,
            width=1,
            opacity=0.6,
            name="Raw sticks",
            marker_line_width=0
        ))
    fig.update_layout(
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Normalized Intensity",
        title="Experimental vs. Theoretical IR Spectrum"
    )
    # Invert the x-axis for IR convention.
    fig.update_xaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    try:
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="⬇️ Download spectrum as PNG",
            data=img_bytes,
            file_name=f"spectrum_UID{uid}.png",
            mime="image/png"
        )
    except Exception:
        st.warning(
            "Could not generate download PNG (kaleido might be missing). "
            "Please install `kaleido` or save manually."
        )
            # after st.plotly_chart(fig,...)
    # Fallback download as HTML
    html_bytes = fig.to_html().encode("utf-8")
    st.download_button(
        label="⬇️ Download spectrum as HTML",
        data=html_bytes,
        file_name=f"spectrum_UID{uid}.html",
        mime="text/html"
    )

else:
    st.info("Theoretical data not loaded. Please fill in the form and click 'Load Theoretical Spectrum'.")
