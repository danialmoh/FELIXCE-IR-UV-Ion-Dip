import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
import re
# from pyteomics import mass
# import IsoSpecPy



st.markdown("# Peak Detection and Candidate Formula Matching")
st.markdown(
    "This page performs three steps on the MegaSum data:\n\n"
    "1. **Peak Detection:** Detect peaks from a baseline‐corrected mass spectrum.\n"
    "2. **Candidate Formula Matching:** For each detected peak, generate candidate formulas assuming the fragment is either CH-only or CHBr (using ^79Br). Chemical rules (saturation, DBE) are used.\n"
    "3. **Plot Candidate Annotations:** Create a static matplotlib plot with arrows annotating the candidate formulas. You can also add custom peaks.\n\n"
    "You can either use the data already stored in session_state or upload a CSV file."
)
# st.markdown(
#     """
#     # Peak Detection and Candidate Formula Matching

#     This interactive Streamlit application provides a comprehensive workflow for analyzing mass spectrometry data—specifically, MegaSum data—by integrating several key processing steps:

#     **Data Source & Signal Selection:**  
#     Users can either upload a CSV file (which must include an `x_mass` column for the mass-to-charge axis) or use existing session data. The app supports selecting between two types of baseline‐corrected signals (with or without IR).

#     **Parameter Settings:**  
#     Prior to analysis, users set parameters such as the minimum peak height, minimum distance between peaks, and the mass tolerance for candidate formula matching. These settings tailor the subsequent analyses to the specific characteristics of the data.

#     **Step 1 – Peak Detection:**  
#     The application uses `scipy.signal.find_peaks` to automatically detect peaks in the chosen mass spectrum. Detected peaks are displayed both in a data table and on an interactive Plotly plot, allowing users to visually inspect the results.

#     **Step 2 – Candidate Formula Matching:**  
#     For each detected peak, the code generates candidate chemical formulas based on two categories:
#     - **CH-only formulas:** Options include alkane, alkene, alkyne, cyclic, and highly unsaturated (PAH) types.
#     - **CHBr formulas:** Multiple candidate types are available (e.g., haloalkanes, haloalkenes, polybrominated compounds) with user-specified parameters such as degrees of unsaturation and maximum Br atoms.
    
#     Custom functions calculate the expected molecular formula based on the input parameters and the measured m/z values. The candidate formulas are then summarized in a table, with an option to download the results as a CSV file.

#     **Step 3 – Plot Candidate Annotations:**  
#     Two types of visualizations are provided:
#     - A **static Matplotlib plot** that shows the full spectrum with annotated candidate peaks and a corresponding legend.
#     - An **interactive Plotly plot** that enhances data exploration by displaying candidate details on hover.
    
#     Users can also customize the plot range and add custom peak annotations.

#     **Additional Visualization – Fragmentation/Parent/Clustering View:**  
#     An extra plot highlights the "parent" peak (the highest intensity peak) and visually delineates the fragmentation (left of the parent) and clustering (right of the parent) regions. This aids in understanding the relationship between different regions of the spectrum.

#     **Optional Isotope Pattern Matching (Commented Out):**  
#     The code includes sections (currently commented out) that implement isotope pattern matching using libraries such as IsoSpecPy and Pyteomics. When enabled, these sections compute theoretical isotopic patterns and compare them with observed data.

#     Overall, this application serves as an all-in-one tool for mass spectrometry analysis—ranging from data upload and preprocessing, through peak detection and candidate formula matching, to detailed spectral visualization and optional isotope analysis.
#     """
# )
#####################################
# Data Source Selection
#####################################
data_source = st.radio("Select Data Source", options=["Use data from session", "Upload CSV file"], key="data_source")
if data_source == "Upload CSV file":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
    if uploaded_file is not None:
        # Expecting a column "x_mass" for the mass axis.
        df = pd.read_csv(uploaded_file)
        if "x_mass" in df.columns:
            x_mass = df["x_mass"].values
            MegaSum = df  # Use the entire DataFrame as MegaSum
            st.session_state["MegaSum"] = MegaSum
            st.session_state["x_mass"] = x_mass
            st.success("CSV file uploaded and data loaded!")
        else:
            st.error("CSV file must contain a column named 'x_mass'.")
else:
    # Use data already stored in session_state
    MegaSum = st.session_state.get("MegaSum", None)
    x_mass = st.session_state.get("x_mass", None)
    if MegaSum is None or x_mass is None:
        st.error("No data found in session_state. Please upload a CSV file or run the previous sections.")
        st.stop()

#####################################
# --- Select the Signal to Analyze ---
#####################################
st.markdown("## Signal Selection")
signal_choice = st.radio("Select which baseline‐corrected signal to analyze:", options=["With IR", "Without IR"], key="signal_choice")
if signal_choice == "With IR":
    if "baseline_corrected_signal_withIR" in MegaSum.columns:
        signal = MegaSum["baseline_corrected_signal_withIR"].values
        signal_label = "Baseline Corrected Signal With IR"
    else:
        st.error("No 'baseline_corrected_signal_withIR' found. Falling back to 'without IR'.")
        if "baseline_corrected_signal_withoutIR" in MegaSum.columns:
            signal = MegaSum["baseline_corrected_signal_withoutIR"].values
            signal_label = "Baseline Corrected Signal Without IR"
        else:
            st.error("No baseline corrected signal available.")
            st.stop()
else:
    if "baseline_corrected_signal_withoutIR" in MegaSum.columns:
        signal = MegaSum["baseline_corrected_signal_withoutIR"].values
        signal_label = "Baseline Corrected Signal Without IR"
    else:
        st.error("No 'baseline_corrected_signal_withoutIR' found. Falling back to 'with IR'.")
        if "baseline_corrected_signal_withIR" in MegaSum.columns:
            signal = MegaSum["baseline_corrected_signal_withIR"].values
            signal_label = "Baseline Corrected Signal With IR"
        else:
            st.error("No baseline corrected signal available.")
            st.stop()

#####################################
# --- Parameter Settings (Always visible) ---
#####################################
st.markdown("## Parameter Settings")
default_height = float(0.1 * np.max(signal))
height_threshold = st.number_input(
    "Set minimum peak height (absolute intensity)",
    value=default_height,
    step=0.1,
    key="height_threshold"
)
min_distance = st.number_input(
    "Set minimum distance between peaks (in index units)",
    value=3,
    step=1,
    key="min_distance"
)
formula_tol = st.number_input(
    "Mass tolerance for formula matching (Da)",
    value=1.00,
    step=0.01,
    key="formula_tol"
)

#####################################
# STEP 1: PEAK DETECTION
#####################################
if st.button("Run Peak Detection"):
    st.markdown("## Step 1: Automated Peak Detection")
    st.markdown("Detecting peaks using `scipy.signal.find_peaks` with the parameters defined above.")
    
    # Detect peaks using the user-specified parameters
    peaks, properties = find_peaks(signal, height=height_threshold, distance=min_distance)
    detected_mz = np.array(x_mass)[peaks]
    
    st.write(f"**Detected {len(detected_mz)} peaks.**")
    peaks_df = pd.DataFrame({
        "Peak Index": peaks,
        "m/z": detected_mz,
        "Intensity": signal[peaks]
    })
    st.dataframe(peaks_df)
    
    # Plot the signal with detected peaks using Plotly
    fig_signal = go.Figure()
    fig_signal.add_trace(go.Scatter(
        x=x_mass, y=signal, mode="lines", name=signal_label
    ))
    fig_signal.add_trace(go.Scatter(
        x=detected_mz, y=signal[peaks],
        mode="markers", marker=dict(color="red", size=8),
        name="Detected Peaks"
    ))
    fig_signal.update_layout(
        title="Signal with Detected Peaks",
        xaxis_title="Mass (amu)",
        yaxis_title="Intensity"
    )
    st.plotly_chart(fig_signal)
    
    # Save detected peaks and signal into session_state
    st.session_state["detected_peaks_df"] = peaks_df
    st.session_state["detected_mz"] = detected_mz
    st.session_state["signal"] = signal
    st.success("Peak detection completed and stored in session_state!")

#####################################
# STEP 2: CANDIDATE FORMULA MATCHING
#####################################

st.markdown("## Candidate Formula Matching Options")

# Let the user choose candidate formula types for CH-only and CHBr
NAPH_CH_OPTION = "Alkylated naphthalene core (C10H8 base + alkyl chains)"

candidate_options_CH = st.multiselect(
    "Choose candidate formula types for CH-only:",
    options=[
        "Alkane (CnH2n+2)",
        "Alkene (CnH2n)",
        "Alkyne (CnH2n-2)",
        "Cyclic (CnH2n)",
        "Highly Unsaturated (PAH: CnH2n+2-2u)",
        NAPH_CH_OPTION,
    ],
    default=[
        "Highly Unsaturated (PAH: CnH2n+2-2u)",
        "Alkane (CnH2n+2)",
        "Alkene (CnH2n)",
        "Alkyne (CnH2n-2)",
        "Cyclic (CnH2n)",
        NAPH_CH_OPTION,
    ],
)

# --- Updated Candidate Options for CHBr ---
candidate_options_CHBr = st.multiselect(
    "Choose candidate formula types for CHBr:",
    options=[
        "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)",
        "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)",
        "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)",
        "Highly Unsaturated (PAH: CnH2n+2-2u)",
        "Polybrominated Alkanes (CnH(2n+2-x)Brx)",
        "Polybrominated Alkenes (CnH(2n-x)Brx)",
        "Polybrominated Alkynes (CnH(2n-2-x)Brx)",
        "Polybrominated Biphenyls (PBBs) (C12H(10-x)Brx)",
        "Polybrominated Aromatic Compounds (CnH(n-x)Brx)",
        "Radicals (Reactive Intermediates) (CnH(2n+1−x)Brx•)",
        "Carbenes (CnH(2n)Br)",
        "Aromatic Halides (Haloarenes) (C6H(6−x)Brx)",
        "Allylic Halides (CnH(2n-1)Br)",
        "Benzylic Halides (C6H5-CH(3−x)Brx)",
        "Perhalogenated Hydrocarbons (CH(4−x)Brx)",
        "Atmospheric Bromine Radicals (CnH(2n+1−x)Brx•)"
    ],
    default=[
        "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)",
        "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)",
        "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)",
    ]
)

# For the "Highly Unsaturated" option, allow specifying degrees of unsaturation (u)
if ("Highly Unsaturated (PAH: CnH2n+2-2u)" in candidate_options_CH or 
    "Highly Unsaturated (PAH: CnH2n+2-2u)" in candidate_options_CHBr):
    min_u = st.number_input("Minimum degrees of unsaturation (u)", value=1, min_value=0, step=1)
    max_u = st.number_input("Maximum degrees of unsaturation (u)", value=8, min_value=0, step=1)
else:
    min_u, max_u = None, None

# Configuration for alkylated naphthalene candidates (C10 core + alkyl chains)
if NAPH_CH_OPTION in candidate_options_CH:
    min_n_subs = int(
        st.number_input(
            "Minimum number of methyl substituents on naphthalene core (n)",
            value=0,
            min_value=0,
            max_value=8,
            step=1,
        )
    )
    max_n_subs = int(
        st.number_input(
            "Maximum number of methyl substituents on naphthalene core (n)",
            value=4,
            min_value=min_n_subs,
            max_value=8,
            step=1,
        )
    )
    min_x_total = int(
        st.number_input(
            "Minimum total added alkyl carbons (x)",
            value=max(2, min_n_subs),
            min_value=0,
            max_value=40,
            step=1,
        )
    )
    max_x_total = int(
        st.number_input(
            "Maximum total added alkyl carbons (x)",
            value=max(6, min_x_total),
            min_value=max(min_x_total, min_n_subs),
            max_value=40,
            step=1,
        )
    )
else:
    min_n_subs = max_n_subs = min_x_total = max_x_total = None

# Maximum number of Br atoms for CHBr candidates
max_Br = st.number_input("Maximum number of Br atoms (for CHBr candidates)", value=3, min_value=1, step=1)

# Element range for carbon is fixed (3 to 30)
min_C = 2
max_C = 30

# Mass tolerance (in Da) for matching candidate formulas
formula_tol = st.number_input("Mass tolerance for formula matching (Da)", value=1.00, step=0.01)

# Define a helper function that returns the expected hydrogen count given a candidate type.
def hydrogen_count(candidate_type, n_C, u=None):
    if candidate_type == "Alkane (CnH2n+2)":
        return 2 * n_C + 2
    elif candidate_type == "Alkene (CnH2n)":
        return 2 * n_C
    elif candidate_type == "Alkyne (CnH2n-2)":
        return 2 * n_C - 2
    elif candidate_type == "Cyclic (CnH2n)":
        return 2 * n_C
    elif candidate_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
        if u is None:
            u = min_u  # default to min_u if not provided
        return 2 * n_C + 2 - 2 * u
    elif candidate_type == NAPH_CH_OPTION:
        # Hydrogen count handled explicitly in candidate generation for this option
        return None
    else:
        return None

# Function to generate candidate formulas for CH-only
def candidate_formulas_CH(target_mz, tol, candidate_types):
    candidates = []
    for cand_type in candidate_types:
        # Handle alkylated naphthalene separately (doesn't use the standard carbon loop)
        if cand_type == NAPH_CH_OPTION and min_n_subs is not None:
            base_c = 10
            base_h = 8
            for n_subs in range(min_n_subs, max_n_subs + 1):
                for x_total in range(max(n_subs, min_x_total), max_x_total + 1):
                    total_c = base_c + x_total
                    total_h = base_h + 2 * x_total
                    calc_mass = total_c * 12.0000 + total_h * 1.007825
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(
                            f"{cand_type}: C{total_c}H{total_h} (n={n_subs}, x={x_total})"
                        )
        else:
            # Standard formula types (alkane, alkene, etc.)
            for n_C in range(min_C, max_C + 1):
                if cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                    for u in range(int(min_u), int(max_u) + 1):
                        n_H = hydrogen_count(cand_type, n_C, u)
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type}: C{n_C}H{n_H} (u={u})")
                else:
                    n_H = hydrogen_count(cand_type, n_C)
                    if n_H is not None and n_H < 1:
                        continue
                    if n_H is not None:
                        calc_mass = n_C * 12.0000 + n_H * 1.007825
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type}: C{n_C}H{n_H}")
    return candidates

# Function to generate candidate formulas for CHBr
# --- Updated Function to Generate CHBr Candidate Formulas ---
def candidate_formulas_CHBr(target_mz, tol, candidate_types):
    Br_mass = 78.9183  # monoisotopic mass for ^79Br
    candidates = []
    for cand_type in candidate_types:
        if cand_type == "Alkyl Halides (Haloalkanes) (CnH(2n+1)Br)":
            # n_C starts at 1 (e.g. CH3Br when n_C=1)
            for n_C in range(1, max_C + 1):
                n_H = 2 * n_C + 1
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
                    
        elif cand_type == "Alkenyl Halides (Haloalkenes) (CnH(2n-1)Br)":
            # At least 2 carbons are needed for a double bond.
            for n_C in range(2, max_C + 1):
                n_H = 2 * n_C - 1
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
                    
        elif cand_type == "Alkynyl Halides (Haloalkynes) (CnH(2n-3)Br)":
            # At least 2 carbons are needed for a triple bond.
            for n_C in range(2, max_C + 1):
                n_H = 2 * n_C - 3
                if n_H < 1:
                    continue
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
                    
        elif cand_type == "Polybrominated Alkanes (CnH(2n+2-x)Brx)":
            # Polybrominated implies more than one Br (x ≥ 2)
            for k in range(2, max_Br + 1):
                for n_C in range(1, max_C + 1):
                    n_H = 2 * n_C + 2 - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
                        
        elif cand_type == "Polybrominated Alkenes (CnH(2n-x)Brx)":
            for k in range(2, max_Br + 1):
                for n_C in range(2, max_C + 1):
                    n_H = 2 * n_C - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
                        
        elif cand_type == "Polybrominated Alkynes (CnH(2n-2-x)Brx)":
            for k in range(2, max_Br + 1):
                for n_C in range(2, max_C + 1):
                    n_H = 2 * n_C - 2 - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
        
        elif cand_type =="Highly Unsaturated (PAH: CnH2n+2-2u)":
            # Existing logic: loop over Br count and carbon numbers
            for k in range(1, max_Br + 1):
                for n_C in range(min_C, max_C + 1):
                    if cand_type == "Highly Unsaturated (PAH: CnH2n+2-2u)":
                        for u in range(int(min_u), int(max_u) + 1):
                            n_H = hydrogen_count(cand_type, n_C, u)
                            if n_H < 1:
                                continue
                            calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                            if abs(calc_mass - target_mz) <= tol:
                                candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k} (u={u})")
                    else:
                        n_H = hydrogen_count(cand_type, n_C)
                        if n_H < 1:
                            continue
                        calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                        if abs(calc_mass - target_mz) <= tol:
                            candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
                
                        
        elif cand_type == "Polybrominated Biphenyls (PBBs) (C12H(10-x)Brx)":
            # Fixed carbon count: C12
            n_C = 12
            for k in range(1, max_Br + 1):
                n_H = 10 - k
                if n_H < 1:
                    continue
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type} (Br{k}): C12H{n_H}Br{k}")
                    
        elif cand_type == "Polybrominated Aromatic Compounds (CnH(n-x)Brx)":
            # Assume an aromatic system with n_C ≥ 6 where each Br replaces one H.
            for n_C in range(6, max_C + 1):
                for k in range(1, max_Br + 1):
                    n_H = n_C - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}")
                        
        elif cand_type == "Radicals (Reactive Intermediates) (CnH(2n+1−x)Brx•)":
            # Loop over variable Br count (x) and adjust hydrogen count.
            for k in range(1, max_Br + 1):
                for n_C in range(1, max_C + 1):
                    n_H = 2 * n_C + 1 - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}•")
                        
        elif cand_type == "Carbenes (CnH(2n)Br)":
            for n_C in range(1, max_C + 1):
                n_H = 2 * n_C
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
                    
        elif cand_type == "Aromatic Halides (Haloarenes) (C6H(6−x)Brx)":
            # Fixed aromatic ring: C6.
            n_C = 6
            for k in range(1, max_Br + 1):
                n_H = 6 - k
                if n_H < 1:
                    continue
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type} (Br{k}): C6H{n_H}Br{k}")
                    
        elif cand_type == "Allylic Halides (CnH(2n-1)Br)":
            for n_C in range(1, max_C + 1):
                n_H = 2 * n_C - 1
                if n_H < 1:
                    continue
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type}: C{n_C}H{n_H}Br")
                    
        elif cand_type == "Benzylic Halides (C6H5-CH(3−x)Brx)":
            # Fixed benzyl structure: a benzene ring (C6H5) plus CH group.
            n_C = 7
            for k in range(1, min(max_Br, 3) + 1):
                n_H = 8 - k
                if n_H < 1:
                    continue
                calc_mass = 7 * 12.0000 + n_H * 1.007825 + k * Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type} (Br{k}): C7H{n_H}Br{k}")
                    
        elif cand_type == "Perhalogenated Hydrocarbons (CH(4−x)Brx)":
            # Single carbon system.
            n_C = 1
            for k in range(1, 5):  # x can vary from 1 to 4.
                n_H = 4 - k
                if n_H < 0:
                    continue
                calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                if abs(calc_mass - target_mz) <= tol:
                    candidates.append(f"{cand_type} (Br{k}): CH{n_H}Br{k}")
                    
        elif cand_type == "Atmospheric Bromine Radicals (CnH(2n+1−x)Brx•)":
            # Similar to the 'Radicals' option.
            for k in range(1, max_Br + 1):
                for n_C in range(1, max_C + 1):
                    n_H = 2 * n_C + 1 - k
                    if n_H < 1:
                        continue
                    calc_mass = n_C * 12.0000 + n_H * 1.007825 + k * Br_mass
                    if abs(calc_mass - target_mz) <= tol:
                        candidates.append(f"{cand_type} (Br{k}): C{n_C}H{n_H}Br{k}•")
                        
    return candidates

# Run candidate matching when the button is pressed
if st.button("Run Candidate Formula Matching"):
    st.markdown("## Step 2: Candidate Formula Matching")
    if "detected_mz" not in st.session_state:
        st.error("No detected peaks found. Please run peak detection first.")
        st.stop()
    
    detected_mz = st.session_state["detected_mz"]
    
    candidate_data = []
    for mz_val in detected_mz:
        ch_candidates = candidate_formulas_CH(mz_val, tol=formula_tol, candidate_types=candidate_options_CH)
        chbr_candidates = candidate_formulas_CHBr(mz_val, tol=formula_tol, candidate_types=candidate_options_CHBr)
        # Only add the candidate if at least one candidate exists in either category
        if ch_candidates or chbr_candidates:
            candidate_data.append({
                "m/z": mz_val,
                "CH candidates": "; ".join(ch_candidates) if ch_candidates else "",
                "CHBr candidates": "; ".join(chbr_candidates) if chbr_candidates else ""
            })
    if candidate_data:
        candidates_df = pd.DataFrame(candidate_data)
        st.dataframe(candidates_df)
        st.session_state["candidates_df"] = candidates_df
    #     raw_candidate_list = []
    # # Loop through each row in the candidates DataFrame.
    #     for _, row in candidates_df.iterrows():
    #         mz_val = row["m/z"]
    #         # If you have an intensity column, use it; otherwise, set to None.
    #         intensity_val = row["Intensity"] if "Intensity" in row else None
            
    #         # Process CH candidates
    #         if row["CH candidates"]:
    #             for cand in row["CH candidates"].split(";"):
    #                 cand = cand.strip()
    #                 if ":" in cand:
    #                     # Only keep the part after the colon (e.g. "C7H14Br")
    #                     cand = cand.split(":", 1)[1].strip()
    #                 if cand:
    #                     raw_candidate_list.append({
    #                         "Formula": cand,
    #                         "m/z": mz_val,
    #                         "Intensity": intensity_val
    #                     })
                        
    #         # Process CHBr candidates
    #         if row["CHBr candidates"]:
    #             for cand in row["CHBr candidates"].split(";"):
    #                 cand = cand.strip()
    #                 if ":" in cand:
    #                     cand = cand.split(":", 1)[1].strip()
    #                 if cand:
    #                     raw_candidate_list.append({
    #                         "Formula": cand,
    #                         "m/z": mz_val,
    #                         "Intensity": intensity_val
    #                     })
        
    #     # Create the raw candidates DataFrame with columns: Formula, m/z, and Intensity.
    #     candidates_raw_df = pd.DataFrame(raw_candidate_list)
    #     st.markdown("### Raw Candidate Formulas with Mass and Intensity")
    #     st.dataframe(candidates_raw_df)
        
    #     # Optionally store in session_state if needed later.
    #     st.session_state["candidates_raw_df"] = candidates_raw_df
    # else:
    #     st.error("candidates_df not found. Please run candidate formula matching first.")

        # Provide a download button for candidate formulas CSV
        csv_candidates = candidates_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Candidate Formulas CSV",
            data=csv_candidates,
            file_name="candidate_formulas.csv",
            mime="text/csv"
        )
        st.success("Candidate formula matching completed!")

#####################################
# STEP 3: PLOT CANDIDATE ANNOTATIONS
#####################################
#####################################
# STEP 3: PLOT CANDIDATE ANNOTATIONS
#####################################
st.markdown("## Plot Candidate Annotations Options")

# Plot range options
x_min = st.number_input("X-axis minimum (amu)", value=float(x_mass[0]))
x_max = st.number_input("X-axis maximum (amu)", value=float(x_mass[-1]))
y_max = st.number_input("Y-axis maximum (Intensity)", value=float(np.max(st.session_state.get('signal', []))), step=1.0)

# OPTIONAL: Minimum intensity threshold for annotating peaks in the static plot
plot_intensity_threshold = st.number_input(
    "Set minimum intensity for plotting annotations (absolute intensity)",
    value=height_threshold,  # defaults to the earlier set minimum peak height
    step=0.1,
    key="plot_intensity_threshold"
)

# OPTIONAL: Allow user to add custom peak values
st.markdown("### Optional: Add Custom Peaks")
st.markdown("Enter custom peaks (one per line) in the format: `m/z, intensity, annotation` (annotation is optional).")
custom_peaks_input = st.text_area("Custom Peaks", value="", height=150)
custom_peaks = []
if custom_peaks_input.strip():
    for line in custom_peaks_input.splitlines():
        parts = line.split(',')
        try:
            mz_val = float(parts[0].strip())
            intensity_val = float(parts[1].strip()) if len(parts) >= 2 else None
            annotation = parts[2].strip() if len(parts) >= 3 else "Custom"
            if intensity_val is None:
                continue
            custom_peaks.append({"m/z": mz_val, "Intensity": intensity_val, "Annotation": annotation})
        except Exception as e:
            st.warning(f"Could not parse line: {line}")

# Annotation style option
annotation_style = st.radio("Select Annotation Style", options=["Mass Value", "Numeric Label"])

if st.button("Plot Candidate Annotations"):
    st.markdown("## Step 3: Static Plot with Candidate Annotations (Two Panels)")
    # Check for required data in session_state
    if ("detected_peaks_df" not in st.session_state or
        "candidates_df" not in st.session_state or
        "signal" not in st.session_state):
        st.error("Please run both peak detection and candidate matching steps first.")
        st.stop()
    
    peaks_df = st.session_state["detected_peaks_df"]
    candidates_df = st.session_state["candidates_df"]
    signal = st.session_state["signal"]
    
    # Filter peaks to include only those with candidate formulas and above the intensity threshold
    matched_peaks = peaks_df[
        peaks_df["m/z"].isin(candidates_df["m/z"]) &
        (peaks_df["Intensity"] >= plot_intensity_threshold)
    ]
    
    # If custom peaks were provided, merge them in
    if custom_peaks:
        custom_df = pd.DataFrame(custom_peaks)
        if "Annotation" not in custom_df.columns:
            custom_df["Annotation"] = "Custom"
        matched_peaks = pd.concat([matched_peaks, custom_df], ignore_index=True)
    
    # Filter out peaks that fall outside the x_min and x_max range
    matched_peaks = matched_peaks[(matched_peaks["m/z"] >= x_min) & (matched_peaks["m/z"] <= x_max)]
    
    # Create a figure with two panels: left for the spectrum, right for the detailed legend
    fig, (ax_spectrum, ax_legend) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(14, 6))
    
    # --- Left Panel: Spectrum ---
    ax_spectrum.plot(x_mass, signal, label=signal_label, color="blue")
    ax_spectrum.scatter(matched_peaks["m/z"], matched_peaks["Intensity"], color="red", zorder=5)
    ax_spectrum.set_xlabel("Mass (amu)")
    ax_spectrum.set_ylabel("Intensity")
    ax_spectrum.set_title("Spectrum with Detected Peaks")
    ax_spectrum.set_xlim(x_min, x_max)
    ax_spectrum.set_ylim(0, y_max)
    
    # Annotate each detected peak according to the chosen style and build legend entries
    offset = 0.05 * y_max  # vertical offset for annotation text
    legend_entries = []
    
    for i, (_, row) in enumerate(matched_peaks.iterrows(), start=1):
        mz_val = row["m/z"]
        intensity_val = row["Intensity"]
        # Retrieve candidate details from candidates_df (if available)
        cand_row = candidates_df[candidates_df["m/z"] == mz_val]
        if not cand_row.empty:
            ch_cand = cand_row["CH candidates"].values[0]
            chbr_cand = cand_row["CHBr candidates"].values[0]
            details = ""
            if ch_cand:
                details += f"CH: {ch_cand}"
            if chbr_cand:
                details += f"\nCHBr: {chbr_cand}"
        else:
            details = row.get("Annotation", "Custom")
        
        if annotation_style == "Numeric Label":
            annotation_text = str(i)
            legend_entry = f"Peak {i}: m/z = {mz_val:.4f}, intensity = {intensity_val:.2f}\n{details}"
        else:  # Mass Value style
            annotation_text = f"{mz_val:.2f}"
            legend_entry = f"m/z = {mz_val:.2f}, intensity = {intensity_val:.2f}\n{details}"
        
        # Annotate on the spectrum
        ax_spectrum.annotate(
            annotation_text,
            xy=(mz_val, intensity_val),
            xytext=(mz_val, intensity_val + offset),
            ha='center',
            fontsize=10,
            fontweight='bold',
            color="black",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5)
        )
        legend_entries.append(legend_entry)
    
    # --- Right Panel: Detailed Legend ---
    ax_legend.axis('off')  # remove axes for a clean text panel
    legend_text = "\n\n".join(legend_entries)
    ax_legend.text(0, 1, legend_text, va='top', ha='left', fontsize=10, wrap=True)
    ax_legend.set_title("Peak Assignments", fontsize=12)
    
    # Adjust layout for proper scaling and spacing
    plt.tight_layout()
    
    # SAVE THE FIGURE TO THE DEFAULT DIRECTORY
    directory = st.session_state.get("file_directory", "./output")
    filename = f"{directory}/spectrum_{x_min:.2f}_{x_max:.2f}_{y_max:.2f}_{annotation_style.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    st.success(f"Figure saved as '{filename}'.")
    
    st.pyplot(fig)

# ------------------------------
# ADDITIONAL INTERACTIVE PLOT
# ------------------------------
if st.button("Interactive Plot with Candidate Annotations"):
    st.markdown("## Interactive Plot: Spectrum with Candidate Annotations")
    
    # Check required session_state data
    if ("detected_peaks_df" not in st.session_state or
        "candidates_df" not in st.session_state or
        "signal" not in st.session_state or
        "x_mass" not in st.session_state):
        st.error("Please run peak detection and candidate matching steps first.")
        st.stop()
    
    # Retrieve stored data
    peaks_df = st.session_state["detected_peaks_df"]
    candidates_df = st.session_state["candidates_df"]
    signal = st.session_state["signal"]
    x_mass = st.session_state["x_mass"]
    
    # Filter peaks: only include those for which candidate formulas exist
    matched_peaks = peaks_df[peaks_df["m/z"].isin(candidates_df["m/z"])]
    
    # Create hover text with candidate details for each peak
    hover_text = []
    for _, row in matched_peaks.iterrows():
        mz_val = row["m/z"]
        intensity_val = row["Intensity"]
        # Look for candidate details by matching m/z value
        cand_row = candidates_df[candidates_df["m/z"] == mz_val]
        if not cand_row.empty:
            details = ""
            if cand_row.iloc[0]["CH candidates"]:
                details += f"CH: {cand_row.iloc[0]['CH candidates']}"
            if cand_row.iloc[0]["CHBr candidates"]:
                details += f"<br>CHBr: {cand_row.iloc[0]['CHBr candidates']}"
        else:
            details = "No candidate details"
        hover_text.append(f"m/z: {mz_val:.4f}<br>Intensity: {intensity_val:.2f}<br>{details}")
    
    # Build an interactive Plotly figure
    fig_int = go.Figure()
    
    # Trace for the full spectrum
    fig_int.add_trace(go.Scatter(
        x=x_mass,
        y=signal,
        mode='lines',
        name=signal_label,
        line=dict(color='blue')
    ))
    
    # Trace for the candidate peaks with hover annotations
    fig_int.add_trace(go.Scatter(
        x=matched_peaks["m/z"],
        y=matched_peaks["Intensity"],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Candidate Peaks',
        text=hover_text,
        hoverinfo='text'
    ))
    
    # Update layout
    fig_int.update_layout(
        title="Interactive Spectrum with Candidate Annotations",
        xaxis_title="Mass (amu)",
        yaxis_title="Intensity",
        hovermode="closest"
    )
    
    st.plotly_chart(fig_int, use_container_width=True)
#####################################
# Custom Range Inputs (Outside the Button)
#####################################
# Ensure you have the data in session_state:
if "x_mass" not in st.session_state or "signal" not in st.session_state:
    st.error("No x_mass or signal found in session_state. Please run previous steps.")
    st.stop()

x_data = st.session_state["x_mass"]
y_data = st.session_state["signal"]

st.markdown("## Fragmentation / Parent / Clustering Plot Range Settings")

custom_xmin = st.number_input("X-axis min (m/z):", value=float(np.min(x_data)))
custom_xmax = st.number_input("X-axis max (m/z):", value=float(np.max(x_data)))
custom_ymin = st.number_input("Y-axis min (Intensity):", value=0.0)
custom_ymax = st.number_input("Y-axis max (Intensity):", value=float(np.max(y_data) * 1.1))

#####################################
# Button to Generate the Plot
#####################################
if st.button("Plot Fragmentation-Parent-Clustering View"):
    st.markdown("## Additional Plot: Fragmentation / Parent / Clustering")

    # 1) Identify the 'parent' peak = highest intensity peak
    parent_index = int(np.argmax(y_data))
    parent_mz = x_data[parent_index]

    # 2) Create a new Matplotlib figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # 3) Plot the entire signal in black (no negative background)
    ax2.plot(x_data, y_data, color='black', label='signal')

    # 4) Color the “fragmentation” region (from custom_xmin to parent_mz) in green
    if parent_mz > custom_xmin:
        ax2.axvspan(custom_xmin, parent_mz, color='green', alpha=0.2, label='fragmentation')

    # 5) Color the “clustering” region (from parent_mz to custom_xmax) in blue
    if parent_mz < custom_xmax:
        ax2.axvspan(parent_mz, custom_xmax, color='blue', alpha=0.2, label='clustering')

    # 6) Annotate the “parent” peak
    parent_intensity = y_data[parent_index]
    offset_parent = 0.05 * parent_intensity  # vertical offset for the label
    ax2.annotate(
        "parent",
        xy=(parent_mz, parent_intensity),
        xytext=(parent_mz, parent_intensity + offset_parent),
        ha='center',
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10,
        fontweight='bold'
    )

    # 7) Apply custom axis limits
    ax2.set_xlim(custom_xmin, custom_xmax)
    ax2.set_ylim(custom_ymin, custom_ymax)
    ax2.set_xlabel("Mass-to-charge (m/z)")
    ax2.set_ylabel("Intensity")
    ax2.set_title("Fragmentation / Parent / Clustering View")

    ax2.legend(loc='upper right')
    plt.tight_layout()

    # 8) (Optional) Save the figure
    directory = st.session_state.get("file_directory", "./output")
    filename = f"{directory}/fragmentation_parent_clustering_custom_range.png"
    plt.savefig(filename, dpi=300)

    st.success(f"Figure saved as '{filename}'.")
    st.pyplot(fig2)
    
####################################
# OPTIONAL: ISOTOPE PATTERN MATCHING (using IsoSpecPy)
#####################################
# st.markdown("## Optional: Isotope Pattern Matching (using IsoSpecPy)")
# perform_isotope_matching = st.checkbox("Enable isotope pattern matching", value=False)
# if perform_isotope_matching:
#     iso_threshold = st.number_input("Minimum isotope match score (0 to 1)", value=0.5, step=0.05)
#     if st.button("Run Isotope Pattern Matching"):
#         # Ensure required data is available in session_state
#         if "candidates_df" not in st.session_state or "signal" not in st.session_state or "x_mass" not in st.session_state:
#             st.error("Please run both peak detection and candidate formula matching steps first.")
#             st.stop()
#         else:
#             candidates_df = st.session_state["candidates_df"]
#             signal = st.session_state["signal"]
#             x_mass = st.session_state["x_mass"]

#             # Helper function to extract a simple molecular formula (e.g., 'C10H8') from a candidate string.
#             import re
#             def extract_formula(candidate_str):
#                 # This pattern captures formulas like "C6H5Br", "C6H5Br2", "C12H22Br", etc.
#                 # Explanation:
#                 #  - C\d+ matches "C" followed by one or more digits
#                 #  - H\d+ matches "H" followed by one or more digits
#                 #  - (Br\d*)? is an optional group for "Br" plus zero or more digits
#                 match = re.search(r'(C\d+H\d+(Br\d*)?)', candidate_str)
#                 if match:
#                     return match.group(1)
#                 return None
#             # for cand in candidate_texts:
#             #     formula = extract_formula(cand)
#             #     print(f"Raw candidate: '{cand}' -> Extracted formula: {formula}")

#             # Try to import IsoSpecPy
#             try:
#                 import IsoSpecPy
#             except ImportError:
#                 st.error("IsoSpecPy package not found. Please install it with 'pip install IsoSpecPy'.")
#                 st.stop()

#             # Calculate the theoretical isotopic pattern using IsoSpecPy.
#             def get_theoretical_pattern(formula, num_peaks=3):
#                 try:
#                     # Create an IsoSpec instance for the given formula.
#                     iso = IsoSpecPy.IsoSpec(formula, threshold=1e-8)
#                     masses, probabilities = iso.getProbabilitiesAndMasses()
#                     # Assume the returned peaks are ordered by increasing mass.
#                     theo = np.array(probabilities[:num_peaks])
#                     if np.max(theo) > 0:
#                         theo = theo / np.max(theo)
#                     return theo
#                 except Exception as e:
#                     return None

#             # Extract the observed isotopic pattern from the experimental signal.
#             def get_observed_pattern(candidate_mz, x_mass, signal, num_peaks=3, window=25):
#                 observed = []
#                 # Isotopic peaks are approximately 1.00335 Da apart (for ^13C).
#                 for i in range(num_peaks):
#                     target = candidate_mz + i * 1.00335
#                     idx = np.where((x_mass >= target - window) & (x_mass <= target + window))[0]
#                     if len(idx) > 0:
#                         observed.append(np.max(signal[idx]))
#                     else:
#                         observed.append(0)
#                 return np.array(observed)

#             # Compare the theoretical and observed patterns using Pearson correlation.
#             def pattern_similarity(theo, obs):
#                 if theo is None or np.std(obs) == 0:
#                     return 0
#                 return np.corrcoef(theo, obs)[0, 1]

#             # Loop over each candidate peak, compute the best isotope match score among its candidate formulas.
#             isotope_scores = []
#             x_mass_arr = np.array(x_mass)
#             for idx, row in candidates_df.iterrows():
#                 candidate_texts = []
#                 if row["CH candidates"]:
#                     candidate_texts.extend([cand.strip() for cand in row["CH candidates"].split(";")])
#                 if row["CHBr candidates"]:
#                     candidate_texts.extend([cand.strip() for cand in row["CHBr candidates"].split(";")])
#                 best_score = 0
#                 for cand in candidate_texts:
#                     formula = extract_formula(cand)
#                     if formula is None:
#                         continue
#                     theo = get_theoretical_pattern(formula, num_peaks=3)
#                     obs = get_observed_pattern(row["m/z"], x_mass_arr, signal, num_peaks=3, window=25)
#                     score = pattern_similarity(theo, obs)
#                     if score > best_score:
#                         best_score = score
#                 isotope_scores.append(best_score)

#             # Add the isotope score to the candidate DataFrame.
#             candidates_df["Isotope Score"] = isotope_scores
#             st.markdown("### Candidate Formulas with Isotope Scores")
#             st.dataframe(candidates_df)

#             # Optionally, filter candidates based on the isotope matching threshold.
#             filtered_df = candidates_df[candidates_df["Isotope Score"] >= iso_threshold]
#             st.markdown(f"### Candidates with Isotope Score ≥ {iso_threshold}")
#             st.dataframe(filtered_df)
# OPTIONAL: ISOTOPE PATTERN MATCHING (using IsoSpecPy)
#####################################
# st.markdown("## Optional: Isotope Pattern Matching (using IsoSpecPy)")
# perform_isotope_matching = st.checkbox("Enable isotope pattern matching", value=False)
# if perform_isotope_matching:
#     iso_threshold = st.number_input("Minimum isotope match score (0 to 1)", value=0.5, step=0.05)
#     if st.button("Run Isotope Pattern Matching"):
#         # Ensure required data is available in session_state
#         if (
#             "candidates_raw_df" not in st.session_state or 
#             "signal" not in st.session_state or 
#             "x_mass" not in st.session_state
#         ):
#             st.error("Please run candidate formula matching (with raw candidates) and ensure that signal and x_mass are available.")
#             st.stop()
#         else:
#             candidates_raw_df = st.session_state["candidates_raw_df"]
#             signal = st.session_state["signal"]
#             x_mass = st.session_state["x_mass"]

#             import re
#             import numpy as np

#             # Helper function to extract a simple molecular formula (e.g., 'C10H8' or 'C6H5Br') from a candidate string.
#             def extract_formula(candidate_str):
#                 # This regex matches formulas like "C6H5Br", "C6H5Br2", "C12H22Br", etc.
#                 match = re.search(r'(C\d+H\d+(Br\d*)?)', candidate_str)
#                 if match:
#                     return match.group(1)
#                 return None

#             # Try to import IsoSpecPy
#             try:
#                 import IsoSpecPy
#             except ImportError:
#                 st.error("IsoSpecPy package not found. Please install it with 'pip install IsoSpecPy'.")
#                 st.stop()

#             # Calculate the theoretical isotopic pattern using IsoSpecPy.
#             def get_theoretical_pattern(formula, num_peaks=3):
#                 try:
#                     iso = IsoSpecPy.IsoSpec(formula, threshold=1e-8)
#                     masses, probabilities = iso.getProbabilitiesAndMasses()
#                     theo = np.array(probabilities[:num_peaks])
#                     if np.max(theo) > 0:
#                         theo = theo / np.max(theo)
#                     return theo
#                 except Exception as e:
#                     return None

#             # Extract the observed isotopic pattern from the experimental signal.
#             def get_observed_pattern(candidate_mz, x_mass, signal, num_peaks=3, window=25):
#                 observed = []
#                 # Isotopic peaks are approximately 1.00335 Da apart (for ^13C).
#                 for i in range(num_peaks):
#                     target = candidate_mz + i * 1.00335
#                     idx = np.where((x_mass >= target - window) & (x_mass <= target + window))[0]
#                     if len(idx) > 0:
#                         observed.append(np.max(signal[idx]))
#                     else:
#                         observed.append(0)
#                 return np.array(observed)

#             # Compare the theoretical and observed patterns using Pearson correlation.
#             def pattern_similarity(theo, obs):
#                 if theo is None or np.std(obs) == 0:
#                     return 0
#                 return np.corrcoef(theo, obs)[0, 1]

#             isotope_scores = []
#             x_mass_arr = np.array(x_mass)

#             # Loop over each raw candidate formula and compute the isotope match score.
#             for idx, row in candidates_raw_df.iterrows():
#                 candidate_str = row["Formula"]
#                 formula = extract_formula(candidate_str)
#                 if formula is None:
#                     isotope_scores.append(0)
#                     continue

#                 # Compute theoretical pattern.
#                 theo = get_theoretical_pattern(formula, num_peaks=3)

#                 # Use IsoSpecPy to get the candidate m/z (assume the first mass is the monoisotopic mass).
#                 try:
#                     iso = IsoSpecPy.IsoSpec(formula, threshold=1e-8)
#                     masses, _ = iso.getProbabilitiesAndMasses()
#                     candidate_mz = masses[0]
#                 except Exception as e:
#                     candidate_mz = None

#                 if candidate_mz is None:
#                     isotope_scores.append(0)
#                     continue

#                 # Get the observed pattern from the experimental data.
#                 obs = get_observed_pattern(candidate_mz, x_mass_arr, signal, num_peaks=3, window=25)
#                 score = pattern_similarity(theo, obs)
#                 isotope_scores.append(score)

#             # Add the isotope score to the raw candidates DataFrame.
#             candidates_raw_df["Isotope Score"] = isotope_scores
#             st.markdown("### Raw Candidate Formulas with Isotope Scores")
#             st.dataframe(candidates_raw_df)

#             # Optionally, filter candidates based on the isotope matching threshold.
#             filtered_df = candidates_raw_df[candidates_raw_df["Isotope Score"] >= iso_threshold]
#             st.markdown(f"### Raw Candidates with Isotope Score ≥ {iso_threshold}")
#             st.dataframe(filtered_df)
# Attempt to import IsoThreshold from IsoSpecPy.
# # Attempt to import IsoThreshold from IsoSpecPy.
# Attempt to import IsoThreshold from IsoSpecPy.
# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # Attempt to import pyteomics.mass.
# try:
#     from pyteomics import mass
# except ImportError:
#     st.error("Pyteomics is not installed. Please install it with 'pip install pyteomics'.")
#     st.stop()

# st.title("Isotopic Pattern Matching with Pyteomics")

# st.markdown(
#     """
#     **Instructions:**
#     1. Enter the molecular formula (e.g., `C4H2Br2`).
#     2. Select the m/z range (x-axis) for your experimental spectrum.
#     3. Set the maximum intensity (y-axis maximum).
#     4. Adjust the window width (in Da) for peak detection.
    
#     Click **Run Isotope Pattern Matching** to generate and compare the theoretical
#     and observed isotopic patterns.
#     """
# )

# # --- User Inputs ---
# formula = st.text_input("Molecular Formula (e.g., C4H2Br2):", value="C4H2Br2")
# mz_min = st.number_input("Minimum m/z value for analysis:", value=100.0)
# mz_max = st.number_input("Maximum m/z value for analysis:", value=400.0)
# y_max = st.number_input("Maximum intensity (y-axis):", value=1.0, step=0.1)
# window_width = st.number_input("Window width (Da) for peak search:", value=0.5, step=0.1)

# # --- Experimental Spectrum Setup ---
# # If no experimental spectrum is loaded, we create dummy data for demonstration.
# if "x_mass" not in st.session_state or "signal" not in st.session_state:
#     x_mass = np.linspace(100, 400, 3000)
#     signal = np.zeros_like(x_mass)
#     # Create dummy peaks (for demonstration) at positions 150, 151.00335, and 152.0067
#     for peak in [150, 151.00335, 152.0067]:
#         signal += np.exp(-0.5 * ((x_mass - peak) / 0.2) ** 2)
#     st.session_state["x_mass"] = x_mass
#     st.session_state["signal"] = signal

# x_mass = np.array(st.session_state["x_mass"])
# signal = np.array(st.session_state["signal"])

# # --- Button to Trigger the Computation ---
# if st.button("Run Isotope Pattern Matching"):
#     try:
#         # --- Validate and Get Monoisotopic Mass Using Pyteomics ---
#         mono_mass = mass.calculate_mass(formula=formula)
#         st.write("**Calculated monoisotopic m/z (using calculate_mass):**", mono_mass)
#         if mono_mass == 0:
#             st.error("Calculated monoisotopic mass is 0. Please check the input formula.")
#             st.stop()
        
#         # --- Theoretical Isotopic Pattern Generation ---
#         # Generate a list of isotopic peaks (mass, abundance) for the formula.
#         theo_peaks = list(mass.isotopologues(formula, abundance_cutoff=1e-8))
#         if not theo_peaks:
#             st.error("No isotopic peaks found for the given formula. Try adjusting the abundance_cutoff.")
#             st.stop()
        
#         # Sort the peaks by mass in ascending order.
#         theo_peaks.sort(key=lambda x: x[0])
        
#         # Use the first 3 peaks for the comparison.
#         num_peaks = 3
#         selected_peaks = theo_peaks[:num_peaks]
#         masses_theo = [p[0] for p in selected_peaks]
#         abundances = np.array([p[1] for p in selected_peaks])
#         st.write("Raw theoretical abundances:", abundances)

#         # Normalize the theoretical abundances.
#         if np.max(abundances) > 0:
#             abundances = abundances / np.max(abundances)
        
#         # Use the calculated monoisotopic mass as the reference.
#         candidate_mz = mono_mass
        
#         st.write("**Theoretical monoisotopic m/z:**", candidate_mz)
#         st.write("**Theoretical isotopic pattern (normalized):**", abundances)
        
#         # --- Observed Pattern Extraction ---
#         def get_observed_pattern(candidate_mz, x_mass, signal, num_peaks=3, window=0.5):
#             observed = []
#             # For each expected isotopic peak, assume an approximate spacing of 1.00335 Da (for ^13C)
#             for i in range(num_peaks):
#                 target = candidate_mz + i * 1.00335
#                 # Only consider if the target is within the selected m/z range.
#                 if target < mz_min or target > mz_max:
#                     observed.append(0)
#                     continue
#                 idx = np.where((x_mass >= target - window) & (x_mass <= target + window))[0]
#                 observed.append(np.max(signal[idx]) if len(idx) > 0 else 0)
#             return np.array(observed)
        
#         obs_pattern = get_observed_pattern(candidate_mz, x_mass, signal, num_peaks, window_width)
#         st.write("**Observed isotopic pattern:**", obs_pattern)
        
#         # --- Similarity Calculation (Pearson correlation) ---
#         def pattern_similarity(theo, obs):
#             if np.std(obs) == 0:
#                 return 0
#             return np.corrcoef(theo, obs)[0, 1]
        
#         similarity_score = pattern_similarity(abundances, obs_pattern)
#         st.write("**Pattern similarity (Pearson correlation):**", similarity_score)
        
#         # --- Plotting the Results ---
#         fig, ax = plt.subplots(figsize=(8, 4))
#         # Plot the experimental spectrum.
#         ax.plot(x_mass, signal, label="Experimental Spectrum")
#         # Shade the selected m/z region.
#         ax.axvspan(mz_min, mz_max, color='gray', alpha=0.3, label="Selected Region")
#         # Mark the expected isotopic peak positions (if they fall within the selected m/z range)
#         for i in range(num_peaks):
#             peak_pos = candidate_mz + i * 1.00335
#             if mz_min <= peak_pos <= mz_max:
#                 ax.axvline(peak_pos, color='red', linestyle='--',
#                            label="Expected Isotopic Peak" if i == 0 else "")
#         ax.set_xlabel("m/z")
#         ax.set_ylabel("Intensity")
#         # Limit the x-axis and y-axis based on user inputs.
#         ax.set_xlim(mz_min, mz_max)
#         ax.set_ylim(0, y_max)
#         ax.legend()
#         st.pyplot(fig)
        
#     except Exception as e:
#         st.error(f"Error processing formula '{formula}': {e}")
####################################
# STEP 4: ISOTOPIC DISTRIBUTION ANALYSIS
####################################
# st.markdown("## Isotopic Distribution Analysis")
# st.markdown(
#     """
#     In this section you can:
#     1. **Select an isotopic distribution module:**  
#        - **Pyteomics:** A library for MS calculations.
#        - **IsoSpecPy:** A fast tool for accurate isotopic pattern calculations.
#     2. **Enter a molecular formula** (e.g. for a PAH such as `C16H10`).
#     3. **Manually input the observed isotopic peaks** (each line in the format: `m/z, intensity`).
#     4. **Adjust the plot range:** Choose the x-axis minimum/maximum and the y-axis maximum.
    
#     When you click **Run Isotopic Distribution Analysis**, the theoretical isotopic pattern is computed and compared to your observed peaks.
#     """
# )

# # List available open-source modules
# st.markdown("### Available Modules for Isotopic Distribution")
# st.write("1. **Pyteomics**: A Python library that includes functions for mass calculation and isotopic distribution.")
# st.write("2. **IsoSpecPy**: An efficient tool for computing high-accuracy isotopic patterns.")

# # Let the user choose which module to use
# iso_module = st.selectbox("Select isotopic distribution module:", options=["Pyteomics", "IsoSpecPy"])

# # Input for molecular formula
# molecule_formula = st.text_input("Enter molecular formula (e.g., C16H10):", value="C16H10")

# # Input for number of theoretical peaks to consider
# num_theo_peaks = st.number_input("Number of theoretical isotopic peaks to display:", value=3, min_value=1, step=1)

# # Text area for manually entering observed isotopic peaks
# st.markdown("**Enter observed isotopic peaks:** (one per line, format: `m/z, intensity`)")
# observed_peaks_input = st.text_area("Observed Peaks", value="")  # leave empty if not available

# # Plot axis parameters
# st.markdown("**Plot Range Settings**")
# x_min_iso = st.number_input("X-axis minimum (m/z):", value=100.0)
# x_max_iso = st.number_input("X-axis maximum (m/z):", value=400.0)
# y_max_iso = st.number_input("Y-axis maximum (Intensity):", value=1.0, step=0.1)

# # Button to run isotopic distribution analysis
# if st.button("Run Isotopic Distribution Analysis"):
    
#     # ----------------------------
#     # 1. Compute the Theoretical Pattern
#     # ----------------------------
#     theoretical_mzs = []
#     theoretical_abundances = []
    
#     try:
#         if iso_module == "Pyteomics":
#             try:
#                 from pyteomics import mass
#             except ImportError:
#                 st.error("Pyteomics is not installed. Please install it with 'pip install pyteomics'.")
#                 st.stop()
#             # Calculate monoisotopic mass (for reference)
#             mono_mass = mass.calculate_mass(formula=molecule_formula)
#             # Get the full isotopic distribution (as list of tuples: (m/z, abundance))
#             iso_peaks = list(mass.isotopologues(molecule_formula, abundance_cutoff=1e-8))
#             if not iso_peaks:
#                 st.error("No isotopic peaks were found. Check the molecular formula or adjust the abundance_cutoff.")
#                 st.stop()
#             # Sort by m/z in ascending order
#             iso_peaks.sort(key=lambda x: x[0])
#             # Select the first N peaks (or as many as available)
#             selected = iso_peaks[:int(num_theo_peaks)]
#             theoretical_mzs = [p[0] for p in selected]
#             theoretical_abundances = np.array([p[1] for p in selected])
            
#         elif iso_module == "IsoSpecPy":
#             try:
#                 import IsoSpecPy
#             except ImportError:
#                 st.error("IsoSpecPy is not installed. Please install it with 'pip install IsoSpecPy'.")
#                 st.stop()
#             # Create an IsoSpec instance for the formula
#             iso = IsoSpecPy.IsoSpec(molecule_formula, threshold=1e-8)
#             masses, probabilities = iso.getProbabilitiesAndMasses()
#             if len(masses) == 0:
#                 st.error("No isotopic peaks were returned by IsoSpecPy.")
#                 st.stop()
#             # Take the first N peaks
#             theoretical_mzs = list(masses[:int(num_theo_peaks)])
#             theoretical_abundances = np.array(probabilities[:int(num_theo_peaks)])
#         else:
#             st.error("Unknown module selected.")
#             st.stop()
#     except Exception as e:
#         st.error(f"Error computing theoretical pattern: {e}")
#         st.stop()
    
#     # Normalize theoretical abundances (if max > 0)
#     if theoretical_abundances.size > 0 and np.max(theoretical_abundances) > 0:
#         theoretical_abundances = theoretical_abundances / np.max(theoretical_abundances)
    
#     st.write("**Theoretical Isotopic Pattern:**")
#     for i, (mz, ab) in enumerate(zip(theoretical_mzs, theoretical_abundances), start=1):
#         st.write(f"Peak {i}: m/z = {mz:.4f}, Relative Abundance = {ab:.3f}")
    
#     # ----------------------------
#     # 2. Parse the Observed Peaks (if provided)
#     # ----------------------------
#     observed_mzs = []
#     observed_intensities = []
    
#     if observed_peaks_input.strip():
#         for line in observed_peaks_input.splitlines():
#             parts = line.split(',')
#             if len(parts) < 2:
#                 st.warning(f"Line skipped (not enough values): {line}")
#                 continue
#             try:
#                 mz_val = float(parts[0].strip())
#                 intensity_val = float(parts[1].strip())
#                 observed_mzs.append(mz_val)
#                 observed_intensities.append(intensity_val)
#             except Exception as e:
#                 st.warning(f"Could not parse line '{line}': {e}")
#     else:
#         st.info("No observed peaks were manually entered. Only the theoretical pattern will be displayed.")
    
#     # ----------------------------
#     # 3. Plotting: Theoretical vs. Observed
#     # ----------------------------
#     fig, ax = plt.subplots(figsize=(8, 4))
    
#     # Plot the theoretical pattern using stem plot
#     markerline, stemlines, baseline = ax.stem(theoretical_mzs, theoretical_abundances, linefmt="C0-", markerfmt="C0o", basefmt="k-")
#     plt.setp(markerline, markersize=8, label="Theoretical Pattern")
    
#     # If observed peaks are available, plot them as red markers
#     if observed_mzs and observed_intensities:
#         ax.scatter(observed_mzs, observed_intensities, color="red", s=50, zorder=5, label="Observed Peaks")
    
#     # Set axis labels and limits
#     ax.set_xlabel("m/z")
#     ax.set_ylabel("Relative Abundance / Intensity")
#     ax.set_title(f"Isotopic Pattern for {molecule_formula} ({iso_module})")
#     ax.set_xlim(x_min_iso, x_max_iso)
#     ax.set_ylim(0, y_max_iso)
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.5)
    
#     st.pyplot(fig)
    
#     # ----------------------------
#     # 4. Optional: Compute a Similarity Metric
#     # ----------------------------
#     if observed_mzs and observed_intensities:
#         # If the number of observed peaks matches the number of theoretical peaks, compute Pearson correlation.
#         n_common = min(len(theoretical_abundances), len(observed_intensities))
#         if n_common > 1:
#             theo_subset = theoretical_abundances[:n_common]
#             obs_subset = np.array(observed_intensities[:n_common])
#             # Normalize observed intensities if max > 0
#             if np.max(obs_subset) > 0:
#                 obs_subset = obs_subset / np.max(obs_subset)
#             corr_coef = np.corrcoef(theo_subset, obs_subset)[0, 1]
#             st.write(f"**Pattern similarity (Pearson correlation) between theoretical and observed peaks:** {corr_coef:.3f}")
#         else:
#             st.write("Not enough peaks to compute a similarity metric.")
