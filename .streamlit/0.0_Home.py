import streamlit as st

st.set_page_config(page_title="FELIX IR-UV Ion-Dip Analysis", page_icon="🔬", layout="wide")

st.title("🔬 FELIX IR-UV Ion-Dip Data Analysis Pipeline")

st.markdown("---")

st.markdown("""
### Welcome to the FELIX Data Analysis Tool

This pipeline processes time-of-flight mass spectrometry data from IR-UV ion-dip experiments 
performed at the FELIX laboratory. The workflow guides you through data import, baseline correction, 
and depletion analysis.
""")

st.markdown("---")

st.markdown("### 📋 Analysis Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 1️⃣ Data Import
    **Page 1.0 - Import Data**
    - Upload HDF5 files
    - Configure step sizes
    - Apply wavelength calibration
    - Group by wavenumbers
    
    **Page 1.3 - Register Parameters**
    - Define molecular complex
    - Set experiment parameters
    - Configure mass axis
    """)

with col2:
    st.markdown("""
    #### 2️⃣ Baseline Correction
    **Page 2.0 - Single Wavenumber**
    - Select baseline method
    - Visualize correction
    - Register parameters
    
    **Page 2.1 - Full Range**
    - Apply baseline to all wavenumbers
    - Compare corrected signals
    """)

with col3:
    st.markdown("""
    #### 3️⃣ Depletion Analysis
    **Page 3.0 - Calculate Depletion**
    - Compute depletion spectra
    - Apply smoothing
    - Generate plots
    - Export results
    """)

st.markdown("---")

st.markdown("### 🚀 Getting Started")

st.info("""
**Quick Start Guide:**
1. Navigate to **1.0 Import Data** using the sidebar
2. Upload your `.h5` files and configure calibration
3. Go to **1.3 Register Parameters** to define your molecular complex
4. Proceed to **2.0 Baseline Correction** to clean your signals
5. Finally, analyze your results in **3.0 Calculate Depletion**
""")

st.markdown("---")

st.markdown("### 📊 Key Features")

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown("""
    **Advanced Baseline Correction**
    - Mean Subtraction
    - iarpls (Improved arPLS)
    - aspls (Adaptive Smoothing)
    - fabc (Fully Automatic)
    """)

with feat_col2:
    st.markdown("""
    **Flexible Calibration**
    - Per-file step sizes
    - Wavelength calibration tables
    - Auto-detection from filenames
    - CSV import/export
    """)

st.markdown("---")

st.success("✨ **Ready to begin?** Start with **1.0 Import Data** in the sidebar!")
