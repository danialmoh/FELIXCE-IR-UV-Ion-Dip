import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

# Configure dashboard
st.set_page_config(layout="wide")

# Hide the default navigation menu
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Configure dashboard with custom CSS for larger logo
st.markdown("""
<style>
[data-testid="stLogo"] {
    height: 8rem !important;
    margin-bottom: -5rem !important;
    padding-bottom: -5rem !important;
}
[data-testid="stLogo"] img {
    height: 8rem !important;
    width: auto !important;
    margin-bottom: -5rem !important;
    padding-bottom: -5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.logo(r"./documentation/logo/logo_FELIXCE_solid.png")

# Sidebar branding and credits
with st.sidebar:
    # st.markdown("---")

    st.markdown(
        """
        # IR/UV Ion-Dip Data Analysis Pipeline
        
        A comprehensive Streamlit application for processing and analyzing FELIX IR/UV ion-dip spectroscopy data.
        
        ---
        
        **Developed by:**  
        • Danial Mohammadi  
        • Kevin Antony Kaw
        
        **Institution:**  
        KU Leuven / Stockholm University
        """
    )
    st.markdown("---")
    
    # Workflow sections with custom styling
    st.markdown("### 📋 Workflow Modules")
    
    # Home button
    if st.button("🏠 Home", use_container_width=True, key="btn_home", type="primary"):
        st.switch_page(".streamlit/0.0_Home.py")
    
    # Section 1: Data Import
    with st.container(border=True):
        st.markdown("#### 📥 Data Import")
        if st.button("📥 Import data", use_container_width=True, key="btn_import"):
            st.switch_page(".streamlit/1.0_ImportData.py")
        if st.button("📊 1.1 Wavenumbers per file (unmodified)", use_container_width=True, key="btn_wav1"):
            st.switch_page(".streamlit/1.1_Wavenumbers_raw.py")
        if st.button("📊 1.2 Wavenumbers per file (rounded)", use_container_width=True, key="btn_wav2"):
            st.switch_page(".streamlit/1.2_Wavenumbers.py")
        if st.button("🔍 1.3 Unique wavenumbers", use_container_width=True, key="btn_wav3"):
            st.switch_page(".streamlit/1.3_UniqueWavenumbers.py")
        if st.button("⚙️ 1.4 Experiment parameters", use_container_width=True, key="btn_exp"):
            st.switch_page(".streamlit/1.4_ExperimentParameters.py")
    
    # Section 2: Baseline Correction
    with st.container(border=True):
        st.markdown("#### 📏 Baseline Correction")
        if st.button("📏 2. Baseline correction", use_container_width=True, key="btn_baseline"):
            st.switch_page(".streamlit/2.0_BaselineCorrection.py")
        if st.button("📈 2.1 Baseline correction - full range", use_container_width=True, key="btn_baseline_full"):
            st.switch_page(".streamlit/2.1_BaselineCorrectionFullRange.py")
    
    # Section 3: Data Analysis
    with st.container(border=True):
        st.markdown("#### 💪 Data Analysis")
        if st.button("💪 3. Data analysis", use_container_width=True, key="btn_depletion"):
            st.switch_page(".streamlit/3.0_CalculateDepletion.py")
        if st.button("🔦 3.01 Laser Normalization (Not working)", use_container_width=True, key="btn_laser1"):
            st.switch_page(".streamlit/3.01_LaserNormalization.py")
        if st.button("🔦 3.02 Laser Normalization-P2 (Not working)", use_container_width=True, key="btn_laser2"):
            st.switch_page(".streamlit/3.02_LaserNormalizationpart2.py")
        if st.button("🗄️ 3.1 Data analysis with NASA PAH Database", use_container_width=True, key="btn_db"):
            st.switch_page(".streamlit/3.1_Database.py")
        if st.button("🗄️ 3.11 Data analysis with NIST Database", use_container_width=True, key="btn_nist"):
            st.switch_page(".streamlit/3.11_DatabaseNIST.py")
    
    # Section 4: Miscellaneous
    with st.container(border=True):
        st.markdown("#### 🌌 Miscellaneous")
        if st.button("🌌 4. Miscellaneous", use_container_width=True, key="btn_misc"):
            st.switch_page(".streamlit/4.0_Misc.py")
        if st.button("📦 4.1 Mega sum", use_container_width=True, key="btn_mega"):
            st.switch_page(".streamlit/4.1_MegaSum.py")
        if st.button("🎯 4.2 Peak Detection", use_container_width=True, key="btn_peak"):
            st.switch_page(".streamlit/4.2_PeakDetection_NEW.py")
        if st.button("📋 4.3 Mass Reference", use_container_width=True, key="btn_mass"):
            st.switch_page(".streamlit/4.3_reference_spectrum.py")
    
    # Section 5: Smoothing
    with st.container(border=True):
        st.markdown("#### 🌊 Smoothing")
        if st.button("🌊 5. Smoothing", use_container_width=True, key="btn_smooth"):
            st.switch_page(".streamlit/5.0_Smoothing.py")
    
    # Section 6: PAH & Comparison
    with st.container(border=True):
        st.markdown("#### 🧬 PAH & Comparison")
        if st.button("🧬 6. PAH Generator", use_container_width=True, key="btn_pah"):
            st.switch_page(".streamlit/6.0_pah_generator.py")
        if st.button("⚖️ 6.1 Data Comparison", use_container_width=True, key="btn_compare"):
            st.switch_page(".streamlit/6.1_Comparison.py")
        if st.button("🔬 6.2 DFT Comparison", use_container_width=True, key="btn_dft"):
            st.switch_page(".streamlit/6.2_DFT_Comparison.py")
    
    # Section 7: Laser Normalization
    with st.container(border=True):
        st.markdown("#### ✨ Advanced")
        if st.button("✨ 7.0 Laser Normalization", use_container_width=True, key="btn_laser_adv"):
            st.switch_page(".streamlit/7.0_LaserNormalization.py")
    
    # Section 8: REMPI
    with st.container(border=True):
        st.markdown("#### 🔬 REMPI (UV-only)")
        if st.button("🔬 8.0 REMPI Import Data", use_container_width=True, key="btn_rempi_import"):
            st.switch_page(".streamlit/8.0_REMPI_ImportData.py")
        if st.button("📊 8.1 REMPI Wavelengths per file", use_container_width=True, key="btn_rempi_wl"):
            st.switch_page(".streamlit/8.1_REMPI_Wavelengths.py")
        if st.button("🔍 8.2 REMPI Unique Wavelengths", use_container_width=True, key="btn_rempi_unique"):
            st.switch_page(".streamlit/8.2_REMPI_UniqueWavelengths.py")
        if st.button("📐 8.3 REMPI Parameters & Baseline", use_container_width=True, key="btn_rempi_baseline"):
            st.switch_page(".streamlit/8.3_REMPI_BaselineCorrection.py")
    
    # Section 9: Results Summary
    with st.container(border=True):
        st.markdown("#### 📋 Results & Export")
        if st.button("📋 9.0 Results Summary & Export", use_container_width=True, key="btn_results_summary"):
            st.switch_page(".streamlit/9.0_ResultsSummary.py")
        if st.button("📚 9.1 Literature Comparison", use_container_width=True, key="btn_lit_comparison"):
            st.switch_page(".streamlit/9.1_LiteratureComparison.py")
    #section 10: 
    with st.container(border=True):
        st.markdown("#### 🧪 Mass Identity Workbench")
        if st.button("🧪 10.0 Mass Identity Workbench", use_container_width=True, key="btn_mass_identity"):
            st.switch_page(".streamlit/10.0_MassIdentity.py")
        if st.button("🧬 11.0 Spectral Decomposition", use_container_width=True, key="btn_spectral_decomp"):
            st.switch_page(".streamlit/11.0_SpectralDecomposition.py")

# Load sections
nav = get_nav_from_toml(".streamlit/pages_sections.toml")
pg = st.navigation(nav) # Loads the contents for each entry in the TOML file
add_page_title(pg) # Loads title from each entry in the TOML file
pg.run()