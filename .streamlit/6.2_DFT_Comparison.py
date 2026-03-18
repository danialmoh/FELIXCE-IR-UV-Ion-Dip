import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime
import re
import os
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

st.title("🔬 DFT Spectrum Comparison")
st.markdown("""
Compare experimental IR spectra with DFT-calculated theoretical spectra.
Upload DFT output files (Gaussian .out, ORCA .out/.dat, or custom parsed files) and compare with experimental data from your session.
""")

# Helper Functions
def broaden_spectrum_felix(frequencies, intensities, x_range=(200, 4000), bw_frac=0.007, npoints=4000):
    """
    Convolve stick spectrum with a Gaussian lineshape whose FWHM scales
    linearly with frequency: FWHM(nu) = bw_frac * nu.
    
    This frequency-proportional resolution is characteristic of FELIX FEL instruments
    and is more physically accurate than constant FWHM broadening.
    
    Parameters:
    -----------
    frequencies : array-like
        Peak frequencies in cm⁻¹
    intensities : array-like
        Peak intensities in km/mol
    x_range : tuple
        (min, max) wavenumber range for output spectrum
    bw_frac : float
        Fractional bandwidth (default 0.007 = 0.7%)
    npoints : int
        Number of points in output spectrum
        
    Returns:
    --------
    x, y : arrays
        Broadened spectrum (wavenumbers, intensities)
    """
    x = np.linspace(x_range[0], x_range[1], npoints)
    y = np.zeros_like(x)
    
    for freq, inten in zip(frequencies, intensities):
        if freq <= 0:
            continue
        # FWHM scales linearly with frequency
        fwhm_local = bw_frac * freq
        # Convert FWHM to Gaussian sigma
        sigma = fwhm_local / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        # Add Gaussian peak
        y += inten * np.exp(-0.5 * ((x - freq) / sigma) ** 2)
    
    return x, y

def parse_custom_report(content):
    """Parse custom DFT report format"""
    frequencies = []
    intensities = []
    metadata = {}
    
    lines = content.split('\n')
    in_spectrum_section = False
    
    for i, line in enumerate(lines):
        # Extract metadata
        if 'Software:' in line:
            metadata['software'] = line.split('Software:')[1].strip()
        elif 'Method:' in line:
            metadata['method'] = line.split('Method:')[1].strip()
        elif 'Final Energy:' in line:
            metadata['energy'] = line.split('Final Energy:')[1].strip().split()[0]
        elif 'Strongest Peak:' in line:
            metadata['strongest_peak'] = line.split('Strongest Peak:')[1].strip()
        
        # Find spectrum data section
        if 'Mode     Frequency (cm⁻¹)     Intensity (km/mol)' in line:
            in_spectrum_section = True
            continue
        
        if in_spectrum_section:
            # Stop at section delimiter or empty lines after data
            if '=====' in line or '----' in line or (line.strip() == '' and frequencies):
                break
            
            # Parse data lines
            parts = line.split()
            if len(parts) >= 3:
                try:
                    mode_num = int(parts[0])
                    freq = float(parts[1])
                    inten = float(parts[2])
                    frequencies.append(freq)
                    intensities.append(inten)
                except ValueError:
                    continue
    
    return np.array(frequencies), np.array(intensities), metadata

def parse_orca_out(content):
    """Parse ORCA .out file for IR frequencies and intensities"""
    frequencies = []
    intensities = []
    
    lines = content.split('\n')
    in_ir_section = False
    
    for i, line in enumerate(lines):
        # Look for IR spectrum section
        if 'IR SPECTRUM' in line:
            in_ir_section = True
            # Skip header lines
            continue
        
        if in_ir_section:
            # Skip separator and header lines
            if '---' in line or 'Mode' in line or 'cm**-1' in line:
                continue
            
            # Stop at empty line after data
            if line.strip() == '':
                if frequencies:
                    break
                continue
            
            # Parse ORCA IR spectrum format: "mode: freq eps Int T**2 ..."
            # Example: "  6:     96.33   0.000147    0.74  0.000477  (-0.000000  0.000000  0.021838)"
            parts = line.split()
            if len(parts) >= 4 and ':' in parts[0]:
                try:
                    freq = float(parts[1])  # frequency in cm⁻¹
                    inten = float(parts[3])  # intensity in km/mol
                    frequencies.append(freq)
                    intensities.append(inten)
                except (ValueError, IndexError):
                    continue
    
    return np.array(frequencies), np.array(intensities), {}

def parse_gaussian_out(content):
    """Parse Gaussian .out/.log file for IR frequencies and intensities"""
    frequencies = []
    intensities = []
    
    lines = content.split('\n')
    
    # Try standard Gaussian format first
    for i, line in enumerate(lines):
        # Look for frequency section
        if 'Frequencies --' in line:
            freqs = [float(x) for x in line.split()[2:]]
            
            # Look for IR intensities a few lines down
            for j in range(i+1, min(i+10, len(lines))):
                if 'IR Inten' in lines[j]:
                    intens = [float(x) for x in lines[j].split()[3:]]
                    frequencies.extend(freqs)
                    intensities.extend(intens)
                    break
    
    return np.array(frequencies), np.array(intensities), {}

def parse_gaussian_anharmonic(content):
    """Parse Gaussian anharmonic frequency output files"""
    frequencies = []
    intensities = []
    
    lines = content.split('\n')
    in_data_section = False
    
    for line in lines:
        # Look for fundamental bands section
        if 'Fundamental Bands' in line or 'Mode(n)' in line:
            in_data_section = True
            continue
        
        if in_data_section:
            # Stop at empty line or section separator
            if line.strip() == '' or '---' in line[:5]:
                if frequencies:
                    break
                continue
            
            # Parse anharmonic data
            # Format: "1(1)                  3481.635   3345.950    118.07814624    102.35434412"
            # We want E(anharm) and I(anharm) - columns 2 and 4
            parts = line.split()
            if len(parts) >= 5 and '(' in parts[0]:
                try:
                    freq_anharm = float(parts[2])  # E(anharm)
                    inten_anharm = float(parts[4])  # I(anharm)
                    frequencies.append(freq_anharm)
                    intensities.append(inten_anharm)
                except (ValueError, IndexError):
                    continue
    
    return np.array(frequencies), np.array(intensities), {'type': 'anharmonic'}

def parse_orca_stick(content):
    """Parse ORCA .out.ir.stk stick spectrum file"""
    frequencies = []
    intensities = []
    
    lines = content.split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                freq = float(parts[0])
                inten = float(parts[1])
                frequencies.append(freq)
                intensities.append(inten)
            except ValueError:
                continue
    
    return np.array(frequencies), np.array(intensities), {'type': 'stick_spectrum'}

def parse_orca_broadened(content):
    """Parse ORCA .out.ir.dat broadened spectrum file (already processed)"""
    frequencies = []
    intensities = []
    
    lines = content.split('\n')
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                freq = float(parts[0])
                inten = float(parts[1])
                frequencies.append(freq)
                intensities.append(inten)
            except ValueError:
                continue
    
    return np.array(frequencies), np.array(intensities), {'type': 'broadened_spectrum'}

def parse_dft_file(uploaded_file):
    """
    Auto-detect and parse DFT output file
    
    Returns:
    --------
    frequencies, intensities, metadata
    """
    content = uploaded_file.read().decode('utf-8', errors='ignore')
    filename = uploaded_file.name.lower()
    
    # Check for specific file extensions first
    if filename.endswith('.ir.stk'):
        return parse_orca_stick(content)
    elif filename.endswith('.ir.dat'):
        return parse_orca_broadened(content)
    elif 'anhar' in filename and ('.txt' in filename or '.log' in filename):
        return parse_gaussian_anharmonic(content)
    # Try different parsers based on content
    elif 'ANALYSIS REPORT' in content[:2000]:
        return parse_custom_report(content)
    elif 'O   R   C   A' in content[:1000]:
        return parse_orca_out(content)
    elif 'Fundamental Bands' in content[:2000]:
        return parse_gaussian_anharmonic(content)
    elif 'Gaussian' in content[:1000] or 'Frequencies --' in content:
        return parse_gaussian_out(content)
    else:
        st.warning("Could not auto-detect file format. Trying custom parser...")
        return parse_custom_report(content)

# Main UI
st.markdown("---")
st.markdown("## 📤 Upload DFT Output File")

uploaded_file = st.file_uploader(
    "Upload DFT calculation output",
    type=['out', 'dat', 'log', 'txt'],
    help="Supported formats: Gaussian .out/.log, ORCA .out/.dat, custom parsed reports"
)

if uploaded_file is not None:
    try:
        frequencies, intensities, metadata = parse_dft_file(uploaded_file)
        
        if len(frequencies) == 0:
            st.error("No IR spectrum data found in file. Please check the file format.")
        else:
            st.success(f"✅ Successfully parsed {len(frequencies)} vibrational modes")
            
            # Display metadata if available
            if metadata:
                st.markdown("### 📋 Calculation Details")
                cols = st.columns(3)
                idx = 0
                for key, value in metadata.items():
                    with cols[idx % 3]:
                        st.metric(key.replace('_', ' ').title(), value)
                    idx += 1
            
            # Display raw stick spectrum data
            with st.expander("📊 View Raw Spectrum Data"):
                df_spectrum = pd.DataFrame({
                    'Mode': range(1, len(frequencies) + 1),
                    'Frequency (cm⁻¹)': frequencies,
                    'Intensity (km/mol)': intensities
                })
                st.dataframe(df_spectrum, height=300)
            
            # Store in session state
            st.session_state['dft_frequencies'] = frequencies
            st.session_state['dft_intensities'] = intensities
            st.session_state['dft_metadata'] = metadata
            
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        st.info("Please ensure the file contains IR spectrum data in a supported format.")

# Broadening and Plotting Section
if 'dft_frequencies' in st.session_state:
    st.markdown("---")
    st.markdown("## 🎨 Spectrum Broadening & Visualization")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_min = st.number_input("Wavenumber Min (cm⁻¹)", value=500.0, step=50.0)
        x_max = st.number_input("Wavenumber Max (cm⁻¹)", value=2200.0, step=50.0)
    with col2:
        bw_percent = st.number_input(
            "FELIX Bandwidth (%)", 
            value=0.7, 
            min_value=0.1,
            max_value=5.0,
            step=0.1,
            format="%.2f",
            help="Default 0.7% represents FELIX FEL characteristic bandwidth. FWHM = bandwidth% × frequency"
        )
        bw_frac = bw_percent / 100.0  # Convert percentage to fraction
    with col3:
        npoints = st.number_input("Number of Points", value=4000, step=100, min_value=100)
    
    # Apply broadening
    x_broad, y_broad = broaden_spectrum_felix(
        st.session_state['dft_frequencies'],
        st.session_state['dft_intensities'],
        x_range=(x_min, x_max),
        bw_frac=bw_frac,
        npoints=int(npoints)
    )
    
    # Store broadened spectrum
    st.session_state['dft_x_broad'] = x_broad
    st.session_state['dft_y_broad'] = y_broad
    
    st.markdown("### 📈 DFT Theoretical Spectrum")
    
    # Interactive plot
    st.markdown("###### *:green[Interactive plot with Plotly]*")
    fig_dft = go.Figure()
    
    # Stick spectrum
    fig_dft.add_trace(go.Scatter(
        x=st.session_state['dft_frequencies'],
        y=st.session_state['dft_intensities'],
        mode='markers',
        marker=dict(size=8, color='red', symbol='line-ns-open'),
        name='Stick Spectrum'
    ))
    
    # Broadened spectrum
    fig_dft.add_trace(go.Scatter(
        x=x_broad,
        y=y_broad,
        mode='lines',
        line=dict(color='blue', width=2),
        name=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)'
    ))
    
    fig_dft.update_layout(
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity (km/mol)",
        title="DFT IR Spectrum",
        hovermode='closest',
        legend=dict(x=0.7, y=0.95)
    )
    
    st.plotly_chart(fig_dft, use_container_width=True)
    
    # Static plot
    st.markdown("###### *:green[Static plot with Matplotlib]*")
    fig_static, ax = plt.subplots(figsize=(12, 6))
    
    # Stick spectrum as stems
    ax.vlines(st.session_state['dft_frequencies'], 0, st.session_state['dft_intensities'], 
              colors='red', alpha=0.6, linewidths=1.5, label='Stick Spectrum')
    
    # Broadened spectrum
    ax.plot(x_broad, y_broad, 'b-', linewidth=2, label=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)')
    
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity (km/mol)", fontsize=12)
    ax.set_title("DFT IR Spectrum", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    fig_static.tight_layout()
    
    st.pyplot(fig_static)
    
    # Add to report button
    add_plot_to_report_button(
        fig_static,
        "DFT IR Spectrum",
        key_suffix="dft_spectrum",
        description="DFT-calculated IR spectrum with FELIX-style broadening"
    )

# Experimental vs Theoretical Comparison
st.markdown("---")
st.markdown("## 🔀 Compare with Experimental Data")
st.info("💡 For IR-UV ion-dip spectroscopy: Spectra are overlaid without scaling. Experimental data shown as -ln(depletion).")

# Check if experimental data is available
fullrange_depletion_data = st.session_state.get("fullrange_depletion_data", None)

if fullrange_depletion_data is not None and 'dft_x_broad' in st.session_state:
    st.markdown("### Experimental Data Available ✅")
    
    # Alignment options only
    st.markdown("#### Alignment Options")
    col1, col2 = st.columns(2)
    with col1:
        shift_theory = st.number_input("Shift Theory (cm⁻¹)", value=0.0, step=1.0, format="%.1f",
                                      help="Shift theoretical spectrum for alignment")
    with col2:
        invert_theory = st.checkbox("Invert Theory", value=False, 
                                   help="Invert theoretical spectrum if needed")
    
    # Plot comparison
    if st.button("📊 Generate Comparison Plot"):
        st.markdown("###### *:green[Interactive Comparison with Plotly]*")
        
        # Experimental data - use -ln(depletion) column
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values  # -ln(depletion) column
        
        # Theoretical data (shifted only, no scaling)
        theory_x_shifted = st.session_state['dft_x_broad'] + shift_theory
        theory_y = st.session_state['dft_y_broad'].copy()
        
        if invert_theory:
            theory_y = -theory_y
        
        # Create figure with secondary y-axis for overlay
        fig_comp = go.Figure()
        
        # Experimental on primary y-axis
        fig_comp.add_trace(go.Scatter(
            x=exp_x,
            y=exp_y,
            mode='lines',
            line=dict(color='black', width=2),
            name='Experimental -ln(depletion)',
            yaxis='y1'
        ))
        
        # Theoretical on secondary y-axis for independent scaling
        fig_comp.add_trace(go.Scatter(
            x=theory_x_shifted,
            y=theory_y,
            mode='lines',
            line=dict(color='red', width=2),
            name='DFT Theory',
            yaxis='y2'
        ))
        
        fig_comp.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis=dict(
                title="-ln(depletion)",
                side='left',
                showgrid=True
            ),
            yaxis2=dict(
                title="Intensity (km/mol)",
                side='right',
                overlaying='y',
                showgrid=False
            ),
            title="IR-UV Ion-Dip: Experimental vs DFT Comparison",
            hovermode='x unified',
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Static comparison plot with dual y-axes
        st.markdown("###### *:green[Static Comparison with Matplotlib]*")
        fig_comp_static, ax1 = plt.subplots(figsize=(14, 7))
        
        # Experimental on left y-axis
        ax1.plot(exp_x, exp_y, 'k-', linewidth=2, label='Experimental -ln(depletion)', alpha=0.8)
        ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax1.set_ylabel("-ln(depletion)", fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        
        # Theoretical on right y-axis
        ax2 = ax1.twinx()
        ax2.plot(theory_x_shifted, theory_y, 'r-', linewidth=2, label='DFT Theory', alpha=0.8)
        ax2.set_ylabel("Intensity (km/mol)", fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
        
        ax1.set_title("IR-UV Ion-Dip: Experimental vs DFT IR Spectrum", fontsize=14, fontweight='bold')
        fig_comp_static.tight_layout()
        
        st.pyplot(fig_comp_static)
        
        # Add to report button
        add_plot_to_report_button(
            fig_comp_static,
            "Experimental vs DFT Comparison",
            key_suffix="exp_vs_dft",
            description="Comparison of experimental and DFT-calculated IR spectra"
        )
        
        # Save comparison data
        if st.checkbox("💾 Save Comparison Data"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            
            comparison_df = pd.DataFrame({
                'Wavenumber_Exp': exp_x,
                'Ln_Depletion_Exp': exp_y,
                'Wavenumber_Theory': theory_x_shifted,
                'Intensity_Theory_km_mol': theory_y
            })
            
            output_filename = os.path.join(file_directory, f"exp_vs_dft_comparison_{timestamp}.csv")
            comparison_df.to_csv(output_filename, index=False)
            st.success(f"Comparison data saved to: {output_filename}")

elif fullrange_depletion_data is None:
    st.info("⚠️ No experimental data found in session. Please run the depletion calculation (Section 3.0) first.")
elif 'dft_x_broad' not in st.session_state:
    st.info("⚠️ Please upload and process a DFT file first.")

# Info box
with st.expander("ℹ️ About FELIX-Style Broadening"):
    st.markdown("""
    ### Frequency-Proportional Broadening
    
    Unlike traditional constant FWHM broadening, FELIX uses FWHM that scales linearly with frequency:
    
    **FWHM(ν) = bw_frac × ν**
    
    Where `bw_frac = 0.007` (0.7%) represents the FELIX FEL's spectral bandwidth.
    
    #### Physical Meaning
    The bandwidth varies across the spectrum:
    - **500 cm⁻¹** → FWHM = 3.5 cm⁻¹
    - **1000 cm⁻¹** → FWHM = 7.0 cm⁻¹
    - **1500 cm⁻¹** → FWHM = 10.5 cm⁻¹
    - **3000 cm⁻¹** → FWHM = 21.0 cm⁻¹
    
    This frequency-proportional resolution is characteristic of FEL instruments and provides
    more physically accurate comparison with FELIX experimental data than constant FWHM broadening.
    """)
