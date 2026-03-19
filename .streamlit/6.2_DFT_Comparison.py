import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import datetime
import re
import os
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from packages.ReportManager import add_plot_to_report_button, init_report_session

init_report_session()

st.title("🔬 DFT Spectrum Comparison & PCC Scoring")
st.markdown("""
Compare experimental IR-UV ion-dip spectra with DFT-calculated theoretical spectra.  
Upload multiple DFT output files to compare different candidate structures using **Pearson Correlation Coefficient (PCC)** scoring.

**Features:**
- Multi-structure comparison
- Region-specific PCC scoring (fingerprint, C≡C stretch, aromatic CH)
- Peak position-based matching (intensity-independent)
""")

# ========================================================================================
# PCC SCORING HELPER FUNCTIONS
# ========================================================================================

# Default diagnostic regions for C11H8 isomer analysis
DEFAULT_DIAGNOSTIC_REGIONS = {
    "Full Overlap":        None,           # entire shared range
    "Fingerprint":         (600,  1500),   # ring deformations, CH bends
    "Mid-IR":              (1500, 2000),   # skeletal stretches
    "C≡C Stretch":         (2050, 2200),   # ethynyl diagnostic
    "Aromatic C-H OOP":    (700,  900),    # out-of-plane CH, isomer-sensitive
}

def get_diagnostic_regions():
    """Get diagnostic regions from session state or use defaults"""
    if 'custom_regions_enabled' in st.session_state and st.session_state['custom_regions_enabled']:
        return st.session_state.get('diagnostic_regions', DEFAULT_DIAGNOSTIC_REGIONS)
    return DEFAULT_DIAGNOSTIC_REGIONS

def normalize_spectrum(y):
    """
    Min-max normalize a spectrum to [0, 1].
    Removes intensity scale bias before PCC calculation.
    """
    y_min, y_max = np.min(y), np.max(y)
    if y_max - y_min < 1e-10:
        return np.zeros_like(y)
    return (y - y_min) / (y_max - y_min)

def interpolate_to_common_grid(x1, y1, x2, y2, n_points=2000):
    """
    Interpolate both spectra onto a shared wavenumber grid
    covering only the overlapping region.
    
    Returns:
    --------
    grid, y1_interp, y2_interp : arrays or (None, None, None) if no overlap
    """
    x_min = max(np.min(x1), np.min(x2))
    x_max = min(np.max(x1), np.max(x2))
    
    if x_min >= x_max:
        return None, None, None  # No overlap
    
    grid = np.linspace(x_min, x_max, n_points)
    
    interp1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=0.0)
    interp2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=0.0)
    
    return grid, interp1(grid), interp2(grid)

def compute_pcc(exp_x, exp_y, theory_x, theory_y, region=None):
    """
    Compute Pearson Correlation Coefficient between experimental
    and theoretical spectra over a given wavenumber region.
    
    Parameters:
    -----------
    exp_x, exp_y : arrays
        Experimental wavenumber and intensity
    theory_x, theory_y : arrays
        Theoretical wavenumber and intensity
    region : tuple (min, max) in cm⁻¹, or None for full overlap
    
    Returns:
    --------
    r : float, PCC score (-1 to 1)
    p : float, p-value for statistical significance
    grid : array, common wavenumber grid used
    exp_norm : array, normalized experimental spectrum on grid
    theory_norm : array, normalized theory spectrum on grid
    """
    grid, exp_interp, theory_interp = interpolate_to_common_grid(
        exp_x, exp_y, theory_x, theory_y
    )
    
    if grid is None:
        return None, None, None, None, None
    
    # Apply region mask if specified
    if region is not None:
        mask = (grid >= region[0]) & (grid <= region[1])
        if mask.sum() < 10:  # not enough points
            return None, None, None, None, None
        grid = grid[mask]
        exp_interp = exp_interp[mask]
        theory_interp = theory_interp[mask]
    
    # Normalize both to [0, 1] - removes intensity scale differences
    exp_norm = normalize_spectrum(exp_interp)
    theory_norm = normalize_spectrum(theory_interp)
    
    # Compute Pearson correlation
    r, p = pearsonr(exp_norm, theory_norm)
    return r, p, grid, exp_norm, theory_norm

def score_label(r):
    """
    Human-readable label based on adjusted thresholds for IR-UV action spectra.
    Note: Thresholds lower than Von der Esch et al. due to action vs absorption differences.
    """
    if r is None:
        return "N/A", "gray"
    elif r >= 0.60:  # Adjusted from 0.75
        return "Excellent ✅", "green"
    elif r >= 0.40:  # Adjusted from 0.50
        return "Good 🟡", "orange"
    elif r >= 0.20:  # Adjusted from 0.30
        return "Weak ⚠️", "orange"
    else:
        return "Poor / Rule Out ❌", "red"

# ========================================================================================
# SPECTRUM PROCESSING HELPER FUNCTIONS
# ========================================================================================

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
st.markdown("## 📤 Upload DFT Output Files")

# Initialize session state for multiple structures
if 'dft_structures' not in st.session_state:
    st.session_state['dft_structures'] = []

uploaded_files = st.file_uploader(
    "Upload DFT calculation outputs (single or multiple for batch comparison)",
    type=['out', 'dat', 'log', 'txt'],
    accept_multiple_files=True,
    help="Supported formats: Gaussian .out/.log, ORCA .out/.dat, custom parsed reports. Upload multiple files to compare different candidate structures."
)

if uploaded_files:
    try:
        structures = []
        
        with st.spinner(f"Parsing {len(uploaded_files)} file(s)..."):
            for uploaded_file in uploaded_files:
                frequencies, intensities, metadata = parse_dft_file(uploaded_file)
                
                if len(frequencies) == 0:
                    st.warning(f"⚠️ No IR spectrum data found in {uploaded_file.name}")
                    continue
                
                structures.append({
                    'filename': uploaded_file.name,
                    'frequencies': frequencies,
                    'intensities': intensities,
                    'metadata': metadata
                })
        
        if structures:
            st.success(f"✅ Successfully parsed {len(structures)} structure(s)")
            
            # Store all structures in session state
            st.session_state['dft_structures'] = structures
            
            # Display summary
            st.markdown("### 📋 Loaded Structures")
            summary_data = []
            for i, struct in enumerate(structures):
                summary_data.append({
                    '#': i + 1,
                    'File': struct['filename'],
                    'Modes': len(struct['frequencies']),
                    'Freq Range (cm⁻¹)': f"{struct['frequencies'].min():.1f} - {struct['frequencies'].max():.1f}"
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Select active structure for detailed view
            if len(structures) > 1:
                st.markdown("### 🔍 Select Structure for Detailed View")
                struct_idx = st.selectbox(
                    "Choose structure:",
                    options=range(len(structures)),
                    format_func=lambda x: structures[x]['filename']
                )
            else:
                struct_idx = 0
            
            selected_struct = structures[struct_idx]
            
            # Display metadata if available
            if selected_struct['metadata']:
                st.markdown(f"#### 📋 {selected_struct['filename']} - Calculation Details")
                cols = st.columns(3)
                idx = 0
                for key, value in selected_struct['metadata'].items():
                    with cols[idx % 3]:
                        st.metric(key.replace('_', ' ').title(), value)
                    idx += 1
            
            # Display raw stick spectrum data
            with st.expander(f"📊 View {selected_struct['filename']} Raw Spectrum Data"):
                df_spectrum = pd.DataFrame({
                    'Mode': range(1, len(selected_struct['frequencies']) + 1),
                    'Frequency (cm⁻¹)': selected_struct['frequencies'],
                    'Intensity (km/mol)': selected_struct['intensities']
                })
                st.dataframe(df_spectrum, height=300)
            
            # Store primary structure (for single file workflow compatibility)
            st.session_state['dft_frequencies'] = selected_struct['frequencies']
            st.session_state['dft_intensities'] = selected_struct['intensities']
            st.session_state['dft_metadata'] = selected_struct['metadata']
            st.session_state['selected_struct_idx'] = struct_idx
        else:
            st.error("❌ No valid DFT data found in uploaded files.")
            
    except Exception as e:
        st.error(f"Error parsing files: {str(e)}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

# Broadening and Plotting Section
if 'dft_frequencies' in st.session_state:
    st.markdown("---")
    st.markdown("## 🎨 Spectrum Broadening & Visualization")
    
    # Frequency scaling section
    st.markdown("### ⚙️ DFT Frequency Scaling")
    col_scale1, col_scale2 = st.columns([2, 1])
    with col_scale1:
        freq_scale_factor = st.number_input(
            "Frequency Scale Factor",
            value=0.967,
            min_value=0.8,
            max_value=1.1,
            step=0.001,
            format="%.3f",
            help="Scale DFT frequencies to match experimental values. Common values: B3LYP/6-31G(d) = 0.967, B3LYP/cc-pVTZ = 0.989"
        )
    with col_scale2:
        st.metric("Applied Scaling", f"{freq_scale_factor:.3f}")
    
    # Apply frequency scaling
    scaled_frequencies = st.session_state['dft_frequencies'] * freq_scale_factor
    
    st.markdown("### Broadening Parameters")
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
            help="Default 0.5-0.7% represents FELIX FEL characteristic bandwidth. FWHM = bandwidth% × frequency"
        )
        bw_frac = bw_percent / 100.0  # Convert percentage to fraction
    with col3:
        npoints = st.number_input("Number of Points", value=4000, step=100, min_value=100)
    
    # Apply broadening with scaled frequencies
    x_broad, y_broad = broaden_spectrum_felix(
        scaled_frequencies,
        st.session_state['dft_intensities'],
        x_range=(x_min, x_max),
        bw_frac=bw_frac,
        npoints=int(npoints)
    )
    
    # Store broadened spectrum, scaled frequencies, and broadening parameters
    st.session_state['dft_x_broad'] = x_broad
    st.session_state['dft_y_broad'] = y_broad
    st.session_state['dft_frequencies_scaled'] = scaled_frequencies
    st.session_state['freq_scale_factor'] = freq_scale_factor
    st.session_state['bw_frac'] = bw_frac
    st.session_state['x_min'] = x_min
    st.session_state['x_max'] = x_max
    
    st.markdown("### 📈 DFT Theoretical Spectrum")
    
    # Show scaling info
    if abs(freq_scale_factor - 1.0) > 0.001:
        st.info(f"📐 Frequencies scaled by factor {freq_scale_factor:.3f} (e.g., {st.session_state['dft_frequencies'][0]:.1f} → {scaled_frequencies[0]:.1f} cm⁻¹)")
    
    # Interactive plot
    st.markdown("###### *:green[Interactive plot with Plotly]*")
    fig_dft = go.Figure()
    
    # Stick spectrum (using scaled frequencies)
    fig_dft.add_trace(go.Scatter(
        x=scaled_frequencies,
        y=st.session_state['dft_intensities'],
        mode='markers',
        marker=dict(size=8, color='red', symbol='line-ns-open'),
        name='Stick Spectrum (Scaled)'
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
    
    # Stick spectrum as stems (using scaled frequencies)
    ax.vlines(scaled_frequencies, 0, st.session_state['dft_intensities'], 
              colors='red', alpha=0.6, linewidths=1.5, label='Stick Spectrum (Scaled)')
    
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

# ========================================================================================
# DIAGNOSTIC REGIONS CONFIGURATION
# ========================================================================================
st.markdown("---")
st.markdown("## ⚙️ Configure Diagnostic Regions for PCC Scoring")

with st.expander("🎯 Customize Spectral Regions (Optional)", expanded=False):
    st.markdown("""
    **Why customize regions?**
    - Your experimental range might not cover all default regions (e.g., no C≡C data above 2000 cm⁻¹)
    - Different molecules have different diagnostic bands (carbonyls, nitriles, etc.)
    - Avoid scoring regions with no experimental coverage → prevents `NaN` in rankings
    """)
    
    use_custom = st.checkbox(
        "Enable Custom Diagnostic Regions",
        value=st.session_state.get('custom_regions_enabled', False),
        help="Override default C₁₁H₈ regions with your own spectral windows"
    )
    st.session_state['custom_regions_enabled'] = use_custom
    
    if use_custom:
        st.markdown("### Define Your Regions:")
        st.caption("Leave a region's min/max empty to disable it. 'Full Overlap' is always included.")
        
        # Initialize custom regions if not exists
        if 'diagnostic_regions' not in st.session_state:
            st.session_state['diagnostic_regions'] = DEFAULT_DIAGNOSTIC_REGIONS.copy()
        
        custom_regions = {"Full Overlap": None}  # Always include full overlap
        
        # Create input fields for each region
        num_regions = st.number_input("Number of custom regions", min_value=1, max_value=8, value=4, step=1)
        
        for i in range(num_regions):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                region_name = st.text_input(
                    f"Region {i+1} Name",
                    value=list(DEFAULT_DIAGNOSTIC_REGIONS.keys())[i+1] if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS) else f"Custom_{i+1}",
                    key=f"region_name_{i}"
                )
            with col2:
                default_min = 500.0
                if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS):
                    default_vals = list(DEFAULT_DIAGNOSTIC_REGIONS.values())[i+1]
                    if default_vals:
                        default_min = float(default_vals[0])
                
                region_min = st.number_input(
                    f"Min (cm⁻¹)",
                    value=default_min,
                    step=50.0,
                    key=f"region_min_{i}",
                    format="%.0f"
                )
            with col3:
                default_max = 1500.0
                if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS):
                    default_vals = list(DEFAULT_DIAGNOSTIC_REGIONS.values())[i+1]
                    if default_vals:
                        default_max = float(default_vals[1])
                
                region_max = st.number_input(
                    f"Max (cm⁻¹)",
                    value=default_max,
                    step=50.0,
                    key=f"region_max_{i}",
                    format="%.0f"
                )
            
            if region_name and region_min < region_max:
                custom_regions[region_name] = (region_min, region_max)
        
        st.session_state['diagnostic_regions'] = custom_regions
        
        # Show preview
        st.markdown("### ✅ Active Regions:")
        preview_df = pd.DataFrame([
            {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
            for name, rng in custom_regions.items()
        ])
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        st.info("💡 **Tip:** Set regions to match your experimental coverage. Batch ranking will only use regions with valid PCC scores.")
    
    else:
        st.markdown("**Using default C₁₁H₈ isomer regions:**")
        default_df = pd.DataFrame([
            {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
            for name, rng in DEFAULT_DIAGNOSTIC_REGIONS.items()
        ])
        st.dataframe(default_df, use_container_width=True, hide_index=True)

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
        
        # ========================================================================================
        # PCC SCORING SECTION
        # ========================================================================================
        st.markdown("---")
        st.markdown("### 📐 Quantitative Similarity - Pearson Correlation (PCC)")
        st.caption("""
        **Position-based scoring for isomer discrimination.** Intensity differences are normalized out.  
        ⚠️ Note: Thresholds adjusted for IR-UV spectra (lower than absorption IR literature values).
        """)
        
        # Compute PCC for all diagnostic regions
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        pcc_results = []
        for region_name, region_range in DIAGNOSTIC_REGIONS.items():
            r, p, grid, exp_norm, theory_norm = compute_pcc(
                exp_x, exp_y,
                theory_x_shifted, theory_y,
                region=region_range
            )
            label, color = score_label(r)
            
            pcc_results.append({
                "Region": region_name,
                "Range (cm⁻¹)": f"{region_range[0]}–{region_range[1]}" if region_range else "Full",
                "PCC (r)": round(r, 4) if r is not None else None,
                "p-value": f"{p:.2e}" if p is not None else None,
                "Verdict": label,
            })
        
        # Display PCC table
        df_pcc = pd.DataFrame(pcc_results)
        
        # Style the dataframe
        def highlight_verdict(row):
            if "Excellent" in str(row["Verdict"]):
                color = "background-color: #d4edda; color: #155724; font-weight: bold"
            elif "Good" in str(row["Verdict"]):
                color = "background-color: #fff3cd; color: #856404"
            elif "Weak" in str(row["Verdict"]):
                color = "background-color: #fff3cd; color: #856404"
            else:
                color = "background-color: #f8d7da; color: #721c24"
            return ['', '', '', '', color]
        
        st.dataframe(
            df_pcc.style.apply(highlight_verdict, axis=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Bar chart of PCC scores per region
        st.markdown("#### � Regional PCC Scores")
        fig_pcc, ax_pcc = plt.subplots(figsize=(10, 4))
        valid = df_pcc.dropna(subset=["PCC (r)"])
        colors_bar = [
            "#28a745" if r >= 0.60 else "#ffc107" if r >= 0.40 else "#dc3545"
            for r in valid["PCC (r)"]
        ]
        bars = ax_pcc.barh(valid["Region"], valid["PCC (r)"], color=colors_bar, alpha=0.8)
        ax_pcc.axvline(0.40, color='orange', linestyle='--', linewidth=1.2, label='Good threshold (0.40)', alpha=0.7)
        ax_pcc.axvline(0.60, color='green', linestyle='--', linewidth=1.2, label='Excellent threshold (0.60)', alpha=0.7)
        ax_pcc.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        ax_pcc.set_xlabel("Pearson r", fontsize=11)
        ax_pcc.set_title("Region-wise PCC Scores (Adjusted for IR-UV Spectra)", fontsize=13, fontweight='bold')
        ax_pcc.set_xlim(-1, 1)
        ax_pcc.legend(fontsize=9, loc='lower right')
        ax_pcc.grid(True, axis='x', alpha=0.3)
        fig_pcc.tight_layout()
        st.pyplot(fig_pcc)
        
        add_plot_to_report_button(
            fig_pcc,
            "PCC Region Scores",
            key_suffix="pcc_scores",
            description="Pearson correlation scores per diagnostic spectral region"
        )
        
        # Structure assignment decision logic
        st.markdown("#### 🧪 Structure Assignment Decision")
        
        full_r = df_pcc[df_pcc["Region"] == "Full Overlap"]["PCC (r)"].values
        full_r = full_r[0] if len(full_r) > 0 and not pd.isna(full_r[0]) else None
        
        fp_r = df_pcc[df_pcc["Region"] == "Fingerprint"]["PCC (r)"].values
        fp_r = fp_r[0] if len(fp_r) > 0 and not pd.isna(fp_r[0]) else None
        
        cc_r = df_pcc[df_pcc["Region"] == "C≡C Stretch"]["PCC (r)"].values
        cc_r = cc_r[0] if len(cc_r) > 0 and not pd.isna(cc_r[0]) else None
        
        if full_r is not None and full_r > 0.60 and fp_r is not None and fp_r > 0.40:
            st.success("🟢 **Candidate Structure:** Strong overall and fingerprint agreement. Consistent with this structure.")
        elif full_r is not None and full_r > 0.40:
            st.warning("🟡 **Tentative Match:** Moderate overall agreement. Check C≡C region and aromatic CH pattern manually for confirmation.")
        elif cc_r is not None and cc_r < 0.20:
            st.error("🔴 **Rule Out:** C≡C stretch region shows very poor agreement — strong evidence against this structure.")
        elif full_r is not None and full_r < 0.20:
            st.error("🔴 **Rule Out:** Overall PCC too low. This structure is inconsistent with the experimental spectrum.")
        else:
            st.info("ℹ️ **Inconclusive:** Mixed scores across regions. Consider visual inspection and additional diagnostic regions.")
        
        with st.expander("ℹ️ About PCC Scoring for IR-UV Spectra"):
            st.markdown("""
            ### Pearson Correlation Coefficient (PCC) for Peak Position Matching
            
            **What it measures:**
            - Correlation between experimental and theoretical peak *positions*
            - Both spectra normalized to [0,1] before comparison → **intensity-independent**
            - Focus on spectral pattern/shape matching
            
            **Adjusted Thresholds (IR-UV Spectra):**
            - **r ≥ 0.60:** Excellent match (structure candidate)
            - **r ≥ 0.40:** Good match (tentative)
            - **r ≥ 0.20:** Weak match
            - **r < 0.20:** Poor match (likely rule out)
            
            *Note: Thresholds lower than absorption IR (Von der Esch 0.75/0.50) due to IR-UV ion dip spectroscopy differences.*
            
            **Best Practices:**
            - Use **regional scores** for isomer discrimination (C≡C stretch, aromatic CH)
            - Compare **multiple candidate structures** - highest PCC wins
            - **Visual inspection** remains critical - PCC is a guide, not absolute truth
            - Low p-values (< 0.05) indicate statistically significant correlation
            
            **Why it works for your C₁₁H₈ isomers:**
            - C≡C stretch position (2050-2200 cm⁻¹) is isomer-specific
            - Aromatic CH out-of-plane bends (700-900 cm⁻¹) vary by substitution pattern
            - Fingerprint region captures overall skeletal differences
            """)
        
        # Save comparison data
        if st.checkbox("💾 Save Comparison Data"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            
            # Save experimental and theoretical data separately (different lengths)
            exp_df = pd.DataFrame({
                'Wavenumber_Exp': exp_x,
                'Ln_Depletion_Exp': exp_y,
            })
            theory_df = pd.DataFrame({
                'Wavenumber_Theory': theory_x_shifted,
                'Intensity_Theory_km_mol': theory_y
            })
            
            exp_filename = os.path.join(file_directory, f"exp_data_{timestamp}.csv")
            theory_filename = os.path.join(file_directory, f"theory_data_{timestamp}.csv")
            exp_df.to_csv(exp_filename, index=False)
            theory_df.to_csv(theory_filename, index=False)
            
            # Save PCC results
            pcc_filename = os.path.join(file_directory, f"pcc_scores_{timestamp}.csv")
            df_pcc.to_csv(pcc_filename, index=False)
            
            st.success(f"✅ Experimental data saved to: {exp_filename}")
            st.success(f"✅ Theory data saved to: {theory_filename}")
            st.success(f"✅ PCC scores saved to: {pcc_filename}")

elif fullrange_depletion_data is None:
    st.info("⚠️ No experimental data found in session. Please run the depletion calculation (Section 3.0) first.")
elif 'dft_x_broad' not in st.session_state:
    st.info("⚠️ Please upload and process a DFT file first.")

# ========================================================================================
# BATCH MULTI-STRUCTURE COMPARISON SECTION
# ========================================================================================

# Batch Multi-Structure Comparison
if fullrange_depletion_data is not None and len(st.session_state.get('dft_structures', [])) > 1:
    st.markdown("---")
    st.markdown("## 🏆 Multi-Structure Batch Comparison")
    st.info("""
    **Rank all candidate structures** using PCC scoring across diagnostic regions.  
    Compare multiple DFT calculations to identify the best match to your experimental spectrum.
    """)
    
    if st.button("🚀 Run Batch PCC Analysis", type="primary"):
        structures = st.session_state['dft_structures']
        
        # Get experimental data
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values
        
        # Get parameters from session (set by the broadening UI section)
        freq_scale = st.session_state.get('freq_scale_factor', 0.967)
        bw_frac = st.session_state.get('bw_frac', 0.007)
        x_min = st.session_state.get('x_min', 500.0)
        x_max = st.session_state.get('x_max', 2200.0)
        shift = 0.0  # No shift for batch comparison
        
        # Get active diagnostic regions
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        
        st.markdown("### 🔄 Processing All Structures...")
        
        # Store results for all structures
        all_results = []
        
        progress_bar = st.progress(0)
        for idx, struct in enumerate(structures):
            with st.spinner(f"Processing {struct['filename']}..."):
                # Scale and broaden
                scaled_freq = struct['frequencies'] * freq_scale
                theory_x, theory_y = broaden_spectrum_felix(
                    scaled_freq,
                    struct['intensities'],
                    x_range=(x_min, x_max),
                    bw_frac=bw_frac,
                    npoints=4000
                )
                theory_x_shifted = theory_x + shift
                
                # Compute PCC for all regions
                struct_pcc = {'filename': struct['filename']}
                for region_name, region_range in DIAGNOSTIC_REGIONS.items():
                    r, p, _, _, _ = compute_pcc(
                        exp_x, exp_y,
                        theory_x_shifted, theory_y,
                        region=region_range
                    )
                    struct_pcc[region_name] = r if r is not None else np.nan
                
                all_results.append(struct_pcc)
                progress_bar.progress((idx + 1) / len(structures))
        
        # Create comparison DataFrame
        df_batch = pd.DataFrame(all_results)
        
        # Calculate average PCC - use only valid (non-NaN) regions, excluding "Full Overlap"
        # Get all region names except "Full Overlap"
        scoring_regions = [r for r in DIAGNOSTIC_REGIONS.keys() if r != "Full Overlap"]
        
        # For each structure, compute mean of non-NaN scores
        df_batch['Average PCC'] = df_batch[scoring_regions].mean(axis=1, skipna=True)
        df_batch['Valid Regions'] = df_batch[scoring_regions].notna().sum(axis=1)  # Count valid scores
        
        # Rank structures
        df_batch['Rank'] = df_batch['Average PCC'].rank(ascending=False, method='min').astype(int)
        df_batch = df_batch.sort_values('Rank')
        
        # Display results
        st.markdown("### 📊 Ranking Results")
        
        # Highlight best match
        best_match = df_batch.iloc[0]
        st.success(f"🥇 **Best Match:** {best_match['filename']} (Avg PCC: {best_match['Average PCC']:.3f}, based on {best_match['Valid Regions']:.0f} regions)")
        
        # Check for regions with no overlap
        nan_counts = df_batch[scoring_regions].isna().sum()
        if nan_counts.sum() > 0:
            st.warning(f"⚠️ Some regions had no experimental coverage: {', '.join([f'{r} ({nan_counts[r]} structures)' for r in nan_counts[nan_counts > 0].index])}")
        
        # Display full table
        st.dataframe(
            df_batch.style.background_gradient(subset=['Average PCC'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True,
            hide_index=True
        )
        
        # Comparison bar chart
        st.markdown("### 📈 Visual Comparison")
        
        fig_batch, ax_batch = plt.subplots(figsize=(12, max(6, len(structures) * 0.5)))
        
        # Plot each region as grouped bars - use all defined regions
        x_pos = np.arange(len(structures))
        regions_to_plot = list(DIAGNOSTIC_REGIONS.keys())
        num_regions = len(regions_to_plot)
        width = 0.8 / num_regions  # Dynamic width based on number of regions
        
        # Generate colors dynamically
        colors_regions = plt.cm.tab10(np.linspace(0, 1, num_regions))
        
        for i, region in enumerate(regions_to_plot):
            if region in df_batch.columns:
                ax_batch.barh(
                    x_pos + i * width,
                    df_batch[region].values,
                    width,
                    label=region,
                    color=colors_regions[i],
                    alpha=0.8
                )
        
        ax_batch.set_yticks(x_pos + width * (num_regions - 1) / 2)
        ax_batch.set_yticklabels(df_batch['filename'].values)
        ax_batch.set_xlabel('PCC Score (r)', fontsize=12)
        ax_batch.set_title('Multi-Structure PCC Comparison Across Diagnostic Regions', fontsize=14, fontweight='bold')
        ax_batch.axvline(0.40, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Good (0.40)')
        ax_batch.axvline(0.60, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.60)')
        ax_batch.legend(loc='lower right', fontsize=8, ncol=2)
        ax_batch.grid(True, axis='x', alpha=0.3)
        ax_batch.set_xlim(-0.2, 1.0)
        fig_batch.tight_layout()
        st.pyplot(fig_batch)
        
        add_plot_to_report_button(
            fig_batch,
            "Multi-Structure PCC Comparison",
            key_suffix="batch_pcc",
            description="Batch comparison of all candidate structures using PCC scoring"
        )
        
        # Alternative view: Heatmap
        st.markdown("### 🗺️ PCC Heatmap")
        
        fig_heat, ax_heat = plt.subplots(figsize=(10, max(6, len(structures) * 0.4)))
        
        # Prepare data for heatmap
        heat_data = df_batch[regions_to_plot].values
        
        im = ax_heat.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
        
        # Set ticks
        ax_heat.set_xticks(np.arange(len(regions_to_plot)))
        ax_heat.set_yticks(np.arange(len(structures)))
        ax_heat.set_xticklabels(regions_to_plot, rotation=45, ha='right')
        ax_heat.set_yticklabels(df_batch['filename'].values)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heat)
        cbar.set_label('PCC Score', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(structures)):
            for j in range(len(regions_to_plot)):
                value = heat_data[i, j]
                if not np.isnan(value):
                    text = ax_heat.text(j, i, f'{value:.2f}',
                                       ha="center", va="center", color="black" if value < 0.5 else "white",
                                       fontsize=9, fontweight='bold')
        
        ax_heat.set_title('PCC Heatmap: All Structures vs Experimental', fontsize=14, fontweight='bold')
        fig_heat.tight_layout()
        st.pyplot(fig_heat)
        
        add_plot_to_report_button(
            fig_heat,
            "PCC Heatmap",
            key_suffix="pcc_heatmap",
            description="Heatmap showing PCC scores for all structures across regions"
        )
        
        # Save batch results
        if st.checkbox("💾 Save Batch Comparison Results", key="save_batch"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            
            batch_filename = os.path.join(file_directory, f"batch_pcc_comparison_{timestamp}.csv")
            df_batch.to_csv(batch_filename, index=False)
            
            st.success(f"✅ Batch comparison results saved to: {batch_filename}")
            
            # Also save a summary report
            summary_text = f"""
Multi-Structure PCC Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Number of structures analyzed: {len(structures)}
Experimental data range: {exp_x.min():.1f} - {exp_x.max():.1f} cm⁻¹

Parameters:
- Frequency scaling: {freq_scale:.3f}
- FELIX bandwidth: {bw_frac*100:.2f}%
- Wavenumber shift: {shift:.1f} cm⁻¹

Rankings:
"""
            for _, row in df_batch.iterrows():
                summary_text += f"\n{row['Rank']:.0f}. {row['filename']}"
                summary_text += f"\n   Average PCC: {row['Average PCC']:.3f}"
                summary_text += f"\n   Full Overlap: {row.get('Full Overlap', np.nan):.3f}"
                summary_text += f"\n   Fingerprint: {row.get('Fingerprint', np.nan):.3f}"
                summary_text += f"\n   C≡C Stretch: {row.get('C≡C Stretch', np.nan):.3f}"
                summary_text += "\n"
            
            report_filename = os.path.join(file_directory, f"batch_pcc_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(summary_text)
            
            st.success(f"✅ Summary report saved to: {report_filename}")

elif len(st.session_state.get('dft_structures', [])) <= 1:
    st.info("💡 **Tip:** Upload multiple DFT files to enable batch comparison mode and rank candidate structures.")

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
