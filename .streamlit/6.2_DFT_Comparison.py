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
st.caption("Compare experimental IR-UV ion-dip spectra with DFT calculations using region-specific Pearson Correlation.")

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

# Default PCC thresholds (adjusted for IR-UV action spectra)
DEFAULT_PCC_THRESHOLDS = {
    "excellent": 0.60,
    "good": 0.40,
    "weak": 0.20,
}

def get_pcc_thresholds():
    """Get PCC thresholds from session state or use defaults"""
    return {
        "excellent": st.session_state.get("pcc_threshold_excellent", DEFAULT_PCC_THRESHOLDS["excellent"]),
        "good": st.session_state.get("pcc_threshold_good", DEFAULT_PCC_THRESHOLDS["good"]),
        "weak": st.session_state.get("pcc_threshold_weak", DEFAULT_PCC_THRESHOLDS["weak"]),
    }

def score_label(r):
    """
    Human-readable label based on adjustable thresholds for IR-UV action spectra.
    Reads thresholds from session state so users can customize them.
    """
    if r is None:
        return "N/A", "gray"
    thresholds = get_pcc_thresholds()
    if r >= thresholds["excellent"]:
        return "Excellent ✅", "green"
    elif r >= thresholds["good"]:
        return "Good 🟡", "orange"
    elif r >= thresholds["weak"]:
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
    """
    Parse Gaussian anharmonic frequency output files.
    
    Reads all three sections produced by Gaussian anharmonic calculations:
      - Fundamental Bands:   Mode(n)        E(harm) E(anharm) I(harm) I(anharm)
      - Overtones:           Mode(n)        E(harm) E(anharm) I(harm) I(anharm)
      - Combination Bands:   Mode(n) Mode(m) E(harm) E(anharm) I(harm) I(anharm)
    
    Returns all bands merged into a single stick spectrum.
    """
    fundamentals = []
    overtones = []
    combinations = []
    
    lines = content.split('\n')
    current_section = None  # 'fundamental', 'overtone', 'combination'
    in_data = False
    
    for line in lines:
        stripped = line.strip()
        
        # Detect section headers
        if 'Fundamental Bands' in line:
            current_section = 'fundamental'
            in_data = False
            continue
        elif 'Overtones' in line and 'Combination' not in line:
            current_section = 'overtone'
            in_data = False
            continue
        elif 'Combination Bands' in line:
            current_section = 'combination'
            in_data = False
            continue
        
        if current_section is None:
            continue
        
        # Skip header/separator lines, then start reading data
        if 'Mode(n)' in line or 'Mode' in line:
            in_data = True
            continue
        if stripped.startswith('---') or stripped.startswith('==='):
            in_data = True
            continue
        
        # Empty line → end of current section data
        if stripped == '':
            if in_data:
                in_data = False
                current_section = None
            continue
        
        if not in_data:
            continue
        
        parts = line.split()
        
        # Need at least a mode identifier to be a data line
        if not parts or '(' not in parts[0]:
            continue
        
        try:
            # Extract numeric values, skipping mode identifiers (contain '(')
            # and text fields like Irrep labels
            nums = []
            for p in parts:
                if '(' in p:
                    continue
                try:
                    nums.append(float(p))
                except ValueError:
                    continue
            
            # nums should be [E_harm, E_anharm, I_harm, I_anharm] (4 values)
            # or             [E_harm, E_anharm, I_anharm]          (3 values)
            if len(nums) >= 4:
                freq = nums[1]    # E(anharm)
                inten = nums[3]   # I(anharm)
            elif len(nums) >= 3:
                freq = nums[1]    # E(anharm)
                inten = nums[2]   # I(anharm) when no I(harm) column
            elif len(nums) >= 2:
                freq = nums[0]
                inten = nums[1]
            else:
                continue
            
            if current_section == 'fundamental':
                fundamentals.append((freq, inten))
            elif current_section == 'overtone':
                overtones.append((freq, inten))
            elif current_section == 'combination':
                combinations.append((freq, inten))
        except (ValueError, IndexError):
            continue
    
    # Merge all bands into single arrays
    all_bands = fundamentals + overtones + combinations
    if not all_bands:
        return np.array([]), np.array([]), {'type': 'anharmonic'}
    
    frequencies = np.array([b[0] for b in all_bands])
    intensities = np.array([b[1] for b in all_bands])
    
    # Build per-band type labels for colour-coded plotting
    band_types = (
        ['fundamental'] * len(fundamentals)
        + ['overtone'] * len(overtones)
        + ['combination'] * len(combinations)
    )
    
    metadata = {
        'type': 'anharmonic',
        'n_fundamentals': len(fundamentals),
        'n_overtones': len(overtones),
        'n_combinations': len(combinations),
        'band_types': band_types,
    }
    
    return frequencies, intensities, metadata

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
    elif 'Fundamental Bands' in content or 'Overtones' in content or 'Combination Bands' in content:
        return parse_gaussian_anharmonic(content)
    elif 'Gaussian' in content[:1000] or 'Frequencies --' in content:
        return parse_gaussian_out(content)
    else:
        st.warning("Could not auto-detect file format. Trying custom parser...")
        return parse_custom_report(content)

# Main UI
st.markdown("---")
st.markdown("## Step 1 — Upload DFT Output Files")

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
                row = {
                    '#': i + 1,
                    'File': struct['filename'],
                    'Modes': len(struct['frequencies']),
                    'Freq Range (cm⁻¹)': f"{struct['frequencies'].min():.1f} - {struct['frequencies'].max():.1f}"
                }
                if struct['metadata'].get('type') == 'anharmonic':
                    nf = struct['metadata'].get('n_fundamentals', 0)
                    no = struct['metadata'].get('n_overtones', 0)
                    nc = struct['metadata'].get('n_combinations', 0)
                    row['Breakdown'] = f"{nf} fund + {no} over + {nc} comb"
                summary_data.append(row)
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
                    if key == 'band_types':  # skip large list
                        continue
                    with cols[idx % 3]:
                        st.metric(key.replace('_', ' ').title(), value)
                    idx += 1
            
            # Display raw stick spectrum data
            with st.expander(f"📊 View {selected_struct['filename']} Raw Spectrum Data"):
                spec_dict = {
                    'Mode': range(1, len(selected_struct['frequencies']) + 1),
                    'Frequency (cm⁻¹)': selected_struct['frequencies'],
                    'Intensity (km/mol)': selected_struct['intensities'],
                }
                if 'band_types' in selected_struct['metadata']:
                    spec_dict['Band Type'] = selected_struct['metadata']['band_types']
                df_spectrum = pd.DataFrame(spec_dict)
                st.dataframe(df_spectrum, height=300)
            
            # Store primary structure (for single file workflow compatibility)
            st.session_state['dft_frequencies'] = selected_struct['frequencies']
            st.session_state['dft_intensities'] = selected_struct['intensities']
            st.session_state['dft_metadata'] = selected_struct['metadata']
            st.session_state['dft_band_types'] = selected_struct['metadata'].get('band_types', None)
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
    st.markdown("## Step 2 — Broadening & Visualization")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        freq_scale_factor = st.number_input(
            "Scale Factor",
            value=0.967, min_value=0.8, max_value=1.1, step=0.001, format="%.3f",
            help="Scale DFT frequencies. Common: B3LYP/6-31G(d) = 0.967, B3LYP/cc-pVTZ = 0.989"
        )
    with col_s2:
        bw_percent = st.number_input(
            "FELIX Bandwidth (%)", value=0.7, min_value=0.1, max_value=5.0, step=0.1, format="%.2f",
            help="FWHM = bandwidth% × frequency. Default 0.7% for FELIX."
        )
        bw_frac = bw_percent / 100.0
    with col_s3:
        x_min = st.number_input("Wavenum. Min (cm⁻¹)", value=500.0, step=50.0)
    with col_s4:
        x_max = st.number_input("Wavenum. Max (cm⁻¹)", value=2200.0, step=50.0)
    
    # Apply frequency scaling
    scaled_frequencies = st.session_state['dft_frequencies'] * freq_scale_factor
    
    # Apply broadening with scaled frequencies
    x_broad, y_broad = broaden_spectrum_felix(
        scaled_frequencies,
        st.session_state['dft_intensities'],
        x_range=(x_min, x_max),
        bw_frac=bw_frac,
        npoints=4000
    )
    
    # Store broadened spectrum, scaled frequencies, and broadening parameters
    st.session_state['dft_x_broad'] = x_broad
    st.session_state['dft_y_broad'] = y_broad
    st.session_state['dft_frequencies_scaled'] = scaled_frequencies
    st.session_state['freq_scale_factor'] = freq_scale_factor
    st.session_state['bw_frac'] = bw_frac
    st.session_state['x_min'] = x_min
    st.session_state['x_max'] = x_max
    
    # Show scaling info
    if abs(freq_scale_factor - 1.0) > 0.001:
        st.caption(f"📐 Scaled by {freq_scale_factor:.3f} — e.g. {st.session_state['dft_frequencies'][0]:.1f} → {scaled_frequencies[0]:.1f} cm⁻¹")
    
    # DFT spectrum in tabs
    tab_plotly_dft, tab_mpl_dft, tab_info_dft = st.tabs(["📈 Interactive Plot", "🖼️ Static Plot (Report)", "ℹ️ About Broadening"])
    
    with tab_plotly_dft:
        fig_dft = go.Figure()
        fig_dft.add_trace(go.Scatter(
            x=scaled_frequencies, y=st.session_state['dft_intensities'],
            mode='markers', marker=dict(size=8, color='red', symbol='line-ns-open'),
            name='Stick Spectrum (Scaled)'
        ))
        fig_dft.add_trace(go.Scatter(
            x=x_broad, y=y_broad, mode='lines', line=dict(color='blue', width=2),
            name=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)'
        ))
        fig_dft.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Intensity (km/mol)",
            title="DFT IR Spectrum", hovermode='closest', legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig_dft, use_container_width=True)
    
    with tab_mpl_dft:
        fig_static, ax = plt.subplots(figsize=(12, 5))
        ax.vlines(scaled_frequencies, 0, st.session_state['dft_intensities'], 
                  colors='red', alpha=0.6, linewidths=1.5, label='Stick Spectrum (Scaled)')
        ax.plot(x_broad, y_broad, 'b-', linewidth=2, label=f'Broadened (FWHM = {bw_frac*100:.2f}% × ν)')
        ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
        ax.set_ylabel("Intensity (km/mol)", fontsize=12)
        ax.set_title("DFT IR Spectrum", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_xlim(x_min, x_max)
        fig_static.tight_layout()
        st.pyplot(fig_static)
        add_plot_to_report_button(fig_static, "DFT IR Spectrum", key_suffix="dft_spectrum",
                                  description="DFT-calculated IR spectrum with FELIX-style broadening")
    
    with tab_info_dft:
        st.markdown("""
        **Frequency-Proportional Broadening:** FWHM scales linearly with frequency — `FWHM(ν) = bw_frac × ν`.
        
        | Wavenumber | FWHM (0.7%) |
        |---|---|
        | 500 cm⁻¹ | 3.5 cm⁻¹ |
        | 1000 cm⁻¹ | 7.0 cm⁻¹ |
        | 1500 cm⁻¹ | 10.5 cm⁻¹ |
        | 3000 cm⁻¹ | 21.0 cm⁻¹ |
        
        This is characteristic of FEL instruments and provides more physically accurate comparison than constant FWHM broadening.
        """)

# ========================================================================================
# PCC CONFIGURATION: Diagnostic Regions & Thresholds (only when DFT loaded)
# ========================================================================================
if 'dft_frequencies' in st.session_state:
    with st.expander("⚙️ PCC Scoring Configuration — Regions, Thresholds & Info", expanded=False):
        cfg_tab_regions, cfg_tab_thresholds, cfg_tab_about = st.tabs(["🎯 Diagnostic Regions", "📏 Score Thresholds", "ℹ️ About PCC"])
        
        with cfg_tab_regions:
            use_custom = st.checkbox(
                "Enable Custom Diagnostic Regions",
                value=st.session_state.get('custom_regions_enabled', False),
                help="Override default C₁₁H₈ regions with your own spectral windows"
            )
            st.session_state['custom_regions_enabled'] = use_custom
            
            if use_custom:
                st.caption("'Full Overlap' is always included. Set regions to match your experimental coverage.")
                
                if 'diagnostic_regions' not in st.session_state:
                    st.session_state['diagnostic_regions'] = DEFAULT_DIAGNOSTIC_REGIONS.copy()
                
                custom_regions = {"Full Overlap": None}
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
                            if default_vals: default_min = float(default_vals[0])
                        region_min = st.number_input(f"Min (cm⁻¹)", value=default_min, step=50.0, key=f"region_min_{i}", format="%.0f")
                    with col3:
                        default_max = 1500.0
                        if i+1 < len(DEFAULT_DIAGNOSTIC_REGIONS):
                            default_vals = list(DEFAULT_DIAGNOSTIC_REGIONS.values())[i+1]
                            if default_vals: default_max = float(default_vals[1])
                        region_max = st.number_input(f"Max (cm⁻¹)", value=default_max, step=50.0, key=f"region_max_{i}", format="%.0f")
                    
                    if region_name and region_min < region_max:
                        custom_regions[region_name] = (region_min, region_max)
                
                st.session_state['diagnostic_regions'] = custom_regions
                preview_df = pd.DataFrame([
                    {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
                    for name, rng in custom_regions.items()
                ])
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("**Using default C₁₁H₈ isomer regions:**")
                default_df = pd.DataFrame([
                    {"Region": name, "Range": f"{rng[0]:.0f}-{rng[1]:.0f} cm⁻¹" if rng else "Full overlap"}
                    for name, rng in DEFAULT_DIAGNOSTIC_REGIONS.items()
                ])
                st.dataframe(default_df, use_container_width=True, hide_index=True)
        
        with cfg_tab_thresholds:
            st.caption("Set the PCC score boundaries for each verdict category. Values are Pearson r (-1 to 1).")
            thr_col1, thr_col2, thr_col3 = st.columns(3)
            with thr_col1:
                st.session_state["pcc_threshold_excellent"] = st.number_input(
                    "Excellent ✅ (r ≥)", value=st.session_state.get("pcc_threshold_excellent", DEFAULT_PCC_THRESHOLDS["excellent"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_exc"
                )
            with thr_col2:
                st.session_state["pcc_threshold_good"] = st.number_input(
                    "Good 🟡 (r ≥)", value=st.session_state.get("pcc_threshold_good", DEFAULT_PCC_THRESHOLDS["good"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_good"
                )
            with thr_col3:
                st.session_state["pcc_threshold_weak"] = st.number_input(
                    "Weak ⚠️ (r ≥)", value=st.session_state.get("pcc_threshold_weak", DEFAULT_PCC_THRESHOLDS["weak"]),
                    min_value=0.0, max_value=1.0, step=0.05, format="%.2f", key="_pcc_thr_weak"
                )
            st.caption("Below **Weak** → **Poor / Rule Out ❌**")
        
        with cfg_tab_about:
            thresholds_info = get_pcc_thresholds()
            st.markdown(f"""
            **PCC** measures correlation between experimental and theoretical peak *positions* after min-max normalization to [0,1] — making it **intensity-independent**.
            
            | Verdict | Threshold |
            |---|---|
            | Excellent ✅ | r ≥ {thresholds_info['excellent']:.2f} |
            | Good 🟡 | r ≥ {thresholds_info['good']:.2f} |
            | Weak ⚠️ | r ≥ {thresholds_info['weak']:.2f} |
            | Poor ❌ | r < {thresholds_info['weak']:.2f} |
            
            *Thresholds lower than absorption IR (Von der Esch 0.75/0.50) due to IR-UV ion dip spectroscopy.*
            
            **Tips:** Use regional scores for isomer discrimination. Compare multiple structures — highest PCC wins. Visual inspection remains critical.
            """)

# Experimental vs Theoretical Comparison
st.markdown("---")
st.markdown("## Step 3 — Compare with Experimental Data")

# Check if experimental data is available
fullrange_depletion_data = st.session_state.get("fullrange_depletion_data", None)

if fullrange_depletion_data is not None and 'dft_x_broad' in st.session_state:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        shift_theory = st.number_input("Shift Theory (cm⁻¹)", value=st.session_state.get("shift_theory", 0.0), step=1.0, format="%.1f",
                                      help="Shift theoretical spectrum for alignment", key="shift_theory")
    with col2:
        invert_theory = st.checkbox("Invert Theory", value=False, key="invert_theory",
                                   help="Invert theoretical spectrum if needed")
    with col3:
        run_comparison = st.button("📊 Compare", type="primary", use_container_width=True)
    
    # Compute comparison when button pressed, store results in session state
    if run_comparison:
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values
        theory_x_shifted = st.session_state['dft_x_broad'] + shift_theory
        theory_y = st.session_state['dft_y_broad'].copy()
        if invert_theory:
            theory_y = -theory_y
        
        # Compute PCC for all diagnostic regions
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        pcc_results = []
        for region_name, region_range in DIAGNOSTIC_REGIONS.items():
            r, p, grid, exp_norm, theory_norm = compute_pcc(
                exp_x, exp_y, theory_x_shifted, theory_y, region=region_range
            )
            label, color = score_label(r)
            pcc_results.append({
                "Region": region_name,
                "Range (cm⁻¹)": f"{region_range[0]}–{region_range[1]}" if region_range else "Full",
                "PCC (r)": round(r, 4) if r is not None else None,
                "p-value": f"{p:.2e}" if p is not None else None,
                "Verdict": label,
            })
        
        # Persist all results in session state
        st.session_state['_comp_exp_x'] = exp_x
        st.session_state['_comp_exp_y'] = exp_y
        st.session_state['_comp_theory_x'] = theory_x_shifted
        st.session_state['_comp_theory_y'] = theory_y
        st.session_state['_comp_pcc_results'] = pcc_results
        st.session_state['_comp_done'] = True
    
    # ---- Display persisted results (survives widget interactions) ----
    if st.session_state.get('_comp_done', False):
        exp_x = st.session_state['_comp_exp_x']
        exp_y = st.session_state['_comp_exp_y']
        theory_x_shifted = st.session_state['_comp_theory_x']
        theory_y = st.session_state['_comp_theory_y']
        pcc_results = st.session_state['_comp_pcc_results']
        
        # --- Comparison Plots in Tabs ---
        tab_comp_plotly, tab_comp_mpl = st.tabs(["📈 Interactive Comparison", "🖼️ Static Plot (Report)"])
        
        with tab_comp_plotly:
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(
                x=exp_x, y=exp_y, mode='lines', line=dict(color='black', width=2),
                name='Experimental -ln(depletion)', yaxis='y1'
            ))
            fig_comp.add_trace(go.Scatter(
                x=theory_x_shifted, y=theory_y, mode='lines', line=dict(color='red', width=2),
                name='DFT Theory', yaxis='y2'
            ))
            fig_comp.update_layout(
                xaxis_title="Wavenumber (cm⁻¹)",
                yaxis=dict(title="-ln(depletion)", side='left', showgrid=True),
                yaxis2=dict(title="Intensity (km/mol)", side='right', overlaying='y', showgrid=False),
                title="Experimental vs DFT Comparison", hovermode='x unified', legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with tab_comp_mpl:
            fig_comp_static, ax1 = plt.subplots(figsize=(14, 6))
            ax1.plot(exp_x, exp_y, 'k-', linewidth=2, label='Experimental -ln(depletion)', alpha=0.8)
            ax1.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
            ax1.set_ylabel("-ln(depletion)", fontsize=12, color='black')
            ax1.tick_params(axis='y', labelcolor='black'); ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax2 = ax1.twinx()
            ax2.plot(theory_x_shifted, theory_y, 'r-', linewidth=2, label='DFT Theory', alpha=0.8)
            ax2.set_ylabel("Intensity (km/mol)", fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='upper left')
            ax1.set_title("Experimental vs DFT IR Spectrum", fontsize=14, fontweight='bold')
            fig_comp_static.tight_layout()
            st.pyplot(fig_comp_static)
            add_plot_to_report_button(fig_comp_static, "Experimental vs DFT Comparison",
                                      key_suffix="exp_vs_dft", description="Comparison of experimental and DFT-calculated IR spectra")
        
        # --- PCC Results ---
        st.markdown("### 📐 PCC Scoring Results")
        df_pcc = pd.DataFrame(pcc_results)
        
        def highlight_verdict(row):
            if "Excellent" in str(row["Verdict"]):
                return ['', '', '', '', "background-color: #d4edda; color: #155724; font-weight: bold"]
            elif "Good" in str(row["Verdict"]) or "Weak" in str(row["Verdict"]):
                return ['', '', '', '', "background-color: #fff3cd; color: #856404"]
            else:
                return ['', '', '', '', "background-color: #f8d7da; color: #721c24"]
        
        st.dataframe(df_pcc.style.apply(highlight_verdict, axis=1), use_container_width=True, hide_index=True)
        
        # Bar chart
        fig_pcc, ax_pcc = plt.subplots(figsize=(10, 4))
        valid = df_pcc.dropna(subset=["PCC (r)"])
        thresholds = get_pcc_thresholds()
        colors_bar = [
            "#28a745" if r >= thresholds["excellent"] else "#ffc107" if r >= thresholds["good"] else "#dc3545"
            for r in valid["PCC (r)"]
        ]
        ax_pcc.barh(valid["Region"], valid["PCC (r)"], color=colors_bar, alpha=0.8)
        ax_pcc.axvline(thresholds["good"], color='orange', linestyle='--', linewidth=1.2, label=f'Good ({thresholds["good"]:.2f})', alpha=0.7)
        ax_pcc.axvline(thresholds["excellent"], color='green', linestyle='--', linewidth=1.2, label=f'Excellent ({thresholds["excellent"]:.2f})', alpha=0.7)
        ax_pcc.axvline(0.0, color='black', linestyle='-', linewidth=0.8)
        ax_pcc.set_xlabel("Pearson r", fontsize=11)
        ax_pcc.set_title("Regional PCC Scores", fontsize=13, fontweight='bold')
        ax_pcc.set_xlim(-1, 1); ax_pcc.legend(fontsize=9, loc='lower right'); ax_pcc.grid(True, axis='x', alpha=0.3)
        fig_pcc.tight_layout()
        st.pyplot(fig_pcc)
        add_plot_to_report_button(fig_pcc, "PCC Region Scores", key_suffix="pcc_scores",
                                  description="Pearson correlation scores per diagnostic spectral region")
        
        # --- Structure Assignment Decision ---
        st.markdown("#### 🧪 Structure Assignment Decision")
        full_r = df_pcc[df_pcc["Region"] == "Full Overlap"]["PCC (r)"].values
        full_r = full_r[0] if len(full_r) > 0 and not pd.isna(full_r[0]) else None
        fp_r = df_pcc[df_pcc["Region"] == "Fingerprint"]["PCC (r)"].values
        fp_r = fp_r[0] if len(fp_r) > 0 and not pd.isna(fp_r[0]) else None
        cc_r = df_pcc[df_pcc["Region"] == "C≡C Stretch"]["PCC (r)"].values
        cc_r = cc_r[0] if len(cc_r) > 0 and not pd.isna(cc_r[0]) else None
        
        thresholds_dec = get_pcc_thresholds()
        if full_r is not None and full_r > thresholds_dec["excellent"] and fp_r is not None and fp_r > thresholds_dec["good"]:
            st.success("🟢 **Candidate Structure:** Strong overall and fingerprint agreement.")
        elif full_r is not None and full_r > thresholds_dec["good"]:
            st.warning("🟡 **Tentative Match:** Moderate agreement. Check diagnostic regions manually.")
        elif cc_r is not None and cc_r < thresholds_dec["weak"]:
            st.error("🔴 **Rule Out:** C≡C stretch shows very poor agreement.")
        elif full_r is not None and full_r < thresholds_dec["weak"]:
            st.error("🔴 **Rule Out:** Overall PCC too low for this structure.")
        else:
            st.info("ℹ️ **Inconclusive:** Mixed scores. Consider visual inspection and additional regions.")
        
        # --- Save ---
        if st.checkbox("💾 Save Comparison Data"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            exp_df = pd.DataFrame({'Wavenumber_Exp': exp_x, 'Ln_Depletion_Exp': exp_y})
            theory_df = pd.DataFrame({'Wavenumber_Theory': theory_x_shifted, 'Intensity_Theory_km_mol': theory_y})
            exp_filename = os.path.join(file_directory, f"exp_data_{timestamp}.csv")
            theory_filename = os.path.join(file_directory, f"theory_data_{timestamp}.csv")
            pcc_filename = os.path.join(file_directory, f"pcc_scores_{timestamp}.csv")
            exp_df.to_csv(exp_filename, index=False)
            theory_df.to_csv(theory_filename, index=False)
            df_pcc.to_csv(pcc_filename, index=False)
            st.success(f"✅ Saved: `{exp_filename}`, `{theory_filename}`, `{pcc_filename}`")

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
    st.markdown("## Step 4 — Multi-Structure Batch Comparison")
    st.caption("Rank all candidate structures using PCC scoring across diagnostic regions.")
    
    if st.button("🚀 Run Batch PCC Analysis", type="primary"):
        structures = st.session_state['dft_structures']
        exp_x = fullrange_depletion_data.iloc[:, 0].values
        exp_y = fullrange_depletion_data.iloc[:, 4].values
        freq_scale = st.session_state.get('freq_scale_factor', 0.967)
        bw_frac = st.session_state.get('bw_frac', 0.007)
        x_min = st.session_state.get('x_min', 500.0)
        x_max = st.session_state.get('x_max', 2200.0)
        shift = st.session_state.get('shift_theory', 0.0)
        DIAGNOSTIC_REGIONS = get_diagnostic_regions()
        
        all_results = []
        progress_bar = st.progress(0)
        for idx, struct in enumerate(structures):
            scaled_freq = struct['frequencies'] * freq_scale
            theory_x, theory_y = broaden_spectrum_felix(
                scaled_freq, struct['intensities'], x_range=(x_min, x_max), bw_frac=bw_frac, npoints=4000
            )
            theory_x_shifted = theory_x + shift
            struct_pcc = {'filename': struct['filename']}
            for region_name, region_range in DIAGNOSTIC_REGIONS.items():
                r, p, _, _, _ = compute_pcc(exp_x, exp_y, theory_x_shifted, theory_y, region=region_range)
                struct_pcc[region_name] = r if r is not None else np.nan
            all_results.append(struct_pcc)
            progress_bar.progress((idx + 1) / len(structures))
        
        df_batch = pd.DataFrame(all_results)
        
        # Average PCC excluding Full Overlap and subset regions
        all_region_names = [r for r in DIAGNOSTIC_REGIONS.keys() if r != "Full Overlap"]
        region_ranges = {r: DIAGNOSTIC_REGIONS[r] for r in all_region_names if DIAGNOSTIC_REGIONS[r] is not None}
        scoring_regions = []
        for name, rng in region_ranges.items():
            is_subset = any(
                other_name != name and other_rng[0] <= rng[0] and other_rng[1] >= rng[1]
                for other_name, other_rng in region_ranges.items()
            )
            if not is_subset:
                scoring_regions.append(name)
        if not scoring_regions:
            scoring_regions = all_region_names
        
        df_batch['Average PCC'] = df_batch[scoring_regions].mean(axis=1, skipna=True)
        df_batch['Valid Regions'] = df_batch[scoring_regions].notna().sum(axis=1)
        df_batch['Rank'] = df_batch['Average PCC'].rank(ascending=False, method='min').astype(int)
        df_batch = df_batch.sort_values('Rank')
        
        # Persist in session state
        st.session_state['_batch_df'] = df_batch
        st.session_state['_batch_regions'] = list(DIAGNOSTIC_REGIONS.keys())
        st.session_state['_batch_scoring_regions'] = scoring_regions
        st.session_state['_batch_params'] = {
            'freq_scale': freq_scale, 'bw_frac': bw_frac, 'shift': shift,
            'n_structures': len(structures)
        }
        st.session_state['_batch_exp_x'] = exp_x
        st.session_state['_batch_done'] = True
    
    # ---- Display persisted batch results ----
    if st.session_state.get('_batch_done', False):
        df_batch = st.session_state['_batch_df']
        regions_to_plot = st.session_state['_batch_regions']
        scoring_regions = st.session_state['_batch_scoring_regions']
        
        best_match = df_batch.iloc[0]
        st.success(f"🥇 **Best Match:** {best_match['filename']} (Avg PCC: {best_match['Average PCC']:.3f}, {best_match['Valid Regions']:.0f} regions)")
        
        nan_counts = df_batch[scoring_regions].isna().sum()
        if nan_counts.sum() > 0:
            st.warning(f"⚠️ Missing coverage: {', '.join([f'{r} ({nan_counts[r]})' for r in nan_counts[nan_counts > 0].index])}")
        
        st.dataframe(
            df_batch.style.background_gradient(subset=['Average PCC'], cmap='RdYlGn', vmin=-1, vmax=1),
            use_container_width=True, hide_index=True
        )
        
        # Batch visualizations in tabs
        tab_batch_bar, tab_batch_heat = st.tabs(["📈 Bar Chart", "🗺️ Heatmap"])
        
        with tab_batch_bar:
            n_structs = len(df_batch)
            fig_batch, ax_batch = plt.subplots(figsize=(12, max(5, n_structs * 0.5)))
            x_pos = np.arange(n_structs)
            num_regions = len(regions_to_plot)
            width = 0.8 / num_regions
            colors_regions = plt.cm.tab10(np.linspace(0, 1, num_regions))
            for i, region in enumerate(regions_to_plot):
                if region in df_batch.columns:
                    ax_batch.barh(x_pos + i * width, df_batch[region].values, width,
                                  label=region, color=colors_regions[i], alpha=0.8)
            ax_batch.set_yticks(x_pos + width * (num_regions - 1) / 2)
            ax_batch.set_yticklabels(df_batch['filename'].values)
            ax_batch.set_xlabel('PCC Score (r)', fontsize=12)
            ax_batch.set_title('Multi-Structure PCC Comparison', fontsize=14, fontweight='bold')
            thresholds_batch = get_pcc_thresholds()
            ax_batch.axvline(thresholds_batch["good"], color='orange', linestyle='--', linewidth=1, alpha=0.5, label=f'Good ({thresholds_batch["good"]:.2f})')
            ax_batch.axvline(thresholds_batch["excellent"], color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Excellent ({thresholds_batch["excellent"]:.2f})')
            ax_batch.legend(loc='lower right', fontsize=8, ncol=2)
            ax_batch.grid(True, axis='x', alpha=0.3); ax_batch.set_xlim(-0.2, 1.0)
            fig_batch.tight_layout()
            st.pyplot(fig_batch)
            add_plot_to_report_button(fig_batch, "Multi-Structure PCC Comparison", key_suffix="batch_pcc",
                                      description="Batch comparison of all candidate structures using PCC scoring")
        
        with tab_batch_heat:
            n_structs = len(df_batch)
            fig_heat, ax_heat = plt.subplots(figsize=(10, max(5, n_structs * 0.4)))
            heat_data = df_batch[regions_to_plot].values
            im = ax_heat.imshow(heat_data, cmap='RdYlGn', aspect='auto', vmin=-0.5, vmax=1.0)
            ax_heat.set_xticks(np.arange(len(regions_to_plot)))
            ax_heat.set_yticks(np.arange(n_structs))
            ax_heat.set_xticklabels(regions_to_plot, rotation=45, ha='right')
            ax_heat.set_yticklabels(df_batch['filename'].values)
            cbar = plt.colorbar(im, ax=ax_heat)
            cbar.set_label('PCC Score', rotation=270, labelpad=20)
            for i in range(n_structs):
                for j in range(len(regions_to_plot)):
                    value = heat_data[i, j]
                    if not np.isnan(value):
                        ax_heat.text(j, i, f'{value:.2f}', ha="center", va="center",
                                     color="black" if value < 0.5 else "white", fontsize=9, fontweight='bold')
            ax_heat.set_title('PCC Heatmap: All Structures vs Experimental', fontsize=14, fontweight='bold')
            fig_heat.tight_layout()
            st.pyplot(fig_heat)
            add_plot_to_report_button(fig_heat, "PCC Heatmap", key_suffix="pcc_heatmap",
                                      description="Heatmap showing PCC scores for all structures across regions")
        
        # Save batch results
        if st.checkbox("💾 Save Batch Results", key="save_batch"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_directory = st.session_state.get("file_directory", "./output")
            batch_filename = os.path.join(file_directory, f"batch_pcc_comparison_{timestamp}.csv")
            df_batch.to_csv(batch_filename, index=False)
            
            params = st.session_state['_batch_params']
            exp_x = st.session_state['_batch_exp_x']
            summary_text = f"""Multi-Structure PCC Comparison Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Structures: {params['n_structures']} | Scale: {params['freq_scale']:.3f} | BW: {params['bw_frac']*100:.2f}% | Shift: {params['shift']:.1f} cm⁻¹
Exp range: {exp_x.min():.1f} - {exp_x.max():.1f} cm⁻¹\n\nRankings:\n"""
            DIAGNOSTIC_REGIONS = get_diagnostic_regions()
            for _, row in df_batch.iterrows():
                summary_text += f"\n{row['Rank']:.0f}. {row['filename']} — Avg PCC: {row['Average PCC']:.3f}"
                for region_name in DIAGNOSTIC_REGIONS.keys():
                    val = row.get(region_name, np.nan)
                    summary_text += f"\n   {region_name}: {val:.3f}" if not np.isnan(val) else f"\n   {region_name}: N/A"
                summary_text += "\n"
            
            report_filename = os.path.join(file_directory, f"batch_pcc_report_{timestamp}.txt")
            with open(report_filename, 'w') as f:
                f.write(summary_text)
            st.success(f"✅ Saved: `{batch_filename}` and `{report_filename}`")

elif len(st.session_state.get('dft_structures', [])) <= 1:
    st.info("💡 Upload multiple DFT files to enable batch comparison and rank candidate structures.")
