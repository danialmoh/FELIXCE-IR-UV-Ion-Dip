'''
DFT output file parsers and spectrum processing utilities.

Supported formats:
  - Gaussian harmonic (.out / .log)
  - Gaussian anharmonic text (fundamental, overtone, combination bands)
  - ORCA harmonic output (.out)
  - ORCA NEARIR / anharmonic output (.out) — includes overtones & combination bands
  - ORCA stick spectrum (.ir.stk)
  - ORCA broadened spectrum (.ir.dat)
  - Pre-convoluted spectrum (_conv_spectrum.txt / _convoluted.txt)
  - MLMD IR spectrum (Freq_MD / Inten_MD two-column .txt)
  - Custom analysis report

All parser functions return (frequencies, intensities, metadata) where
frequencies and intensities are numpy arrays and metadata is a dict.
'''

import re
import numpy as np

__all__ = [
    'broaden_spectrum_felix',
    'parse_custom_report',
    'parse_orca_out',
    'parse_orca_out_anharmonic',
    'parse_gaussian_out',
    'parse_gaussian_anharmonic',
    'parse_orca_stick',
    'parse_orca_broadened',
    'parse_convoluted_spectrum',
    'parse_mlmd_ir',
    'parse_dft_file',
    'parse_orca_vpt2_out',
]


# ========================================================================================
# SPECTRUM BROADENING
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


# ========================================================================================
# INDIVIDUAL PARSERS
# ========================================================================================

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


def parse_orca_out_anharmonic(content):
    """
    Parse ORCA NEARIR / anharmonic .out file — extracts fundamentals from
    the 'IR SPECTRUM' block AND overtones + combination bands from the
    'OVERTONES AND COMBINATION BANDS' block.

    Returns merged stick spectrum with band_types metadata (same convention
    as parse_gaussian_anharmonic).
    """
    fundamentals = []
    overtones_combinations = []

    lines = content.split('\n')
    section = None  # 'ir' or 'oc'

    for i, line in enumerate(lines):
        stripped = line.strip()

        # --- section headers ---
        if stripped == 'IR SPECTRUM':
            section = 'ir'
            continue
        if stripped == 'OVERTONES AND COMBINATION BANDS':
            section = 'oc'
            continue

        if section is None:
            continue

        # Skip header / separator lines
        if stripped.startswith('---') or stripped.startswith('==='):
            continue
        if stripped.startswith('Mode') or 'cm**-1' in stripped:
            continue
        if stripped.startswith('*'):
            # Footnote lines → end of block
            section = None
            continue

        # Empty line → end of current block if we already collected data
        if stripped == '':
            if section == 'ir' and fundamentals:
                section = None
            elif section == 'oc' and overtones_combinations:
                section = None
            continue

        # --- data lines ---
        # IR SPECTRUM:  "  6:     98.39   0.000361    1.82  0.001145  (...)"
        if section == 'ir':
            parts = line.split()
            if len(parts) >= 4 and ':' in parts[0]:
                try:
                    freq = float(parts[1])
                    inten = float(parts[3])  # km/mol column
                    fundamentals.append((freq, inten))
                except (ValueError, IndexError):
                    continue

        # OVERTONES AND COMBINATION BANDS:
        # "  6+   6:   196.78   0.000009    0.04  0.000014  (...)"
        # "  6+   7:   238.21   0.000001    0.00  0.000001  (...)"
        elif section == 'oc':
            parts = line.split()
            # Find the token that ends with ':' → index of mode label end
            colon_idx = None
            for pi, p in enumerate(parts):
                if p.endswith(':'):
                    colon_idx = pi
                    break
            if colon_idx is None:
                continue
            nums = parts[colon_idx + 1:]  # everything after "N:"
            if len(nums) < 3:
                continue
            try:
                freq = float(nums[0])   # freq cm⁻¹
                inten = float(nums[2])  # km/mol (3rd numeric column)
                # Determine overtone vs combination from mode label
                # overtone: "N+ N:" (same mode twice), combination: "N+ M:" (different)
                mode_tokens = parts[:colon_idx + 1]
                mode_nums = [t.replace('+', '').replace(':', '') for t in mode_tokens]
                mode_nums = [m for m in mode_nums if m.isdigit()]
                if len(mode_nums) == 2 and mode_nums[0] == mode_nums[1]:
                    overtones_combinations.append((freq, inten, 'overtone'))
                else:
                    overtones_combinations.append((freq, inten, 'combination'))
            except (ValueError, IndexError):
                continue

    # Merge
    all_freqs = [f for f, i in fundamentals]
    all_intens = [i for f, i in fundamentals]
    band_types = ['fundamental'] * len(fundamentals)

    ot = [(f, i) for f, i, t in overtones_combinations if t == 'overtone']
    cb = [(f, i) for f, i, t in overtones_combinations if t == 'combination']

    all_freqs += [f for f, i in ot]
    all_intens += [i for f, i in ot]
    band_types += ['overtone'] * len(ot)

    all_freqs += [f for f, i in cb]
    all_intens += [i for f, i in cb]
    band_types += ['combination'] * len(cb)

    metadata = {
        'type': 'anharmonic',
        'software': 'ORCA',
        'n_fundamentals': len(fundamentals),
        'n_overtones': len(ot),
        'n_combinations': len(cb),
        'band_types': band_types,
    }

    return np.array(all_freqs), np.array(all_intens), metadata


def parse_orca_vpt2_out(content):
    """
    Parse ORCA VPT2+K .out file.

    VPT2 output does not use the standard 'IR SPECTRUM' block. Instead it has
    repeated 'IR Intensities' tables and a final 'Fundamental transitions [1/cm]'
    table. This parser extracts:

      * anharmonic fundamental frequencies from 'Fundamental transitions [1/cm]'
      * corresponding IR intensities from the last 'IR Intensities' table
      * overtones and combination bands from the final
        'Overtones and combination bands' table

    Returns merged stick spectrum with band_types metadata.
    """
    lines = content.splitlines()

    def _read_table(start):
        """
        Yield the data rows of an ORCA table that begins just after ``start``.

        ORCA brackets its column names with dashed separators, e.g.::

            -------------------------------
            Mode freq     Int          T2
                 cm-1    km/mol        a.u.
            -------------------------------
            0   -1.58   -nan       -nan

        so we cannot simply stop at the first dashed line. Instead we skip any
        separator/units/column-name line and treat every line whose first token
        is an integer as data. The table ends at the first blank line, or at a
        dashed line once data has been seen.
        """
        rows = []
        j = start
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped:
                if rows:
                    break
                j += 1
                continue
            if stripped.startswith('---') or stripped.startswith('==='):
                if rows:
                    break
                j += 1
                continue
            if stripped.startswith('*'):
                break
            parts = stripped.split()
            try:
                int(parts[0])
            except (ValueError, IndexError):
                # Column-name or units line (e.g. 'Mode freq Int', 'cm-1 km/mol')
                if rows:
                    break
                j += 1
                continue
            rows.append(parts)
            j += 1
        return rows, j

    # ------------------------------------------------------------------
    # 1. Collect intensities from every 'IR Intensities' block.
    #    Later blocks overwrite earlier ones, so we end up with the last one.
    # ------------------------------------------------------------------
    intensities_by_mode = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == 'IR Intensities':
            rows, i = _read_table(i + 1)
            for parts in rows:
                if len(parts) >= 3:
                    try:
                        mode = int(parts[0])
                        freq = float(parts[1])
                        inten = float(parts[2])
                        intensities_by_mode[mode] = (freq, inten)
                    except (ValueError, IndexError):
                        pass
        else:
            i += 1

    # ------------------------------------------------------------------
    # 2. Read anharmonic fundamental frequencies.
    # ------------------------------------------------------------------
    fund_freq_by_mode = {}
    for i, line in enumerate(lines):
        if 'Fundamental transitions [1/cm]' in line:
            rows, _ = _read_table(i + 1)
            for parts in rows:
                if len(parts) >= 3:
                    try:
                        mode = int(parts[0])
                        v_fund = float(parts[2])  # v(fund) column
                        fund_freq_by_mode[mode] = v_fund
                    except (ValueError, IndexError):
                        pass
            break

    # ------------------------------------------------------------------
    # 3. Read overtones and combination bands.
    # ------------------------------------------------------------------
    overtones_combinations = []
    for i, line in enumerate(lines):
        if 'Overtones and combination bands' in line:
            rows, _ = _read_table(i + 1)
            for parts in rows:
                if len(parts) >= 5:
                    try:
                        m1 = int(parts[0])
                        m2 = int(parts[1])
                        freq = float(parts[2])
                        inten = float(parts[4])  # km/mol column
                        btype = 'overtone' if m1 == m2 else 'combination'
                        overtones_combinations.append((freq, inten, btype))
                    except (ValueError, IndexError):
                        pass
            break

    # ------------------------------------------------------------------
    # 4. Merge fundamentals. The IR tables index all 3N modes (the first
    #    6 are translations/rotations), while the fundamental table indexes
    #    only vibrations, so the usual offset is +6.
    # ------------------------------------------------------------------
    fundamentals = []
    if fund_freq_by_mode:
        min_ir = min(intensities_by_mode.keys()) if intensities_by_mode else 0
        offset = 6 if min_ir == 0 and 6 in intensities_by_mode else 0
        for mode, freq in sorted(fund_freq_by_mode.items()):
            ir_idx = mode + offset
            if ir_idx in intensities_by_mode:
                _, inten = intensities_by_mode[ir_idx]
            elif mode in intensities_by_mode:
                _, inten = intensities_by_mode[mode]
            else:
                inten = 0.0
            if freq > 0 and np.isfinite(inten) and inten >= 0:
                fundamentals.append((freq, inten))
    elif intensities_by_mode:
        # Fallback: use the last IR Intensities block directly.
        for _, (freq, inten) in sorted(intensities_by_mode.items()):
            if freq > 0 and np.isfinite(inten) and inten >= 0:
                fundamentals.append((freq, inten))

    ot = [(f, i) for f, i, t in overtones_combinations if t == 'overtone']
    cb = [(f, i) for f, i, t in overtones_combinations if t == 'combination']

    all_freqs = (
        [f for f, i in fundamentals]
        + [f for f, i in ot]
        + [f for f, i in cb]
    )
    all_intens = (
        [i for f, i in fundamentals]
        + [i for f, i in ot]
        + [i for f, i in cb]
    )
    band_types = (
        ['fundamental'] * len(fundamentals)
        + ['overtone'] * len(ot)
        + ['combination'] * len(cb)
    )

    metadata = {
        'type': 'vpt2',
        'software': 'ORCA',
        'n_fundamentals': len(fundamentals),
        'n_overtones': len(ot),
        'n_combinations': len(cb),
        'band_types': band_types,
    }

    return np.array(all_freqs), np.array(all_intens), metadata



def parse_mlmd_ir(content):
    """
    Parse MLMD (Machine Learning Molecular Dynamics) IR spectrum text file.

    Expected format::

        # Freq_MD (cm^-1) Inten_MD (Normalized intensity)
        300.0 4.0775e-04
        301.0 0.0000e+00
        ...

    Lines starting with '#' are skipped as comments.
    Returns (frequencies, intensities, metadata).
    The spectrum is already broadened/convoluted on a 1 cm⁻¹ grid.
    """
    frequencies = []
    intensities = []

    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                freq = float(parts[0])
                inten = float(parts[1])
                frequencies.append(freq)
                intensities.append(inten)
            except ValueError:
                continue

    return (
        np.array(frequencies),
        np.array(intensities),
        {'type': 'mlmd_ir', 'broadened': True},
    )


def parse_convoluted_spectrum(content):
    """
    Parse a pre-convoluted / pre-broadened spectrum text file.

    Expected format (ORCA orca_ir or Gaussian utility output)::

        Frequencies    Intensities
        (cm-1)          (km/mol)

             0              0
             1              0
             ...
           600     0.00963703

    Columns: integer-spaced wavenumber grid, broadened intensity.
    Returns (frequencies, intensities, metadata).
    """
    frequencies = []
    intensities = []

    for line in content.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        # Skip header lines
        if stripped.startswith('Freq') or stripped.startswith('(cm'):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            try:
                freq = float(parts[0])
                inten = float(parts[1])
                frequencies.append(freq)
                intensities.append(inten)
            except ValueError:
                continue

    return (
        np.array(frequencies),
        np.array(intensities),
        {'type': 'convoluted_spectrum'},
    )


# ========================================================================================
# AUTO-DETECT DISPATCHER
# ========================================================================================

def parse_dft_file(content, filename):
    """
    Auto-detect and parse DFT output file.

    Detection priority:
      1. Filename-based (extension / naming convention)
      2. Content-based (header signatures)

    The returned metadata always contains 'detected_as' describing the
    format that was selected.

    Parameters:
    -----------
    content : str
        File content as a string
    filename : str
        Original filename (used for format detection)

    Returns:
    --------
    frequencies : np.ndarray
    intensities : np.ndarray
    metadata : dict
    """
    fname_lower = filename.lower()

    def _tag(freqs, intens, meta, label):
        meta['detected_as'] = label
        return freqs, intens, meta

    # ---- 1. Filename-based checks ----

    # ORCA auxiliary files
    if fname_lower.endswith('.ir.stk'):
        return _tag(*parse_orca_stick(content), 'ORCA stick (.ir.stk)')
    if fname_lower.endswith('.ir.dat'):
        return _tag(*parse_orca_broadened(content), 'ORCA broadened (.ir.dat)')

    # Pre-convoluted / pre-broadened spectrum (by filename convention)
    if ('_conv_spectrum' in fname_lower or '_convoluted' in fname_lower) and fname_lower.endswith('.txt'):
        return _tag(*parse_convoluted_spectrum(content), 'Pre-convoluted spectrum (.txt)')

    # MLMD IR spectrum (by filename convention: contains '_qm' suffix before .txt)
    if fname_lower.endswith('.txt') and '_qm' in fname_lower:
        return _tag(*parse_mlmd_ir(content), 'MLMD IR spectrum (.txt)')

    # Gaussian anharmonic extracted text (by filename convention)
    if 'anhar' in fname_lower and fname_lower.endswith('.txt'):
        return _tag(*parse_gaussian_anharmonic(content), 'Gaussian anharmonic text (.txt)')

    # ---- 2. Content-based checks ----

    header = content[:3000]

    # Custom analysis report
    if 'ANALYSIS REPORT' in header:
        return _tag(*parse_custom_report(content), 'Custom analysis report')

    # ORCA output — decide harmonic vs NEARIR/anharmonic vs VPT2
    if 'O   R   C   A' in header:
        if 'Fundamental transitions [1/cm]' in content:
            return _tag(*parse_orca_vpt2_out(content), 'ORCA VPT2 (.out)')
        if 'OVERTONES AND COMBINATION BANDS' in content:
            return _tag(*parse_orca_out_anharmonic(content), 'ORCA NEARIR/anharmonic (.out)')
        return _tag(*parse_orca_out(content), 'ORCA harmonic (.out)')

    # Gaussian anharmonic (text block, may arrive without .txt extension)
    if 'Fundamental Bands' in content and ('Overtones' in content or 'Combination Bands' in content):
        return _tag(*parse_gaussian_anharmonic(content), 'Gaussian anharmonic (content)')

    # MLMD IR spectrum (content-based: comment header with 'Freq_MD')
    if 'Freq_MD' in content[:500]:
        return _tag(*parse_mlmd_ir(content), 'MLMD IR spectrum (header)')

    # Pre-convoluted spectrum (header-based fallback)
    if content.lstrip().startswith('Frequencies') or content.lstrip().startswith(' Frequencies'):
        return _tag(*parse_convoluted_spectrum(content), 'Pre-convoluted spectrum (header)')

    # Gaussian harmonic (.log / .out)
    if 'Gaussian' in header or 'Frequencies --' in content:
        return _tag(*parse_gaussian_out(content), 'Gaussian harmonic (.log/.out)')

    # Fallback
    return _tag(*parse_custom_report(content), 'Unknown (fallback)')
