'''
Pearson Correlation Coefficient (PCC) scoring for DFT vs experimental IR spectra.

Provides region-based PCC analysis for structure assignment:
  - Interpolation of spectra onto a common grid
  - Min-max normalization (intensity-independent comparison)
  - Per-region and batch PCC computation
  - Score labelling with adjustable thresholds
  - Automatic optimal scaling factor search
  - Dual scaling factor support (high / low frequency domains)

No Streamlit dependency — all functions accept explicit parameters.

Reference
---------
Von der Esch, B.; Peters, L. D. M.; Sauerland, L.; Ochsenfeld, C.
"Quantitative Comparison of Experimental and Computed IR-Spectra
Extracted from Ab Initio Molecular Dynamics."
J. Chem. Theory Comput. 2021, 17, 985–995.
https://doi.org/10.1021/acs.jctc.0c01279
'''

import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

__all__ = [
    'DEFAULT_DIAGNOSTIC_REGIONS',
    'DEFAULT_PCC_THRESHOLDS',
    'normalize_spectrum',
    'interpolate_to_common_grid',
    'compute_pcc',
    'score_label',
    'compute_batch_pcc',
    'rank_batch_results',
    'apply_dual_scaling',
    'find_optimal_scaling_factor',
]


# ========================================================================================
# DEFAULT CONSTANTS
# ========================================================================================

# Default diagnostic regions for C11H8 isomer analysis
DEFAULT_DIAGNOSTIC_REGIONS = {
    "Full Overlap":        None,           # entire shared range
    "Fingerprint":         (600,  1500),   # ring deformations, CH bends
    "Mid-IR":              (1500, 2000),   # skeletal stretches
    "C≡C Stretch":         (2050, 2200),   # ethynyl diagnostic
    "Aromatic C-H OOP":    (700,  900),    # out-of-plane CH, isomer-sensitive
}

# Default PCC thresholds (adjusted for IR-UV action spectra)
DEFAULT_PCC_THRESHOLDS = {
    "excellent": 0.60,
    "good": 0.40,
    "weak": 0.20,
}


# ========================================================================================
# CORE FUNCTIONS
# ========================================================================================

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
    
    interp_1 = interp1d(x1, y1, kind='linear', bounds_error=False, fill_value=0.0)
    interp_2 = interp1d(x2, y2, kind='linear', bounds_error=False, fill_value=0.0)
    
    return grid, interp_1(grid), interp_2(grid)


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


def score_label(r, thresholds=None):
    """
    Human-readable label based on adjustable thresholds for IR-UV action spectra.
    
    Parameters:
    -----------
    r : float or None
        PCC score
    thresholds : dict, optional
        Dict with keys 'excellent', 'good', 'weak'. Uses DEFAULT_PCC_THRESHOLDS if None.
    
    Returns:
    --------
    label : str
    color : str
    """
    if r is None:
        return "N/A", "gray"
    if thresholds is None:
        thresholds = DEFAULT_PCC_THRESHOLDS
    if r >= thresholds["excellent"]:
        return "Excellent ✅", "green"
    elif r >= thresholds["good"]:
        return "Good 🟡", "orange"
    elif r >= thresholds["weak"]:
        return "Weak ⚠️", "orange"
    else:
        return "Poor / Rule Out ❌", "red"


# ========================================================================================
# BATCH COMPARISON
# ========================================================================================

def compute_batch_pcc(structures, exp_x, exp_y, diagnostic_regions,
                      freq_scale=0.967, bw_frac=0.007, x_range=(500.0, 2200.0),
                      shift=0.0, broaden_func=None):
    """
    Run PCC analysis for multiple DFT structures against experimental data.
    
    Parameters:
    -----------
    structures : list of dict
        Each dict must have 'filename', 'frequencies', 'intensities' keys.
    exp_x, exp_y : arrays
        Experimental spectrum.
    diagnostic_regions : dict
        Region name → (min, max) tuple or None for full overlap.
    freq_scale : float
        Frequency scaling factor.
    bw_frac : float
        FELIX fractional bandwidth.
    x_range : tuple
        (min, max) wavenumber range for broadening.
    shift : float
        Rigid shift applied to theoretical spectrum (cm⁻¹).
    broaden_func : callable, optional
        Broadening function with signature (freqs, intens, x_range, bw_frac, npoints).
        If None, imports broaden_spectrum_felix from DFT_Parsers.
    
    Returns:
    --------
    all_results : list of dict
        Per-structure PCC scores for each region.
    """
    if broaden_func is None:
        from .DFT_Parsers import broaden_spectrum_felix
        broaden_func = broaden_spectrum_felix

    all_results = []
    for struct in structures:
        scaled_freq = struct['frequencies'] * freq_scale
        theory_x, theory_y = broaden_func(
            scaled_freq, struct['intensities'],
            x_range=x_range, bw_frac=bw_frac, npoints=4000
        )
        theory_x_shifted = theory_x + shift
        
        struct_pcc = {'filename': struct['filename']}
        for region_name, region_range in diagnostic_regions.items():
            r, p, _, _, _ = compute_pcc(exp_x, exp_y, theory_x_shifted, theory_y, region=region_range)
            struct_pcc[region_name] = r if r is not None else np.nan
        all_results.append(struct_pcc)
    
    return all_results


def rank_batch_results(all_results, diagnostic_regions):
    """
    Given batch PCC results, compute average PCC, valid region counts, and rank.
    
    Parameters:
    -----------
    all_results : list of dict
        Output from compute_batch_pcc.
    diagnostic_regions : dict
        Region definitions used in the batch analysis.
    
    Returns:
    --------
    df_batch : pd.DataFrame
        Ranked results with 'Average PCC', 'Valid Regions', 'Rank' columns.
    scoring_regions : list of str
        Region names used for averaging (excludes Full Overlap and subset regions).
    """
    import pandas as pd

    df_batch = pd.DataFrame(all_results)

    # Determine scoring regions: exclude Full Overlap and regions that are
    # subsets of other regions
    all_region_names = [r for r in diagnostic_regions.keys() if r != "Full Overlap"]
    region_ranges = {
        r: diagnostic_regions[r]
        for r in all_region_names
        if diagnostic_regions[r] is not None
    }
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

    return df_batch, scoring_regions


# ========================================================================================
# SCALING FACTOR OPTIMISATION
# (Von der Esch et al., J. Chem. Theory Comput. 2021, 17, 985–995)
# ========================================================================================

def apply_dual_scaling(frequencies, split_at=2200.0, factor_low=1.0, factor_high=1.0):
    """
    Apply separate scaling factors to the low- and high-frequency domains.

    The frequency regions are split at *split_at* cm⁻¹ (default 2200, as
    recommended by Von der Esch et al., JCTC 2021, 17, 985–995, Table 3).
    Fewer peaks are observed around 2200 cm⁻¹ so the split avoids
    discontinuities at occupied spectral regions.

    Parameters
    ----------
    frequencies : array-like
        Unscaled DFT frequencies in cm⁻¹.
    split_at : float
        Boundary wavenumber separating low and high domains.
    factor_low : float
        Scaling factor applied to frequencies < split_at.
    factor_high : float
        Scaling factor applied to frequencies >= split_at.

    Returns
    -------
    scaled : np.ndarray
        Scaled frequencies (same length, order preserved).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    scaled = np.empty_like(frequencies)
    low_mask = frequencies < split_at
    scaled[low_mask] = frequencies[low_mask] * factor_low
    scaled[~low_mask] = frequencies[~low_mask] * factor_high
    return scaled


def find_optimal_scaling_factor(
    exp_x, exp_y, frequencies, intensities,
    factor_range=(0.82, 1.05), n_steps=230,
    broaden_func=None, bw_frac=0.007, x_range=(500.0, 2200.0),
    regions=None, shift=0.0,
    dual=False, split_at=2200.0,
):
    """
    Sweep scaling factor(s) and return the value that maximises the mean PCC
    across the requested diagnostic regions.

    Implements the scaling factor search procedure described in
    Von der Esch et al., J. Chem. Theory Comput. 2021, 17, 985–995,
    Section 2.2.1 & Figure 3.

    Parameters
    ----------
    exp_x, exp_y : arrays
        Experimental spectrum (wavenumber, intensity).
    frequencies, intensities : arrays
        Raw (unscaled) DFT stick spectrum.
    factor_range : tuple (min, max)
        Range of scaling factors to test (default 0.82–1.05).
    n_steps : int
        Number of factors to evaluate.
    broaden_func : callable, optional
        Broadening function (freqs, intens, x_range, bw_frac, npoints) → (x, y).
        Defaults to broaden_spectrum_felix from DFT_Parsers.
    bw_frac : float
        Fractional bandwidth for broadening.
    x_range : tuple
        (min, max) wavenumber for broadened spectrum.
    regions : dict or None
        Diagnostic regions to average over. If None, uses Full Overlap only.
    shift : float
        Rigid wavenumber shift applied after scaling.
    dual : bool
        If True, perform a 2-D grid search for independent low/high factors
        (split at *split_at* cm⁻¹). n_steps applies per axis (total = n_steps²),
        so keep n_steps modest (≤40) for dual mode.
    split_at : float
        Boundary for dual scaling (default 2200 cm⁻¹).

    Returns
    -------
    result : dict
        'best_factor' : float or tuple  — optimal factor (single or (low, high))
        'best_mean_pcc' : float         — mean PCC at optimum
        'factors' : array               — tested factor values
        'mean_pcc' : array              — mean PCC at each factor (1-D sweep only)
        'per_region_pcc' : dict         — {region_name: array} at each factor (1-D)
    """
    if broaden_func is None:
        from .DFT_Parsers import broaden_spectrum_felix
        broaden_func = broaden_spectrum_felix

    if regions is None:
        regions = {"Full Overlap": None}

    factors = np.linspace(factor_range[0], factor_range[1], n_steps)

    # ------------------------------------------------------------------
    # Single scaling factor sweep (1-D)
    # ------------------------------------------------------------------
    if not dual:
        per_region = {name: np.full(n_steps, np.nan) for name in regions}
        mean_pcc = np.full(n_steps, np.nan)

        for i, f in enumerate(factors):
            scaled = frequencies * f
            tx, ty = broaden_func(scaled, intensities, x_range=x_range,
                                  bw_frac=bw_frac, npoints=4000)
            tx_shifted = tx + shift

            scores = []
            for rname, rrange in regions.items():
                r, _, _, _, _ = compute_pcc(exp_x, exp_y, tx_shifted, ty,
                                            region=rrange)
                val = r if r is not None else np.nan
                per_region[rname][i] = val
                if not np.isnan(val):
                    scores.append(val)
            mean_pcc[i] = np.nanmean(scores) if scores else np.nan

        best_idx = int(np.nanargmax(mean_pcc))
        return {
            'best_factor': float(factors[best_idx]),
            'best_mean_pcc': float(mean_pcc[best_idx]),
            'factors': factors,
            'mean_pcc': mean_pcc,
            'per_region_pcc': per_region,
        }

    # ------------------------------------------------------------------
    # Dual scaling factor grid search (2-D)
    # ------------------------------------------------------------------
    grid_pcc = np.full((n_steps, n_steps), np.nan)

    for i, f_low in enumerate(factors):
        for j, f_high in enumerate(factors):
            scaled = apply_dual_scaling(frequencies, split_at=split_at,
                                        factor_low=f_low, factor_high=f_high)
            tx, ty = broaden_func(scaled, intensities, x_range=x_range,
                                  bw_frac=bw_frac, npoints=4000)
            tx_shifted = tx + shift

            scores = []
            for rname, rrange in regions.items():
                r, _, _, _, _ = compute_pcc(exp_x, exp_y, tx_shifted, ty,
                                            region=rrange)
                if r is not None:
                    scores.append(r)
            grid_pcc[i, j] = np.nanmean(scores) if scores else np.nan

    best_idx = np.unravel_index(np.nanargmax(grid_pcc), grid_pcc.shape)
    return {
        'best_factor': (float(factors[best_idx[0]]), float(factors[best_idx[1]])),
        'best_mean_pcc': float(grid_pcc[best_idx]),
        'factors': factors,
        'grid_pcc': grid_pcc,
        'split_at': split_at,
    }
