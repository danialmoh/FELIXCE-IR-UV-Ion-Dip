'''
Spectral similarity scoring for DFT vs experimental IR spectra.

Provides region-based analysis for structure assignment:
  - PCC  — Pearson Correlation Coefficient (Von der Esch et al.)
  - SEC  — Squared Euclidean Cosine (intensity-pattern similarity)
  - SFEC — Squared First-Difference Euclidean Cosine (Samuel et al.)
           Robust to baseline distortion & noise in action spectra.
  - Preprocessing: smoothing, baseline clipping, derivative transform
  - Interpolation of spectra onto a common grid
  - Min-max normalization (intensity-independent comparison)
  - Per-region and batch computation (any metric)
  - Score labelling with adjustable thresholds
  - Automatic optimal scaling factor search
  - Dual scaling factor support (high / low frequency domains)

No Streamlit dependency — all functions accept explicit parameters.

References
----------
Von der Esch, B.; Peters, L. D. M.; Sauerland, L.; Ochsenfeld, C.
  J. Chem. Theory Comput. 2021, 17, 985–995.
  https://doi.org/10.1021/acs.jctc.0c01279

Samuel, A. Z. et al.
  ACS Omega 2021, 6, 2060–2065.
  https://doi.org/10.1021/acsomega.0c05041
'''

import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine as _cosine_distance
from scipy.signal import savgol_filter

__all__ = [
    'DEFAULT_DIAGNOSTIC_REGIONS',
    'DEFAULT_PCC_THRESHOLDS',
    'AVAILABLE_METRICS',
    'normalize_spectrum',
    'interpolate_to_common_grid',
    'preprocess_spectrum',
    'compute_pcc',
    'compute_sec',
    'compute_sfec',
    'compute_similarity',
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


def compute_sec(exp_x, exp_y, theory_x, theory_y, region=None):
    """
    Squared Euclidean Cosine (SEC) similarity.

    SEC = cos²(θ) where θ is the angle between the two spectra treated
    as vectors.  Uses scipy.spatial.distance.cosine internally.

    Returns 0–1 (1 = identical shape).  Unlike PCC, SEC is always ≥ 0
    and is insensitive to additive offsets only when spectra are
    non-negative.

    Returns
    -------
    score, None, grid, exp_norm, theory_norm
        (p-value slot is None — not applicable for cosine metric)
    """
    grid, exp_interp, theory_interp = interpolate_to_common_grid(
        exp_x, exp_y, theory_x, theory_y
    )
    if grid is None:
        return None, None, None, None, None

    if region is not None:
        mask = (grid >= region[0]) & (grid <= region[1])
        if mask.sum() < 10:
            return None, None, None, None, None
        grid = grid[mask]
        exp_interp = exp_interp[mask]
        theory_interp = theory_interp[mask]

    exp_norm = normalize_spectrum(exp_interp)
    theory_norm = normalize_spectrum(theory_interp)

    # scipy cosine() returns distance; similarity = 1 - distance
    # Guard against zero-vectors
    if np.allclose(exp_norm, 0) or np.allclose(theory_norm, 0):
        return 0.0, None, grid, exp_norm, theory_norm

    cos_sim = 1.0 - _cosine_distance(exp_norm, theory_norm)
    sec = cos_sim ** 2
    return float(sec), None, grid, exp_norm, theory_norm


def compute_sfec(exp_x, exp_y, theory_x, theory_y, region=None, sg_window=51):
    """
    Squared First-Difference Euclidean Cosine (SFEC) similarity.

    Compares the **first derivative** of both spectra via the cosine angle.
    This naturally removes:
      - Constant baseline offsets
      - Linear baseline slopes
      - Slow baseline curvature
    and emphasises peak *shapes* over flat noisy regions.

    The derivative is computed with a Savitzky-Golay filter (``savgol_filter``
    with ``deriv=1``), which simultaneously **smooths** the spectrum and
    differentiates — making SFEC robust to both noise and baseline artefacts
    in a single step.

    Particularly effective for noisy action spectra (IR-UV ion-dip, IRMPD).

    Parameters
    ----------
    sg_window : int
        Savitzky-Golay window length for the derivative (odd, ≥ 5).
        Larger = more smoothing of noise.  Default 51.

    Reference: Samuel et al., ACS Omega 2021, 6, 2060–2065.

    Returns
    -------
    score, None, grid, exp_norm, theory_norm
    """
    grid, exp_interp, theory_interp = interpolate_to_common_grid(
        exp_x, exp_y, theory_x, theory_y
    )
    if grid is None:
        return None, None, None, None, None

    if region is not None:
        mask = (grid >= region[0]) & (grid <= region[1])
        if mask.sum() < 10:
            return None, None, None, None, None
        grid = grid[mask]
        exp_interp = exp_interp[mask]
        theory_interp = theory_interp[mask]

    # Savitzky-Golay first derivative (smooths + differentiates)
    if sg_window % 2 == 0:
        sg_window += 1
    sg_window = max(sg_window, 5)
    sg_window = min(sg_window, len(exp_interp) - 1)
    if sg_window % 2 == 0:
        sg_window -= 1

    d_exp = savgol_filter(exp_interp, window_length=sg_window, polyorder=2, deriv=1)
    d_theory = savgol_filter(theory_interp, window_length=sg_window, polyorder=2, deriv=1)

    if np.allclose(d_exp, 0) or np.allclose(d_theory, 0):
        return 0.0, None, grid, normalize_spectrum(exp_interp), normalize_spectrum(theory_interp)

    cos_sim = 1.0 - _cosine_distance(d_exp, d_theory)
    sfec = cos_sim ** 2

    return float(sfec), None, grid, normalize_spectrum(exp_interp), normalize_spectrum(theory_interp)


def preprocess_spectrum(x, y, smooth_window=0, clip_negative=False):
    """
    Optional preprocessing before similarity scoring.

    Parameters
    ----------
    x, y : arrays
        Wavenumber and intensity.
    smooth_window : int
        Savitzky-Golay window length (odd). 0 = no smoothing.
    clip_negative : bool
        If True, set negative intensities to zero (removes dip artefacts).

    Returns
    -------
    x, y_processed : arrays (same length)
    """
    y_out = np.array(y, dtype=float).copy()
    if smooth_window > 0:
        if smooth_window % 2 == 0:
            smooth_window += 1
        polyorder = min(2, smooth_window - 1)
        y_out = savgol_filter(y_out, window_length=smooth_window, polyorder=polyorder)
    if clip_negative:
        y_out = np.clip(y_out, 0.0, None)
    return np.asarray(x), y_out


# Metric registry — maps short name → function
AVAILABLE_METRICS = {
    'PCC':  compute_pcc,
    'SEC':  compute_sec,
    'SFEC': compute_sfec,
}


def compute_similarity(exp_x, exp_y, theory_x, theory_y,
                       region=None, metric='SFEC'):
    """
    Unified interface: compute spectral similarity using the chosen metric.

    Parameters
    ----------
    metric : str
        One of 'PCC', 'SEC', 'SFEC'.

    Returns
    -------
    score, p_value, grid, exp_norm, theory_norm
    """
    func = AVAILABLE_METRICS.get(metric.upper())
    if func is None:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(AVAILABLE_METRICS)}")
    return func(exp_x, exp_y, theory_x, theory_y, region=region)


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
                      shift=0.0, broaden_func=None, metric='PCC'):
    """
    Run similarity analysis for multiple DFT structures against experimental data.
    
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
    metric : str
        Similarity metric — 'PCC', 'SEC', or 'SFEC'.
    
    Returns:
    --------
    all_results : list of dict
        Per-structure similarity scores for each region.
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
        
        struct_scores = {'filename': struct['filename']}
        for region_name, region_range in diagnostic_regions.items():
            score, _, _, _, _ = compute_similarity(
                exp_x, exp_y, theory_x_shifted, theory_y,
                region=region_range, metric=metric,
            )
            struct_scores[region_name] = score if score is not None else np.nan
        all_results.append(struct_scores)
    
    return all_results


def rank_batch_results(all_results, diagnostic_regions, metric='PCC'):
    """
    Given batch similarity results, compute average score, valid region counts, and rank.
    
    Parameters:
    -----------
    all_results : list of dict
        Output from compute_batch_pcc.
    diagnostic_regions : dict
        Region definitions used in the batch analysis.
    metric : str
        Name of the metric used (for column labelling).
    
    Returns:
    --------
    df_batch : pd.DataFrame
        Ranked results with 'Average <metric>', 'Valid Regions', 'Rank' columns.
    scoring_regions : list of str
        Region names used for averaging (excludes Full Overlap and subset regions).
    avg_col : str
        Name of the average score column.
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

    avg_col = f'Average {metric}'
    df_batch[avg_col] = df_batch[scoring_regions].mean(axis=1, skipna=True)
    df_batch['Valid Regions'] = df_batch[scoring_regions].notna().sum(axis=1)
    df_batch['Rank'] = df_batch[avg_col].rank(ascending=False, method='min').astype(int)
    df_batch = df_batch.sort_values('Rank')

    return df_batch, scoring_regions, avg_col


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
