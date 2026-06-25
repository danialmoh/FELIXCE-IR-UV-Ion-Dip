"""
SpectralDecomposition.py — Core functions for NNLS mixture decomposition
=========================================================================
Extracted from spectral_analysis_v3.py for use in the Streamlit pipeline.

Pipeline steps:
  0. Diagnostics (collinearity, condition number, Gram matrix)
  1. Ranking (Pearson derivative, cosine similarity)
  2. Full NNLS with polynomial baseline
  3. Forward stepwise subset selection
  4. Model selection (BIC + blocked CV)
  5. Exhaustive search (small k)
  6. Final fit + block bootstrap CIs
  7. Peak-resolved residual analysis
  8. Sensitivity to DFT scaling factor
"""

import numpy as np
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import linkage, fcluster
from itertools import combinations


# ════════════════════════════════════════════════════════════════════════
# DESIGN MATRIX & FITTING
# ════════════════════════════════════════════════════════════════════════

def build_design_matrix(dft_sub, exp_wn, poly_order=1):
    """Build design matrix: [DFT columns | polynomial baseline columns].

    Parameters
    ----------
    dft_sub : ndarray, shape (n_components, n_wn)
    exp_wn  : ndarray, shape (n_wn,)
    poly_order : int

    Returns
    -------
    A : ndarray, shape (n_wn, n_components + poly_order + 1)
    n_dft : int — number of DFT columns
    n_base : int — number of baseline columns
    """
    wn_c = 2 * (exp_wn - exp_wn.min()) / (exp_wn.max() - exp_wn.min()) - 1
    base_cols = [wn_c ** p for p in range(poly_order + 1)]
    B = np.column_stack(base_cols)
    A = np.hstack([dft_sub.T, B])
    return A, dft_sub.shape[0], B.shape[1]


def fit_nnls(dft_sub, exp_wn, exp_norm, poly_order=1):
    """NNLS fit: DFT coefficients >= 0, baseline unconstrained.

    Returns
    -------
    dft_coeffs : ndarray — DFT component coefficients
    base_coeffs : ndarray — baseline polynomial coefficients
    rss : float — residual sum of squares
    residuals : ndarray — fit residuals
    """
    A, nd, nb = build_design_matrix(dft_sub, exp_wn, poly_order)
    lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
    ub = np.full(nd + nb, np.inf)
    res = lsq_linear(A, exp_norm, bounds=(lb, ub), method="bvls")
    dc = res.x[:nd]
    bc = res.x[nd:]
    resid = exp_norm - A @ res.x
    rss = np.sum(resid ** 2)
    return dc, bc, rss, resid


# ════════════════════════════════════════════════════════════════════════
# STEP 0: DIAGNOSTICS
# ════════════════════════════════════════════════════════════════════════

def compute_diagnostics(dft_matrix):
    """Collinearity diagnostics on the DFT library.

    Returns
    -------
    dict with keys: cond_number, gram_matrix, max_cosine, max_pair,
                    n_pairs_above_90, n_pairs_above_95,
                    cluster_labels, n_clusters
    """
    cond_num = np.linalg.cond(dft_matrix.T)
    norms = np.linalg.norm(dft_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dft_normed = dft_matrix / norms
    gram = dft_normed @ dft_normed.T
    gram_full = gram.copy()
    np.fill_diagonal(gram, 0)
    max_cos = gram.max()
    i_mc, j_mc = np.unravel_index(gram.argmax(), gram.shape)

    n_95 = int((gram > 0.95).sum() // 2)
    n_90 = int((gram > 0.90).sum() // 2)

    # Hierarchical clustering
    dist = 1 - np.abs(gram_full)
    np.fill_diagonal(dist, 0)
    n_mol = dft_matrix.shape[0]
    condensed = dist[np.triu_indices(n_mol, k=1)]
    Z = linkage(condensed, method="average")
    cluster_labels = fcluster(Z, t=0.10, criterion="distance")
    n_clusters = len(set(cluster_labels))

    return {
        "cond_number": cond_num,
        "gram_matrix": gram_full,
        "max_cosine": max_cos,
        "max_pair": (i_mc, j_mc),
        "n_pairs_above_90": n_90,
        "n_pairs_above_95": n_95,
        "cluster_labels": cluster_labels.tolist(),
        "n_clusters": n_clusters,
        "linkage": Z,
    }


# ════════════════════════════════════════════════════════════════════════
# STEP 1: RANKING
# ════════════════════════════════════════════════════════════════════════

def pearson_derivative_scores(exp_norm, dft_matrix, exp_wn, sigma=2.0):
    """Pearson correlation on Gaussian-smoothed first derivatives."""
    es = gaussian_filter1d(exp_norm, sigma)
    ed = np.gradient(es, exp_wn)
    ed_c = ed - ed.mean()
    ed_n = np.linalg.norm(ed_c)
    scores = []
    for i in range(dft_matrix.shape[0]):
        ds = gaussian_filter1d(dft_matrix[i], sigma)
        dd = np.gradient(ds, exp_wn)
        dd_c = dd - dd.mean()
        dd_n = np.linalg.norm(dd_c)
        if ed_n > 0 and dd_n > 0:
            scores.append(float(np.dot(ed_c, dd_c) / (ed_n * dd_n)))
        else:
            scores.append(0.0)
    return np.array(scores)


def cosine_scores(exp_norm, dft_matrix):
    """Cosine similarity between experimental and each DFT spectrum."""
    ne = np.linalg.norm(exp_norm)
    return np.array([
        float(np.dot(exp_norm, d) / (ne * np.linalg.norm(d)))
        if np.linalg.norm(d) > 0 else 0.0
        for d in dft_matrix
    ])


# ════════════════════════════════════════════════════════════════════════
# STEP 3: FORWARD STEPWISE SELECTION
# ════════════════════════════════════════════════════════════════════════

def forward_stepwise(dft_matrix, exp_wn, exp_norm, max_k, poly_order=1,
                     progress_callback=None):
    """At each step, add the candidate that maximally reduces RSS.

    Parameters
    ----------
    progress_callback : callable or None
        If provided, called with (step, max_k, history) after each step.

    Returns
    -------
    history : list of (selected_indices, rss)
    """
    available = set(range(dft_matrix.shape[0]))
    selected = []
    history = []
    for step in range(max_k):
        best_rss, best_i = np.inf, None
        for cand in available:
            trial = selected + [cand]
            _, _, rss, _ = fit_nnls(dft_matrix[trial], exp_wn, exp_norm, poly_order)
            if rss < best_rss:
                best_rss, best_i = rss, cand
        if best_i is None:
            break
        selected.append(best_i)
        available.remove(best_i)
        history.append((list(selected), best_rss))
        if progress_callback:
            progress_callback(step + 1, max_k, history)
    return history


def exhaustive_search(dft_matrix, exp_wn, exp_norm, k, poly_order=1):
    """Exhaustive search for best k-component subset.

    Returns
    -------
    best_subset : list of int
    best_rss : float
    """
    best_rss, best_sub = np.inf, None
    n_mol = dft_matrix.shape[0]
    for sub in combinations(range(n_mol), k):
        _, _, rss, _ = fit_nnls(dft_matrix[list(sub)], exp_wn, exp_norm, poly_order)
        if rss < best_rss:
            best_rss, best_sub = rss, list(sub)
    return best_sub, best_rss


# ════════════════════════════════════════════════════════════════════════
# STEP 4: MODEL SELECTION
# ════════════════════════════════════════════════════════════════════════

def bic_neff(rss, k, n_eff):
    """BIC with effective number of independent points."""
    return n_eff * np.log(rss / n_eff) + k * np.log(n_eff)


def blocked_cv(dft_matrix, selected, exp_wn, exp_norm, n_blocks=10,
               poly_order=1):
    """Blocked k-fold cross-validation with contiguous wavenumber blocks."""
    n = len(exp_norm)
    edges = np.linspace(0, n, n_blocks + 1, dtype=int)
    cv_err = 0.0
    sub = dft_matrix[selected]
    for b in range(n_blocks):
        test_mask = np.zeros(n, dtype=bool)
        test_mask[edges[b]:edges[b + 1]] = True
        train_mask = ~test_mask
        A_tr, nd, nb = build_design_matrix(sub[:, train_mask],
                                           exp_wn[train_mask], poly_order)
        lb = np.concatenate([np.zeros(nd), np.full(nb, -np.inf)])
        ub = np.full(nd + nb, np.inf)
        res = lsq_linear(A_tr, exp_norm[train_mask], bounds=(lb, ub),
                         method="bvls")
        A_te, _, _ = build_design_matrix(sub[:, test_mask],
                                         exp_wn[test_mask], poly_order)
        pred = A_te @ res.x
        cv_err += np.sum((exp_norm[test_mask] - pred) ** 2)
    return cv_err


def select_model(sw_history, exp_wn, exp_norm, dft_matrix, n_eff,
                 n_cv_blocks=10, poly_order=1):
    """Compute BIC and CV error for each step in forward stepwise history.

    Returns
    -------
    bic_vals : list of float
    cv_vals : list of float
    best_k_bic : int
    best_k_cv : int
    """
    bic_vals, cv_vals = [], []
    for step, (sel, rss) in enumerate(sw_history):
        k = step + 1
        bv = bic_neff(rss, k, n_eff)
        bic_vals.append(bv)
        cve = blocked_cv(dft_matrix, sel, exp_wn, exp_norm, n_cv_blocks,
                         poly_order)
        cv_vals.append(cve)
    best_k_bic = int(np.argmin(bic_vals)) + 1
    best_k_cv = int(np.argmin(cv_vals)) + 1
    return bic_vals, cv_vals, best_k_bic, best_k_cv


# ════════════════════════════════════════════════════════════════════════
# STEP 6: BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════

def _block_bootstrap_indices(n, block_len, rng):
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(n - block_len + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block_len, n)) for s in starts])
    return idx[:n]


def run_bootstrap(dft_matrix, selected, exp_wn, exp_norm, n_boot,
                  block_len, poly_order=1, progress_callback=None):
    """Block bootstrap to get confidence intervals on spectral weights.

    Returns
    -------
    all_weights : ndarray, shape (n_boot, n_selected)
        Spectral weight fractions per bootstrap sample.
    """
    rng = np.random.default_rng(42)
    n_sel = len(selected)
    all_weights = np.zeros((n_boot, n_sel))
    sub = dft_matrix[selected]
    for b in range(n_boot):
        idx = _block_bootstrap_indices(len(exp_norm), block_len, rng)
        dc, _, _, _ = fit_nnls(sub[:, idx], exp_wn[idx], exp_norm[idx],
                               poly_order)
        s = dc.sum()
        all_weights[b] = dc / s if s > 0 else dc
        if progress_callback and (b + 1) % 50 == 0:
            progress_callback(b + 1, n_boot)
    return all_weights


def bootstrap_summary(boot_weights, selected, molecules):
    """Summarize bootstrap results.

    Returns
    -------
    list of dicts with keys: idx, cid, name, weight_median, ci_lo, ci_hi,
                             sel_freq
    """
    results = []
    for i, idx in enumerate(selected):
        w = boot_weights[:, i]
        results.append({
            "idx": idx,
            "cid": molecules[idx].get("cid", str(idx)),
            "name": molecules[idx].get("name", f"Component {idx}"),
            "weight_median": float(np.median(w)),
            "ci_lo": float(np.percentile(w, 2.5)),
            "ci_hi": float(np.percentile(w, 97.5)),
            "sel_freq": float(np.mean(w > 1e-4)),
        })
    return results


def rank_stability_matrix(boot_weights):
    """Pairwise rank stability: fraction of bootstraps where i > j.

    Returns
    -------
    matrix : ndarray, shape (n_sel, n_sel)
    """
    n_sel = boot_weights.shape[1]
    mat = np.zeros((n_sel, n_sel))
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            frac = float(np.mean(boot_weights[:, i] > boot_weights[:, j]))
            mat[i, j] = frac
            mat[j, i] = 1 - frac
    return mat


# ════════════════════════════════════════════════════════════════════════
# STEP 7: PEAK-RESOLVED RESIDUALS
# ════════════════════════════════════════════════════════════════════════

def peak_residual_analysis(exp_wn, exp_norm, reconstruction,
                           height=0.1, distance=5, prominence=0.05):
    """Detect peaks and compute residuals at each peak.

    Returns
    -------
    dict with keys: peak_wn, peak_exp, peak_fit, peak_resid, peak_indices
    """
    peaks, _ = find_peaks(exp_norm, height=height, distance=distance,
                          prominence=prominence)
    if len(peaks) == 0:
        return {"peak_wn": np.array([]), "peak_exp": np.array([]),
                "peak_fit": np.array([]), "peak_resid": np.array([]),
                "peak_indices": np.array([])}
    return {
        "peak_wn": exp_wn[peaks],
        "peak_exp": exp_norm[peaks],
        "peak_fit": reconstruction[peaks],
        "peak_resid": exp_norm[peaks] - reconstruction[peaks],
        "peak_indices": peaks,
    }


# ════════════════════════════════════════════════════════════════════════
# STEP 8: SCALING SENSITIVITY
# ════════════════════════════════════════════════════════════════════════

def scaling_sensitivity(dft_matrix, exp_wn, exp_norm, molecules,
                        base_scale, test_scales, max_k=8,
                        n_cv_blocks=10, poly_order=1):
    """Test sensitivity of decomposition to DFT scaling factor.

    Returns
    -------
    list of dicts with keys: scale, best_k, r2, cv_err, top_cid, top_weight,
                             selected
    """
    ss_tot = np.sum((exp_norm - exp_norm.mean()) ** 2)
    n_mol = dft_matrix.shape[0]
    results = []
    for sf in test_scales:
        shift = sf / base_scale
        dft_shifted = np.zeros_like(dft_matrix)
        for i in range(n_mol):
            dft_shifted[i] = np.interp(exp_wn, exp_wn * shift, dft_matrix[i],
                                       left=0, right=0)
        hist_s = forward_stepwise(dft_shifted, exp_wn, exp_norm, max_k,
                                  poly_order)
        cv_s = [blocked_cv(dft_shifted, h[0], exp_wn, exp_norm, n_cv_blocks,
                           poly_order)
                for h in hist_s]
        bk = int(np.argmin(cv_s)) + 1
        sel_s = hist_s[bk - 1][0]
        dc_s, _, rss_s, _ = fit_nnls(dft_shifted[sel_s], exp_wn, exp_norm,
                                     poly_order)
        r2_s = 1 - rss_s / ss_tot
        top_i = sel_s[int(np.argmax(dc_s))]
        top_w = float(dc_s.max() / dc_s.sum()) if dc_s.sum() > 0 else 0
        results.append({
            "scale": sf, "best_k": bk, "r2": r2_s,
            "cv_err": cv_s[bk - 1],
            "top_cid": molecules[top_i].get("cid", str(top_i)),
            "top_weight": top_w,
            "selected": sel_s,
        })
    return results


# ════════════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════

def run_full_pipeline(exp_wn, exp_norm, dft_matrix, molecules,
                      fwhm_cm=10.0, scaling_factor=0.95, poly_order=1,
                      max_k_forward=12, max_k_exhaustive=5,
                      n_cv_blocks=10, n_bootstrap=1000,
                      scaling_test_factors=None,
                      progress_callback=None):
    """Run the complete spectral decomposition pipeline.

    Parameters
    ----------
    exp_wn : ndarray — experimental wavenumber grid
    exp_norm : ndarray — experimental spectrum (normalized)
    dft_matrix : ndarray, shape (n_candidates, n_wn)
    molecules : list of dict — each with 'cid', 'name'
    fwhm_cm : float
    scaling_factor : float — base DFT scaling factor used
    poly_order : int
    max_k_forward : int
    max_k_exhaustive : int
    n_cv_blocks : int
    n_bootstrap : int
    scaling_test_factors : list of float or None
    progress_callback : callable or None — (step_name, detail)

    Returns
    -------
    results : dict with all pipeline outputs
    """
    n_wn = len(exp_wn)
    n_mol = dft_matrix.shape[0]
    wn_step = exp_wn[1] - exp_wn[0] if n_wn > 1 else 1.0
    wn_span = exp_wn.max() - exp_wn.min()
    n_eff = max(int(wn_span / fwhm_cm), 2)
    block_len = max(int(3 * fwhm_cm / wn_step), 5)
    ss_tot = float(np.sum((exp_norm - exp_norm.mean()) ** 2))

    out = {
        "n_wn": n_wn, "wn_step": wn_step, "n_eff": n_eff,
        "block_len": block_len, "ss_tot": ss_tot,
    }

    # Step 0: Diagnostics
    if progress_callback:
        progress_callback("diagnostics", "Computing collinearity...")
    out["diagnostics"] = compute_diagnostics(dft_matrix)
    if progress_callback:
        _d = out["diagnostics"]
        progress_callback("diagnostics",
                          f"cond(A) = {_d['cond_number']:.2e}, "
                          f"max cos = {_d['max_cosine']:.4f}, "
                          f"{_d['n_clusters']} clusters from {n_mol} candidates")

    # Step 1: Ranking
    if progress_callback:
        progress_callback("ranking", f"Scoring {n_mol} candidates (Pearson + cosine)...")
    out["pearson_scores"] = pearson_derivative_scores(exp_norm, dft_matrix,
                                                     exp_wn)
    out["cosine_scores"] = cosine_scores(exp_norm, dft_matrix)
    if progress_callback:
        _top_p = int(np.argmax(out["pearson_scores"]))
        _top_c = int(np.argmax(out["cosine_scores"]))
        progress_callback("ranking",
                          f"Top Pearson: {molecules[_top_p].get('name','?')[:25]} "
                          f"({out['pearson_scores'][_top_p]:.4f}) | "
                          f"Top Cosine: {molecules[_top_c].get('name','?')[:25]} "
                          f"({out['cosine_scores'][_top_c]:.4f})")

    # Step 2: Full NNLS
    if progress_callback:
        progress_callback("full_nnls", f"Full NNLS fit ({n_mol} candidates + baseline)...")
    dc_full, bc_full, rss_full, resid_full = fit_nnls(
        dft_matrix, exp_wn, exp_norm, poly_order)
    r2_full = 1 - rss_full / ss_tot
    wt_full = dc_full / dc_full.sum() if dc_full.sum() > 0 else dc_full
    out["full_nnls"] = {
        "coeffs": dc_full, "baseline": bc_full, "rss": rss_full,
        "r2": r2_full, "weights": wt_full, "residuals": resid_full,
    }
    if progress_callback:
        _n_nz = int(np.sum(dc_full > 0))
        progress_callback("full_nnls",
                          f"R² = {r2_full:.4f}, {_n_nz}/{n_mol} non-zero components")

    # Step 3: Forward stepwise
    if progress_callback:
        progress_callback("stepwise", f"Forward stepwise (max k={max_k_forward})...")

    def _sw_progress(step, total, history):
        if progress_callback and history:
            _last_sel, _last_rss = history[-1]
            _added = molecules[_last_sel[-1]].get('name', '?')[:20]
            _r2_s = 1 - _last_rss / ss_tot
            progress_callback("stepwise",
                              f"Step {step}/{total}: added {_added} → R² = {_r2_s:.4f}")

    sw_history = forward_stepwise(dft_matrix, exp_wn, exp_norm,
                                  max_k_forward, poly_order, _sw_progress)
    out["stepwise_history"] = sw_history

    # Step 4: Model selection
    if progress_callback:
        progress_callback("model_selection",
                          f"Computing BIC (n_eff={n_eff}) + {n_cv_blocks}-fold blocked CV...")
    bic_vals, cv_vals, best_k_bic, best_k_cv = select_model(
        sw_history, exp_wn, exp_norm, dft_matrix, n_eff, n_cv_blocks,
        poly_order)
    out["bic_vals"] = bic_vals
    out["cv_vals"] = cv_vals
    out["best_k_bic"] = best_k_bic
    out["best_k_cv"] = best_k_cv

    best_k = best_k_cv
    best_sel_sw = sw_history[best_k - 1][0]
    if progress_callback:
        progress_callback("model_selection",
                          f"Best k: BIC → {best_k_bic}, CV → {best_k_cv}. Using k = {best_k}")

    # Step 5: Exhaustive (if feasible)
    if best_k <= max_k_exhaustive:
        from math import comb
        _n_combos = comb(n_mol, best_k)
        if progress_callback:
            progress_callback("exhaustive",
                              f"Testing C({n_mol},{best_k}) = {_n_combos} subsets...")
        exh_sub, exh_rss = exhaustive_search(dft_matrix, exp_wn, exp_norm,
                                             best_k, poly_order)
        sw_rss = sw_history[best_k - 1][1]
        if exh_rss < sw_rss - 1e-6:
            best_sel = exh_sub
            if progress_callback:
                progress_callback("exhaustive",
                                  f"Exhaustive RSS={exh_rss:.4f} < stepwise RSS={sw_rss:.4f} → using exhaustive")
        else:
            best_sel = best_sel_sw
            if progress_callback:
                progress_callback("exhaustive",
                                  f"Stepwise matched exhaustive (RSS={sw_rss:.4f})")
        out["exhaustive"] = {"subset": exh_sub, "rss": exh_rss}
    else:
        best_sel = best_sel_sw
        out["exhaustive"] = None
        if progress_callback:
            progress_callback("exhaustive",
                              f"k={best_k} > {max_k_exhaustive}, skipping exhaustive")

    out["best_k"] = best_k
    out["best_selection"] = best_sel

    # Step 6: Final fit + bootstrap
    if progress_callback:
        _sel_names = ", ".join(molecules[i].get('name', '?')[:20] for i in best_sel)
        progress_callback("final_fit", f"Fitting k={best_k}: [{_sel_names}]")
    dc_best, bc_best, rss_best, resid_best = fit_nnls(
        dft_matrix[best_sel], exp_wn, exp_norm, poly_order)
    recon_best = exp_norm - resid_best
    r2_best = 1 - rss_best / ss_tot
    wt_best = dc_best / dc_best.sum() if dc_best.sum() > 0 else dc_best

    out["final_fit"] = {
        "coeffs": dc_best, "baseline": bc_best, "rss": rss_best,
        "r2": r2_best, "weights": wt_best, "residuals": resid_best,
        "reconstruction": recon_best,
    }
    if progress_callback:
        _wt_str = ", ".join(f"{molecules[best_sel[i]].get('cid','?')}:{wt_best[i]:.0%}"
                            for i in range(len(best_sel)))
        progress_callback("final_fit", f"R² = {r2_best:.4f} | Weights: {_wt_str}")

    if progress_callback:
        progress_callback("bootstrap",
                          f"Block bootstrap ({n_bootstrap} iters, block={block_len}pts)...")

    def _boot_progress(b, total):
        if progress_callback:
            progress_callback("bootstrap", f"Iteration {b}/{total}")

    boot_w = run_bootstrap(dft_matrix, best_sel, exp_wn, exp_norm,
                           n_bootstrap, block_len, poly_order, _boot_progress)
    out["bootstrap_weights"] = boot_w
    out["bootstrap_summary"] = bootstrap_summary(boot_w, best_sel, molecules)
    out["rank_stability"] = rank_stability_matrix(boot_w)
    if progress_callback:
        _sel_freq = np.mean(boot_w > 1e-4, axis=0)
        _sf_str = ", ".join(f"{molecules[best_sel[i]].get('cid','?')}:{_sel_freq[i]:.0%}"
                            for i in range(len(best_sel)))
        progress_callback("bootstrap", f"Selection frequencies: {_sf_str}")

    # Step 7: Peak residuals
    if progress_callback:
        progress_callback("peak_residuals", "Detecting peaks & computing residuals...")
    out["peak_analysis"] = peak_residual_analysis(exp_wn, exp_norm,
                                                  recon_best)
    if progress_callback:
        _n_pk = len(out["peak_analysis"]["peak_wn"])
        if _n_pk > 0:
            _mean_res = float(np.mean(np.abs(out["peak_analysis"]["peak_resid"])))
            progress_callback("peak_residuals",
                              f"{_n_pk} peaks detected, mean |residual| = {_mean_res:.4f}")
        else:
            progress_callback("peak_residuals", "No peaks above threshold")

    # Step 8: Scaling sensitivity
    if scaling_test_factors:
        if progress_callback:
            progress_callback("sensitivity",
                              f"Testing {len(scaling_test_factors)} scaling factors: "
                              f"{scaling_test_factors}...")
        out["sensitivity"] = scaling_sensitivity(
            dft_matrix, exp_wn, exp_norm, molecules,
            scaling_factor, scaling_test_factors,
            max_k=min(8, max_k_forward), n_cv_blocks=n_cv_blocks,
            poly_order=poly_order)
        if progress_callback:
            _top_cids = [r["top_cid"] for r in out["sensitivity"]]
            if len(set(_top_cids)) == 1:
                progress_callback("sensitivity",
                                  f"✓ Stable: top assignment = CID {_top_cids[0]} across all scales")
            else:
                progress_callback("sensitivity",
                                  f"⚠ Top assignment varies: {_top_cids}")
    else:
        out["sensitivity"] = None

    if progress_callback:
        progress_callback("done", "Pipeline complete.")

    return out
