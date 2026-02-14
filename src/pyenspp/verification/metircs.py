"""
vericication/metrics.py
===================

A comprehensive collection of step-wise and aggregate verification metrics 
for hydroclimatic ensemble forecasts.

This module provides both atomic functions (returning series for plotting) 
and an EnsembleEvaluator class (for batch processing).

Features:
1. Provides step-wise error series for scatterplots and temporal diagnostics.
2. Provides aggregate metrics covering deterministic, ensemble, and categorical forecasts.
3. Supports Pseudo-PIT randomization for zero-precipitation handling.

Metrics included (25 total):
- Deterministic: RB, PCC, SCC, RMSE, RMSESS, MAE, MAESS, NSE, KGE, NPE
- Ensemble: CRPS, CRPSS, PIT, Alpha-Index, IQR
- Categorical: BS(REL/RES/UNC), BSS, ROCSS
- Event-based: POD, FAR, CSI, Bias, ETS

References
----------
1. Huang, Z., & Zhao, T. (2022). Predictive performance of ensemble hydroclimatic 
   forecasts: Verification metrics, diagnostic plots and forecast attributes. 
   WIREs Water, 9, e1580. https://doi.org/10.1002/wat2.1580
2. Murphy, A. H. (1973). A new vector partition of the probability score. 
   Journal of Applied Meteorology, 12, 595–600.
3. Gupta, H. V., et al. (2009). Decomposition of the MSE and NSE. 
   Journal of Hydrology, 377, 80–91.
4. Pool, S., et al. (2018). Non-parametric variant of KGE. 
   Hydrological Sciences Journal, 63, 1941–1953.
5. Hersbach, H. (2000). Decomposition of the CRPS. 
   Weather and Forecasting, 15, 559–570.
et al.
"""

import numpy as np
from scipy.stats import rankdata
from typing import Union, List, Dict, Optional, Tuple

# =============================================================================
# 1. Basic helper functions
# =============================================================================

def _get_mask(f: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Return boolean mask where both forecast and observation are not NaN"""
    return ~(np.isnan(f) | np.isnan(o))

def _calc_probs(ensemble: np.ndarray, threshold: float) -> np.ndarray:
    """Compute exceedance probability for an ensemble given a threshold"""
    condition = (ensemble > threshold).astype(float)
    condition[np.isnan(ensemble)] = np.nan
    return np.nanmean(condition, axis=1)

# =============================================================================
# 2. Deterministic metrics
# =============================================================================

def bias_series(f, o):
    """Point-wise bias (mean error) series"""
    return f - o

def rb_series(f, o):
    """Relative bias (%) series"""
    with np.errstate(divide='ignore', invalid='ignore'):
        res = (f - o) / o * 100
    res[np.isinf(res)] = np.nan
    return res

def ae_series(f, o): return np.abs(f - o)
def se_series(f, o): return (f - o)**2

def pcc(f, o):
    """Pearson correlation coefficient"""
    mask = _get_mask(f, o)
    return np.corrcoef(f[mask], o[mask])[0, 1] if np.sum(mask) > 1 else np.nan

def scc(f, o):
    """Spearman rank correlation coefficient"""
    mask = _get_mask(f, o)
    return np.corrcoef(rankdata(f[mask]), rankdata(o[mask]))[0, 1] if np.sum(mask) > 1 else np.nan

def nse(f, o):
    """Nash-Sutcliffe Efficiency"""
    mask = _get_mask(f, o)
    f_m, o_m = f[mask], o[mask]
    return 1 - np.sum((f_m - o_m)**2) / np.sum((o_m - np.mean(o_m))**2)

def kge(f, o):
    """Kling-Gupta Efficiency"""
    mask = _get_mask(f, o)
    f_m, o_m = f[mask], o[mask]
    if len(o_m) < 2: return np.nan
    r = np.corrcoef(f_m, o_m)[0, 1]
    beta = np.mean(f_m) / np.mean(o_m)
    gamma = (np.std(f_m)/np.mean(f_m)) / (np.std(o_m)/np.mean(o_m))
    return 1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)

def npe(f, o):
    """Non-parametric Efficiency"""
    mask = _get_mask(f, o)
    f_m, o_m = f[mask], o[mask]
    if len(o_m) < 2: return np.nan
    r_s = scc(f_m, o_m)
    beta = np.mean(f_m) / np.mean(o_m)
    fs, os = np.sort(f_m)[::-1], np.sort(o_m)[::-1]
    eps = 1 - 0.5 * np.mean(np.abs(fs/np.mean(f_m) - os/np.mean(o_m)))
    return 1 - np.sqrt((r_s-1)**2 + (beta-1)**2 + (eps-1)**2)

# =============================================================================
# 3. Ensemble metrics
# =============================================================================

def crps_series(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Compute step-wise CRPS for an ensemble forecast.
    Method: Sorting-based Probability Weighted Moments (PWM)
    Complexity: O(N*M log M)
    """
    n, m = ensemble.shape
    term1 = np.nanmean(np.abs(ensemble - obs[:, None]), axis=1)
    
    ens_sorted = np.sort(ensemble, axis=1)
    valid_counts = np.sum(~np.isnan(ens_sorted), axis=1)
    term2 = np.zeros(n)
    
    unique_vcounts = np.unique(valid_counts)
    for mv in unique_vcounts:
        if mv < 2: continue
        idx = (valid_counts == mv)
        sub_ens = ens_sorted[idx, :mv]
        w = (2.0 * np.arange(1, mv + 1) - mv - 1.0)
        term2[idx] = np.sum(sub_ens * w[None, :], axis=1) / (mv * mv)
    
    res = term1 - term2
    res[valid_counts == 0] = np.nan
    return res

def pit_series(ensemble, obs, method='traditional', threshold=0.01, seed=None):
    """
    Compute PIT (Probability Integral Transform) for an ensemble forecast.
    Supports 'traditional' and 'censored' methods for zero precipitation.
    """
    v_cnt = np.sum(~np.isnan(ensemble), axis=1)
    if method == 'traditional':
        return (np.sum(ensemble < obs[:, None], axis=1) + 0.5 * np.sum(ensemble == obs[:, None], axis=1)) / v_cnt
    elif method == 'censored':
        oc, ec = obs.copy(), ensemble.copy()
        oc[oc <= threshold], ec[ec <= threshold] = 0.0, 0.0
        raw_ecdf = np.sum((ec <= oc[:, None]), axis=1) / v_cnt
        pit_vals = raw_ecdf.copy()
        idx = (oc <= 0.0) & (~np.isnan(raw_ecdf))
        if np.any(idx):
            pit_vals[idx] = np.random.default_rng(seed).random(size=np.sum(idx)) * raw_ecdf[idx]
        return pit_vals
    return np.nan

def alpha_index(pits):
    """Alpha-index for ensemble calibration"""
    p = pits[~np.isnan(pits)]
    n = len(p)
    if n < 2: return np.nan
    p_s = np.sort(p)
    return 1 - (2/n) * np.sum(np.abs(p_s - np.arange(1, n+1)/(n+1)))

# =============================================================================
# 4. Categorical and contingency metrics
# =============================================================================

def bs_series(p, o_bin): return (p - o_bin)**2

def bs_decomposition(p, o_bin, n_bins=10):
    """Decompose Brier Score into REL, RES, UNC"""
    o_bar = np.nanmean(o_bin)
    unc = o_bar * (1 - o_bar)
    bins = np.linspace(0, 1, n_bins + 1)
    b_ids = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    rel, res = 0.0, 0.0
    for i in range(n_bins):
        idx = (b_ids == i)
        if np.any(idx):
            ni, pi_b, oi_b = np.sum(idx), np.mean(p[idx]), np.mean(o_bin[idx])
            rel += ni * (pi_b - oi_b)**2
            res += ni * (oi_b - o_bar)**2
    return {"REL": rel/len(o_bin), "RES": res/len(o_bin), "UNC": unc}

def rocss(p, o_bin):
    """Ranked Probability Skill Score (ROC-based)"""
    mask = _get_mask(p, o_bin)
    p, o = p[mask], o_bin[mask]
    if len(np.unique(o)) < 2: return np.nan
    pos, neg = p[o==1], p[o==0]
    r = rankdata(np.concatenate([pos, neg]))
    auc = (np.sum(r[:len(pos)]) - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))
    return 2 * auc - 1

def calc_contingency(p, o_bin, prob_threshold=0.5):
    """Compute contingency table counts H, F, M, CN"""
    f_bin = (p >= prob_threshold).astype(int)
    H = np.sum((f_bin == 1) & (o_bin == 1))
    F = np.sum((f_bin == 1) & (o_bin == 0))
    M = np.sum((f_bin == 0) & (o_bin == 1))
    CN = np.sum((f_bin == 0) & (o_bin == 0))
    return H, F, M, CN

# =============================================================================
# 5. EnsembleEvaluator class
# =============================================================================

class EnsembleEvaluator:
    """
    High-level interface for deterministic, ensemble, and categorical metrics.
    """
    def __init__(self, obs: np.ndarray, ensemble: np.ndarray, ref_ensemble: Optional[np.ndarray] = None):
        self.obs = np.asarray(obs)
        self.ensemble = np.asarray(ensemble)
        self.ens_mean = np.nanmean(self.ensemble, axis=1)
        if ref_ensemble is None:
            self.ref_ensemble = np.full_like(self.ensemble, np.nanmean(self.obs))
        else:
            self.ref_ensemble = np.asarray(ref_ensemble)
        self.ref_mean = np.nanmean(self.ref_ensemble, axis=1)

    # Deterministic metrics
    def bias(self): return bias_series(self.ens_mean, self.obs)
    def rb(self): return rb_series(self.ens_mean, self.obs)
    def pcc(self): return pcc(self.ens_mean, self.obs)
    def scc(self): return scc(self.ens_mean, self.obs)
    def nse(self): return nse(self.ens_mean, self.obs)
    def kge(self): return kge(self.ens_mean, self.obs)
    def npe(self): return npe(self.ens_mean, self.obs)
    def ae(self): return ae_series(self.ens_mean, self.obs)
    def se(self): return se_series(self.ens_mean, self.obs)
    def rmse(self): return np.sqrt(np.nanmean(self.se()))

    # Ensemble metrics
    def crps(self): return crps_series(self.ensemble, self.obs)
    def crpss(self):
        cf = np.nanmean(self.crps())
        cr = np.nanmean(crps_series(self.ref_ensemble, self.obs))
        if cr == 0 or np.isnan(cr): return np.nan
        return (1 - cf / cr) * 100
    def pit(self, method='traditional', threshold=0.01, seed=None):
        return pit_series(self.ensemble, self.obs, method, threshold, seed)
    def alpha_index(self, method='traditional', threshold=0.01):
        return alpha_index(self.pit(method, threshold))
    def iqr(self): return np.nanpercentile(self.ensemble, 75, axis=1) - np.nanpercentile(self.ensemble, 25, axis=1)

    # Categorical metrics
    def bs(self, threshold: float):
        p_f = _calc_probs(self.ensemble, threshold)
        return bs_series(p_f, (self.obs > threshold).astype(float))
    def bss(self, threshold: float):
        p_f, p_r = _calc_probs(self.ensemble, threshold), _calc_probs(self.ref_ensemble, threshold)
        o_b = (self.obs > threshold).astype(float)
        bf, br = bs_series(p_f, o_b), bs_series(p_r, o_b)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (1 - bf / br) * 100
    def rocss(self, threshold: float):
        return rocss(_calc_probs(self.ensemble, threshold), (self.obs > threshold).astype(float))