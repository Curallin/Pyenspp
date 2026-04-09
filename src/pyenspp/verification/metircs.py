"""
vericication/metrics.py
===================

A comprehensive collection of step-wise and aggregate verification metrics 
for hydroclimatic ensemble forecasts.

This module provides both atomic functions (returning series for plotting) 
and an EnsembleEvaluator class (for batch processing).

Metrics included:
- Deterministic: RB, PCC, SCC, RMSE, RMSESS, MAE, MAESS, NSE, KGE, NPKGE
- Ensemble: CRPS, CRPSS, PIT, Alpha-Index, IQR, twCRPS
- Categorical: BS(REL/RES/UNC), BSS, AUC, ROCSS
- Event-based: POD, FAR, CSI, Bias, ETS

References
----------
1. Huang, Z., & Zhao, T. (2022). Predictive performance of ensemble hydroclimatic 
   forecasts: Verification metrics, diagnostic plots and forecast attributes. 
   WIREs Water, 9, e1580.
2. Murphy, A. H. (1973). A new vector partition of the probability score. 
   Journal of Applied Meteorology, 12, 595–600.
3. Gupta, H. V., et al. (2009). Decomposition of the MSE and NSE. 
   Journal of Hydrology, 377, 80–91.
4. Pool, S., et al. (2018). Non-parametric variant of KGE. 
   Hydrological Sciences Journal, 63, 1941–1953.
5. Hersbach, H. (2000). Decomposition of the CRPS. 
   Weather and Forecasting, 15, 559–570.
6. Wessel, N., et al. (2025). Improving probabilistic forecasts of extreme wind speeds by training statistical postprocessing models with weighted scoring rules.   Monthly Weather Review, 1489-1511
7. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review
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
def mb_series(f, o):
    """Mean bias series"""
    mask = _get_mask(f, o)
    return np.mean(f[mask] - o[mask]) if np.sum(mask) > 0 else np.nan

def rb_series(f, o):
    """Relative bias (%) series"""
    mask = _get_mask(f, o)
    return np.mean((f[mask] - o[mask]) / o[mask] * 100) if np.sum(mask) > 0 else np.nan

def mae_series(f, o): 
    """Mean absolute error series"""
    mask = _get_mask(f, o)
    return np.mean(np.abs(f - o)) if np.sum(mask) > 0 else np.nan
def mse_series(f, o): 
    """Mean squared error series"""
    mask = _get_mask(f, o)
    return np.mean((f - o)**2) if np.sum(mask) > 0 else np.nan

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

def npkge(f, o):
    """Non-parametric Kling-Gupta Efficiency"""
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
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (n_forecasts, m_members). The raw ensemble values.
    obs : np.ndarray
        Shape (n_forecasts,). The actual observations.
    
    Returns:
    --------
    np.ndarray
        Shape (n_forecasts,). The step-wise CRPS values.
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

def twcrps_series(ensemble: np.ndarray, obs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Compute step-wise threshold-weighted CRPS (twCRPS) for an ensemble forecast.
    Method: Censoring + Sorting-based PWM (Following Wessel et al. 2025)
    
    Parameters:
    -----------
    ensemble : np.ndarray
        Shape (n_forecasts, m_members). The raw ensemble values.
    obs : np.ndarray
        Shape (n_forecasts,). The actual observations.
    threshold : float or np.ndarray
        The threshold tau. Can be a single value or an array of shape (n_forecasts,).
    """
    # 1. 预处理：应用阈值审查 (Censoring)
    # 根据论文引理 4.1：twCRPS(F, y) = CRPS(max(F, tau), max(y, tau))
    # 将所有低于阈值的值全部替换为阈值
    ens_censored = np.maximum(ensemble, threshold if np.isscalar(threshold) else threshold[:, None])
    obs_censored = np.maximum(obs, threshold)
    
    n, m = ens_censored.shape
    
    # 2. 计算第一项 (准确度项): 1/M * sum |Z_i - y_tilde|
    # 这里 Z_i 是经过阈值处理后的预报成员，y_tilde 是经过阈值处理后的观测
    term1 = np.nanmean(np.abs(ens_censored - obs_censored[:, None]), axis=1)
    
    # 3. 对处理后的预报成员进行排序
    ens_sorted = np.sort(ens_censored, axis=1)
    valid_counts = np.sum(~np.isnan(ens_sorted), axis=1)
    term2 = np.zeros(n)
    
    # 4. 计算第二项 (离散度项): 1/M^2 * sum (2i - M - 1) * Z_i
    # 逻辑与标准 CRPS 一致，但输入的是排序后的 Z_i (ens_sorted)
    unique_vcounts = np.unique(valid_counts)
    for mv in unique_vcounts:
        if mv < 2: continue
        idx = (valid_counts == mv)
        sub_ens = ens_sorted[idx, :mv]
        
        # 计算权重系数 (2i - M - 1)
        # i 从 1 到 M
        w = (2.0 * np.arange(1, mv + 1) - mv - 1.0)
        
        # 利用矩阵乘法快速计算加权和
        term2[idx] = np.sum(sub_ens * w[None, :], axis=1) / (mv * mv)
    
    # 5. 最终结果
    res = term1 - term2
    
    # 处理全为空值的情况
    res[valid_counts == 0] = np.nan
    return res

def pit_series(ensemble, obs, method='censored', threshold=0.01, seed=42):
    """
    Compute PIT (Probability Integral Transform) for ensemble forecasts.
    
    Parameters
    ----------
    ensemble : np.ndarray, shape (n_forecasts, n_members)
        Ensemble forecasts
    obs : np.ndarray, shape (n_forecasts,)
        Observations
    method : str
        'traditional': Mid-P method (deterministic, uses midpoint)
        'randomized': Randomized PIT (stochastic, properly handles discreteness)
        'censored': Alias for 'randomized' (deprecated)
    threshold : float
        Precipitation threshold for zero events (default: 0.01 mm)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pit : np.ndarray, shape (n_forecasts,)
        PIT values
    """
    rng = np.random.default_rng(seed)
    
    # 1. 计算有效成员数
    v_cnt = np.sum(~np.isnan(ensemble), axis=1)
    
    # 2. 处理微量降水阈值（审查）
    oc = obs.copy()
    ec = ensemble.copy()
    oc[oc <= threshold] = 0.0
    ec[ec <= threshold] = 0.0
    
    # 3. 计算 F(y-) 和 F(y)
    f_minus = np.sum(ec < oc[:, None], axis=1) / v_cnt
    f_plus = np.sum(ec <= oc[:, None], axis=1) / v_cnt
    
    # 4. 处理全为 NaN 的情况
    f_minus[v_cnt == 0] = np.nan
    f_plus[v_cnt == 0] = np.nan
    
    if method in ['traditional', 'midp']:
        # Mid-P method (deterministic)
        return (f_minus + f_plus) / 2.0
    
    elif method in ['randomized', 'censored', 'stochastic']:
        # Randomized PIT (stochastic, proper for discrete distributions)
        # Sample uniformly from [F(y-), F(y)]
        pit_vals = f_minus + rng.random(size=len(obs)) * (f_plus - f_minus)
        return pit_vals
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'traditional' or 'randomized'")
    
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
def bs_series(ensemble, obs, threshold):
    """
    计算 Brier Score
    ensemble: (n_forecasts, n_members)  - 原始集合预报
    obs: (n_forecasts,)             - 原始观测
    threshold: float          - 降水阈值

    返回: (n_forecasts,) - 每个预报的 Brier Score
    """
    # 1. 将集合预报转换为概率 (p)
    # 假设最后一个维度是集合成员维度
    p = (ensemble >= threshold).mean(axis=-1)
    
    # 2. 将观测转换为二值化结果 (o)
    o_bin = (obs >= threshold).astype(float)
    
    # 3. 计算 BS 序列
    return (p - o_bin)**2

def decompose_bs(ensemble, obs, threshold, n_bins=11):
    """
    对 Brier Score 进行 REL, RES, UNC 三项分解
    ensemble: (time, members) 或 (space, time, members)
    obs: (time,) 或 (space, time)
    threshold: float
    n_bins: 分箱数量

    return: dict - REL, RES, UNC 三项以及总 BS 和 BSS 值
    """
    # --- 核心转换 ---
    # 计算概率 p: [0, 1]
    p = (ensemble >= threshold).mean(axis=-1).flatten()
    # 计算二值观测 o_bin: {0, 1}
    o_bin = (obs >= threshold).astype(float).flatten()
    
    if len(p) != len(o_bin):
        raise ValueError("Flattened ensemble and observation must have the same length.")

    # --- 基础统计 ---
    o_bar = np.mean(o_bin)
    unc = o_bar * (1 - o_bar)
    
    # --- 分箱逻辑 (Vectorized) ---
    bins = np.linspace(0, 1, n_bins + 1)
    # 将概率映射到 bin 索引 (0 到 n_bins-1)
    bin_ids = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    
    # 使用 bincount 统计每个 bin
    counts = np.bincount(bin_ids, minlength=n_bins)
    p_sum = np.bincount(bin_ids, weights=p, minlength=n_bins)
    o_sum = np.bincount(bin_ids, weights=o_bin, minlength=n_bins)
    
    # 过滤掉没有样本的 bin
    mask = counts > 0
    p_bin_mean = p_sum[mask] / counts[mask]
    o_bin_freq = o_sum[mask] / counts[mask]
    
    # --- 分解计算 ---
    total_n = len(o_bin)
    rel = np.sum(counts[mask] * (p_bin_mean - o_bin_freq)**2) / total_n
    res = np.sum(counts[mask] * (o_bin_freq - o_bar)**2) / total_n
    
    bs = rel - res + unc
    
    return {
        "BS": bs,
        "REL": rel,
        "RES": res,
        "UNC": unc,
        "BSS": 1 - bs/unc if unc > 0 else 0.0
    }

def calc_contingency(p, o_bin, prob_threshold=0.5):
    """Compute contingency table counts H, F, M, CN"""
    f_bin = (p >= prob_threshold).astype(int)
    H = np.sum((f_bin == 1) & (o_bin == 1))
    F = np.sum((f_bin == 1) & (o_bin == 0))
    M = np.sum((f_bin == 0) & (o_bin == 1))
    CN = np.sum((f_bin == 0) & (o_bin == 0))
    return H, F, M, CN

def roc_calc(ensemble, obs, threshold: float, n_prob_bins: int = 21):
    """
    Compute ROC curve points, exact AUC (Area Under Curve), and ROC Skill Score (ROCSS)
    for ensemble forecasts.

    This function evaluates the discrimination ability of ensemble forecasts
    for a binary event defined by a threshold (e.g., precipitation exceeding
    a certain value). The ensemble is converted to exceedance probabilities,
    and ROC curve points are computed by scanning probability thresholds.

    Additionally, the AUC is computed exactly using a rank-based method
    (Mann–Whitney U statistic), which is more accurate than numerical integration.

    Parameters
    ----------
    ensemble : ndarray of shape (n_samples, n_members)
        Ensemble forecast values.
    obs : ndarray of shape (n_samples,)
        Observed values.
    threshold : float
        Threshold defining the binary event (e.g., precipitation > threshold).
    n_prob_bins : int, optional
        Number of probability thresholds used to construct ROC curve.
        If None, unique forecast probabilities are used instead.

    Returns
    -------
    fpr : ndarray
        False Positive Rate (FAR) values for ROC curve.
    tpr : ndarray
        True Positive Rate (Hit Rate) values for ROC curve.
    auc_exact : float
        Area Under the ROC Curve computed using rank statistics.
    rocss : float
        ROC Skill Score (ROCSS = 2*AUC - 1).

    Notes
    -----
    - Ensemble forecasts are converted into probabilities using the fraction
      of members exceeding the threshold.
    - AUC is computed using the Mann–Whitney U statistic, which is equivalent
      to the probability that a randomly chosen event case has a higher forecast
      probability than a non-event case.
    - ROC curve evaluates discrimination ability, not calibration.
    """

    # Step 1: Convert ensemble forecasts to exceedance probabilities
    p = _calc_probs(ensemble, threshold)

    # Step 2: Convert observations to binary events
    o_bin = (obs > threshold).astype(int)

    # Step 3: Remove missing values
    mask = _get_mask(p, o_bin)
    p, o_bin = p[mask], o_bin[mask]

    # Step 4: Compute exact AUC using rank statistics (Mann–Whitney U)
    if len(np.unique(o_bin)) < 2:
        # Cannot compute ROC if only one class exists
        auc_exact = np.nan
        rocss = np.nan
    else:
        pos = p[o_bin == 1]
        neg = p[o_bin == 0]

        n_pos = len(pos)
        n_neg = len(neg)

        # Combine and rank
        combined = np.concatenate([pos, neg])
        ranks = rankdata(combined, method='average')

        # Compute AUC from rank sum
        auc_exact = (
            np.sum(ranks[:n_pos]) - n_pos * (n_pos + 1) / 2
        ) / (n_pos * n_neg)

        # ROC Skill Score
        rocss = 2 * auc_exact - 1

    # Step 5: Define probability thresholds for ROC curve
    if n_prob_bins is not None:
        prob_thresholds = np.linspace(0, 1.001, n_prob_bins)
    else:
        prob_thresholds = np.unique(p)
        prob_thresholds = np.sort(np.append(prob_thresholds, [0.0, 1.1]))

    tpr_list, fpr_list = [], []

    # Step 6: Compute ROC points
    for pc in prob_thresholds:
        f_bin = (p >= pc).astype(int)

        tp = np.sum((f_bin == 1) & (o_bin == 1))
        fp = np.sum((f_bin == 1) & (o_bin == 0))
        fn = np.sum((f_bin == 0) & (o_bin == 1))
        tn = np.sum((f_bin == 0) & (o_bin == 0))

        # True Positive Rate (Hit Rate)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # False Positive Rate (False Alarm Rate)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr_array = np.array(fpr_list)
    tpr_array = np.array(tpr_list)

    # Step 7: Ensure ROC curve endpoints exist
    if not np.any((fpr_array == 0) & (tpr_array == 0)):
        fpr_array = np.append(fpr_array, 0.0)
        tpr_array = np.append(tpr_array, 0.0)

    if not np.any((fpr_array == 1) & (tpr_array == 1)):
        fpr_array = np.append(fpr_array, 1.0)
        tpr_array = np.append(tpr_array, 1.0)

    # Step 8: Sort by FPR (ascending)
    idx = np.argsort(fpr_array)

    return fpr_array[idx], tpr_array[idx], auc_exact, rocss

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
    def mb(self): return mb_series(self.ens_mean, self.obs)
    def rb(self): return rb_series(self.ens_mean, self.obs)
    def pcc(self): return pcc(self.ens_mean, self.obs)
    def scc(self): return scc(self.ens_mean, self.obs)
    def nse(self): return nse(self.ens_mean, self.obs)
    def kge(self): return kge(self.ens_mean, self.obs)
    def npkge(self): return npkge(self.ens_mean, self.obs)
    def mae(self): return mae_series(self.ens_mean, self.obs)
    def mse(self): return mse_series(self.ens_mean, self.obs)
    def rmse(self): return np.sqrt(np.nanmean(self.mse()))

    # Ensemble metrics
    def crps(self): return crps_series(self.ensemble, self.obs)
    
    def crpss(self):
        cf = np.nanmean(self.crps())
        cr = np.nanmean(crps_series(self.ref_ensemble, self.obs))
        if cr == 0 or np.isnan(cr): return np.nan
        return (1 - cf / cr) * 100
    def pit(self, method='censored', threshold=0.01, seed=42):
        return pit_series(self.ensemble, self.obs, method, threshold, seed)
    def alpha_index(self, method='censored', threshold=0.01):
        return alpha_index(self.pit(method, threshold))
    def iqr(self): return np.nanpercentile(self.ensemble, 75, axis=1) - np.nanpercentile(self.ensemble, 25, axis=1)

    # Categorical metrics
    def bs(self, threshold: float):
        """
        Compute mean Brier Score.
        """
        bs_vals = bs_series(self.ensemble, self.obs, threshold)
        return np.nanmean(bs_vals)
    def bss(self, threshold: float):
        """
        Compute Brier Skill Score (%) relative to reference ensemble.
        """
        bs_f = np.nanmean(bs_series(self.ensemble, self.obs, threshold))
        bs_r = np.nanmean(bs_series(self.ref_ensemble, self.obs, threshold))

        if bs_r > 0:
            return (1 - bs_f / bs_r) * 100
        else:
            return np.nan
    
    def bs_decomposition(self, threshold: float, n_bins: int = 11):
        """
        Perform Brier Score decomposition (REL, RES, UNC).

        Parameters
        ----------
        threshold : float
            Event threshold.
        n_bins : int
            Number of probability bins.

        Returns
        -------
        dict with keys:
            BS, REL, RES, UNC, BSS
        """
        return decompose_bs(self.ensemble, self.obs, threshold, n_bins)
        
    def roc_curve(self, threshold: float, n_prob_bins: int = 21):
        """
        Return ROC curve points (FPR, TPR).
        """
        fpr, tpr, _, _ = roc_calc(self.ensemble, self.obs, threshold, n_prob_bins)
        return fpr, tpr

    def auc(self, threshold: float, n_prob_bins: int = 21):
        """
        Return exact AUC computed using rank statistics (Mann–Whitney U).
        """
        _, _, auc_exact, _ = roc_calc(self.ensemble, self.obs, threshold, n_prob_bins)
        return auc_exact

    def rocss(self, threshold: float, n_prob_bins: int = 21):
        """
        Return ROC Skill Score (ROCSS = 2*AUC - 1).
        """
        _, _, _, rocss_val = roc_calc(self.ensemble, self.obs, threshold, n_prob_bins)
        return rocss_val