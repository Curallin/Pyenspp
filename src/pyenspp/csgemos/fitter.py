import numpy as np
import scipy.stats as stats
from scipy.stats import gamma
from typing import Optional, Tuple, Union
from .climparams import calculate_clim_par
from .condparams import calculate_cond_par

class CSGDFitter:
    """
    Censored Shifted Gamma Distribution (CSGD) 适配器类
    包含气候态拟合、条件回归拟合、分位数采样预报及 QQ 置信区间评估。
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.clim_params_ = None  # [muc, sigmac, shiftc]
        self.cond_params_ = None  # [a1, a2, a3, a4, a5, a6]
        self.f_bar_cl_ = None     # 用于保存集合预报的气候态均值 (S7公式中的 \bar{f}_{cl})
        self.is_clim_fit = False
        self.is_cond_fit = False

    # ========================== 1. 拟合模块 ==========================
    def fit_climatology(self, obs: np.ndarray, **kwargs):
        self.clim_params_ = calculate_clim_par(obs, **kwargs)
        self.is_clim_fit = True

    def fit_conditional(self, obs: np.ndarray, ensmean: np.ndarray, md: np.ndarray, **kwargs):
        """
        注意：外部传入的 ensmean 是原始的降水集合均值序列
        """
        if not self.is_clim_fit:
            raise RuntimeError("请先执行 fit_climatology。")
        
        # 1. 计算当前格点在训练期内的集合平均值的气候态 f_bar_cl
        self.f_bar_cl_ = np.mean(ensmean)
        
        # 防止极端干旱区全是0导致的除零警告
        eps = 1e-5
        f_bar_safe = np.maximum(self.f_bar_cl_, eps)
        
        # 2. 提前计算好比值序列
        ensmean_ratio = (ensmean+eps) / f_bar_safe
            
        # 3. 将比值序列喂给你的底层优化函数
        self.cond_params_ = calculate_cond_par(obs, ensmean_ratio, md, self.clim_params_, **kwargs)
        self.is_cond_fit = True

    # ========================== 2. 预测与采样模块 ==========================

    def predict_climatology(self, n_members: int = 500) -> np.ndarray:
        if not self.is_clim_fit: raise RuntimeError("气候态未拟合。")
        return self._quantile_sampling(*self.clim_params_, n_members)

    def predict_condition(self, ensmean: float, md: float, n_members: int = 500) -> np.ndarray:
        if not self.is_cond_fit: raise RuntimeError("条件分布回归未拟合。")
        mu, sigma, shift = self._get_cond_dist_params(ensmean, md)
        return self._quantile_sampling(mu, sigma, shift, n_members)

    def rvs(self, mu: float, sigma: float, shift: float, size: int = 1) -> np.ndarray:
        """生成随机样本（包含阈值抖动逻辑）"""
        k, theta = self._get_k_theta(mu, sigma)
        # 基础 CSGD 采样
        raw = gamma.rvs(a=k, scale=theta, size=size) + shift
        samples = np.maximum(raw, 0.0)
        # 阈值抖动 (Jitter)
        mask = samples < self.threshold
        if np.any(mask):
            samples[mask] = np.random.uniform(0, self.threshold, size=np.sum(mask))
        return samples

    # ========================== 3. 评估模块 (QQ Confidence Band) ==========================

    def qq_confidence_band(
        self, 
        obs_data: np.ndarray, 
        n_sim: int = 1000, 
        alpha: float = 0.05, 
        mode: str = 'climatology',
        ensmean: float = None, 
        md: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        利用蒙特卡洛模拟生成 QQ 图的置信区间带。
        
        参数:
        obs_data: 用于对比的实况观测序列
        n_sim: 模拟次数 (默认 1000)
        alpha: 显著性水平 (0.05 对应 95% 置信区间)
        mode: 'climatology' 或 'condition'
        """
        if mode == 'climatology':
            if not self.is_clim_fit: raise RuntimeError("气候态未拟合")
            mu, sigma, shift = self.clim_params_
        else:
            if not self.is_cond_fit: raise RuntimeError("条件分布未拟合")
            mu, sigma, shift = self._get_cond_dist_params(ensmean, md)

        n_obs = len(obs_data)
        k, theta = self._get_k_theta(mu, sigma)
        
        # 存储所有模拟的顺序统计量
        sim_results = np.zeros((n_sim, n_obs))
        
        for i in range(n_sim):
            # 模拟采样并包含 jitter 逻辑，保持与预报一致
            sim_samples = self.rvs(mu, sigma, shift, size=n_obs)
            sim_results[i, :] = np.sort(sim_samples)
            
        # 计算每个分位点上的置信边界
        lower = np.percentile(sim_results, 100 * alpha / 2, axis=0)
        upper = np.percentile(sim_results, 100 * (1 - alpha / 2), axis=0)
        
        return lower, upper

    # ========================== 4. 分布函数 ==========================

    def cdf(self, x: Union[float, np.ndarray], mu: float, sigma: float, shift: float) -> np.ndarray:
        k, theta = self._get_k_theta(mu, sigma)
        z = (np.asarray(x) - shift) / theta
        z = np.maximum(z, 0.0)
        return np.where(x < 0, 0.0, gamma.cdf(z, a=k))

    def ppf(self, q: Union[float, np.ndarray], mu: float, sigma: float, shift: float) -> np.ndarray:
        k, theta = self._get_k_theta(mu, sigma)
        q = np.clip(q, 1e-7, 1.0 - 1e-7)
        val = theta * gamma.ppf(q, a=k) + shift
        return np.maximum(val, 0.0)

    # ========================== 5. 内部工具 ==========================
    def _get_cond_dist_params(self, ensmean: float, md: float) -> Tuple[float, float, float]:
        """根据回归方程计算推断期的 μ, σ, δ"""
        muc, sigmac, shiftc = self.clim_params_
        a = self.cond_params_
        eps = 1e-5
        
        f_bar_safe = np.maximum(self.f_bar_cl_, 1e-5)  # 取出训练期存下来的均值
        ratio_f = (ensmean+eps) / f_bar_safe           # 把今天测试期的预报值除以它，得到比值！
        
        # 🌟【代入公式 S7】使用的是转换后的比值 ratio_f，而不是原始的 ensmean
        mu = muc * np.log1p(np.expm1(a[0]) * (a[1] + a[2] * ratio_f)) / a[0]
        
        # 对应 S8 公式计算 σ
        sigma = a[3] * sigmac * (mu / (muc + eps))**a[4] + a[5] * md
        
        return mu, sigma, shiftc

    def _get_k_theta(self, mu, sigma):
        sigma = np.maximum(sigma, 1e-8)
        mu = np.maximum(mu, 1e-8)
        shape = (mu / sigma) ** 2
        scale = mu / shape
        return shape, scale

    def _quantile_sampling(self, mu: float, sigma: float, shift: float, n_members: int) -> np.ndarray:
        """等概率分位数采样 + 阈值抖动"""
        quantiles = np.linspace(1 / (n_members + 1), n_members / (n_members + 1), n_members)
        samples = self.ppf(quantiles, mu, sigma, shift)
        
        mask = samples < self.threshold
        if np.any(mask):
            # 抖动处理：保证小于阈值的成员分布在 [0, threshold]
            samples[mask] = np.random.uniform(0, self.threshold, size=np.sum(mask))
        
        return np.sort(samples)