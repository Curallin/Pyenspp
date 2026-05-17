import numpy as np
from scipy.stats import gamma
from scipy.special import beta
from ..optimization.sceua import sceua

# 1. 定义链接方程的CRPS计算公式
def crpsCond(par, obs, ensmean_ratio, md, muc, sigmac, shiftc, zero_threshold=0.1):
    """
    最小化CRPS, 拟合线性回归的参数

    μ = μc * log1p[ expm1(a1) * (a2 + a3*ensmean_ratio) ] / a1
    σ = a4 * σc * (μ/μc)**a5 + a6 * MD
    δ = δc
    """
    # 1. 回归系数提取与保护
    # a1-a3: 均值参数; a4-a6: 尺度参数
    a = par 
    esp = 1e-6 # 适当调小一点提升精度

    # 预处理观测：统一精度阈值
    obs = np.where(obs > zero_threshold, obs, 0.0)
    # 判定哪些日子是实况无雨的
    is_dry = (obs < 1e-6)

    # 2. 回归方程 (公式 S7, S8)
    # 使用向量化计算每天的 mu 和 sigma
    try:
        # np.expm1(a[0]) 对于大的 a[0] 比 np.exp(a[0])-1 更稳健
        mu = muc * np.log1p(np.expm1(a[0]) * (a[1] + a[2] * ensmean_ratio)) / (a[0] + esp)
        sigma = a[3] * sigmac * (mu / (muc + esp))**a[4] + a[5] * md
        
        # 惩罚项：mu 或 sigma 必须为正
        if np.any(mu <= 0) or np.any(sigma <= 0):
            return 1e10
    except:
        return 1e10

    # 3. 分布参数转换
    shape = np.square(mu / (sigma + esp))
    scale = np.square(sigma) / (mu + esp)
    shift = shiftc
    
    # 4. CRPS 核心项计算
    # cstd 和相关的 Fck 是每天都必须算的（因为预报每天在变）
    cstd = -shift / scale
    Fck = gamma.cdf(cstd, shape)
    FckP1 = gamma.cdf(cstd, shape + 1.0)
    F2c2k = gamma.cdf(2 * cstd, 2 * shape)
    betaf = beta(0.5, shape + 0.5)

    # --- 关键优化点：只对有雨的日子计算 Fyk 和 FykP1 ---
    # 先初始化 ystd 和 F 序列
    ystd = (obs - shift) / scale
    # 对于无雨日，ystd 理论上等于 cstd，但由于精度处理，我们强制对齐
    ystd = np.maximum(ystd, cstd) 
    
    Fyk = Fck.copy()
    FykP1 = FckP1.copy()

    # 只在有雨索引下更新
    is_rainy = ~is_dry
    if np.any(is_rainy):
        Fyk[is_rainy] = gamma.cdf(ystd[is_rainy], shape[is_rainy])
        FykP1[is_rainy] = gamma.cdf(ystd[is_rainy], shape[is_rainy] + 1.0)

    # 5. 解析式合成 (根据变换后的积分公式)
    # 该公式对每一天计算一个 CRPS 值
    crps = (ystd * (2.0 * Fyk - 1.0) 
            - cstd * np.square(Fck) 
            + shape * (1.0 + 2.0 * Fck * FckP1 - np.square(Fck) - 2.0 * FykP1)
            - (shape / np.pi) * betaf * (1.0 - F2c2k))

    # 返回所有天数的平均值乘以各自当天的 scale
    return np.mean(scale * crps)

# 2. 链接方程的参数估计
def calculate_cond_par(obs, ensmean_ratio, md, par_clima, ngs=5, max_iter=500, seed=42):
    """
    利用历史训练集拟合 CSG-EMOS 的 6 个回归参数
    
    参数:
    obs: 历史观测序列
    ensmean_ratio: 历史集合预报均值与气候均值的比值 (f_bar / f_cl)
    MD: 历史集合离散度序列
    par_clima: [muc, sigmac, shiftc] 第一阶段算出的气候态参数
    """
    muc, sigmac, shiftc = par_clima
    
    # 1. 定义回归参数的搜索边界 (基于 Scheuerer & Hamill 2015 经验值)
    # a1: 链接函数形状, a2, a3: 均值系数
    # a4, a5, a6: 尺度/方差系数
    bounds = [
        (0.001, 10),   # a1
        (0.0001, 1),   # a2
        (0.0001, 3),   # a3
        (0.1, 10),     # a4
        (0.0001, 1),   # a5 (指数项不宜过大)
        (0.0001, 1.5)  # a6
    ]
    
    # 2. 封装目标函数
    target_func = lambda p: crpsCond(p, obs, ensmean_ratio, md, muc, sigmac, shiftc)
    
    # 3. 调用 SCE-UA 寻找最优回归系数
    # 对于 6 维参数空间，ngs=5-10 是比较合理的
    best_a, _ = sceua(
        target_func, 
        bounds, 
        ngs=ngs, 
        max_iter=max_iter, 
        seed=seed
    )
    
    return best_a