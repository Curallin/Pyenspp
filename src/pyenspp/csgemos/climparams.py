import numpy as np
from scipy.stats import gamma
from scipy.special import beta
from ..optimization.sceua import sceua

# 1. 定义气候态模型待优化目标函数及参数
def crpsClim(par, obs, zero_threshold=0.1):
    """
    高性能优化版 CSGD CRPS 计算

    计算三参数 (shape / scale / shift) climatologyCSGD分布与真实观测分布之间的CRPS 
    考虑观测仪器误差, 设置 zero_threshold 以下的值为零

    使用前提: 观测降水的降水发生概率pop<1
    par[0]: mu
    par[1]: sigma
    par[2]: shift

    输入参数: 
    obs: 观测降水序列
    """

    # 1. 鲁棒性检查 (针对优化算法如 SCE-UA)
    mu, sigma, shift = par

    # sigma 过小会导致 shape 爆炸，设置一个合理的下限
    if sigma < 1e-4 or mu < 1e-4 or shift > 0:
        return 1e10

    # 清理数据
    obs = obs[~np.isnan(obs)]
    n = len(obs)

    # 考虑仪器观测精度
    obs = np.where(obs > zero_threshold, obs, 0)
    
    # 提取零值和非零值
    is_zero = (obs < 1e-6) # 考虑浮点数误差
    k0 = np.sum(is_zero)
    obs_nonzero = obs[~is_zero]

    # 参数转换
    # 限制 mu/sigma 的比值防止 shape 过大导致计算卡死
    ratio = mu / sigma
    if ratio > 20: return 1e10 # 经验值，可根据业务调整

    shape = ratio ** 2
    scale = mu / shape
    
    # 2. 计算公共常数项 (这些项只依赖于分布参数和截断点 c)
    c_std = -shift / scale
    F_k_c = gamma.cdf(c_std, a=shape)
    F_kp1_c = gamma.cdf(c_std, a=shape + 1.0)
    F_2k_2c = gamma.cdf(2.0 * c_std, a=2.0 * shape)
    B_05_kp05 = beta(0.5, shape + 0.5)
    
    # 公共偏移项 (对应积分中与截断点相关的部分)
    # 这部分在原公式中对所有 y 都是一样的
    common_offset = (
        - shape * (F_k_c**2 - 2. * F_kp1_c * F_k_c)
        - c_std * F_k_c**2
        - (shape / np.pi) * B_05_kp05 * (1. - F_2k_2c)
    )

    # 3. 处理零值样本的具体项
    # 当 obs == 0 时, y_std == c_std
    term_zero = c_std * (2. * F_k_c - 1.) - shape * (2. * F_kp1_c - 1.)
    
    # 4. 处理非零值样本的具体项 (向量化)
    y_std_nonzero = (obs_nonzero - shift) / scale
    F_k_y = gamma.cdf(y_std_nonzero, a=shape)
    F_kp1_y = gamma.cdf(y_std_nonzero, a=shape + 1.0)
    
    terms_nonzero = y_std_nonzero * (2. * F_k_y - 1.) - shape * (2. * F_kp1_y - 1.)

    # 5. 合并计算平均 CRPS
    # 总和 = k0 * (零值项 + 公共项) + sum(非零值项 + 公共项)
    # 等价于: 公共项 + (k0 * 零值项 + sum(非零值项)) / n
    total_crps_std = common_offset + (k0 * term_zero + np.sum(terms_nonzero)) / n
    
    return scale * total_crps_std

# 2. 使用 SCEUA 进行参数优化
# Reference: Liu et al.(2026). Statistical postprocessing of subseasonal cumulative precipitation forecasts using a spatial heterogeneity-aware U-net. Journal of Hydrology
def calculate_clim_par(obs, zero_threshold=0.1,ngs=5, max_iter=500, seed=42):
    """
    调用适配后的 sceua 函数率定 ClimaCSGD 参数，最终仅返回参数数组
    """
    obs_mean = np.mean(obs)
    # 统计降水发生频率 (阈值设为 zero_threshold)
    obs_pop = np.mean(obs > zero_threshold)
    sigma = obs_mean if obs_mean > 0 else zero_threshold  # 初始假设 sigma = mu

    # 1. 启发式初始化 mu, sigma, shift
    # 通过频率匹配法快速寻找一个合理的起始 shift
    mu_found = 0.1
    shift_found = -0.1
    for mu in (np.arange(40, 0, -1) * (sigma / 40)):
        if mu <= 0: continue
        shape = (mu / sigma) ** 2
        scale = mu / shape
        # 利用无雨频率匹配计算 shift 初始值
        shift = -gamma.ppf(1. - obs_pop, a=shape, scale=scale, loc=0)
        if shift > -mu / 2.:
            mu_found = mu
            shift_found = shift
            break
    
    par0 = np.array([mu_found, sigma, shift_found])
        
    # 2. 极端干旱情况处理 (样本太少，优化算法易失效)
    if obs_pop < 0.005:
        # 返回预设的经验参数
        return np.array([0.0005, 0.0182, -0.00049])

    # 3. 较干旱情况处理
    if obs_pop < 0.02:
        # 直接返回初始化参数，不进行复杂优化
        return par0

    # 4. 正常情况：调用适配后的 sceua 进行参数优化
    else:
        # 定义目标函数闭包，只接受 par 一个参数
        target_func = lambda p: crpsClim(p, obs, zero_threshold=zero_threshold)
        
        # 构造边界 [(min, max), ...]
        # 注意：因为 par0[2] (shift) 为负数，所以 5 * par0[2] 是更小的值（下界）
        shift_upper = min(0.1 * par0[2], -1e-5)
        bounds = [
            (0.1 * par0[0], 5.0 * par0[0]),  # mu 的范围
            (0.1 * par0[1], 5.0 * par0[1]),  # sigma 的范围
            (5.0 * par0[2] - 0.1, shift_upper)  # shift 的范围 (减0.1防止shift为0时边界重合)
        ]
        
        # 调用你的 sceua 函数
        # ngs 和 max_iter 比较敏感，其他参数可以保持不变这里不再进行设置
        best_par, _ = sceua(
            target_func, 
            bounds, 
            ngs=ngs, 
            max_iter=max_iter, 
            seed=seed
        )
        
        return best_par