import numpy as np

def sceua(func, bounds, ngs=5, max_iter=500, kstop=10, pcento=0.01, peps=1e-5, seed=None):
    """
    SCE-UA (Shuffled Complex Evolution - University of Arizona) 算法高效串行实现

    该算法结合了单纯形搜索的高效性和群体进化的全局性，适用于处理多峰值、非线性的全局优化问题。

    算法目标:最小化目标函数值

    参数:
    ----------
    func : callable
        待优化的目标函数，接收一个一维 numpy 数组并返回一个标量值。
    bounds : list of tuples
        参数的取值范围，例如: [(min1, max1), (min2, max2), ...]。
    ngs : int
        复合形 (Complexes) 的数量。增加该值可提高全局搜索能力，但增加计算量。
    max_iter : int
        最大迭代 (洗牌) 次数。
    kstop : int
        收敛触发窗口。若连续 kstop 次迭代目【标函数改进极小，则停止。
    pcento : float
        收敛改进比例。用于判断目标函数值是否在窗口内有显著变化。
    peps : float
        参数空间收敛容差。若群体各维度的标准差小于此值，则认为已收敛。
    seed : int, optional
        随机数种子。
    
    返回:
    -------
    best_x : ndarray
        找到的最优参数组合。
    best_f : float
        最优参数对应的目标函数值。
    """
    
    if seed is not None:
        np.random.seed(seed)

    # 1. 初始化基础参数
    bounds = np.array(bounds)
    low, high = bounds[:, 0], bounds[:, 1]
    dim = len(bounds)

    npg = 2 * dim + 1       # 每个复合形中的点数 (建议 2n+1)
    nps = dim + 1           # 子复合形中的点数 (建议 n+1)
    nspl = npg              # 每个复合形在洗牌前的进化次数
    npt = ngs * npg         # 总群体大小

    # 2. 生成初始种群并排序
    population = np.random.uniform(low, high, (npt, dim))
    fitness = np.array([func(x) for x in population])
    
    # 升序排列 (寻找最小值)
    idx = np.argsort(fitness)
    population, fitness = population[idx], fitness[idx]

    best_history = []

    # 3. 进化循环
    for it in range(max_iter):
        # 对每个复合形独立进行进化
        for igs in range(ngs):
            # 提取第 igs 个复合形 (采用等间隔抽样保证多样性)
            cp_idx = np.arange(igs, npt, ngs)
            cp_pop = population[cp_idx]
            cp_fit = fitness[cp_idx]
            
            # 执行竞争复合形进化 (CCE)
            cp_pop, cp_fit = _cce(func, cp_pop, cp_fit, nspl, nps, low, high)
            
            # 将进化后的结果写回原种群
            population[cp_idx] = cp_pop
            fitness[cp_idx] = cp_fit

        # 4. 洗牌 (Shuffling) - 重新对全种群进行整体排序
        idx = np.argsort(fitness)
        population, fitness = population[idx], fitness[idx]

        best_f = fitness[0]
        best_history.append(best_f)

        # 打印进度
        if it % 10 == 0:
            print(f"Iteration {it}: Best Fitness = {best_f:.6e}")

        # 5. 收敛检测
        # 准则 A: 目标函数值在 kstop 次窗口内改进微小
        if len(best_history) >= kstop:
            improvement = abs(best_history[-kstop] - best_f)
            if improvement < pcento * abs(best_f) + 1e-12:
                print(f"Converged: Function value stability reached at iteration {it}.")
                break

        # 准则 B: 种群在参数空间已高度收敛 (各维度标准差足够小)
        if np.std(population, axis=0).max() < peps:
            print(f"Converged: Parameter space stability reached at iteration {it}.")
            break

    return population[0], fitness[0]


def _cce(func, cp_pop, cp_fit, nspl, nps, low, high):
    """
    竞争复合形进化 (Competitive Complex Evolution) 核心逻辑。
    """
    m, dim = cp_pop.shape
    
    # 计算基于排名的选择概率 (排名越靠前，选入子复合形的概率越大)
    weights = 2.0 * (m + 1 - np.arange(1, m + 1)) / (m * (m + 1))

    for _ in range(nspl):
        # 1. 按照概率权重随机选取 nps 个点构成子复合形
        sub_idx = np.sort(np.random.choice(m, nps, replace=False, p=weights))
        s_pop = cp_pop[sub_idx]
        s_fit = cp_fit[sub_idx]
        
        # 2. 找到最差点 (由于 cp_pop 始终保持排序，s_idx 最后的点就是子复合形里最差的)
        worst_in_sub = -1
        xw, fw = s_pop[worst_in_sub], s_fit[worst_in_sub]
        
        # 3. 计算质心 (排除最差点后的几何中心)
        centroid = np.mean(s_pop[:-1], axis=0)
        
        # 4. 尝试 反射 (Reflection)
        xr = 2.0 * centroid - xw
        # 越界检查: 若越界则随机生成
        if np.any(xr < low) or np.any(xr > high):
            xr = np.random.uniform(low, high)
        
        fr = func(xr)
        
        if fr < fw:
            # 反射点比最差点好，替换它
            cp_pop[sub_idx[worst_in_sub]] = xr
            cp_fit[sub_idx[worst_in_sub]] = fr
        else:
            # 5. 尝试 收缩 (Contraction)
            xc = 0.5 * (centroid + xw)
            fc = func(xc)
            if fc < fw:
                cp_pop[sub_idx[worst_in_sub]] = xc
                cp_fit[sub_idx[worst_in_sub]] = fc
            else:
                # 6. 反射和收缩都失败，执行 随机跳跃
                xn = np.random.uniform(low, high)
                fn = func(xn)
                cp_pop[sub_idx[worst_in_sub]] = xn
                cp_fit[sub_idx[worst_in_sub]] = fn
        
        # 7. 每步更新后对复合形内进行重新排序，维持排名权重
        idx_recalc = np.argsort(cp_fit)
        cp_pop, cp_fit = cp_pop[idx_recalc], cp_fit[idx_recalc]
    
    return cp_pop, cp_fit