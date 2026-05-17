import numpy as np

def genetic_algorithm(func, bounds, pop_size=100, max_gen=500, 
                           pc=0.8, pm=0.1, tour_size=3, b=3.0, return_history=False):
    """
    基于 锦标赛选择 + 非均匀变异 + 边界反射 的高效全向量化遗传算法
    参数:
    func: 目标函数 (最小化)
    bounds: 边界[(L1, U1), (L2, U2), ...]
    pop_size: 种群大小 (必须为偶数，方便交叉配对)
    max_gen: 最大迭代代数 T
    pc: 交叉概率
    pm: 变异概率
    tour_size: 锦标赛规模 K
    b: 非均匀变异的形状参数 (通常为2~5, 越大前期变异越大, 后期衰减越快)
    return_history: bool, 是否返回历史记录
    """
    assert pop_size % 2 == 0, "种群大小 pop_size 必须为偶数"
    
    bounds = np.array(bounds)
    dim = len(bounds)
    lower_bounds = bounds[:, 0]
    upper_bounds = bounds[:, 1]
    
    # 1. 向量化初始化种群 (pop_size, dim)
    pop = np.random.uniform(lower_bounds, upper_bounds, (pop_size, dim))
    
    # 评估初始种群
    fitness = np.apply_along_axis(func, 1, pop)
    
    best_idx = np.argmin(fitness)
    global_best_x = pop[best_idx].copy()
    global_best_f = fitness[best_idx]
    
    history = [global_best_f] if return_history else[]
    
    for t in range(max_gen):
        # ==========================================
        # 1. 锦标赛选择 (Tournament Selection)
        # ==========================================
        tour_indices = np.random.randint(0, pop_size, size=(pop_size, tour_size))
        best_in_tours = np.argmin(fitness[tour_indices], axis=1)
        parents = pop[tour_indices[np.arange(pop_size), best_in_tours]]
        
        # ==========================================
        # 2. 算术交叉 (Arithmetic Crossover)
        # ==========================================
        p1 = parents[0::2] 
        p2 = parents[1::2] 
        
        cross_mask = np.random.rand(len(p1), 1) < pc
        alpha = np.random.rand(len(p1), dim)
        
        c1 = np.where(cross_mask, alpha * p1 + (1 - alpha) * p2, p1)
        c2 = np.where(cross_mask, alpha * p2 + (1 - alpha) * p1, p2)
        offspring = np.vstack((c1, c2))
        
        # ==========================================
        # 3. 非均匀变异 (Non-uniform Mutation)
        # ==========================================
        mut_mask = np.random.rand(pop_size, dim) < pm
        dir_mask = np.random.rand(pop_size, dim) < 0.5
        
        y = np.where(dir_mask, upper_bounds - offspring, offspring - lower_bounds)
        r = np.random.rand(pop_size, dim) 
        annealing_power = (1.0 - t / max_gen) ** b
        delta = y * (1.0 - r ** annealing_power)
        
        offspring = np.where(mut_mask & dir_mask, offspring + delta, offspring)
        offspring = np.where(mut_mask & ~dir_mask, offspring - delta, offspring)
        
        # ==========================================
        # 4. 越界反射处理 (Reflecting Boundary) 🌟 改进点
        # ==========================================
        mask_low = offspring < lower_bounds
        mask_up = offspring > upper_bounds
        
        # 像台球撞壁一样弹回搜索空间内部
        offspring = np.where(mask_low, 2 * lower_bounds - offspring, offspring)
        offspring = np.where(mask_up, 2 * upper_bounds - offspring, offspring)
        
        # 二次保险：防止极端变异步长导致弹射后依然越界，最终用 clip 兜底
        offspring = np.clip(offspring, lower_bounds, upper_bounds)
        
        # ==========================================
        # 重新评估适应度
        # ==========================================
        offspring_fitness = np.apply_along_axis(func, 1, offspring)
        
        # ==========================================
        # 5. 精英保留策略 (Elitism)
        # ==========================================
        worst_idx = np.argmax(offspring_fitness)
        offspring[worst_idx] = global_best_x
        offspring_fitness[worst_idx] = global_best_f
        
        # ==========================================
        # 6. 更新全局最优与种群
        # ==========================================
        pop = offspring
        fitness = offspring_fitness
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < global_best_f:
            global_best_f = fitness[current_best_idx]
            global_best_x = pop[current_best_idx].copy()
            
        if return_history:
            history.append(global_best_f)

    if return_history:
        return global_best_x, global_best_f, history
    return global_best_x, global_best_f