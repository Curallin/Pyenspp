import numpy as np

def nelder_mead(f, initial_simplex, max_iter=100, tol=1e-10, 
                          alpha=1.0, beta=0.5, gamma=2.0, sigma=0.5, return_history=False):
    """
    高效的 Nelder-Mead 优化算法实现
    目的：优化函数的参数，使得函数值最小化
    alpha: 反射系数 (标准为 1.0)
    gamma: 扩展系数 (标准为 2.0)
    beta: 收缩系数 (标准为 0.5)
    sigma: 缩点系数 (标准为 0.5)
    """
    simplex = np.array(initial_simplex, dtype=float)
    n = simplex.shape[1]
    
    # 1. 初始计算：仅在最开始计算所有点的函数值
    f_vals = np.array([f(x) for x in simplex])
    
    simplex_history = [simplex.copy()] if return_history else[]
    f_val_history = []

    if return_history:
        simplex_history.append(simplex.copy())
        f_val_history.append(f_vals.min())

    for _ in range(max_iter):
        # 2. 根据函数值排序
        order = np.argsort(f_vals)
        simplex = simplex[order]
        f_vals = f_vals[order]
        
        # 检查收敛 (使用最优与最差点的距离作为简单判据)
        if np.linalg.norm(simplex[0] - simplex[-1]) < tol: # 计算范数(高维线性空间的距离)
            break
            
        x_best, f_best = simplex[0], f_vals[0]
        x_second_worst, f_second_worst = simplex[-2], f_vals[-2]
        x_worst, f_worst = simplex[-1], f_vals[-1]
        
        # 计算形心（不包含最差点）
        centroid = np.mean(simplex[:-1], axis=0)
        
        # 3. 反射 (Reflection)
        x_r = centroid + alpha * (centroid - x_worst)
        f_r = f(x_r)  # 只计算一次
        
        # 标志位：判断本轮是否需要执行整体缩点
        shrink = False
        
        if f_best <= f_r < f_second_worst:
            # 介于最优和次差之间：接受反射点
            simplex[-1] = x_r
            f_vals[-1] = f_r
            
        elif f_r < f_best:
            # 比最优还要好：尝试扩展 (Expansion)
            x_e = centroid + gamma * (x_r - centroid)
            f_e = f(x_e)
            if f_e < f_r: # 扩展成功 替换最差点为当前扩展点
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else: # 扩展失败 替换最差点为反射点
                simplex[-1] = x_r
                f_vals[-1] = f_r
                
        else: # f_r >= f_second_worst
            # 比次差还要差：执行收缩 (Contraction)
            if f_r < f_worst:
                # 外部收缩 (Outside Contraction)
                x_c = centroid + beta * (x_r - centroid)
                f_c = f(x_c)
                if f_c <= f_r:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    shrink = True
            else:
                # 内部收缩 (Inside Contraction)
                x_c = centroid - beta * (centroid - x_worst) # 注意方向
                f_c = f(x_c)
                if f_c < f_worst:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    shrink = True
                    
        # 4. 整体缩点 (Shrink)
        if shrink:
            for i in range(1, n + 1):
                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                f_vals[i] = f(simplex[i]) # 缩点时必须重新计算新点的值

        # 记录历史
        if return_history:
            simplex_history.append(simplex.copy())
            f_val_history.append(f_vals.min())

    # --- 修正返回值逻辑 ---
    best_x = simplex[0]
    best_f = f_vals[0]
    
    if return_history:
        # 返回 4 个值：单纯形历史，函数值历史，最优x，最优f
        return np.array(simplex_history), np.array(f_val_history), best_x, best_f
    
    else:
        # 返回 2 个值：最优坐标x，最优函数值f
        return best_x, best_f