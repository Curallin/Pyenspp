from .marginal_fitter import CSGDFitter
from .qm import QuantileMapping
from .kan import KANCSGDRegression

# 可选：定义 __all__ 控制 from ... import * 的行为
__all__ = [
    'CSGDFitter',
    'QuantileMapping',
    'KANCSGDRegression'
]