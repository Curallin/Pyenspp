import numpy as np
from abc import ABC, abstractmethod
from scipy import stats, optimize
import warnings


# ======================================================
# Abstract base
# ======================================================
class TransformationStrategy(ABC):
    """
    Abstract base class for distributional transformations.
    Only changes distributional shape (no standardization).
    """

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass


# ======================================================
# Log Transformation
# ======================================================
class LogTransformation(TransformationStrategy):
    """
    Shifted log transform:
        y = log(x + alpha)

    alpha is estimated via profile likelihood assuming
    y ~ N(mu, sigma^2), including Jacobian term.
    """

    def __init__(self):
        self.alpha_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = X[~np.isnan(X)]

        if np.any(X < 0):
            raise ValueError("LogTransformation requires non-negative data.")

        def nll(alpha):
            if alpha <= 0:
                return np.inf

            Y = np.log(X + alpha)
            var = np.var(Y)
            if var <= 0:
                return np.inf

            # Profile likelihood (Gaussian) + Jacobian
            n = len(Y)
            return (n / 2.0) * np.log(var) + np.sum(np.log(X + alpha))

        res = optimize.minimize_scalar(
            nll,
            bounds=(1e-6, np.max(X) + 1.0),
            method="bounded",
        )

        self.alpha_ = res.x
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Transformation not fitted.")
        return np.log(np.asarray(X, dtype=float) + self.alpha_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X) - self.alpha_

    def get_params(self):
        return {"method": "log", "alpha": self.alpha_}


# ======================================================
# Asinh (Log-Sinh) Transformation
# ======================================================
class LogSinhTransformation(TransformationStrategy):
    """
    Asinh transformation:
        y = asinh(x / alpha)

    Behaves linearly near zero and logarithmically for extremes.
    alpha is estimated via MLE with Jacobian correction.
    """

    def __init__(self):
        self.alpha_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = X[~np.isnan(X)]

        if np.any(X < 0):
            raise ValueError("Asinh transformation requires non-negative data.")

        def nll(alpha):
            if alpha <= 0:
                return np.inf

            Y = np.arcsinh(X / alpha)
            var = np.var(Y)
            if var <= 0:
                return np.inf

            n = len(Y)

            # 正确的 log-Jacobian 项（∑ log|dy/dx|）
            log_jac_sum = -n * np.log(alpha) - 0.5 * np.sum(np.log(1.0 + (X / alpha)**2))

            # NLL ∝ n/2 log(var) - ∑ log|dy/dx|
            return (n / 2.0) * np.log(var) - log_jac_sum

        res = optimize.minimize_scalar(
            nll,
            bounds=(1e-6, np.max(X) + 1.0),
            method="bounded",
        )

        self.alpha_ = res.x
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.alpha_ is None:
            raise RuntimeError("Transformation not fitted.")
        return np.arcsinh(np.asarray(X, dtype=float) / self.alpha_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.alpha_ * np.sinh(X)

    def get_params(self):
        return {"method": "asinh", "alpha": self.alpha_}


# ======================================================
# Box-Cox Transformation (with shift)
# ======================================================
class BoxCoxTransformation(TransformationStrategy):
    """
    Shifted Box-Cox transformation:
        y = boxcox(x + alpha, lambda)

    alpha and lambda are estimated via maximum likelihood.
    """

    def __init__(self):
        self.alpha_ = None
        self.lmbda_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        X = X[~np.isnan(X)]

        if np.any(X < 0):
            raise ValueError("Box-Cox requires non-negative data.")

        def neg_ll(alpha):
            if alpha <= 0:
                return np.inf
            try:
                _, lmbda = stats.boxcox(X + alpha)
                return -stats.boxcox_llf(lmbda, X + alpha)
            except Exception:
                return np.inf

        res = optimize.minimize_scalar(
            neg_ll,
            bounds=(1e-6, np.max(X) + 1.0),
            method="bounded",
        )

        self.alpha_ = res.x
        _, self.lmbda_ = stats.boxcox(X + self.alpha_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.alpha_ is None or self.lmbda_ is None:
            raise RuntimeError("Transformation not fitted.")
        return stats.boxcox(np.asarray(X, dtype=float) + self.alpha_, self.lmbda_)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        from scipy.special import inv_boxcox

        return inv_boxcox(X, self.lmbda_) - self.alpha_

    def get_params(self):
        return {
            "method": "box-cox",
            "alpha": self.alpha_,
            "lambda": self.lmbda_,
        }


# ======================================================
# Context class
# ======================================================
class DistributionTransformer:
    """
    Context class for precipitation distribution transformation.
    """

    _STRATEGIES = {
        "log": LogTransformation,
        "asinh": LogSinhTransformation,
        "box-cox": BoxCoxTransformation,
    }

    def __init__(self, method: str = "asinh"):
        if method not in self._STRATEGIES:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.strategy = self._STRATEGIES[method]()

    def fit(self, X: np.ndarray):
        self.strategy.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = self.strategy.transform(X)
        return np.nan_to_num(out)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        out = self.strategy.inverse_transform(X)
        return np.maximum(out, 0.0)

    def get_fit_params(self):
        return self.strategy.get_params()