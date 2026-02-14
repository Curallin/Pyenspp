# calibration/marginal_fitter.py
import numpy as np
import scipy.special as sp
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Optional, Union
import warnings
from .base import ProbabilisticFitter

class CSGDFitter(ProbabilisticFitter):
    """
    Censored Shifted Gamma Distribution (CSGD) fitter.

    Assumes latent variable Z ~ ShiftedGamma(k, theta, delta),
    observed variable Y = max(0, Z).

    Parameters are estimated by minimizing the CRPS.
    Reference: Scheuerer and Hamill (2015).

    Usage:
        fitter = CSGDFitter()
        fitter.fit(obs_data)

        # Access fitted parameters
        mu, sigma, delta = fitter.params

        # Compute probabilities
        prob = fitter.cdf(values)
        vals = fitter.ppf(quantiles)
    """

    def __init__(self):
        self.mu_: Optional[float] = None
        self.sigma_: Optional[float] = None
        self.delta_: Optional[float] = None
        self.k_: Optional[float] = None
        self.theta_: Optional[float] = None
        self.is_fitted: bool = False

    @property
    def params(self) -> Tuple[float, float, float]:
        """Return fitted (mu, sigma, delta) parameters."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return (self.mu_, self.sigma_, self.delta_)

    def get_gamma_params(self, mu: float, sigma: float) -> Tuple[float, float]:
        """Convert mu and sigma to Gamma shape (k) and scale (theta)."""
        sigma = np.maximum(sigma, 1e-6)
        mu = np.maximum(mu, 1e-6)
        k = (mu / sigma) ** 2
        theta = (sigma ** 2) / mu
        return k, theta

    def _csgd_crps_loss(self, params: np.ndarray, data: np.ndarray) -> float:
        """Compute mean CRPS loss for optimizer (Scheuerer and Hamill, 2015)."""
        mu, sigma, delta = params
        k, theta = self.get_gamma_params(mu, sigma)
        y_tilde = (data - delta) / theta
        c_tilde = -delta / theta
        y_tilde = np.maximum(y_tilde, 0)
        c_tilde = np.maximum(c_tilde, 0)
        F_k_y = sp.gammainc(k, y_tilde)
        F_k_c = sp.gammainc(k, c_tilde)
        F_k1_y = sp.gammainc(k + 1, y_tilde)
        F_k1_c = sp.gammainc(k + 1, c_tilde)
        F_2k_2c = sp.gammainc(2 * k, 2 * c_tilde)
        beta_val = sp.beta(0.5, k + 0.5)
        term1 = theta * y_tilde * (2 * F_k_y - 1)
        term2 = -theta * c_tilde * (F_k_c ** 2)
        term3 = theta * k * (1 + 2 * F_k_c * F_k1_c - F_k_c**2 - 2 * F_k1_y)
        term4 = -(theta * k / np.pi) * beta_val * (1 - F_2k_2c)
        
        return np.mean(term1 + term2 + term3 + term4)

    def fit(self, data: np.ndarray) -> "CSGDFitter":
        """
        Fit CSGD parameters by minimizing CRPS.

        Parameters
        ----------
        data : np.ndarray
            1D non-negative observations (e.g., precipitation)
        """
        data = np.clip(np.asarray(data), 0, None)

        # Initialization
        mu_init = np.mean(data) 
        sigma_init = np.std(data)
        if mu_init < 1e-3: mu_init = 0.1
        if sigma_init < 1e-3: sigma_init = 0.1
        p_zero_obs = np.mean(data == 0)
        delta_init = -0.1 * mu_init if p_zero_obs > 0 else 0.0
        x0 = [mu_init, sigma_init, delta_init]
        # Constraints and bounds
        constraints = [{'type': 'ineq', 'fun': lambda x: x[0] + x[2] + 1e-5}]
        bounds = [(1e-3, None), (1e-3, None), (None, 0.0)]

        res = minimize(
            self._csgd_crps_loss,
            x0,
            args=(data,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-6}
        )

        if not res.success:
            warnings.warn(f"CSGD fitting failed: {res.message}")

        self.mu_, self.sigma_, self.delta_ = res.x
        self.k_, self.theta_ = self.get_gamma_params(self.mu_, self.sigma_)
        self.is_fitted = True
        return self

    def cdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Compute CDF of fitted CSGD."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before using cdf()")
        val = (x - self.delta_) / self.theta_
        val = np.maximum(val, 0)
        cdf_vals = sp.gammainc(self.k_, val)
        return np.where(x < 0, 0.0, cdf_vals)

    def ppf(self, q: Union[float, np.ndarray]) -> np.ndarray:
        """Compute PPF / inverse CDF of fitted CSGD."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before using ppf()")
        q = np.asarray(q)
        q = np.clip(q, 1e-6, 1-1e-6)
        p_zero = sp.gammainc(self.k_, -self.delta_ / self.theta_)
        result = np.zeros_like(q, dtype=float)
        # Case 1: q <= p_zero -> 降水量为 0
        mask_zero = q <= p_zero
        result[mask_zero] = 0.0
        # Case 2: q > p_zero -> 反解 Gamma
        # q = gammainc(k, (y - delta)/theta)
        # => (y - delta)/theta = gammaincinv(k, q)
        mask_pos = ~mask_zero
        if np.any(mask_pos):
            inv_vals = sp.gammaincinv(self.k_, q[mask_pos])
            result[mask_pos] = self.theta_ * inv_vals + self.delta_
            
        return result

    def pdf(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Compute PDF of fitted CSGD (x>0 part)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before using pdf()")
        x = np.asarray(x)
        pdf_vals = np.zeros_like(x, dtype=float)
        mask_pos = x > 0
        if np.any(mask_pos):
            #  scipy.stats.gamma.pdf
            # scale=theta, loc=delta
            pdf_vals[mask_pos] = stats.gamma.pdf(
                x[mask_pos], 
                a=self.k_, 
                scale=self.theta_, 
                loc=self.delta_
            )
            
        return pdf_vals

    def probability_at_zero(self) -> float:
        """Return model-predicted probability of zero precipitation."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return float(sp.gammainc(self.k_, -self.delta_ / self.theta_))

    def rvs(self, size: int = 1) -> np.ndarray:
        """Generate random samples from fitted CSGD."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        raw_gamma = np.random.gamma(self.k_, self.theta_, size=size)
        return np.maximum(raw_gamma + self.delta_, 0.0)

    def evaluate_score(self, data: np.ndarray) -> float:
        """Compute mean CRPS for given data under current fit."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self._csgd_crps_loss(np.array([self.mu_, self.sigma_, self.delta_]), data)

    def qq_confidence_band(
        self,
        n_obs: int,
        n_sim: int = 1000,
        alpha: float = 0.05,
        random_state: Optional[int] = None
    ):
        """
        Monte Carlo QQ-plot confidence band for fitted CSGD.

        Parameters
        ----------
        n_obs : int
            Number of observations in the sample.
        n_sim : int
            Number of Monte Carlo simulations.
        alpha : float
            Significance level (default 0.05 for 95% CI).
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        lower : np.ndarray
        upper : np.ndarray
            Pointwise confidence band for order statistics.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        if random_state is not None:
            np.random.seed(random_state)

        raw = np.random.gamma(
            shape=self.k_,
            scale=self.theta_,
            size=(n_sim, n_obs)
        )
        sim = np.maximum(raw + self.delta_, 0.0)
        sim.sort(axis=1)

        lower = np.percentile(sim, 100 * alpha / 2, axis=0)
        upper = np.percentile(sim, 100 * (1 - alpha / 2), axis=0)

        return lower, upper

    def predict_climatology(self, n_members: int = 500, mode: str = 'quantile') -> np.ndarray:
        """
        Generate climatological forecast samples using fitted CSGD parameters.

        Parameters
        ----------
        n_members : int
            Number of ensemble members (default 500)
        mode : str
            'quantile' - equally spaced quantiles (benchmark)
            'random' - random samples

        Returns
        -------
        np.ndarray
            Climatological forecast samples of shape (n_members,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if mode == 'quantile':
            quantiles = np.linspace(1 / (n_members + 1), n_members / (n_members + 1), n_members)
            return self.ppf(quantiles)
        elif mode == 'random':
            return self.rvs(size=n_members)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'quantile' or 'random'.")