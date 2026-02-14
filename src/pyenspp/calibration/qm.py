# calibration/qm.py
from .base import BaseCorrector
from .marginal_fitter import CSGDFitter
import numpy as np
from typing import Optional, Any, Tuple
import copy

class QuantileMapping(BaseCorrector):
    """
    Quantile Mapping (QM) corrector based on CSGD.

    Assumes precipitation follows CSGD and handles zero and positive values
    in a unified parametric framework. No threshold specification is needed.

    Principle:
        x_corr = F_obs^{-1}(F_fcst(x_fcst))
    where F and F^{-1} are the CSGD CDF and PPF.

    Usage:
        qm = QuantileMapping()
        qm.fit(obs, fcst)
        corrected = qm.correct(new_fcst)
    """

    def __init__(
        self, 
        base_fitter: Optional[Any] = None, 
        random_state: Optional[int] = None
    ):
        """
        Parameters
        ----------
        base_fitter : object
            Prototype fitter implementing fit/cdf/ppf.
            Cloned for obs and forecast fitting. Default: CSGDFitter().
        random_state : int, optional
            Random seed.
        """
        super().__init__()
        self.random_state = random_state
        self.base_fitter = base_fitter if base_fitter is not None else CSGDFitter()
        self._fitter_obs = None
        self._fitter_fcst = None

    def fit(self, obs: np.ndarray, fcst: np.ndarray) -> "QuantileMapping":
        """
        Fit CSGD parameters for observation and forecast.

        Parameters
        ----------
        obs : np.ndarray, shape (n,)
            Historical observations.
        fcst : np.ndarray, shape (m,)
            Historical forecasts (length may differ from obs).
        """
        if obs.ndim != 1 or fcst.ndim != 1:
            raise ValueError("obs and fcst must be 1D arrays")

        # Clone base fitter for obs and forecast
        self._fitter_obs = copy.deepcopy(self.base_fitter)
        self._fitter_fcst = copy.deepcopy(self.base_fitter)

        self._fitter_obs.fit(obs)
        self._fitter_fcst.fit(fcst)

        self.is_fitted = True
        return self

    def correct(self, fcst: np.ndarray) -> np.ndarray:
        """
        Apply quantile mapping correction.

        Parameters
        ----------
        fcst : np.ndarray
            Forecast values to correct.

        Returns
        -------
        np.ndarray
            Corrected forecast.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before correct()")
        
        fcst = np.asarray(fcst)
        if fcst.ndim != 1:
            raise ValueError("fcst must be 1D array")
        
        # Compute CDF in forecast distribution
        u = np.clip(self._fitter_fcst.cdf(fcst), 1e-9, 1-1e-9)
        
        # Map probability back using obs PPF
        corrected = self._fitter_obs.ppf(u)
        return corrected
    
    def get_params(self) -> Tuple[tuple, tuple]:
        """
        Return fitted parameters for analysis.

        Returns
        -------
        (obs_params, fcst_params)
            obs_params: (mu, sigma, delta) for observation
            fcst_params: (mu, sigma, delta) for forecast
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        return (self._fitter_obs.params, self._fitter_fcst.params)

    def cross_validate_ensemble(
        self,
        obs: np.ndarray,
        fcst_ens: np.ndarray,
        random_state: Optional[int] = None
    ) -> np.ndarray:
        """
        Leave-One-Out cross-validation for ensemble QM correction.

        Parameters
        ----------
        obs : np.ndarray, shape (n,)
            Historical observations.
        fcst_ens : np.ndarray, shape (n, m)
            Historical ensemble forecasts.

        Returns
        -------
        np.ndarray
            Corrected ensemble forecasts (LOO-CV).
        """
        obs = np.asarray(obs)
        fcst_ens = np.asarray(fcst_ens)
        if obs.ndim != 1 or fcst_ens.ndim != 2:
            raise ValueError("obs must be 1D, fcst_ens must be 2D")
        
        n_days, n_members = fcst_ens.shape
        corrected_ens = np.full_like(fcst_ens, np.nan)

        # Set random seed if provided
        seed = random_state if random_state is not None else self.random_state
        if seed is not None:
            np.random.seed(seed)

        for i in range(n_days):
            # Training set excluding the i-th day
            trn_obs = np.delete(obs, i)
            trn_fcst = np.delete(fcst_ens, i, axis=0).flatten()

            # Fit new QM instance
            qm_cv = QuantileMapping(base_fitter=self.base_fitter, random_state=seed)
            qm_cv.fit(trn_obs, trn_fcst)

            # Correct the i-th day's forecast
            corrected_ens[i, :] = qm_cv.correct(fcst_ens[i, :])

        return corrected_ens