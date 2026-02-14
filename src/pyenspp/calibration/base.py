# calibration/base.py
from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np

class ProbabilisticFitter(Protocol):
    """Interface that all fitters must implement."""
    
    def fit(self, data: np.ndarray) -> "ProbabilisticFitter": ...
    def cdf(self, x: np.ndarray) -> np.ndarray: ...
    def ppf(self, q: np.ndarray) -> np.ndarray: ...

class BaseCorrector(ABC):
    """
    Base class for all precipitation correctors.
    Enforces implementation of fit and correct methods in subclasses.
    """

    def __init__(self):
        self.is_fitted = False  # Flag indicating whether the model has been fitted

    @abstractmethod
    def fit(self, obs: np.ndarray, fcst: np.ndarray) -> None:
        """
        Fit the correction model using historical observations and forecasts.

        Parameters:
            obs: Observed data, shape (N_obs,)
            fcst: Forecast data, shape (N_fcst,)
        """
        pass

    @abstractmethod
    def correct(self, fcst: np.ndarray) -> np.ndarray:
        """
        Correct new forecast data.

        Parameters:
            fcst: Forecast data to be corrected, shape (M,)

        Returns:
            corrected_fcst: Corrected forecast data, shape (M,)
        """
        pass