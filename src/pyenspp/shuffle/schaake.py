import numpy as np

def schaake_shuffle(X, Y, axis=1, random_state=42):
    """
    Schaake Shuffle with tie-breaking jitter.

    Parameters
    ----------
    X : ndarray
        Forecast ensemble to be reordered.
    Y : ndarray
        Reference data providing rank structure.
    axis : int, default=1
        Axis along which to reorder (1=ensemble members).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Reordered forecast ensemble.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    if X.shape != Y.shape:
        raise ValueError(f"Shape mismatch: X{X.shape} vs Y{Y.shape}")
    
    rng = np.random.default_rng(random_state)
    jitter = rng.uniform(-1e-9, 1e-9, size=Y.shape)  # break ties
    Y_grading = Y + jitter
    
    X_sorted = np.sort(X, axis=axis)
    indices = np.argsort(np.argsort(Y_grading, axis=axis), axis=axis)
    X_shuffled = np.take_along_axis(X_sorted, indices, axis=axis)
    
    return X_shuffled


def schaake_shuffle_block(X, Y, axis=1, random_state=None, jitter_eps=1e-6):
    """
    Block-based Schaake Shuffle preserving X distribution.

    Parameters
    ----------
    X : ndarray
        Forecast data to reorder.
    Y : ndarray
        Reference block structure (used to determine ranks).
    axis : int
        Axis along which to apply blocks (0=row, 1=column).
    random_state : int or None
        Random seed for tie-breaking.
    jitter_eps : float
        Small random perturbation to break ties.

    Returns
    -------
    ndarray
        Reordered X preserving rank correlation per block.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    result = X.copy()
    rng = np.random.default_rng(random_state)

    # Axis 0: block along rows
    if axis == 0:
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have same number of columns")
        block = Y.shape[0]
        if X.shape[0] % block != 0:
            raise ValueError("X.shape[0] must be divisible by Y.shape[0]")
        
        Y_grading = np.where(np.isnan(Y), np.nan, Y + rng.uniform(-jitter_eps, jitter_eps, Y.shape))
        indices = np.argsort(np.argsort(Y_grading, axis=0), axis=0)
        for i in range(0, X.shape[0], block):
            X_block = X[i:i+block, :]
            result[i:i+block, :] = np.take_along_axis(np.sort(X_block, axis=0), indices, axis=0)

    # Axis 1: block along columns
    elif axis == 1:
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have same number of rows")
        block = Y.shape[1]
        if X.shape[1] % block != 0:
            raise ValueError("X.shape[1] must be divisible by Y.shape[1]")
        
        Y_grading = np.where(np.isnan(Y), np.nan, Y + rng.uniform(-jitter_eps, jitter_eps, Y.shape))
        indices = np.argsort(np.argsort(Y_grading, axis=1), axis=1)
        for j in range(0, X.shape[1], block):
            X_block = X[:, j:j+block]
            result[:, j:j+block] = np.take_along_axis(np.sort(X_block, axis=1), indices, axis=1)
    else:
        raise ValueError("axis must be 0 or 1")

    return result