import numpy as np
def exponential_halflife(length: int, half_life: int) -> np.ndarray:
    """
    exponential halflife decay function of certain lenghth.
    Args:
        length: length of the decay
        half_life: half life of the decay
    Returns:
        np.ndarray: exponential decay function. it will start with small values and end with 1
    Examples:
    a = exponential_halflife(10, 3)
    print(a)
    Output:
        ndarray:
        0,0.125000
        1,0.157490
        2,0.198425
        3,0.250000
        4,0.314980
        5,0.396850
        6,0.500000
        7,0.629961
        8,0.793701
        9,1.000000
    """
    return np.exp(-np.log(2) / half_life * np.arange(length))[::-1]


def wls(x, y, w, intercept=False):
    """
    Weighted least squares regression for multivariate x, including R^2 and residuals.

    Args:
        x (list of lists or numpy.ndarray): 2D list or array where each inner list or
        row represents a single observation's features.
        y (list or numpy.ndarray): Output variable values, one for each observation.
        w (list or numpy.ndarray): Weights, one for each observation.
        intercept (bool): Whether to include an intercept in the model. Default is False.
    Returns:
        dict: A dictionary containing coefficients, intercept, R^2, and residuals.
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)

    # Check if lengths of x, y, and w are the same
    if len(x) != len(y) or len(y) != len(w):
        raise ValueError("The lengths of x, y, and w must be the same")

    # Ensure x is two-dimensional (for a single predictor case, it should still work)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Create diagonal matrix for weights
    W = np.diag(w)

    # Reshape y as a column vector
    Y = y.reshape(-1, 1)
    if intercept:
        # Augment x with a column of ones for intercept
        X = np.hstack([np.ones((len(x), 1)), x])
    else:
        X = x

    # Compute X'WX and X'WY
    XTWX = X.T @ W @ X
    XTWY = X.T @ W @ Y

    try:
        # Solve for beta (coefficients)
        beta = np.linalg.solve(XTWX, XTWY)
        beta = beta.flatten()  # Flatten the array to 1D

        # Calculate fitted values
        y_fitted = X @ beta.reshape(-1, 1)

        # Calculate residuals
        residuals = Y - y_fitted

        # Calculate weighted R^2
        SS_res = residuals.T @ W @ residuals
        SS_tot = (Y - np.mean(Y)).T @ W @ (Y - np.mean(Y))
        r_squared = 1 - (SS_res / SS_tot).item()  # Extract scalar value
        if intercept:
            return {
                "coefficients": beta[1:],  # coefficients for predictors
                "intercept": beta[0],  # intercept
                "R2": r_squared,  # R^2 value
                "residuals": residuals.flatten(),  # residuals
            }
        else:
            return {
                "coefficients": beta,  # coefficients for predictors
                "R2": r_squared,  # R^2 value
                "residuals": residuals.flatten(),  # residuals
            }
    except np.linalg.LinAlgError:
        raise "Matrix is singular and cannot be inverted."


def ols(x, y, intercept=False):
    """
    Ordinary least squares regression for multivariate x, including R^2 and residuals.

    Args:
        x (list of lists or numpy.ndarray): 2D list or array where each inner list or
         row represents a single observation's features.
        y (list or numpy.ndarray): Output variable values, one for each observation.
        intercept (bool): Whether to include an intercept in the model. Default is False.
    Returns:
        dict: A dictionary containing coefficients, intercept, R^2, and residuals.
    """
    # Convert inputs to numpy arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)

    # Ensure x is two-dimensional (for a single predictor case, it should still work)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if intercept:
        # Augment x with a column of ones for intercept
        X = np.hstack([np.ones((len(x), 1)), x])
    else:
        X = x

    # Compute X'X and X'Y
    XTX = X.T @ X
    XTY = X.T @ y.reshape(-1, 1)

    try:
        # Solve for beta (coefficients)
        beta = np.linalg.solve(XTX, XTY)
        beta = beta.flatten()  # Flatten the array to 1D

        # Calculate fitted values
        y_fitted = X @ beta.reshape(-1, 1)

        # Calculate residuals
        residuals = y.reshape(-1, 1) - y_fitted

        # Calculate R^2
        SS_res = residuals.T @ residuals
        SS_tot = (y.reshape(-1, 1) - np.mean(y)).T @ (y.reshape(-1, 1) - np.mean(y))
        r_squared = 1 - (SS_res / SS_tot).item()  # Extract scalar value
        if intercept:
            return {
                "coefficients": beta[1:],  # coefficients for predictors
                "intercept": beta[0],  # intercept
                "R2": r_squared,  # R^2 value
                "residuals": residuals.flatten(),  # residuals
            }
        else:
            return {
                "coefficients": beta,  # coefficients for predictors
                "R2": r_squared,  # R^2 value
                "residuals": residuals.flatten(),  # residuals
            }
    except np.linalg.LinAlgError:
        raise "Matrix is singular and cannot be inverted."
