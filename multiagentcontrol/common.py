from typing import Any, Callable, List
import numpy as np
import matplotlib.pyplot as plt
import scipy

__epsilon = 1e-4

def is_scalar(val: Any) -> bool:
    """Check if a variable is a scalar

    Args:
        val (Any): variable to be checked

    Returns:
        bool: True if val is a scalar
    """
    try:
        _ = list(val)
    except TypeError:
        return True
    return False

def is_integer(val: Any) -> bool:
    """Check if a variable is an integer

    Args:
        val (Any): variable to be checked

    Returns:
        bool: True if val is an integer
    """
    return isinstance(val, int)

def length(val: Any) -> int:
    """Return length of the variable

    Args:
        val (Any): variable to be checked

    Returns:
        int: length of val
    """
    try:
        return len(val)
    except TypeError:
        return 1

def calculate_gradient(f: Callable, x: List[float]) -> np.array:
    """Return the gradient of a function at a given point

    Args:
        f (Callable): function in consideration
        x (List[float]): point at which gradient is evaluated

    Returns:
        np.array: gradient of f(x)
    """
    dim = length(x)
    gradient = np.zeros(dim)
    for i in range(dim):
        xi = x[i]
        hi = np.abs(xi) * __epsilon
        if hi == 0:
            hi = __epsilon
        ei = np.zeros(dim)
        ei[i] = 1
        gradient[i] = (f(x + hi*ei) - f(x - hi*ei))/(2 * hi)
    return gradient

def calculate_hessian(f: Callable, x: List[float]) -> np.array:
    """Return the hessian matrix of a function at a given point

    Args:
        f (Callable): function in consideration
        x (List[float]): point at which hessian is evaluated

    Returns:
        np.array: hessian matrix of f(x)
    """
    dim = length(x)
    hessian = np.zeros((dim, dim))
    for i in range(dim):
        xi = x[i]
        absxi = np.abs(xi)
        hi = absxi * __epsilon if absxi > 0 else __epsilon
        ei = np.zeros(dim)
        ei[i] = 1
        for j in range(dim):
            if i == j:
                hessian[i,i] = (-f(x + 2*hi*ei) + 16*f(x + hi*ei) - 30*f(x) + 16*f(x - hi*ei) - f(x - 2*hi*ei))/(12*(hi ** 2))
            else:
                xj = x[j]
                absxj = np.abs(xj)
                hj = absxj * __epsilon if absxj > 0 else __epsilon
                ej = np.zeros(dim)
                ej[j] = 1
                hessian[i,j] = (f(x + hi*ei + hj*ej) - f(x + hi*ei - hj*ej) - f(x - hi*ei + hj*ej) + f(x - hi*ei - hj*ej))/(4 * hi * hj)

    return 0.5 * (hessian + np.transpose(hessian))

def calculate_convex_optimal(f: Callable, dim: int) -> np.array:
    """Return the optimal solution to an unconstrained convex minimisation problem

    Args:
        f (Callable): objective function of the minimisation problem
        dim (int): dimension of the function output

    Returns:
        np.array: optimal solution
    """
    x = np.random.uniform(size=(dim))
    return np.array(scipy.optimize.minimize(fun=f, x0=x)['x'])

def lowpass_filter(arr: List[float], alpha: float = 0.8) -> List[float]:
    """First order low pass filter for timeseries data

    Args:
        arr (List[float]): timeseries data to be filtered
        alpha (float, optional): filter parameter, alpha = 1 means no filtering. Defaults to 0.8.

    Returns:
        List[float]: filtered data
    """
    data = np.copy(arr)
    for i in range(1, length(data)):
        data[i] = (1-alpha) * data[i-1] + alpha * data[i]
    return data

def show_plots():
    """Alias function for plt.show() from matplotlib
    """
    plt.show()