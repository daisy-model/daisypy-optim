"""Scalar loss functions.
Signature is (actual : numpy.ndarray, target : numpy.ndarray) -> float
"""

def mse(actual, target):
    """Mean squared error"""
    return ((actual - target)**2).mean()

def mae(actual, target):
    """Mean absolute error"""

available_loss_fns = {
    "mse" : mse,
    "mae" : mae
}
