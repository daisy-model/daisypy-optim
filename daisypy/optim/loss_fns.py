"""Loss functions."""

def mse(actual, target):
    """Mean squared error"""
    return ((actual - target)**2).mean()

available_loss_fns = {
    "mse" : mse
}