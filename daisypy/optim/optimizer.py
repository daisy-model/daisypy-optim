# pylint: disable=wrong-import-position
available_optimizers = {}
try:
    from daisypy.optim.cma_optimizer import DaisyCMAOptimizer
    available_optimizers["cma"] = DaisyCMAOptimizer
except ImportError:
    pass

try:
    from daisypy.optim.skopt_optimizer import DaisySkoptOptimizer
    available_optimizers["skopt"] = DaisySkoptOptimizer
except ImportError:
    pass

from daisypy.optim.sequential_optimizer import DaisySequentialOptimizer
available_optimizers["sequential"] = DaisySequentialOptimizer
