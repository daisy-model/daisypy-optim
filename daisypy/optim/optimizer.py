# pylint: disable=wrong-import-position
available_optimizers = {}
try:
    from daisypy.optim.cma_optimizer import DaisyCMAOptimizer
    available_optimizers["cma"] = DaisyCMAOptimizer
except ImportError:
    pass

try:
    from daisypy.optim.ax_optimizer import DaisyAxOptimizer, AxResult #pylint: disable=unused-import
    available_optimizers["ax"] = DaisyAxOptimizer
except ImportError:
    pass

from daisypy.optim.sequential_optimizer import DaisySequentialOptimizer
available_optimizers["sequential"] = DaisySequentialOptimizer
