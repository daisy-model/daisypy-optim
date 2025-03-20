available_loggers = {}
try:
    from daisypy.optim.tensorboard_logger import TensorBoardLogger
    available_loggers["tensorboard"] = TensorBoardLogger
except ImportError:
    pass
from daisypy.optim.csv_logger import CsvLogger
available_loggers["csv"] = CsvLogger
