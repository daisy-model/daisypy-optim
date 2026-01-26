import os
from daisypy.optim.terminal_log import TerminalLog
from daisypy.optim.csv_log import CsvLog
from daisypy.optim.logger import Logger

available_logs = {
    "terminal" : TerminalLog,
    "csv" : CsvLog
}

def DefaultLogger(outdir): # pylint: disable=invalid-name # (It should look like a class)
    '''Return a logger instance with the following predefined logs.

      'default' : Log to stdout
      'warning', 'error' : Log to stderr
      'parameters' : Log to csv file 'parameters.csv' in outdir
      'result' : Log to csv file 'retult.csv' in outdir

    Parameters
    ----------
    outdir : str
      Directory to store log files in

    Returns
    -------
    daisypy.optim.Logger
    '''
    logs = {
        'default' : TerminalLog(),
        'warning' : TerminalLog(error=True),
        'error' : TerminalLog(error=True),
        'parameters' : CsvLog(os.path.join(outdir, 'parameters.csv')),
        'result' : CsvLog(os.path.join(outdir, 'result.csv')),
    }
    return Logger(**logs)
