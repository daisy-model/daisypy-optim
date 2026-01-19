import os
from daisypy.optim.terminal_log import TerminalLog
from daisypy.optim.csv_log import CsvLog
from daisypy.optim.logger import Logger

available_logs = {
    "terminal" : TerminalLog,
    "csv" : CsvLog
}

def DefaultLogger(outdir):
    logs = {
        'default' : TerminalLog(),
        'warning' : TerminalLog(error=True),
        'error' : TerminalLog(error=True),
        'result' : CsvLog(os.path.join(outdir, 'result.csv')),
    }
    return Logger(**logs)
