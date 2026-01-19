import sys
from .log import Log

class TerminalLog(Log):
    '''A simple terminal log that prints to stdout or stderr'''

    def __init__(self, error=False):
        self.is_open = True
        self.out = sys.stderr if error else sys.stdout

    def log(self, *args, **kwargs):
        if self.is_open:
            args_msg = ','.join([str(a) for a in args])
            kwargs_msg = ','.join([f'{k}={v}' for k,v in kwargs.items()])
            if len(args_msg) > 1 and len(kwargs_msg) > 1:
                msg = args_msg + ',' + kwargs_msg
            else:
                msg = args_msg + kwargs_msg
            print(msg, file=self.out)

    def close(self):
        self.is_open = False

    def __del__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
