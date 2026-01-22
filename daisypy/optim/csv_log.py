import os
from .log import Log
from .formatters import quote_if_string

class CsvLog(Log):
    '''A file backed csv log.'''

    def __init__(self, path, columns=None, default_formatter=quote_if_string):
        '''
        Parameters
        ----------
        path: str
          Where to store log

        columns : list of string or dict of (str, callable)
          Column names and formats. If a list use default format function for all columns (`str`).
          If None determine columns from first call to self.log

        default_formatter : callable [Object -> str]
          Default function for formatting column values. If None use self.quote_if_string
        '''
        _dir = os.path.dirname(path)
        os.makedirs(_dir, exist_ok=True)
        self._log = open(path, 'w', encoding='utf-8')
        if columns is None:
            # Deferred setting of columns such that they can be set on first write
            self.columns = None
            self.default_formatter = default_formatter
        else:
            self._setup_columns(columns, default_formatter)

    def log(self, flush=True, **kwargs):
        '''Log a row.

        Parameters
        ----------
        flush : Bool
          If True flush the log after writing.

        **kwargs : dict
          If the log has a column specification, then this dict must contain the columns defined
          there. Otherwise the dict keys are used to create a column specification.
        '''
        if self._log.closed:
            raise RuntimeError('Writing to closed CsvLog')
        if self.columns is None:
            self._setup_columns(list(kwargs.keys()))
        row = []
        for col, formatter in self.columns.items():
            row.append(formatter(kwargs[col]))
        self._write(','.join(row), flush)

    def close(self):
        '''Close the underlying file'''
        self._log.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _write(self, msg, flush):
        self._log.write(msg + "\n")
        if flush:
            self.flush()

    def flush(self):
        '''Flush the log so it is written to disk'''
        self._log.flush()
        os.fsync(self._log.fileno())

    def _setup_columns(self, columns, default_formatter=None):
        if default_formatter is None:
            default_formatter = self.default_formatter
        if not isinstance(columns, dict):
            self.columns = { col : default_formatter for col in columns }
        else:
            self.columns = {
                k : default_formatter if v is None else v for k,v in columns.items()
            }
        self._write(','.join(self.columns.keys()), True)
