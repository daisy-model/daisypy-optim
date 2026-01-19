import os
from .log import Log
from .formatters import quote_if_string

class CsvLog(Log):
    '''A file backed csv log.'''

    def __init__(self, path, columns, default_formatter=quote_if_string):
        '''
        Parameters
        ----------
        path: str
          Where to store log

        columns : list of string or dict of (str, callable)
          Column names and formats. If a list use default format function for all columns (`str`).

        default_formatter : callable [Object -> str]
          Default function for formatting column values. If None use self.quote_if_string
        '''
        self._log = open(path, 'w', encoding='utf-8')
        if not isinstance(columns, dict):
            self.columns = { col : default_formatter for col in columns }
        else:
            self.columns = { k : default_formatter if v is None else v for k,v in columns.items() }
        self._write(','.join(self.columns.keys()), True)

    def log(self, flush=True, **kwargs):
        '''Log a row.

        Parameters
        ----------
        flush : Bool
          If True flush the log after writing.

        **kwargs :
          Must contain the columns defined when constructing the log
        '''
        if self._log.closed:
            raise RuntimeError('Writing to closed CsvLog')
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
        self._log.flush()
        os.fsync(self._log.fileno())
