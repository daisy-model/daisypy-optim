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
        self.path = path
        self._log = open(path, 'w', encoding='utf-8') # pylint: disable=consider-using-with
        if columns is None:
            # Deferred setting of columns such that they can be set on first write
            self.columns = None
            self.default_formatter = default_formatter
        else:
            self._setup_columns(columns, default_formatter)

    def log(self, *args, flush=True, **kwargs):
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
        if len(args) > 0:
            raise RuntimeError('CsvLog requires all log arguments to be passed as keywords')
        if self.columns is None:
            self._setup_columns(list(kwargs.keys()))
        row = []
        for col, formatter in self.columns.items():
            row.append(formatter(kwargs[col]))
        self._write(','.join(row), flush)

    def log_rows(self, rows, flush=True):
        '''Log many rows.

        Parameters
        ----------
        rows : list of dict
          List of rows to log. It is assumed each row has the same keys

        flush : Bool
          If True flush the log after writing.
        '''
        if self._log.closed:
            raise RuntimeError('Writing to closed CsvLog')
        rows = list(rows)
        if len(rows) == 0:
            return
        if self.columns is None:
            self._setup_columns(list(rows[0].keys()))
        lines = []
        for row in rows:
            line = []
            for col, formatter in self.columns.items():
                line.append(formatter(row[col]))
            lines.append(','.join(line))
        self._write('\n'.join(lines), flush)

    def close(self):
        '''Close the underlying file'''
        if not self._log.closed:
            self.persist()
            self._log.close()

    def persist(self):
        '''Force write to disk'''
        self.flush(True)

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

    def flush(self, durable=False):
        '''Flush buffered data; optionally force it to disk'''
        self._log.flush()
        if durable:
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
