class Logger:
    """A logger that handles a number of logs.
    Use as context manager OR call close method explicitly OR ensure the object is destroyed
    """
    def __init__(self, **kwargs):
        self.logs = kwargs

    def add_log(self, name, log):
        '''Add a named log

        Parameters
        ----------
        name : str
          Name of log

        log : daisypy.optim.Log
          A Log instance

        Raises
        ------
        ValueError if a log with the given name already exists
        '''
        if name in self.logs:
            raise ValueError(f"Log {name} already exists")
        self.logs[name] = log

    def info(self, *args, **kwargs):
        '''Log to the info log'''
        self.log('info', *args, **kwargs)

    def warning(self, *args, **kwargs):
        '''Log to the warning log'''
        self.log('warning', *args, **kwargs)

    def error(self, *args, **kwargs):
        '''Log to the error log'''
        self.log('error', *args, **kwargs)

    def parameters(self, *args, **kwargs):
        '''Log to the parameters log'''
        self.log('parameters', *args, **kwargs)

    def samples(self, *args, **kwargs):
        '''Log to the samples log'''
        self.log('samples', *args, **kwargs)

    def outcome(self, *args, **kwargs):
        '''Log extracted prediction rows to the outcome log.'''
        self.log('outcome', *args, **kwargs)

    def target(self, *args, **kwargs):
        '''Log target rows to the target log.'''
        self.log('target', *args, **kwargs)

    def log(self, name, *args, **kwargs):
        '''Log to a specific named log. Fallback to `default` log if the named log does not exist

        Parameters
        ----------
        name : str
          Name of log

        Raises
        ------
        ValueError if no log with the given name exists AND no default log is defined
        '''
        if name not in self.logs:
            if 'default' in self.logs:
                self.logs['default'].log(*args, **kwargs)
            else:
                raise ValueError(f"No log named {name} exists")
        else:
            self.logs[name].log(*args, **kwargs)

    def log_rows(self, name, rows, **kwargs):
        '''Log many rows to a specific named log. Fallback to `default` log if the named log does
        not exist

        Parameters
        ----------
        name : str
          Name of log

        rows : [{}]
          List of rows. Each row is a dict. It is assume keys are the same for all rows

        Raises
        ------
        ValueError if no log with the given name exists AND no default log is defined
        '''
        if name not in self.logs:
            if 'default' in self.logs:
                log = self.logs['default']
            else:
                raise ValueError(f"No log named {name} exists")
        else:
            log = self.logs[name]
        if hasattr(log, 'log_rows'):
            log.log_rows(rows, **kwargs)
        else:
            for row in rows:
                log.log(**row, **kwargs)

    def close(self):
        '''Close all logs for writing'''
        for log in self.logs.values():
            log.close()

    def persist(self):
        '''Force log data to disk'''
        for log in self.logs.values():
            if hasattr(log, 'persist'):
                log.persist()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
