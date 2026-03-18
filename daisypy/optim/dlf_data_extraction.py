# pylint: disable=too-few-public-methods
'''Classes for extracting and procesing data in Daisy log files'''
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
import pandas as pd
from daisypy.io.dlf import read_dlf

class DlfDataExtractor:
    '''Class for extracting data from Daisy log files (dlf)'''
    def __init__(self, logs_and_variables, post_processor=None):
        '''
        Parameters
        ----------
        logs_and_variables : dict of [str, [str]]
          Map from log names to variable names in the log that should be extracted

        post_processor : Callable implementing DlfPostProcessor interface or None
          Function mapping extracted data to a single pandas.DataFrame with columns 'time' and
          'value'. It is called with a list of pd.DataFrames with a "time" column and one or more
          value columns named with '<log-name>/<var-name>'.
          If None then logs_and_variables must contain exactly one key, and that key must map to a
          list of length 1 or a single str
        '''
        if not isinstance(logs_and_variables, Mapping):
            raise ValueError("`logs_and_variables` must be a Mapping type")
        self.logs_and_variables = logs_and_variables
        if post_processor is None:
            keys = list(self.logs_and_variables.keys())
            msg = ("When no `post_processor` is provided, `logs_and_variables` must contain "
                "exactly one key, and that key must map to a list of length 1 or a single str")
            if len(keys) != 1:
                raise ValueError(msg)
            val = self.logs_and_variables[keys[0]]
            if not isinstance(val, str) and len(val) != 1:
                raise ValueError(msg)
            self.post_processor = DlfSingleton()
        else:
            self.post_processor = post_processor

    def __call__(self, daisy_output_directory):
        '''Extract data from log files in the given directory

        Returns
        -------
        pandas.DataFrame with columns 'time' and 'value'
        '''
        out_dir = Path(daisy_output_directory)
        dfs = []
        for log_name, var_names in self.logs_and_variables.items():
            if isinstance(var_names, str):
                var_names = [var_names]
            dlf = read_dlf(out_dir / log_name)
            dlf.body['time'] = pd.to_datetime(
                dlf.body[['year', 'month', 'mday', 'hour']].rename(columns={'mday' : 'day'})
            )
            dfs.append(dlf.body[['time'] + var_names].rename(columns={
                var_name : f'{log_name}/{var_name}' for var_name in var_names
            }))
        processed = self.post_processor(dfs)
        if set(processed.columns) != {'time', 'value'}:
            raise RuntimeError(
                "`post_processor` must return a DataFrame with columns 'time' and 'value'"
            )
        return processed


class DlfPostProcessor(ABC):
    '''Interface for post processors'''
    @abstractmethod
    def __call__(self, data_frames):
        '''
        Parameters
        ----------
        data_frames : [pandas.DataFrame]
          A list of data frames containing a 'time' column and one or more value columns

        Returns
        -------
        pandas.DataFrame with columns 'time' and 'value'
        '''

class DlfSum(DlfPostProcessor):
    '''Compute sum of all extracted variables'''
    def __call__(self, data_frames):
        df = data_frames[0]
        for other in data_frames[1:]:
            df = pd.merge(df, other, on='time', validate='1:1')
        cols = [col for col in df.columns if col != 'time']
        return pd.DataFrame({'time' : df['time'], 'value' : df[cols].sum(axis=1)})

class DlfSingleton(DlfPostProcessor):
    '''Unpack a single extracted variable'''
    def __call__(self, data_frames):
        df = data_frames[0]
        value_col = [col for col in df.columns if col != 'time'][0]
        return df[['time', value_col]].rename(columns={value_col : 'value'})
