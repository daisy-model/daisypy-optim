from abc import ABC, abstractmethod

class Log(ABC):
    '''Log interface'''
    @abstractmethod
    def log(self, *args, **kwargs):
        '''Handle any kind of log mesage'''

    @abstractmethod
    def close(self):
        '''Explicitly close the log'''

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
