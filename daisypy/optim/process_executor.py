import os
from concurrent.futures import ProcessPoolExecutor

class DaisyProcessExecutor(ProcessPoolExecutor):
    '''Wrapper around ProcessPoolExecutor that stores max_processes so it can be used for scheduling
    '''
    def __init__(self, max_processes=None):
        if max_processes is None:
            self.max_processes = os.process_cpu_count()
        else:
            self.max_processes = max_processes
        super().__init__(max_processes)
