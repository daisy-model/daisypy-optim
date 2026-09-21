import os
from concurrent.futures import ProcessPoolExecutor

class DaisyProcessExecutor(ProcessPoolExecutor):
    '''Wrapper around ProcessPoolExecutor that stores max_processes so it can be used for scheduling
    '''
    def __init__(self, max_processes=None):
        if max_processes is None:
            self.max_processes = _process_cpu_count()
        else:
            self.max_processes = max_processes
        super().__init__(self.max_processes)

def _process_cpu_count():
    # pylint: disable=no-member
    # os.process_cpu_count is available from python 3.13
    if hasattr(os, 'process_cpu_count'):
        return os.process_cpu_count()
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()
