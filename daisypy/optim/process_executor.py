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
    count = None
    if hasattr(os, 'process_cpu_count'):
        count = os.process_cpu_count()
    elif hasattr(os, 'sched_getaffinity'):
        count = len(os.sched_getaffinity(0))
    return count or os.cpu_count() or 1
