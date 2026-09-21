import os
from daisypy.optim.process_executor import DaisyProcessExecutor


def test_max_processes_set():
    '''Test that DaisyProcessExecutor respects max_processes argument'''
    with DaisyProcessExecutor(3) as e:
        assert e.max_processes == 3

def test_max_processes_not_set():
    '''Test the DaisyProcessExecutor does not allocate more processes than there are logical CPUs'''
    with DaisyProcessExecutor() as e:
        assert e.max_processes <= os.cpu_count()
