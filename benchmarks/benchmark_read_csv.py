import os
import time
from pathlib import Path
import pandas as pd
from generate_time_series import generate_time_series # pylint: disable=import-error

def char_range(n):
    '''Generate a range of characters'''
    return [chr(ord('a') + i) for i in range(n)]

def generate_benchmark_data():
    '''Generate varying sized csv files for benchmarking purposes'''
    data_dir = Path(__file__).parent / 'benchmark-data'
    os.makedirs(data_dir, exist_ok=True)
    print('Generating small', flush=True)
    small = generate_time_series('2000-01-01', '2020-01-01', 200, char_range(2))
    small.to_csv(data_dir / 'small.csv')

    print('Generating medium', flush=True)
    medium = generate_time_series('2000-01-01', '2020-01-01', 2000, char_range(5))
    medium.to_csv(data_dir / 'medium.csv')

    print('Generating large', flush=True)
    large = generate_time_series('2000-01-01', '2020-01-01', 20000, char_range(20))
    large.to_csv(data_dir / 'large.csv')

    print('Generating very large', flush=True)
    very_large = generate_time_series('2000-01-01', '2020-01-01', 200000, char_range(50))
    very_large.to_csv(data_dir / 'very-large.csv')

def benchmark_read_csv():
    '''Compare pandas.read_csv using C engine with sep=',' and python engine with sep=None'''
    in_dir = Path(__file__).parent / 'benchmark-data'
    names = ['small', 'medium', 'large', 'very-large']
    data_files = [ in_dir / f'{name}.csv' for name in names ]

    c_timings = []
    for data_file in data_files:
        print(f'Start C {data_file}', flush=True)
        start = time.perf_counter()
        df = pd.read_csv(data_file, sep=',', engine='c') # pylint: disable=unused-variable
        elapsed = time.perf_counter() - start
        c_timings.append(elapsed)

    py_timings = []
    for data_file in data_files:
        print(f'Start python {data_file}', flush=True)
        start = time.perf_counter()
        df = pd.read_csv(data_file, sep=None, engine='python') # pylint: disable=unused-variable
        elapsed = time.perf_counter() - start
        py_timings.append(elapsed)


    results = pd.DataFrame({'name' : names, 'C' : c_timings, 'python' : py_timings})
    results['python/C'] = results['python'] / results['C']
    print(results)


if __name__ == '__main__':
    #print('Generating data')
    #generate_benchmark_data()
    print('Running benchmarks')
    benchmark_read_csv()
