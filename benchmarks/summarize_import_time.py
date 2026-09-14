import argparse
import pandas as pd
import numpy as np

def summarize_slow_package_imports(path):
    '''Compute the relative package import times and summarize the slowest'''
    df = pd.read_csv(path)
    df['base_package'] = df['importedpackage'].str.split('.', regex=False, n=1, expand=True)[0]
    total_time = df['cumulative'].iloc[-1]
    rel_time = df['self[us]']/total_time
    rel_time.sort_values(inplace=True, ascending=False)
    acc = np.cumsum(rel_time.values)
    slow_idx = rel_time.iloc[:np.argmax(acc > 0.5)]
    slow_imports = df.iloc[slow_idx.index]
    base_package_cost = \
        slow_imports[['base_package', 'self[us]']].groupby('base_package').aggregate(sum)
    rel_base_package_cost = base_package_cost / sum(base_package_cost['self[us]'])
    print(rel_base_package_cost.sort_values(by='self[us]',ascending=False))
    print(f'Total time: {total_time*1e-6:.3} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", type=str)
    args = parser.parse_args()
    summarize_slow_package_imports(args.inpath)
