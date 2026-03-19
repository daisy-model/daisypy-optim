import pandas as pd
import numpy as np

def generate_time_series(start_date, end_date, steps, columns):
    '''
    Generate a time series. Data will be either stationary, cyclic or increasing with standard
    normal noise
    '''
    rng = np.random.default_rng()
    data = {
        'time' : pd.date_range(start_date, end_date, steps)
    }

    for col in columns:
        t = rng.random()
        if t < 1/3:
            # Stationary
            data[col] = rng.normal(size=steps)
        elif t < 2/3:
            # cyclic
            data[col] = 2 * np.sin(np.linspace(0, 4*np.pi, steps)) +  rng.normal(size=steps)
        else:
            # Increasing
            data[col] = np.linspace(0, 4, steps) + rng.normal(size=steps)
    return pd.DataFrame(data)
