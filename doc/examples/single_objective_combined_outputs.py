# pylint: disable=too-many-locals,R0801
"""Example showing how to combine multiple outputs"""
import argparse
from pathlib import Path
import pandas as pd
from daisypy.optim import (
    ContinuousParameter,
    DefaultLogger,
    DaisyCMAOptimizer,
    DlfDataExtractor,
    DaiFileGenerator,
    MultiFileGenerator,
    DaisyRunner,
    DaisyOptimizationProblem,
    ScalarObjective,
    mse
)

# The function that combines log files is called with a list of DataFrames with a 'time' column
# and some value columns.
# It should return a single DataFrame with columns 'time' and 'value'
# And it has to be defined at the top-level because we cannot pass local functions due to
# multiprocessing.
def combine_logs(data_frames):
    '''
    Combine outputs by summing "Residuals-*" in field_nitrogen.dlf and multiplying by
    "Soil matrix water" in field_water.dlf

    '''
    df = data_frames[0]
    for df2 in data_frames[1:]:
        df = pd.merge(df, df2, on='time', validate='1:1')
    # The name of value columns follow this pattern <log-name>/<variable-name>
    # This value does not really makes sense, but it illustrates how we can combine outputs
    value = df['field_water.dlf/Soil matrix water'] * (
        df['field_nitrogen.dlf/Residuals-Soil'] + df['field_nitrogen.dlf/Residuals-Surface']
    )
    return pd.DataFrame({'time' : df['time'], 'value' : value })

def single_objective_combined_outputs(daisy_path, daisy_home=None):
    '''Optimize by combining multiple outputs

    daisy_path: str
      Path to daisy binary

    daisy_home: str
      Path to daisy home. If None let the Daisy binary figure it out
    '''
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'single-objective-combined-outputs'
    data_dir = base_dir / 'example-data' / 'combined-outputs'
    file_generator = MultiFileGenerator({
        'dai' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'template.dai'),
    })

    data_extractor = DlfDataExtractor({
        'field_water.dlf' : 'Soil matrix water',
        'field_nitrogen.dlf' : ['Residuals-Soil', 'Residuals-Surface']
    }, combine_logs)

    parameters = {
        'dai' : [
            ContinuousParameter('fertilize_weight_1', 20, (0, 200)),
            ContinuousParameter('fertilize_weight_2', 20, (0, 200))
        ],
    }

    runner = DaisyRunner(daisy_path, daisy_home)

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective(
        name="objective",
        data_extractor=data_extractor,
        target=target_file,
        target_name="target",
        loss_fn=mse
    )

    problem = DaisyOptimizationProblem(
        runner, file_generator, objective, parameters, out_dir
    )
    cma_options = {
        "maxfevals" : 200
    }
    with DefaultLogger(out_dir) as logger:
        optimizer = DaisyCMAOptimizer(problem, logger, cma_options)
        result = optimizer.optimize()

    # Optimum at
    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')
    print('Optimum at',
          '  fertilize_weight_1 = 100',
          '  fertilize_weight_2 = 80',
          sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('--daisy-home', type=str,
                        help='Path to daisy home directory containing lib/ and sample/',
                        default=None)
    args = parser.parse_args()
    single_objective_combined_outputs(args.daisy_path, args.daisy_home)
