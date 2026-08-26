# pylint: disable=too-many-locals,R0801
"""Example showing how to combine multiple outputs"""
import argparse
from pathlib import Path
import pandas as pd
from daisypy.optim import (
    ContinuousParameter,
    DaiFileGenerator,
    DaisyCMAOptimizer,
    DaisyOptimizationProblem,
    DaisyRunner,
    DefaultLogger,
    OutputSpec,
    PostProcessor,
    ScalarObjective,
    Simulation,
    mse
)


class CombinedOutput(PostProcessor):
    # pylint: disable=too-few-public-methods
    """Combine multiple outcomes into a single outcome"""
    def __call__(self, outcomes):
        df = pd.merge(
            outcomes['field-water'],
            outcomes['field-nitrogen-soil'],
            on='time',
            validate='1:1',
            suffixes=('-water', '-soil')
        )
        df = pd.merge(
            df,
            outcomes['field-nitrogen-surface'],
            on='time',
            validate='1:1'
        )
        # This value does not really makes sense, but it illustrates how we can combine outputs
        value = df['value-water'] * (df['value-soil'] + df['value'])
        return pd.DataFrame({'time' : df['time'], 'value' : value})


def run(daisy_path):
    '''Optimize by combining multiple outputs

    daisy_path: str
      Path to daisy binary
    '''
    base_dir = Path(__file__).parent
    out_dir = base_dir / 'out' / 'single-objective-combined-outputs'
    data_dir = base_dir / 'example-data' / 'combined-outputs'
    file_generators = {
        'runfile' : DaiFileGenerator('run.dai', template_file_path=data_dir / 'template.dai'),
    }

    outputs = {
        'water' : OutputSpec('field_water.dlf', 'Soil matrix water'),
        'nitrogen' : OutputSpec('field_nitrogen.dlf', ['Residuals-Soil', 'Residuals-Surface'])
    }

    simulations = {
        'sim' : Simulation(file_generators, outputs)
    }

    outcome_specs = {
        'field-water' : ('sim', 'water', 'Soil matrix water'),
        'field-nitrogen-soil' : ('sim', 'nitrogen', 'Residuals-Soil'),
        'field-nitrogen-surface' : ('sim', 'nitrogen', 'Residuals-Surface'),
    }

    post_processing = {
        'combined-output' : CombinedOutput()
    }

    parameters = {
        'runfile' : [
            ContinuousParameter('fertilize_weight_1', 20, (0, 200)),
            ContinuousParameter('fertilize_weight_2', 20, (0, 200))
        ],
    }

    runner = DaisyRunner(daisy_path)

    target_file = data_dir / 'target.csv'
    objective = ScalarObjective(
        name='objective',
        target=target_file,
        target_col='target',
        outcome_name='combined-output',
        loss_fn=mse
    )

    problem = DaisyOptimizationProblem(
        runner, simulations, outcome_specs, post_processing, objective, parameters, out_dir
    )
    cma_options = {
        'maxfevals' : 200
    }
    with DefaultLogger(out_dir) as logger:
        optimizer = DaisyCMAOptimizer(problem, logger, cma_options)
        result = optimizer.optimize()

    for name, res in result.items():
        print(name)
        for k, v in res.items():
            print('    ', k, ' : ', v, sep='')
    print('Optimum at',
          '  fertilize_weight_1 = 100',
          '  fertilize_weight_2 = 80',
          sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    args = parser.parse_args()
    run(args.daisy_path)
