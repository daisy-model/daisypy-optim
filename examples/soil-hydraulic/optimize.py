"""Example showing how to do parameter optimization in Daisy.
The optimization consists of four steps
  1. Setup 
  2. Optimize
  3. Evaluate
  4. Analyze
"""
# pylint: disable=missing-function-docstring
import argparse
import os
import json
import warnings
with warnings.catch_warnings():
    # We dont want the Pyarrow depedency warning
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import pandas as pd
import matplotlib.pyplot as plt
from daisypy.io.dlf import read_dlf
from daisypy.optim import (
    DaiFileGenerator,
    available_loggers,
    available_optimizers,
    DaisyLoss,
    DaisyObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('daisy_path', type=str, help='Path to daisy binary')
    parser.add_argument('daisy_home', type=str,
                        help='Path to daisy home directory containing lib/ and sample/')
    parser.add_argument('outdir', type=str, help='Directory to store results in')
    parser.add_argument('--optimizer', type=str, choices=available_optimizers, default='sequential')
    parser.add_argument('--logger', type=str, choices=available_loggers, default='csv')
    parser.add_argument('--run-id', type=int, default=-1,
                        help="Id to use for this run. Set to -1 to generate automatically")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.run_id == -1:
        run_id = get_run_id(args.outdir)
    else:
        run_id = str(args.run_id)
    problem = setup(args.daisy_path, args.daisy_home, args.outdir, run_id)
    result = optimize(problem, args.optimizer, args.logger, args.outdir, run_id)
    eval_dir = evaluate(result, problem, args.outdir, run_id)
    analyze(eval_dir, problem, args.outdir, run_id)


# We will use mean squared error as the loss
# We use the multiprocessing module, which uses pickle, so we cannot use local functions
def mse(actual, target):
    return ((actual - target)**2).mean()

def setup(daisy_path, daisy_home, base_outdir, run_id):
    # Setup a runner that knows how to execute daisy
    runner = DaisyRunner(daisy_path, daisy_home)

    # Setup a generator that can generate a valid .dai file from a template and parameter values
    dai_template = 'soil_template.dai'
    dai_file_generator = DaiFileGenerator(template_file_path=dai_template)

    # Define parameters, their initial value and their valid range
    # The name of the parameters MUST match the names used in the template file
    parameters = [
        ContinuousParameter(
            name='K_Bt1',
            initial_value=5.5,
            valid_range=(2.001, 10.001)
        ),
        ContinuousParameter(
            name='K_Bt2',
            initial_value=5.5,
            valid_range=(2.0, 10.0)
        ),
        ContinuousParameter(
            name='K_C1',
            initial_value=5.5,
            valid_range=(2.0, 10.0)
        ),
        ContinuousParameter(
            name='K_aquitard',
            initial_value=1.0,
            valid_range=(0.1, 2)
        ),
        ContinuousParameter(
            name='pressure_table',
            initial_value=-2,
            valid_range=(-3.01, -1.01)
        ),
        ContinuousParameter(
            name='pipe_position',
            initial_value=-100,
            valid_range=(-130.01, -50.01)
        ),
    ]

    # Create a dirctory for storing setup files for debug/inspection
    outdir = os.path.join(base_outdir, run_id, 'setup')
    os.makedirs(outdir, exist_ok=True)
    
    # Generate a dai file for debug runs
    params = { p.name : p.initial_value  for p in parameters }
    dai_file_generator(outdir, params)    

    # We need to know the name of the variable we are optimizing for, and we need to know the name
    # of the Daisy log file where we can find the variable.
    variable_name = "table_low"
    log_name = "groundwater.csv"
    
    # Cleanup the measurements file
    # The measurement file contains hourly measurements, but we only log daily at hour 0 so we need
    # to remove the other measurements from the target file.
    # Cant get read_csv to parse the dates , so we do it afterwards
    target = pd.read_csv('GW_South.csv', comment='#').drop(columns="sensor") 
    target["time"] = pd.to_datetime(target["time"])
    target = target[target["time"].dt.hour == 0]
    target.rename(columns={"GW" : variable_name}, inplace=True)

    # Write the cleaned target so we can check it
    target.to_csv(os.path.join(outdir, 'target.csv'))

    # We need to wrap the loss function with the DaisyLoss class
    # You can substitute any loss function with the following signature
    # (actual : numpy.ndarray, target : numpy.ndarray) -> float
    loss_fn = DaisyLoss(mse)

    # Now we can create the objective and wrap it as a DaisyOptimizationProblem that knows how to
    # run daisy, get the output data and measure the loss
    objective_fn = DaisyObjective(log_name, variable_name, target, loss_fn)
    problem = DaisyOptimizationProblem(runner, dai_file_generator, objective_fn, parameters)
    return problem


def optimize(problem, optimizer_name, logger_name, base_outdir, run_id):
    outdir = os.path.join(base_outdir, run_id, "optimize")
    os.makedirs(outdir, exist_ok=True)

    # Define some default parameters for each of the optimizers
    optimizer_options = {
        "sequential" : {
            "num_samples" : 20, # Number of samples from each continuous parameter
        },
        "cma" : {
            "maxfevals" : 1000, # Maximum number of function evaluations
        },
        "skopt" : {
            "maxfevals" : 200,
        }        
    }

    options = optimizer_options[optimizer_name]
    logdir = os.path.join(base_outdir, 'logs')
    logger = available_loggers[logger_name](logdir, f'{run_id}-{optimizer_name}')
    optimizer = available_optimizers[optimizer_name](problem, logger, options)

    # Optimize
    result = optimizer.optimize()

    # Print the result
    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')

    # Save the result
    with open(os.path.join(outdir, 'result.json'), 'w') as out:
        json.dump(result, out)
            
    return result 


def evaluate(result, problem, base_outdir, run_id):
    # Create a directory for outputs
    outdir = os.path.join(base_outdir, run_id, "evaluate")
    os.makedirs(outdir, exist_ok=True)

    # Generate a dai file with the best parameters
    dai_file_generator = problem.dai_file_generator
    parameters = { k : v['best'] for k,v in result.items() }    
    dai_path = dai_file_generator(outdir, parameters)

    # Run daisy with the newly generated dai file
    problem.runner(dai_path, outdir)
    return outdir

    
def analyze(eval_dir, problem, base_outdir, run_id):
    outdir = os.path.join(base_outdir, run_id, 'analyze')
    os.makedirs(outdir, exist_ok=True)

    # Extract variable, log name and target from the problem
    var = problem.objective_fn.variable_name    
    target = problem.objective_fn.target
    log_name = problem.objective_fn.log_name

    # Get the simulated results
    sim_dlf = read_dlf(os.path.join(eval_dir, log_name))
    sim_value = sim_dlf.body[var]
    sim_time = pd.to_datetime(
        sim_dlf.body[['year', 'month', 'mday', 'hour']].rename(columns={'mday' : 'day'})
    )
    sim = pd.DataFrame({'time' : sim_time, 'value' : sim_value})

    # Make a time series plot of the target and simulated values
    fig, ax = plt.subplots()
    ax.plot(target["time"], target['value'])
    ax.plot(sim["time"], sim['value']) 
    ax.legend(["target", "simulation"])
    ax.set_ylabel(var)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'time-series.pdf'))

    # Make a scatter plot of the tarvet vs simulated value
    fig, ax = plt.subplots()
    merged = pd.merge(
        target, sim,
        how='left',
        on='time',
        suffixes=('_target', '_sim')
    )
    ax.scatter(merged['value_target'], merged['value_sim'])
    ax.axis('equal')
    ax.axline((merged['value_target'][0],)*2, slope=1, color='red')
    ax.set_xlabel('Target')
    ax.set_ylabel('Simulated')
    ax.set_title(var)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'scatter.pdf'))
    
    plt.show()

def get_run_id(base_outdir):
    used = set()
    for entry in os.scandir(base_outdir):
        try:
            used.add(int(entry.name))
        except ValueError:
            pass
    runid = 0
    try:
        while True:
            used.remove(runid)
            runid += 1
    except KeyError:
        return str(runid)
                    
if __name__ == '__main__':
    main()
