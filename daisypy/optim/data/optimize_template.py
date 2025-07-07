"""Example showing how to do parameter optimization in Daisy.
The optimization consists of four steps
  1. Setup 
  2. Optimize
  3. Evaluate
  4. Analyze
"""
# pylint: disable=missing-function-docstring
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
    available_loss_fns,
    available_optimizers,
    DaisyLoss,
    DaisyObjective,
    DaisyOptimizationProblem,
    ContinuousParameter,
    DaisyRunner,
)

def main():
    daisy_path = "{daisy_path}"
    daisy_home = "{daisy_home}"
    
    outdir = "{outdir}"
    os.makedirs(outdir, exist_ok=True)

    loss_fn = available_loss_fns["{loss_fn}"]
    optimizer = available_optimizers["{optimizer}"]
    logger = available_loggers["{logger}"]
    dai_template = "{dai_template}"
    parameters = read_parameters("{parameter_file}")
    target = pd.read_csv("{target_file}")
    variable_name = "{variable_name}"
    log_name = "{log_name}"
    run_id = get_run_id("{outdir}")
    
    
    problem = setup(daisy_path, daisy_home, outdir, dai_template, parameters, variable_name, log_name, target, loss_fn, run_id)
    result = optimize(problem, optimizer, logger, outdir, run_id)
    eval_dir = evaluate(result, problem, outdir, run_id)
    analyze(eval_dir, problem, outdir, run_id)

def read_parameters(path):
    with open(path, 'r', encoding='utf-8') as infile:
        parameter_specs = json.load(infile)
    parameters = []
    for spec in parameter_specs:
        assert "type" in spec, 'Missing "type" key'
        assert "name" in spec, 'Missing "name" key'
        assert "initial_value" in spec, 'Missing "initial_value" key'
        assert "valid_range" in spec, 'Missing "valid_range" key'
        if spec["type"] == "continuous":
            parameters.append(
                ContinuousParameter(
                    name=spec["name"],
                    initial_value=spec["initial_value"],
                    valid_range=(spec["valid_range"][0], spec["valid_range"][1])
                )
            )
        else:
            raise ValueError(f'Parameter type: "{{spec["type"]}}" not supported.')
    return parameters



def setup(daisy_path, daisy_home, base_outdir, dai_template, parameters, variable_name, log_name, target, loss_fn, run_id):
    # Setup a runner that knows how to execute daisy
    runner = DaisyRunner(daisy_path, daisy_home)

    # Setup a generator that can generate a valid .dai file from a template and parameter values
    dai_file_generator = DaiFileGenerator(template_file_path=dai_template)

    # Create a dirctory for storing setup files for debug/inspection
    outdir = os.path.join(base_outdir, run_id, 'setup')
    os.makedirs(outdir, exist_ok=True)
    
    # Generate a dai file for debug runs
    params = {{ p.name : p.initial_value  for p in parameters }}
    dai_file_generator(outdir, params)    

    # We need to wrap the loss function with the DaisyLoss class
    # You can substitute any loss function with the following signature
    # (actual : numpy.ndarray, target : numpy.ndarray) -> float
    wrapped_loss_fn = DaisyLoss(loss_fn)

    # Now we can create the objective and wrap it as a DaisyOptimizationProblem that knows how to
    # run daisy, get the output data and measure the loss
    objective_fn = DaisyObjective(log_name, variable_name, target, wrapped_loss_fn)
    problem = DaisyOptimizationProblem(runner, dai_file_generator, objective_fn, parameters)
    return problem


def optimize(problem, optimizer_name, logger_name, base_outdir, run_id):
    outdir = os.path.join(base_outdir, run_id, "optimize")
    os.makedirs(outdir, exist_ok=True)

    # Define some default parameters for each of the optimizers
    optimizer_options = {{
        "sequential" : {{
            "num_samples" : 20, # Number of samples from each continuous parameter
        }},
        "cma" : {{
            "maxfevals" : 1000, # Maximum number of function evaluations
        }},
        "skopt" : {{
            "maxfevals" : 200,
        }} 
    }}

    options = optimizer_options[optimizer_name]
    logdir = os.path.join(base_outdir, 'logs')
    logger = available_loggers[logger_name](logdir, f'{{run_id}}-{{optimizer_name}}')
    optimizer = available_optimizers[optimizer_name](problem, logger, options)

    # Optimize
    result = optimizer.optimize()

    # Print the result
    for name, res in result.items():
        print(name)
        for k,v in res.items():
            print('    ', k, ' : ', v, sep='')

    # Save the result
    with open(os.path.join(outdir, 'result.json'), 'w', encoding='utf-8') as out:
        json.dump(result, out)
            
    return result 


def evaluate(result, problem, base_outdir, run_id):
    # Create a directory for outputs
    outdir = os.path.join(base_outdir, run_id, "evaluate")
    os.makedirs(outdir, exist_ok=True)

    # Generate a dai file with the best parameters
    dai_file_generator = problem.dai_file_generator
    parameters = {{ k : v['best'] for k,v in result.items() }}
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
        sim_dlf.body[['year', 'month', 'mday', 'hour']].rename(columns={{'mday' : 'day'}})
    )
    sim = pd.DataFrame({{'time' : sim_time, 'value' : sim_value}})

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
