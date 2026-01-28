"Setup input files and script for running a Daisy optimization"
import argparse
import os
import sys
import json
import csv
import importlib.resources
import platform
import subprocess
import shutil

from daisypy.optim import (
    available_aggregate_fns,
    available_loss_fns,
    available_optimizers,
)

def main():
    '''Entry point for create'''
    # We need argument parsing in the entry point function
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    # pylint: disable=broad-exception-caught
    # Regardless of what happens we want to try and inform the user
    try:
        run(args.example)
        sys.exit(0)
    except Exception as e:
        print("Error creating project", e)
        sys.exit(1)

def run(example=False):
    '''
    Parameters
    ----------
    example : bool
      If True generate example optimization, otherwise prompt user for inputs
    '''
    if example:
        config = get_example_config()
    else:
        config = get_config()
    create_optim(config)
    finalize()

def get_example_config():
    '''Generate example configuration. User is prompted for Daisy home and Daisy executable paths'''
    config = {
        'name': 'example',
        'out_dir': 'out',
        'optimizer': 'sequential',
        'dai_template': 'input/template.dai',
        'parameter_file': 'input/parameters.json',
        'num_continuous_parameters': 0,
        'objectives' : {
            0 : {
                'variable_name': 'N2O-Nitrification',
                'log_name': 'field_nitrogen.dlf',
                'target_file': 'input/target.csv',
                'loss_fn': 'mse',
            },
            1 : {
                'variable_name': 'N2O-Nitrification',
                'log_name': 'field_nitrogen.dlf',
                'target_file': 'input/target.csv',
                'loss_fn': 'mae'
            }
        },
        'aggregate_fn' : 'sum'
    }
    config["daisy_home"] = get_daisy_home()
    config["daisy_path"] = get_daisy_path(config['daisy_home'])
    return config

def get_config():
    '''Prompt user for full configuration'''
    config = {}
    config["name"] = input_with_default("Name", "my-optimization", lambda x: not os.path.exists(x))
    config["daisy_home"] = get_daisy_home()
    config["daisy_path"] = get_daisy_path(config["daisy_home"])
    config["out_dir"] = input_with_default("Output directory", "out")

    config["optimizer"] = input_with_choices("Optimization method",
                                             list(available_optimizers.keys()),
                                             None,
                                             False)
    ##  TODO: Get optimizer specific parameters
    config["dai_template"] = input_with_default("Dai template file", "input/template.dai")
    config["parameter_file"] = input_with_default("Parameter file", "input/parameters.json")
    if not os.path.exists(config["parameter_file"]):
        config["num_continuous_parameters"] = int(
            input_with_default("Number of continuous parameters", 0, is_int)
        )
        # config["num_categorical_parameters"] = int(
        #     input_with_default("Number of categorical parameters", 0, is_int)
        # )
    objectives = {}
    add_more = True
    while add_more:
        objective, add_more = _prompt_for_objective()
        objectives[len(objectives)] = objective
    config["objectives"] = objectives
    if len(objectives) > 1:
        config["aggregate_fn"] = input_with_choices("Aggregate function",
                                                    list(available_aggregate_fns.keys()))
    else:
        config["aggregate_fn"] = "sum"

    return config

def _prompt_for_objective():
    objective = {}
    objective["variable_name"] = input_no_default("Name of variable to optimize for")
    objective["log_name"] = input_no_default("Name of log file where the variable is logged")
    objective["target_file"] = input_with_default("Path to target file", "input/target.csv")
    objective["loss_fn"] = input_with_choices("Loss function", list(available_loss_fns.keys()))
    add_more = 'y' == input_no_default("Add more objectives? (y/n)",
                                       check=lambda s: s in ('y', 'n'))
    return objective, add_more

def create_optim(config):
    '''Create an optimization project from a configuration'''
    config["daisy_home"] = sanitize_path(os.path.abspath(config["daisy_home"]))
    config["daisy_path"] = sanitize_path(os.path.abspath(config["daisy_path"]))
    basedir = os.path.abspath(config["name"])
    os.makedirs(basedir, exist_ok=False)
    # This is two calls because we want to fail if the basedir already exists
    os.makedirs(os.path.join(basedir, "input"))

    if not os.path.isabs(config["out_dir"]):
        config["out_dir"] = os.path.join(basedir, config["out_dir"])
    config["out_dir"] = sanitize_path(config["out_dir"])

    # Copy or create parameter file
    config["parameter_file"] = _copy_or_create_param(config, basedir)

    # Copy or create target files fot objectives
    # And transform from dict of objectives to lists
    objectives = config.pop("objectives")
    for objective_id, objective in objectives.items():
        # Copy or create target file
        objective["target_file"] = _copy_or_create_target(objective_id, objective, basedir)

    config['variable_names'] = [ obj["variable_name"] for obj in objectives.values() ]
    config['log_names'] = [ obj["log_name"] for obj in objectives.values() ]
    config['target_files'] = [ obj["target_file"] for obj in objectives.values() ]
    config['loss_fns'] = [ obj["loss_fn"] for obj in objectives.values() ]

    # Copy or create template file
    config["dai_template"] = _copy_or_create_template(config, basedir)

    # Instantiate an optimize.py file
    optimize_template = _read_optimize_template()
    optimize_program = optimize_template.format(**config)
    optimize_path = os.path.join(basedir, "optimize.py")
    with open(optimize_path, "w", encoding='utf-8') as outfile:
        outfile.write(optimize_program)

    # Copy the analyze script
    _copy_script('analyze.py', basedir)

def _copy_or_create_param(config, basedir):
    param_path = os.path.join(basedir, "input", "parameters.json")
    if os.path.exists(config["parameter_file"]):
        try:
            shutil.copyfile(config["parameter_file"], param_path)
        except shutil.SameFileError:
            pass # This is not a problem
    else:
        if config["num_continuous_parameters"] == 0:
            params = get_example_params()
        else:
            params = [{
                "type" : "continuous",
                "name" : f"p{i}",
                "initial_value" : 0,
                "valid_range" : (0, 1)
                } for i in range(config["num_continuous_parameters"])
            ]
        with open(param_path, 'w', encoding='utf-8') as outfile:
            json.dump(params, outfile, indent="  ")
    return sanitize_path(param_path)

def _copy_or_create_target(objective_id, objective, basedir):
    target_path = os.path.join(basedir, "input", f"target-{objective_id}.csv")
    if os.path.exists(objective["target_file"]):
        try:
            shutil.copyfile(objective["target_file"], target_path)
        except shutil.SameFileError:
            pass # This is not a problem
    else:
        with open(target_path, 'w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in get_example_target():
                writer.writerow(row)
    return sanitize_path(target_path)

def _copy_or_create_template(config, basedir):
    template_path = os.path.join(basedir, "input", "template.dai")
    if os.path.exists(config["dai_template"]):
        try:
            shutil.copyfile(config["dai_template"], template_path)
        except shutil.SameFileError:
            pass # This is not a problem
    else:
        # We have a default with some instructions
        # From python 3.13 we can do
        # with importlib.resources.path("daisypy.optim",
        #                               "data",
        #                               "default-template.dai") as default_path:
        with importlib.resources.path("daisypy.optim", "data") as datadir:
            shutil.copyfile(datadir / "default-template.dai", template_path)
    return sanitize_path(template_path)

def _read_optimize_template():
    # From python 3.13 we can do
    # optimize_template = importlib.resources.read_text("daisypy.optim",
    #                                                   "data",
    #                                                   "optimize_template.py",
    #                                                   encoding="utf-8")
    with importlib.resources.path("daisypy.optim", "data") as datadir:
        optimize_inpath = os.path.join(datadir, "optimize_template.py")
    with open(optimize_inpath, 'r', encoding='utf-8') as infile:
        optimize_template = infile.read()
    return optimize_template

def _copy_script(file_name, out_dir):
    out_path = os.path.join(out_dir, file_name)
    with importlib.resources.path("daisypy.optim", "data") as datadir:
        in_path = os.path.join(datadir, file_name)
        try:
            shutil.copyfile(in_path, out_path)
        except (shutil.SameFileError, IOError):
            print(f'Could not copy {file_name}')

def finalize():
    '''Try to add dependencies with uv. Inform user if it fails'''
    cmd = [
        "uv",
        "add",
        "daisypy-optim@git+https://github.com/daisy-model/daisypy-optim",
        "pandas",
        "matplotlib"
    ]
    if 'cma' in available_optimizers:
        cmd.append("cma")
    if 'skopt' in available_optimizers:
        cmd += ["scikit-optimize", "joblib"]

    result = subprocess.run(cmd, text=True, check=False)
    if result.returncode != 0:
        print("Error adding dependencies. Returncode", result.returncode)
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        print("Please verify/add manually")
        print(*cmd)

def get_daisy_home():
    '''Prompt user for Daisy home location. Try to guess based on platform'''
    daisy_candidates = []
    match platform.system():
        case "Windows":
            try:
                for entry in os.scandir("C:/Program Files"):
                    if entry.name.startswith("daisy"):
                        daisy_candidates.append(entry.path)
            except FileNotFoundError:
                pass
        case "Linux":
            # Try to get Daisy home from flatpak install
            path = os.path.expanduser('~/.var/app/dk.ku.daisy/data')
            if os.path.exists(path):
                daisy_candidates.append(path)
        case _:
            pass

    match len(daisy_candidates):
        case 0:
            return input_no_default("Path to daisy directory", os.path.isdir)
        case 1:
            return input_with_default("Path to daisy directory", daisy_candidates[0], os.path.isdir)
        case _:
            return input_with_choices("Path to daisy directory", daisy_candidates, os.path.isdir)

def get_daisy_path(daisy_home):
    '''Prompt user for Daisy executable path. Try to guess based on daisy_home/platform'''
    match platform.system():
        case "Windows":
            default = os.path.join(daisy_home, "bin", "daisy.exe")
            if os.path.exists(default):
                return input_with_default("Path to daisy.exe", default, os.path.exists)
            return input_no_default("Path to daisy.exe", os.path.exists)

        case "Linux":
            default = os.path.expanduser("~/.local/bin/daisy")
            if os.path.exists(default):
                return input_with_default("Path to daisy binary", default, os.path.exists)
            return input_no_default("Path to daisy binary", os.path.exists)

        case _:
            default = os.path.join(daisy_home, "bin", "daisy")
            if os.path.exists(default):
                return input_with_default("Path to daisy binary", default, os.path.exists)
            return input_no_default("Path to daisy binary", os.path.exists)


def input_with_default(prompt, default, check=None):
    '''Prompt for user input with a default.
    Optionally validate input.'''
    while True:
        response = input(f"{prompt} [{default}]: ")
        if len(response) == 0:
            print_selected(default)
            return default
        if check is None or check(response):
            print_selected(response)
            return response

def input_no_default(prompt, check=None):
    '''Prompt for user input without a default.
    Optionally validate input.'''
    while True:
        response = input(f"{prompt}: ")
        if check is None or check(response):
            print_selected(response)
            return response

def input_with_choices(prompt, choices, check=None, user_supplied_ok=True):
    '''Prompt for user input with a list of choices.
    Optionally validate input.
    Optinally allow user supply their own choice'''
    while True:
        print(prompt, *[f"  {i} {choice}" for i, choice in enumerate(choices)], sep='\n')
        response = input("Choice: ")
        try:
            i = int(response)
            if 0 <= i < len(choices):
                print_selected(choices[i])
                return choices[i]
        except ValueError:
            if user_supplied_ok and (check is None or check(response)):
                print_selected(response)
                return response

def print_selected(selected):
    '''Print user selection'''
    print(f'  "{selected}"')

def is_int(s):
    '''Check if s can be converted to an integer'''
    try:
        int(s)
        return True
    except ValueError:
        return False

def sanitize_path(path):
    '''Get rid of backslashes in paths'''
    return path.replace("\\", "/")

def get_example_params():
    '''Parameters to optimize in example'''
    return [
        {
            "type" : "continuous",
            "name" : "K_aquitard",
            "initial_value": 1.0,
            "valid_range" : [0.1, 2]
        },
        {
            "type" : "continuous",
            "name" : "Z_aquitard",
            "initial_value" : 150,
            "valid_range" : [50, 250]
        }
    ]

def get_example_target():
    '''The output from running with K_aquitard = 0.456 and Z_aquitard = 236'''
    return [
        ["time", "N2O-Nitrification"],
        ["2000-01-01", "0.000011"],
        ["2000-02-01", "0.029358"],
        ["2000-03-01", "0.077190"],
        ["2000-04-01", "0.312162"],
        ["2000-05-01", "1.034610"],
        ["2000-06-01", "0.490148"],
        ["2000-07-01", "0.214898"],
        ["2000-08-01", "0.217915"],
        ["2000-09-01", "0.180642"],
        ["2000-10-01", "0.163631"],
        ["2000-11-01", "0.280886"],
        ["2000-12-01", "0.204323"],
    ]


if __name__ == '__main__':
    main()
