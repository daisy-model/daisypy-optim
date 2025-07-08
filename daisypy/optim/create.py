import argparse
import os
import json
import csv
import importlib.resources
import subprocess
import shutil

import daisypy.optim as daisypy_optim

from daisypy.optim import (
    available_loggers,
    available_loss_fns,
    available_optimizers,
)

def main():
    parser = argparse.ArgumentParser(description="Setup input files and script for running a Daisy optimization")
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    if args.example:
        config = get_example_config()
    else:
        config = get_config()
    print(config)
    create_optim(config)
    finalize()

def get_example_config():
    return {
        'name': 'example',
        'daisy_home': 'C:/Program Files/daisy 7.1.0',
        'daisy_path': 'C:/Program Files/daisy 7.1.0/bin/daisy.exe',
        'outdir': 'out',
        'optimizer': 'sequential',
        'logger': 'csv',
        'dai_template': 'input/template.dai',
        'parameter_file': 'input/parameters.json',
        'num_continuous_parameters': 0,
        'variable_name': 'N2O-Nitrification',
        'log_name': 'field_nitrogen.dlf',
        'target_file': 'input/target.csv',
        'loss_fn': 'mse',
    }


def get_config():
    config = {}
    config["name"] = input_with_default("Name", "my-optimization", lambda x: not os.path.exists(x))

    daisy_candidates = []
    for entry in os.scandir("C:/Program Files"):
        if entry.name.startswith("daisy"):
            daisy_candidates.append(entry.path)
    match len(daisy_candidates):
        case 0:
            config["daisy_home"] = input_no_default("Path to daisy directory", os.path.isdir) 
        case 1:
            config["daisy_home"] = input_with_default("Path to daisy directory", daisy_candidates[0], os.path.isdir) 
        case _:
            config["daisy_home"] = input_with_choices("Path to daisy directory", daisy_candidates, os.path.isdir) 
    
    config["daisy_path"] = input_with_default("Path to daisy.exe", os.path.join(config["daisy_home"], "bin", "daisy.exe"), os.path.exists)
    config["outdir"] = input_with_default("Output directory", "out")

    config["optimizer"] = input_with_choices("Optimization method", list(available_optimizers.keys()), None, False)
    ##  TODO: Get optimizer specific parameters

    config["logger"] = input_with_choices("Logger", list(available_loggers.keys()), None, False)
    config["dai_template"] = input_with_default("Dai template file", "input/template.dai")
    config["parameter_file"] = input_with_default("Parameter file", "input/parameters.json")
    if not os.path.exists(config["parameter_file"]):
        config["num_continuous_parameters"] = int(input_with_default("Number of continuous parameters", 0, is_int))
        #config["num_categorical_parameters"] = input_with_default("Number of categorical parameters", 0, is_int)

    config["variable_name"] = input_no_default("Name of variable to optimize for")
    config["log_name"] = input_no_default("Name of log file where the variable is logged")
    config["target_file"] = input_with_default("Path to target file", "input/target.csv")

    config["loss_fn"] = input_with_choices("Loss function", list(available_loss_fns.keys()))

    return config

def create_optim(config):
    basedir = os.path.abspath(config["name"])
    os.makedirs(basedir, exist_ok=False)
    # This is two calls because we want to fail if the basedir already exists
    os.makedirs(os.path.join(basedir, "input"))

    if not os.path.isabs(config["outdir"]):
        config["outdir"] = os.path.join(basedir, config["outdir"])
    config["outdir"] = sanitize_path(config["outdir"])

    # Copy or create parameter file
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
    config["parameter_file"] = sanitize_path(param_path)

    # Copy or create target file
    target_path = os.path.join(basedir, "input", "target.csv")
    if os.path.exists(config["target_file"]):
        try:
            shutil.copyfile(config["target_file"], target_path)
        except shutil.SameFileError:
            pass # This is not a problem
    else:
        with open(target_path, 'w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for row in get_example_target():
                writer.writerow(row)
    config["target_file"] = sanitize_path(target_path)

    # Copy or create template file
    template_path = os.path.join(basedir, "input", "template.dai")
    if os.path.exists(config["dai_template"]):
        try:
            shutil.copyfile(config["dai_template"], template_path)
        except shutil.SameFileError:
            pass # This is not a problem
    else:
        # We have a default with some instructions
        # From python 3.13 we can do
        # with importlib.resources.path("daisypy.optim", "data", "default-template.dai") as default_path:
        with importlib.resources.path("daisypy.optim", "data") as datadir:
            default_path = datadir / "default-template.dai"
            shutil.copyfile(default_path, template_path)
    config["dai_template"] = sanitize_path(template_path)

    # Instantiate an optimize.py file
    # From python 3.13 we can do
    # optimize_template = importlib.resources.read_text("daisypy.optim", "data", "optimize_template.py", encoding="utf-8")
    with importlib.resources.path("daisypy.optim", "data") as datadir:
        optimize_inpath = os.path.join(datadir, "optimize_template.py")
    with open(optimize_inpath, 'r', encoding='utf-8') as infile:
        optimize_template = infile.read()
    optimize_program = optimize_template.format(**config)
    optimize_path = os.path.join(basedir, "optimize.py")
    with open(optimize_path, "w", encoding='utf-8') as outfile:
         outfile.write(optimize_program)

def finalize():
    # Try to add dependencies with uv
    cmd = ["uv", "add"]
    cmd = [
        '"daisypy-optim @ git+https://github.com/daisy-model/daisypy-optim"',
        "pandas",
        "matplotlib"
    ]
    if 'cma' in available_optimizers:
        cmd.append("cma")
    if 'skopt' in available_optimizers:
        cmd += ["scikit-optimize", "joblib"]
    if "tensorboard" in available_loggers:
        cmd += ["scipy", "tensorboard", "torch"]
    try:
        result = subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        print("Error while adding dependencies:", e)
        print("Please verify/add manually")
        print(*cmd)

def input_with_default(prompt, default, check=None):
    while True:
        response = input(f"{prompt} [{default}]: ")
        if len(response) == 0:
            print_selected(default)
            return default
        elif check is None or check(response):
            print_selected(response)
            return response

def input_no_default(prompt, check=None):
    while True:
        response = input(f"{prompt}: ")
        if check is None or check(response):
            print_selected(response)
            return response
        
def input_with_choices(prompt, choices, check=None, user_supplied_ok=True):
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
    print(f'  "{selected}"')

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def sanitize_path(path):
    return path.replace("\\", "/")

def get_example_params():
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
    return [
        ["time", "N2O-Nitrification"],
        ["2000-12-13", "3.0"]
    ]

if __name__ == '__main__':
    main()
