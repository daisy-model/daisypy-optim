# Getting started with daisypy-optim on Windows

0. Prerequisites
1. Install the package manager `uv`
2. Install the tool `daisypy_optim_create`
3. Create and run an example optimization project
4. Create a skeleton for your own optimization project

## Prerequisites
1. Install Daisy by following the instructions from [https://github.com/daisy-model/daisy/blob/main/doc/getting-started-windows.md](https://github.com/daisy-model/daisy/blob/main/doc/getting-started-windows.md).
2. Install `git` from [https://git-scm.com/downloads/win](https://git-scm.com/downloads/win)

## Install the python package manager `uv`
1. Install `uv` by following the instructions from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).
    - We recommend you use the standalone installer.
3. Check which, if any, python versions are available with `uv python list`
4. If necesary, install python by following the instructions from [https://docs.astral.sh/uv/guides/install-python/](https://docs.astral.sh/uv/guides/install-python/).

## Install the tool `daisypy_optim_create`
The behaviour of `daisypy_optim_create` depends on the availability of a number of other packages. We recommend you include the CMA-ES method. To do so, open PowerShell and run 
```
uv tool install --from git+https://github.com/daisy-model/daisypy-optim daisypy_optim --with cma
```

### Detailed installation instructions
For a minimal install open PowerShell and run
```
uv tool  install --from git+https://github.com/daisy-model/daisypy-optim daisypy_optim
```

* To include the CMA-ES method add the flag `--with cma`
* To include tensorboard logging add the flag `--with scipy,tensorboard,torch`
* To include the `scikit-optimize` method add the flag `--with scikit-optimize,joblib`

## Create and run an example optimization project
1. Open a PowerShell and navigate to a directory where you want to store your project
2. Initialize the project with `uv init daisypy-optim-example --bare`
    - This will create some scaffolding and initialize a git repository for version control.
3. Navigate to the newly created project with `cd daisypy-optim-example`
4. Generate files for the optimization with `daisypy_optim_create --example`
   - This should add dependencies to the project as needed, if not you will have to it manually using the command that is printed
5. Run the optimization with `uv run example/optimize.py`

```{PowerShell}
uv init daisypy-optim-example --bare
cd daisypy-optim-example
daisypy_optim_create --example
uv run example/optimize.py
```

While running this should produce log files in `example/out/0/logs`. The only one of interest when using the sequential optimizer is `result.csv` that contains the parameter values and objective values.

Once the optimization is done it should produce a couple of plots summarizing the optimization. These can be found in `example/out/0/analyze`.

The numbering in `example/out` is just consecutive integers starting from 0. Every time you rerun the optimization the number is incremented and a new directory created to store the output.

### Visualize optimization
There is a small script for visualizing the optimization result, run it with

    uv run example/analyze.py example/out/0
    
It will make an animation, two parameter plots and a convergence plot.

### Switch to CMA
1. Open `example/optimize.py` in your favorite editor
2. Replace `optimizer_name = "sequential"` with `optimizer_name = "cma"`
3. Rerun the optimization with `uv run example/optimize.py`

This will use CMA-ES to optimize and produce output in `example/out/1`, assuming this is the second time you run the the optimization.

## Create a skeleton for your own optimization project
Replace `<project-name>` with the name of your project.
```{PowerShell}
uv init <project-name> --bare
cd <project-name>
daisypy_optim_create
```
The last step will prompt for a bunch of input. Fill this out as best you can. Existing files will be copied to the project directory and can be edited as necesary afterwards.

You might want to change the optimization parameters in `optimize.py`, specifically the following that controls the maximum number of Daisy executions.
```{python}
    # Define some default parameters for each of the optimizers
    optimizer_options = {
        "sequential" : {
            "num_samples" : 20, # Number of samples from each continuous parameter
        },
        "cma" : {
            "maxfevals" : 100, # Maximum number of function evaluations
        },
        "skopt" : {
            "maxfevals" : 50,
        } 
    }
```
Once you have edited them you can run the optimization with
```
uv run <name>/optimize.py
```
Where `<name>` is the name you provided when running `daisypy_optim_create`
