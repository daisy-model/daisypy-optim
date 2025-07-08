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
uv tool install --from git+https://github.com/daisy-model/daisypy-optim@getting-started daisypy_optim --with cma
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
2. Initialize the project with `uv init daisypy-optim-example`
    - This will create some scaffolding and initialize a git repository for version control.
3. Navigate to the newly created project with `cd daisypy-optim-example`
4. Add `daisypy-optim` as a dependency with `uv add "daisypy-optim @ git+https://github.com/daisy-model/daisypy-optim"`
5. Add optional dependencies
   - CMA-ES : `uv add cma`
   - tensorboard : `uv add scipy tensorboard torch`
   - scikit-optimize : `uv add scikit-optimize joblib`
6. Generate files for the optimization with `daisypy_optim_create --example`
7. Run the optimization with `uvx run example/optimize.py`

```{PowerShell}
uv init daisypy-optim-example
cd daisypy-optim-example
uv add "daisypy-optim @ git+https://github.com/daisy-model/daisypy-optim"
uv add cma
daisypy_optim_create --example
uv run example/optimize.py
```

## Create a skeleton for your own optimization project
Replace `<project-name>` with the name of your project.
```{bash}
uv init <project-name>
cd <project-name>
uv add "daisypy-optim @ git+https://github.com/daisy-model/daisypy-optim"
uv add cma
daisypy_optim_create
```
The last step will prompt for a bunch of input. Fill this out as best you can. Existing files will be copied to the project directory and can be edited as necesary afterwards. Once you have edited them you can run the optimization with
```
uv run <name>
```
Where `<name>` is the name you provided when running daisypy_optim_create
