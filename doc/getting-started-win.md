# Getting started with daisypy-optim on Windows
1. Setup a python environment
2. Create an optimization project

## Setup python environment
1. Download and run an installer from [https://www.python.org/](https://www.python.org/)
    - For convenience, let the installer add python.exe to ypur path
2. Install the python package manager `uv` from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
    - We recommend you use the standalone installer

## Create an optimization project
1. Open a PowerShell and navigate to a directory where you want to store your project
2. Run `uv init daisypy-optim-example`
    - This will create a new project with some scaffolding and initialize a git repository for version control.
3. Run `cd daisypy-optim-example` to change directory to the newly created project
4. Verify that things work with `uv run main.py`
    - This should display "Hello from daisypy-optim-example!"
5. Add `daisypy-optim` as a dependency with `uv add "daisypy-optim @ git+https://github.com/daisy-model/daisypy-optim"`
6. Run `uvx daispy_optim_create` from the project directory and follow the pormpts