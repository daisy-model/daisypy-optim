# Copilot session notes

`daisypy-optim` is a Python 3.11+ library for optimizing Daisy model parameters. Core code lives in `daisypy/optim/`; tests are in `tests/`; runnable examples and user docs are in `doc/examples/` and `doc/`.

When changing code, follow existing lightweight Python style and keep public behavior stable. Prefer targeted `pytest` runs for touched areas; use `pytest .` for broader validation. CI also runs `pylint --disable fixme .`.

Optional optimizers are feature-gated by extras: `.[cma]` enables CMA-ES and `.[ax]` enables Ax/Bayesian optimization. If work touches those paths, make sure the needed extra is installed before testing.

Do not commit generated or local-only artifacts such as `__pycache__/`, `tests/tmp/`, or editor backup files like `*.py~`.
