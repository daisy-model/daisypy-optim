# Development

## Tests

There are two kinds of tests in this repository:

1. **Automated tests** in `tests/`. These are regular `pytest` tests and should be used for
   normal development.
2. **Manual tests** in `manual-tests/`. These are small interactive scripts for visual inspection
   and other exploratory checks.

### Automated tests

Install the development dependencies and run the test suite:

    pip install -e .[dev]
    pytest

For a smaller run, execute a single test file:

    pytest tests/test_output_store.py

Some tests require a working `daisy` installation and are skipped automatically when `daisy` is
not available.

### Manual tests

Manual tests are not run by `pytest`. Run them directly with Python.

The current manual visualization test continuously writes `samples.csv` and updates the
`plot_samples` window:

    python manual-tests/test_plot_samples.py

Useful options:

    python manual-tests/test_plot_samples.py --standardized
    python manual-tests/test_plot_samples.py --write-interval 0.25
    python manual-tests/test_plot_samples.py --duration 10

The generated `samples.csv` is written to `manual-tests/out/plot-samples/` by default.

## Coverage

    pip install coverage
    coverage run
    coverage report
