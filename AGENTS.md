# Copilot session notes

`daisypy-optim` is a Python 3.12+ library for optimizing Daisy model parameters. Core code lives in `daisypy/optim/`; tests are in `tests/`; runnable examples and user docs are in `doc/examples/` and `doc/`.

When changing code, follow existing lightweight Python style and keep public behavior stable. Prefer targeted `pytest` runs for touched areas; use `pytest .` for broader validation. CI also runs `pylint --disable fixme .`.

Keep a narrow scope on actions. For example, if the user asks for a test, then implement the test and run it. It is not required that the feature exists or conforms to the test specification.

## Minimal-first interactive workflow

For requests that could be implemented in multiple reasonable ways, pause early and ask enough
clarifying questions to converge on the smallest useful version before making changes.

Default to a minimal implementation:
- Satisfy only the explicitly requested requirements.
- Do not add extra features, configurability, generalization, or robustness work unless
  requested.
- Do not add automated tests, documentation, CLI options, environment-variable switches, or
  integration hooks unless they are part of the ask.
- For visualization, demo, or exploratory tasks, prefer standalone/manual scripts over
  pytest/integration unless the user asks for automated coverage.
- Avoid speculative edge-case handling. Handle the direct use case first.

If there is ambiguity, prefer interaction over inference:
- Ask brief, concrete questions when the answer changes scope or implementation shape.
- If the request can be satisfied by a simpler manual artifact or a more integrated solution, ask
  which is wanted instead of choosing the larger one.
- If you notice an unrelated bug, inconsistency, or unresolved behavior while working on a narrow
  task, do not ignore it and do not fix it without approval. Pause and ask the user how they want
  to handle it.

Validation should also be minimal:
- Use the smallest check that confirms the requested behavior.
- Do not expand into broader robustness or regression work until the user confirms the minimal
  version is satisfactory.

After delivering the minimal version, stop. Improvements such as robustness, polish, broader
tests, and extra options should happen only in later iterations if requested.

When possible, implement tests before rewriting or adding new features. These tests should fail so we can verify expectations before implementing or rewriting features.

Optional optimizers are feature-gated by extras: `.[cma]` enables CMA-ES and `.[ax]` enables Ax/Bayesian optimization. If work touches those paths, make sure the needed extra is installed before testing.

Do not commit generated or local-only artifacts such as `__pycache__/`, `tests/tmp/`, or editor backup files like `*.py~`.
