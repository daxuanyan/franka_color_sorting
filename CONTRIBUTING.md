# Contributing

Thank you for helping improve the project. Keep changes focused, reproducible,
and safe for robot execution.

## Development setup

Create a virtual environment with Python 3.10 or newer, then install the test
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Isaac Sim is not required for the unit tests. Simulator integration tests must
be run with Isaac Sim's bundled Python as documented in the README.

## Checks

Run the complete unit test suite before opening a pull request:

```bash
python -m pytest -q
python -m compileall -q -x '(^|/)(outputs|vla_datasets|benchmark_results|observations)/' .
```

For controller or perception changes, include the command, seed, configuration,
and generated summary report used for validation. Do not commit raw datasets,
camera captures, checkpoints, API keys, or simulator logs.

## Safety boundary

The language model may select only locally registered skills. Metric poses,
joint targets, collision checks, retry budgets, and success verification remain
under deterministic local control. Changes must preserve this boundary unless
the pull request explicitly documents and tests a replacement safety model.
