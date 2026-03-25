"""Invoke tasks for the skin-type-classifier project."""

from invoke import task


@task
def learning_curve(c, overrides=""):
    """Run the learning curve experiment.

    Args:
        c: Invoke context.
        overrides: Space-separated Hydra overrides (e.g. "training.max_epochs=5 model.dropout=0.5").
    """
    c.run(f"uv run python scripts/run_learning_curve.py {overrides}")


@task
def test(c, markers="unit"):
    """Run tests with pytest.

    Args:
        c: Invoke context.
        markers: Pytest marker expression (default: "unit").
    """
    c.run(f"uv run pytest tests/ -m {markers}")


@task
def lint(c):
    """Run ruff linter and formatter."""
    c.run("uv run ruff check . --fix")
    c.run("uv run ruff format .")
