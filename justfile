# Install the project and dev tools from the lock file
sync:
    uv sync --locked

# Fix lint and format
format:
    uv run ruff check --fix .
    uv run ruff format .

# Check lint, format and types without changing files
lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run basedpyright

# Run tests with coverage
test:
    uv run pytest -n auto --cov --cov-report=term-missing
