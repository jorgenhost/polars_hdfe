# polars-hdfe

## Development

Create the environment and install dependencies (skipping the package build):

```bash
uv sync --dev --no-install-project

# windows
.venv\Scripts\activate

# linux/mac
source .venv/bin/activate

```

```bash
maturin develop --release 

# include timings
maturin develop --release --features timing -v
```

Run tests:

```bash
# create datasets
python tests/_gen_data.py

# run tests
python tests/test_ols.py

## can also do
uv run --no-sync tests/_gen_data.py
uv run --no-sync tests/test_ols.py
```
