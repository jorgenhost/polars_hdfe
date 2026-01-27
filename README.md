# polars-hdfe

## Development

Create the environment with dev dependencies:

```bash
uv sync --dev
```

Build and install the extension (when you change any rust code) in the environment:

```bash
maturin develop --release --uv

# include timings

maturin develop --release --uv --features timing
```

Run tests:

```bash
# create datasets
uv run tests/_gen_data.py

uv run tests/test_ols.py
```
