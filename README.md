# polars-hdfe

## Development

Create the environment with dev dependencies:

```bash
uv sync --dev
```

Build and install the extension in the environment:

```bash
maturin develop --release --uv

# include timings

maturin develop --release --uv --features timing
```

Run tests:

```bash
uv run tests/test_ols.py
```
