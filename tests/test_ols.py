import os
from pathlib import Path
import polars as pl
import polars.selectors as cs
import polars_hdfe

TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = TEST_DIR.parent
DATA_DIR = f'{PROJECT_ROOT}/data'

FE_DATA = f'{DATA_DIR}/data_fe.pq'
LDFE_DATA = f'{DATA_DIR}/data_ldfe.pq'
HDFE_DATA = f'{DATA_DIR}/data_hdfe.pq'
UHDFE_DATA = f'{DATA_DIR}/data_uhdfe.pq'
MEGA_FE_DATA = f'{DATA_DIR}/data_mega_fe.pq'


# TODO: Allow for more than just Float64
df = pl.read_parquet(FE_DATA).with_columns(cs.integer().cast(pl.Float64))


# Usage: pl.col("target").namespace.ols(features=[list_of_features])
result = df.select(
    pl.col("log_wage").least_squares.ols(
        features=[pl.col("experience"), pl.col("education"), pl.col("age"), pl.col("age_sq")],
        add_intercept=True
    ).alias("ols_result")
)

# Inspect output
print(result.unnest("ols_result"))
