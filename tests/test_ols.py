import os
from pathlib import Path
import polars as pl
import polars.selectors as cs
import polars_hdfe
import polars_ols as pls
import time

TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = TEST_DIR.parent
DATA_DIR = f'{PROJECT_ROOT}/data'

FE_DATA = f'{DATA_DIR}/data_fe.pq'
LDFE_DATA = f'{DATA_DIR}/data_ldfe.pq'
HDFE_DATA = f'{DATA_DIR}/data_hdfe.pq'
UHDFE_DATA = f'{DATA_DIR}/data_uhdfe.pq'
MEGA_FE_DATA = f'{DATA_DIR}/data_mega_fe.pq'


# TODO: Allow for more than just Float64
df_fe = pl.scan_parquet(FE_DATA).with_columns(cs.integer().cast(pl.Float64)).collect()

def test_ols_basic():
    start = time.perf_counter()

    result = df_fe.select(
        pl.col("log_wage").hdfe_least_squares.ols(
            features=[
                pl.col("experience"),
                pl.col("education"),
                pl.col("age"),
                pl.col("age_sq"),
            ],
        ).alias("ols_result")
    )
    # Inspect output
    print(result.unnest("ols_result"))
    elapsed = time.perf_counter() - start
    print(f"test_ols_basic: {elapsed:.3f} s")


test_ols_basic()

def test_ols_basic_pls():
    start = time.perf_counter()
    coefficients = df_fe.select(pl.col("log_wage").least_squares.from_formula("experience + education + age + age_sq", mode="coefficients")
                         .alias("coefficients"))
    # Inspect output
    print(coefficients)
    elapsed = time.perf_counter() - start
    print(f"test_ols_basic_pls: {elapsed:.3f} s")


test_ols_basic_pls()

del df_fe

df_uhdfe = pl.scan_parquet(UHDFE_DATA).with_columns(cs.integer().cast(pl.Float64)).collect()

def test_ols_uhdfe():
    start = time.perf_counter()

    result = df_uhdfe.select(
        pl.col("log_wage").hdfe_least_squares.ols(
            features=[
                pl.col("experience"),
                pl.col("education"),
                pl.col("age"),
                pl.col("age_sq"),
            ],
        ).alias("ols_result")
    )
    # Inspect output
    print(result.unnest("ols_result"))
    elapsed = time.perf_counter() - start
    print(f"test_ols_uhdfe: {elapsed:.3f} s")

test_ols_uhdfe()

def test_ols_uhdfe_pls():
    start = time.perf_counter()
    coefficients = df_uhdfe.select(pl.col("log_wage").least_squares.from_formula("experience + education + age + age_sq", mode="coefficients")
                         .alias("coefficients"))
    # Inspect output
    print(coefficients)
    elapsed = time.perf_counter() - start
    print(f"test_ols_uhdfe_pls: {elapsed:.3f} s")

test_ols_uhdfe_pls()