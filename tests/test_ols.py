import os
from pathlib import Path
import polars as pl
import polars.selectors as cs
import polars_hdfe
import polars_ols as pls
import pyfixest as pf
import time
import tracemalloc
import gc

TEST_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = TEST_DIR.parent
DATA_DIR = f'{PROJECT_ROOT}/data'

FE_DATA = f'{DATA_DIR}/data_fe.pq'
LDFE_DATA = f'{DATA_DIR}/data_ldfe.pq'
HDFE_DATA = f'{DATA_DIR}/data_hdfe.pq'
UHDFE_DATA = f'{DATA_DIR}/data_uhdfe.pq'
MEGA_FE_DATA = f'{DATA_DIR}/data_mega_fe.pq'

print("VERSION:", pl.__version__)
print(pl.build_info())
print("Threads:", pl.thread_pool_size())

def start_benchmark():
    print("***" * 20)
    tracemalloc.start()
    start_time = time.perf_counter()
    return start_time

def end_benchmark(label: str, start_time: float):
    elapsed = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("===" * 20)
    print(f"{label}: {elapsed:.3f} s")
    print(f"{label} peak memory (tracemalloc): {peak / 1024**2:.2f} MiB")
    print("===" * 20)

# TODO: Allow for more than just Float64
lf_fe = pl.scan_parquet(FE_DATA)
df_fe = lf_fe.collect()

def test_ols_basic():
    start = start_benchmark()

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
    end_benchmark("test_ols_basic", start)
    print(result.unnest("ols_result"))

def test_ols_basic_lazy():
    start = start_benchmark()

    result = lf_fe.select(
        pl.col("log_wage").hdfe_least_squares.ols(
            features=[
                pl.col("experience"),
                pl.col("education"),
                pl.col("age"),
                pl.col("age_sq"),
            ],
        ).alias("ols_result")
    ).collect()
    end_benchmark("test_ols_basic_lazy", start)
    # Inspect output
    print(result.unnest("ols_result"))


def test_ols_basic_pyfixest():
    start = start_benchmark()
    mod=pf.feols(
        fml = "log_wage ~ experience + education + age + age_sq",
        data = df_fe,
        lean=True,
        store_data=False,
        copy_data=False
    )
    end_benchmark("test_ols_basic_pyfixest", start)
    # Inspect output
    print(mod.summary())
    

def test_ols_basic_pls():
    start = start_benchmark()
    coefficients = df_fe.select(pl.col("log_wage").least_squares.from_formula("experience + education + age + age_sq", mode="coefficients")
                         .alias("coefficients"))

    end_benchmark("test_ols_basic_pls", start)
    # Inspect output
    print(coefficients)

test_ols_basic()
test_ols_basic_lazy()
test_ols_basic_pyfixest()
test_ols_basic_pls()

del df_fe
gc.collect()
lf_uhdfe = pl.scan_parquet(UHDFE_DATA)
df_uhdfe = lf_uhdfe.collect()

def test_ols_uhdfe():
    start = start_benchmark()
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

    end_benchmark("test_ols_uhdfe", start)

    # Inspect output
    print(result.unnest("ols_result"))
    
def test_ols_uhdfe_lazy():
    start = start_benchmark()

    result = lf_uhdfe.select(
        pl.col("log_wage").hdfe_least_squares.ols(
            features=[
                pl.col("experience"),
                pl.col("education"),
                pl.col("age"),
                pl.col("age_sq"),
            ],
        ).alias("ols_result")
    ).collect()

    end_benchmark("test_ols_uhdfe_lazy", start)

    # Inspect output
    print(result.unnest("ols_result"))

def test_ols_uhdfe_pls():
    start = start_benchmark()
    coefficients = df_uhdfe.select(pl.col("log_wage").least_squares.from_formula("experience + education + age + age_sq", mode="coefficients")
                         .alias("coefficients"))
    # Inspect output
    print(coefficients)
    end_benchmark("test_ols_uhdfe_pls", start)

def test_ols_uhdfe_pyfixest():
    start = start_benchmark()
    mod=pf.feols(
        fml = "log_wage ~ experience + education + age + age_sq",
        data = df_uhdfe,
        lean=True,
        store_data=False,
        copy_data=False
    )

    end_benchmark("test_ols_uhdfe_pyfixest", start)

    # Inspect output
    print(mod.summary())

test_ols_uhdfe()
test_ols_uhdfe_lazy()
test_ols_uhdfe_pls()
# test_ols_uhdfe_pyfixest() # OOM