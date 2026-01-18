mod expressions;
mod ols_engine;

use ols_engine::{build_xy_data, OlsRegressor};
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn _internal(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

fn ols_output(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new("names".into(), DataType::List(Box::new(DataType::String))),
        Field::new("beta".into(), DataType::List(Box::new(DataType::Float64))),
        Field::new("se".into(), DataType::List(Box::new(DataType::Float64))),
        Field::new("t_stat".into(), DataType::List(Box::new(DataType::Float64))),
        Field::new("p_val".into(), DataType::List(Box::new(DataType::Float64))),
    ];
    Ok(Field::new("result".into(), DataType::Struct(fields)))
}

#[polars_expr(output_type_func=ols_output)]
fn pl_ols(inputs: &[Series]) -> PolarsResult<Series> {
    let with_intercept = true;

    let (x_mat, y_mat, mut x_names) = build_xy_data(inputs, 0, 1)?;

    let model = OlsRegressor::builder()
        .with_intercept(with_intercept)
        .build();

    let fitted = model
        .fit(&x_mat, &y_mat)
        .map_err(|e| PolarsError::ComputeError(e.into()))?;

    let result = fitted.result();

    if with_intercept && !x_names.is_empty() {
        x_names.pop();
    }

    let name_s = Series::new("names".into(), x_names);
    let beta_s = Series::new("beta".into(), result.betas.clone());
    let se_s = Series::new("se".into(), result.std_errors.clone());
    let t_s = Series::new("t_stat".into(), result.t_stats.clone());
    let p_s = Series::new("p_val".into(), result.p_values.clone());

    let df = DataFrame::new(vec![
        name_s.into(),
        beta_s.into(),
        se_s.into(),
        t_s.into(),
        p_s.into(),
    ])?;

    let out_chunked = df.into_struct("result".into());
    Ok(out_chunked.into_series())
}