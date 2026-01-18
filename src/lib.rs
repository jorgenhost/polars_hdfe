mod expressions;
mod ols_engine;
use ols_engine::solve_ols;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3::prelude::*;
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
    let y_series = &inputs[0];
    let x_series_list = &inputs[1..];

    // Convert Series to strict f64 Vec, assuming no nulls
    let y: Vec<f64> = y_series.f64()?.into_no_null_iter().collect();
    let n = y.len();

    let mut x_vecs: Vec<Vec<f64>> = Vec::with_capacity(x_series_list.len());
    let mut x_names: Vec<String> = Vec::with_capacity(x_series_list.len());

    for s in x_series_list {
        let s_vals: Vec<f64> = s.f64()?.into_no_null_iter().collect();
        
        // Broadcast scalar values (like intercept) to match n
        if s_vals.len() == 1 && n > 1 {
            x_vecs.push(vec![s_vals[0]; n]);
        } else {
            x_vecs.push(s_vals);
        }
        
        x_names.push(s.name().to_string());
    }

    let result = solve_ols(&y, &x_vecs).map_err(|e| PolarsError::ComputeError(e.into()))?;

    let name_s = Series::new("names".into(), x_names);
    let beta_s = Series::new("beta".into(), result.betas);
    let se_s = Series::new("se".into(), result.std_errors);
    let t_s = Series::new("t_stat".into(), result.t_stats);
    let p_s = Series::new("p_val".into(), result.p_values);

    let df = DataFrame::new(vec![
        name_s.into(), 
        beta_s.into(), 
        se_s.into(), 
        t_s.into(), 
        p_s.into()
    ])?;
    
    let out_chunked = df.into_struct("result".into());

    Ok(out_chunked.into_series())
}