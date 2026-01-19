use faer::prelude::*;
use faer::Mat;
use polars::prelude::*;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::time::Instant;

#[derive(Debug)]
pub struct OlsResult {
    pub betas: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub _n_observations: usize,
    pub _r_squared: f64,
    pub _adj_r_squared: f64,
}

pub struct FittedOls {
    _intercept: f64,
    _coefficients: Vec<f64>,
    result: OlsResult,
}

impl FittedOls {
    pub fn _intercept(&self) -> f64 {
        self._intercept
    }
    pub fn _coefficients(&self) -> &[f64] {
        &self._coefficients
    }
    pub fn result(&self) -> &OlsResult {
        &self.result
    }
}

/// Thin wrapper type, like `OlsRegressor` in polars-statistics.
pub struct OlsRegressor {
    with_intercept: bool,
}

pub struct OlsRegressorBuilder {
    with_intercept: bool,
}

impl OlsRegressor {
    pub fn builder() -> OlsRegressorBuilder {
        OlsRegressorBuilder {
            with_intercept: true,
        }
    }

    pub fn fit(&self, x: &Mat<f64>, y: &Mat<f64>) -> Result<FittedOls, String> {
        let t_total = Instant::now();
        
        let n = x.nrows();
        let k = x.ncols();
        println!("fit() called: n={}, k={}", n, k);

        // ... validation ...

        let t0 = Instant::now();
        let (xtx, xty) = compute_xtx_xty(x, y);
        println!("  XtX/Xty: {:?}", t0.elapsed());

        let t1 = Instant::now();
        let llt = xtx
            .llt(faer::Side::Lower)
            .map_err(|_| "Matrix is singular (collinear columns)".to_string())?;
        println!("  Cholesky: {:?}", t1.elapsed());

        let t2 = Instant::now();
        let beta_mat = llt.solve(&xty);
        let betas_full: Vec<f64> = (0..k).map(|i| beta_mat[(i, 0)]).collect();
        println!("  Solve beta: {:?}", t2.elapsed());
        // Treat last column as intercept if `with_intercept == true`
        let (_intercept, betas) = if self.with_intercept && k > 0 {
            let _intercept = *betas_full.last().unwrap();
            (_intercept, betas_full[..k - 1].to_vec())
        } else {
            // keep betas_full available for residuals and inference
            (0.0, betas_full.clone())
        };

        let effective_k = if self.with_intercept && k > 0 { k - 1 } else { k };

        // --- Residuals, RSS, TSS, R², adj-R² ---

        let t3 = Instant::now();
        // RSS/TSS loop
        let mut rss = 0.0;
        let mut tss = 0.0;
        let mut y_mean = 0.0;
        for row in 0..n {
            y_mean += y[(row, 0)];
        }
        y_mean /= n as f64;

        for row in 0..n {
            let mut y_hat = 0.0;
            for j in 0..k {
                y_hat += x[(row, j)] * betas_full[j];
            }
            let resid = y[(row, 0)] - y_hat;
            rss += resid * resid;
            let diff = y[(row, 0)] - y_mean;
            tss += diff * diff;
        }
        println!("  RSS/TSS: {:?}", t3.elapsed());

        let _r_squared = if tss > 0.0 { 1.0 - rss / tss } else { f64::NAN };
        let _dof_model = effective_k as f64;
        let dof_resid = (n - k) as f64;

        let mse = if dof_resid > 0.0 { rss / dof_resid } else { f64::NAN };
        let _rmse = mse.sqrt();

        let _adj_r_squared = if (n as f64 - 1.0) > 0.0 && (n as f64 - effective_k as f64 - 1.0) > 0.0
        {
            1.0 - (1.0 - _r_squared)
                * ((n as f64 - 1.0) / (n as f64 - effective_k as f64 - 1.0))
        } else {
            f64::NAN
        };

        // --- Standard errors, t-stats, p-values ---

        let t4 = Instant::now();
        let sigma2 = rss / (n - k) as f64;
        let identity = Mat::<f64>::identity(k, k);
        let inv_xtx = llt.solve(&identity);
        println!("  Invert XtX: {:?}", t4.elapsed());

        let mut std_errors_full = Vec::with_capacity(k);
        let mut t_stats_full = Vec::with_capacity(k);
        let mut p_values_full = Vec::with_capacity(k);

        let t_dist = StudentsT::new(0.0, 1.0, (n - k) as f64)
            .map_err(|e| format!("Failed to construct t distribution: {e}"))?;

        for i in 0..k {
            let var_beta = sigma2 * inv_xtx[(i, i)];
            let se = var_beta.sqrt();
            let beta = betas_full[i];
            let t = beta / se;
            let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));

            std_errors_full.push(se);
            t_stats_full.push(t);
            p_values_full.push(p);
        }

        println!("  TOTAL fit(): {:?}", t_total.elapsed());

        // Drop intercept row from inference vectors if with_intercept.
        let (std_errors, t_stats, p_values) = if self.with_intercept && k > 0 {
            (
                std_errors_full[..k - 1].to_vec(),
                t_stats_full[..k - 1].to_vec(),
                p_values_full[..k - 1].to_vec(),
            )
        } else {
            (std_errors_full, t_stats_full, p_values_full)
        };

        let result = OlsResult {
            betas,
            std_errors,
            t_stats,
            p_values,
            _n_observations: n,
            _r_squared,
            _adj_r_squared,
        };

        Ok(FittedOls {
            _intercept,
            _coefficients: result.betas.clone(),
            result,
        })
    }
}

impl OlsRegressorBuilder {
    pub fn with_intercept(mut self, yes: bool) -> Self {
        self.with_intercept = yes;
        self
    }

    pub fn build(self) -> OlsRegressor {
        OlsRegressor {
            with_intercept: self.with_intercept,
        }
    }
}

/// Helper like `build_xy_data` in polars-statistics.
/// Here we assume:
///   - inputs[y_idx] is y,
///   - inputs[x_start..] are predictors,
///   - no nulls, all castable to Float64.
pub fn build_xy_data(
    inputs: &[Series],
    y_idx: usize,
    x_start: usize,
) -> Result<(Mat<f64>, Mat<f64>, Vec<String>), PolarsError> {
    let y_series = &inputs[y_idx];
    let x_series_list = &inputs[x_start..];

    // y: cast to Float64 and collect
    let y_f = y_series.cast(&DataType::Float64)?;
    let y_ca = y_f.f64().unwrap();
    let n = y_ca.len();

    let mut y_mat = Mat::<f64>::zeros(n, 1);
    for (row, val) in y_ca.into_no_null_iter().enumerate() {
        y_mat[(row, 0)] = val;
    }

    // X: (n, k), each column from one Series, scalar columns broadcast
    let k = x_series_list.len();
    let mut x_mat = Mat::<f64>::zeros(n, k);
    let mut names = Vec::with_capacity(k);

    for (j, s) in x_series_list.iter().enumerate() {
        let s_f = s.cast(&DataType::Float64)?;
        let ca = s_f.f64().unwrap();

        if ca.len() == 1 && n > 1 {
            let val = ca.get(0).unwrap();
            for row in 0..n {
                x_mat[(row, j)] = val;
            }
        } else {
            if ca.len() != n {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Column '{}' has length {}, expected {}",
                        s.name(),
                        ca.len(),
                        n
                    )
                    .into(),
                ));
            }
            for (row, val) in ca.into_no_null_iter().enumerate() {
                x_mat[(row, j)] = val;
            }
        }

        names.push(s.name().to_string());
    }

    Ok((x_mat, y_mat, names))
}

fn compute_xtx_xty(x: &Mat<f64>, y: &Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let n = x.nrows();
    let k = x.ncols();
    
    let mut xtx = Mat::<f64>::zeros(k, k);
    let mut xty = Mat::<f64>::zeros(k, 1);
    
    for row in 0..n {
        let y_val = y[(row, 0)];
        for i in 0..k {
            let xi = x[(row, i)];
            xty[(i, 0)] += xi * y_val;
            for j in 0..=i {
                xtx[(i, j)] += xi * x[(row, j)];
            }
        }
    }
    
    // Mirror lower triangle to upper
    for i in 0..k {
        for j in (i + 1)..k {
            xtx[(i, j)] = xtx[(j, i)];
        }
    }
    
    (xtx, xty)
}