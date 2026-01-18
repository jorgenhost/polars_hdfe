use faer::Mat;
use faer::prelude::*; 
use statrs::distribution::{StudentsT, ContinuousCDF};

#[derive(Debug)]
pub struct OlsResult {
    pub betas: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub t_stats: Vec<f64>,
    pub p_values: Vec<f64>,
}

pub fn solve_ols(y: &[f64], x: &[Vec<f64>]) -> Result<OlsResult, String> {
    let n = y.len();
    let k = x.len();

    if n <= k {
        return Err("Insufficient degrees of freedom".to_string());
    }

    // --- Build X as an faer::Mat of shape (n, k) from column vectors x[j][row] ---
    // Assume x[j].len() == n for all j
    let mut x_mat = Mat::<f64>::zeros(n, k);
    for (j, col) in x.iter().enumerate() {
        debug_assert_eq!(col.len(), n);
        for (row, &val) in col.iter().enumerate() {
            // faer is row-major: (row, col)
            x_mat[(row, j)] = val;
        }
    }

    // --- Build y as an faer column vector (n, 1) ---
    let mut y_vec = Mat::<f64>::zeros(n, 1);
    for (row, &val) in y.iter().enumerate() {
        y_vec[(row, 0)] = val;
    }

    // --- Compute xtx = X^T * X and xty = X^T * y using faer operations ---
    // X has shape (n, k), so:
    // X^T has shape (k, n)
    // X^T * X has shape (k, k)
    // X^T * y has shape (k, 1)
    let x_t = x_mat.transpose(); // (k, n)

    let xtx = &x_t * &x_mat;     // (k, k)
    let xty = &x_t * &y_vec;     // (k, 1)

    // --- Solve (X^T X) beta = X^T y via Cholesky ---
    let llt = xtx
        .llt(faer::Side::Lower)
        .map_err(|_| "Matrix is singular (collinear columns)".to_string())?;

    let beta_mat = llt.solve(&xty); // (k, 1)
    let betas: Vec<f64> = (0..k).map(|i| beta_mat[(i, 0)]).collect();

    // --- Compute residuals, RSS, TSS, etc., like you already do ---
    let mut rss = 0.0;
    let mut tss = 0.0;
    let y_mean = y.iter().sum::<f64>() / n as f64;

    for row_idx in 0..n {
        let mut y_hat = 0.0;
        for (j, beta) in betas.iter().enumerate() {
            y_hat += x[j][row_idx] * beta;
        }
        let resid = y[row_idx] - y_hat;
        rss += resid * resid;
        tss += (y[row_idx] - y_mean).powi(2);
    }

    let _r2 = 1.0 - (rss / tss);
    let sigma2 = rss / (n - k) as f64;

    // Variances: Var(beta) = sigma^2 * diag((X^T X)^{-1})
    let identity = Mat::<f64>::identity(k, k);
    let inv_xtx = llt.solve(&identity);

    let mut std_errors = Vec::with_capacity(k);
    let mut t_stats = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);

    let t_dist = StudentsT::new(0.0, 1.0, (n - k) as f64).unwrap();

    for i in 0..k {
        let var_beta = sigma2 * inv_xtx[(i, i)];
        let se = var_beta.sqrt();
        let beta = betas[i];
        let t = beta / se;
        let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));

        std_errors.push(se);
        t_stats.push(t);
        p_values.push(p);
    }

    Ok(OlsResult {
        betas,
        std_errors,
        t_stats,
        p_values,
    })
}