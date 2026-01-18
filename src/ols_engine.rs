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

    let mut xtx = Mat::<f64>::zeros(k, k);
    let mut xty = Mat::<f64>::zeros(k, 1);

    for i in 0..k {
        let dot_y: f64 = x[i].iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        xty[(i, 0)] = dot_y;

        for j in i..k {
            let dot_x: f64 = x[i].iter().zip(x[j].iter()).map(|(a, b)| a * b).sum();
            xtx[(i, j)] = dot_x;
            if i != j { xtx[(j, i)] = dot_x; }
        }
    }

    let llt = xtx.llt(faer::Side::Lower)
        .map_err(|_| "Matrix is singular (collinear columns)".to_string())?;
    
    let beta_mat = llt.solve(&xty);
    let betas: Vec<f64> = (0..k).map(|i| beta_mat[(i, 0)]).collect();

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

    let _r2 = 1.0 - (rss/tss);
    let sigma2 = rss/(n-k) as f64;

    let identity = Mat::<f64>::identity(k, k);
    let inv_xtx = llt.solve(&identity);

    let mut std_errors = Vec::with_capacity(k);
    let mut t_stats = Vec::with_capacity(k);
    let mut p_values = Vec::with_capacity(k);

    let t_dist = StudentsT::new(0.0, 1.0, (n - k) as f64).unwrap();

    for i in 0..k {
        let var_beta = sigma2 * inv_xtx[(i, i)];
        let se  = var_beta.sqrt();
        let beta = betas[i];
        let t = beta/se;
        let p = 2.0 * (1.0 - t_dist.cdf(t.abs()));
        
        std_errors.push(se);
        t_stats.push(t);
        p_values.push(p);
    }

    Ok(OlsResult { betas, std_errors, t_stats, p_values})
}