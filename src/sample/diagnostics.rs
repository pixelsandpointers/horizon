//! Convergence diagnostics for several chains of draws of one scalar: split R-hat and the
//! effective sample size, computed as in Stan.

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0)
}

/// Each chain split into halves, which also exposes trends within a chain.
fn split(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    chains
        .iter()
        .flat_map(|c| {
            let half = c.len() / 2;
            [c[..half].to_vec(), c[c.len() - half..].to_vec()]
        })
        .collect()
}

/// The potential scale reduction factor on split chains; values near 1 indicate convergence.
pub fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let chains = split(chains);
    let n = chains[0].len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let within = mean(&chains.iter().map(|c| variance(c)).collect::<Vec<_>>());
    let between = n * variance(&means);
    if within == 0.0 {
        return if between == 0.0 { 1.0 } else { f64::INFINITY };
    }
    let var_plus = (n - 1.0) / n * within + between / n;
    (var_plus / within).sqrt()
}

/// Autocovariance of one chain at a lag.
fn autocovariance(x: &[f64], m: f64, lag: usize) -> f64 {
    let n = x.len();
    (0..n - lag)
        .map(|i| (x[i] - m) * (x[i + lag] - m))
        .sum::<f64>()
        / n as f64
}

/// The effective sample size over split chains, with Geyer's initial monotone sequence.
pub fn effective_sample_size(chains: &[Vec<f64>]) -> f64 {
    let chains = split(chains);
    let (m, n) = (chains.len(), chains[0].len());
    if n < 4 {
        return f64::NAN;
    }
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let within = mean(&chains.iter().map(|c| variance(c)).collect::<Vec<_>>());
    let between = n as f64 * variance(&means);
    let var_plus = (n as f64 - 1.0) / n as f64 * within + between / n as f64;
    if var_plus == 0.0 {
        return (m * n) as f64;
    }
    let rho = |lag: usize| {
        let acov: f64 = chains
            .iter()
            .zip(&means)
            .map(|(c, &mu)| autocovariance(c, mu, lag))
            .sum::<f64>()
            / m as f64;
        1.0 - (within - acov) / var_plus
    };
    // Sum pairs of autocorrelations while they stay positive, never letting them increase.
    let mut tau = -1.0;
    let mut previous_pair = f64::INFINITY;
    let mut lag = 0;
    while lag + 1 < n {
        let pair = (rho(lag) + rho(lag + 1)).min(previous_pair);
        if pair <= 0.0 {
            break;
        }
        tau += 2.0 * pair;
        previous_pair = pair;
        lag += 2;
    }
    (m * n) as f64 / tau.max(1.0 / ((m * n) as f64).log10())
}

/// The `q`-quantile of a sample, by linear interpolation.
pub fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = q * (sorted.len() - 1) as f64;
    let (lo, hi) = (position.floor() as usize, position.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo as f64)
}
