use super::diagnostics::{effective_sample_size, quantile, split_rhat};
use super::summary::{Summary, SummaryRow};
use crate::model::VarId;

/// Draws from the posterior, chain by chain.
#[derive(Clone, Debug)]
pub struct Samples {
    pub(super) names: Vec<String>,
    pub(super) vars: Vec<VarId>,
    /// `chains[c][d][k]`: the entries of recorded variable `k` in draw `d` of chain `c`.
    pub(super) chains: Vec<Vec<Vec<Vec<f64>>>>,
    /// Divergent transitions after warmup, over all chains.
    pub divergences: usize,
}

impl Samples {
    fn index(&self, v: VarId) -> usize {
        self.vars
            .iter()
            .position(|&x| x == v)
            .expect("the variable is data or not part of the model")
    }

    /// Entry `i` of a variable's draws, chain by chain.
    pub fn component(&self, v: VarId, i: usize) -> Vec<Vec<f64>> {
        let k = self.index(v);
        self.chains
            .iter()
            .map(|c| c.iter().map(|d| d[k][i]).collect())
            .collect()
    }

    /// The posterior mean of every entry of a variable.
    pub fn mean(&self, v: VarId) -> Vec<f64> {
        let k = self.index(v);
        let len = self.chains[0][0][k].len();
        (0..len)
            .map(|i| {
                let all: Vec<f64> = self.component(v, i).into_iter().flatten().collect();
                all.iter().sum::<f64>() / all.len() as f64
            })
            .collect()
    }

    /// NumPyro-style summary statistics for every entry of every variable.
    pub fn summary(&self) -> Summary {
        let mut rows = Vec::new();
        for (k, name) in self.names.iter().enumerate() {
            let len = self.chains[0][0][k].len();
            for i in 0..len {
                let chains = self.component(self.vars[k], i);
                let all: Vec<f64> = chains.iter().flatten().copied().collect();
                let n = all.len() as f64;
                let mean = all.iter().sum::<f64>() / n;
                let std = (all.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
                rows.push(SummaryRow {
                    name: if len == 1 {
                        name.clone()
                    } else {
                        format!("{name}[{i}]")
                    },
                    mean,
                    std,
                    median: quantile(&all, 0.5),
                    q05: quantile(&all, 0.05),
                    q95: quantile(&all, 0.95),
                    ess: effective_sample_size(&chains),
                    rhat: split_rhat(&chains),
                });
            }
        }
        Summary {
            rows,
            divergences: self.divergences,
        }
    }
}
