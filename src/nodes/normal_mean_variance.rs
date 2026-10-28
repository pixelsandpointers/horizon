use crate::node::{Node, Support};
use crate::tensor::Tensor;

/// `out ~ Normal(mean, variance)`, mean and variances are broadcasted against `out`.
#[derive(Clone, Copy, Debug)]
pub struct NormalMeanVariance;

impl Node for NormalMeanVariance {
    fn name(&self) -> &str {
        "NormalMeanVariance"
    }

    fn arity(&self) -> usize {
        3
    }

    fn support(&self, interface: usize) -> Option<Support> {
        match interface {
            2 => Some(Support::Positive),
            _ => Some(Support::Real),
        }
    }

    fn log_density(&self, values: &[Tensor<1>]) -> Tensor<1> {
        let [y, m, v] = [&values[0], &values[1], &values[2]].map(Clone::clone);
        let residual = y - m;
        let log_norm = v.clone().mul_scalar(2.0 * std::f64::consts::PI).log();
        (log_norm + residual.clone() * residual / v)
            .sum()
            .mul_scalar(-0.5)
    }

    fn log_density_grad(&self, values: &[&[f64]]) -> Option<(f64, Vec<Vec<f64>>)> {
        let [y, m, v] = [values[0], values[1], values[2]];
        let mut grads: Vec<Vec<f64>> = values.iter().map(|x| vec![0.0; x.len()]).collect();
        let at = |x: &[f64], i: usize| if x.len() == 1 { 0 } else { i };
        let mut total = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let (mi, vi) = (at(m, i), at(v, i));
            let residual = yi - m[mi];
            let z = residual / v[vi];
            total -= 0.5 * ((2.0 * std::f64::consts::PI * v[vi]).ln() + residual * z);
            grads[0][i] -= z;
            grads[1][mi] += z;
            grads[2][vi] -= 0.5 * (1.0 - residual * z) / v[vi];
        }
        Some((total, grads))
    }
}
