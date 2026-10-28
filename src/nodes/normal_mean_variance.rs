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
}
