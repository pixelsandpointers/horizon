use std::sync::Arc;

use crate::node::{Node, Support};
use crate::tensor::Tensor;

type TensorFn = Arc<dyn Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync>;

/// Arbitrary log-density term `log f(arg0, ...)`, like NumPyro's `factor`
#[derive(Clone)]
pub struct Potential {
    arity: usize,
    density: TensorFn,
    supports: Vec<Option<Support>>,
}

impl Potential {
    pub fn new(
        arity: usize,
        density: impl Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync + 'static,
    ) -> Self {
        Self {
            arity,
            density: Arc::new(density),
            supports: vec![None; arity],
        }
    }

    pub fn with_support(mut self, k: usize, support: Support) -> Self {
        self.supports[k] = Some(support);
        self
    }
}

impl Node for Potential {
    fn name(&self) -> &str {
        "Potential"
    }

    fn arity(&self) -> usize {
        self.arity
    }

    fn support(&self, interface: usize) -> Option<Support> {
        self.supports[interface]
    }

    fn log_density(&self, values: &[Tensor<1>]) -> Tensor<1> {
        (self.density)(values)
    }
}
