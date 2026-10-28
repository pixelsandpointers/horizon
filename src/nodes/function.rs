use std::sync::Arc;

use crate::node::Node;
use crate::tensor::Tensor;

type TensorFn = Arc<dyn Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync>;

/// Deterministic variable, like NumPyro's `deterministic`.
#[derive(Clone)]
pub struct Function {
    arity: usize,
    forward: TensorFn,
}

impl Function {
    pub fn new(
        arity: usize,
        func: impl Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync + 'static,
    ) -> Self {
        Self {
            arity,
            forward: Arc::new(func),
        }
    }
}

impl Node for Function {
    fn name(&self) -> &str {
        "Function"
    }

    fn arity(&self) -> usize {
        self.arity + 1
    }

    fn is_deterministic(&self) -> bool {
        true
    }

    fn forward(&self, inputs: &[Tensor<1>]) -> Tensor<1> {
        (self.forward)(inputs)
    }
}
