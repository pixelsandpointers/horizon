use std::sync::Arc;

use super::elementwise::Elementwise;
use crate::node::Node;
use crate::tensor::Tensor;

type TensorFn = Arc<dyn Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync>;

#[derive(Clone)]
enum Forward {
    Tensor(TensorFn),
    Scalar(Elementwise),
}

/// Deterministic variable, like NumPyro's `deterministic`.
#[derive(Clone)]
pub struct Function {
    arity: usize,
    forward: Forward,
}

impl Function {
    /// A function written with tensors, differentiated by burn.
    pub fn new(
        arity: usize,
        func: impl Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync + 'static,
    ) -> Self {
        Self {
            arity,
            forward: Forward::Tensor(Arc::new(func)),
        }
    }

    /// A function of scalar inputs, applied to each entry of inputs broadcast against each
    /// other. Its gradient comes from central differences unless given by
    /// [`Function::with_gradient`].
    pub fn scalar(arity: usize, func: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            arity,
            forward: Forward::Scalar(Elementwise::new(func)),
        }
    }

    /// The exact partial derivatives of a [`Function::scalar`], one per input.
    /// # Panics
    /// Panics if the function is written with tensors.
    pub fn with_gradient(
        mut self,
        gradient: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        let Forward::Scalar(f) = self.forward else {
            panic!("`with_gradient` needs a `Function::scalar`");
        };
        self.forward = Forward::Scalar(f.with_partials(gradient));
        self
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
        match &self.forward {
            Forward::Tensor(f) => f(inputs),
            Forward::Scalar(_) => unimplemented!("a `Function::scalar` has no tensor forward"),
        }
    }

    fn forward_f64(&self, inputs: &[&[f64]]) -> Option<Vec<f64>> {
        match &self.forward {
            Forward::Tensor(_) => None,
            Forward::Scalar(f) => Some(f.values(inputs)),
        }
    }

    fn vjp(&self, inputs: &[&[f64]], adjoint: &[f64]) -> Option<Vec<Vec<f64>>> {
        match &self.forward {
            Forward::Tensor(_) => None,
            Forward::Scalar(f) => Some(f.weighted_grad(inputs, |i| adjoint[i])),
        }
    }
}
