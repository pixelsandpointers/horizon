use std::sync::Arc;

use super::elementwise::Elementwise;
use crate::node::{Node, Support};
use crate::tensor::Tensor;

type TensorFn = Arc<dyn Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync>;

#[derive(Clone)]
enum Density {
    Tensor(TensorFn),
    Scalar(Elementwise),
}

/// Arbitrary log-density term `log f(arg0, ...)`, like NumPyro's `factor`
#[derive(Clone)]
pub struct Potential {
    arity: usize,
    density: Density,
    supports: Vec<Option<Support>>,
}

impl Potential {
    /// A term written with tensors, differentiated by burn.
    pub fn new(
        arity: usize,
        density: impl Fn(&[Tensor<1>]) -> Tensor<1> + Send + Sync + 'static,
    ) -> Self {
        Self::with_density(arity, Density::Tensor(Arc::new(density)))
    }

    /// A term over scalar arguments, summed over the entries of arguments broadcast against
    /// each other. Its gradient comes from central differences unless given by
    /// [`Potential::with_gradient`].
    pub fn scalar(arity: usize, density: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
        Self::with_density(arity, Density::Scalar(Elementwise::new(density)))
    }

    fn with_density(arity: usize, density: Density) -> Self {
        Self {
            arity,
            density,
            supports: vec![None; arity],
        }
    }

    pub fn with_support(mut self, k: usize, support: Support) -> Self {
        self.supports[k] = Some(support);
        self
    }

    /// The exact partial derivatives of a [`Potential::scalar`] term, one per argument.
    /// # Panics
    /// Panics if the term is written with tensors.
    pub fn with_gradient(
        mut self,
        gradient: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        let Density::Scalar(f) = self.density else {
            panic!("`with_gradient` needs a `Potential::scalar`");
        };
        self.density = Density::Scalar(f.with_partials(gradient));
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
        match &self.density {
            Density::Tensor(f) => f(values),
            Density::Scalar(_) => unimplemented!("a `Potential::scalar` has no tensor density"),
        }
    }

    fn log_density_grad(&self, values: &[&[f64]]) -> Option<(f64, Vec<Vec<f64>>)> {
        let Density::Scalar(f) = &self.density else {
            return None;
        };
        let total = f.values(values).iter().sum();
        Some((total, f.weighted_grad(values, |_| 1.0)))
    }
}
