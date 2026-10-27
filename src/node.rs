//! The factor node interface.

use crate::tensor::Tensor;

/// The set of values a variable can take, which sampling maps to an unconstrained space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Real,
    Positive,
}

/// A factor node. Each interface's value arrives as a one-dimensional tensor of its entries.
/// Stochastic nodes give a log density.
pub trait Node: Send + Sync {
    /// Name used in error messages.
    fn name(&self) -> &str;

    /// The number of interfaces/variables attached.
    fn arity(&self) -> usize;

    /// The support this node gives the variable on interface if it constrains it.
    fn support(&self, interface: usize) -> Option<Support> {
        let _ = interface;
        None
    }

    /// `log f(values)` up to a constant, as a tensor with one entry.
    fn log_density(&self, values: &[Tensor<1>]) -> Tensor<1> {
        let _ = values;
        unimplemented!("`{}` has no log density", self.name())
    }

    fn is_deterministic(&self) -> bool {
        false
    }

    fn forward(&self, inputs: &[Tensor<1>]) -> Tensor<1> {
        let _ = inputs;
        unimplemented!("`{}` is not deterministic", self.name())
    }
}
