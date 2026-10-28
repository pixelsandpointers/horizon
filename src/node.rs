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

    /// Whether interface 0 is a function of the others.
    fn is_deterministic(&self) -> bool {
        false
    }

    /// The output of a deterministic node from the values of its other interfaces.
    fn forward(&self, inputs: &[Tensor<1>]) -> Tensor<1> {
        let _ = inputs;
        unimplemented!("`{}` is not deterministic", self.name())
    }

    /// `log f(values)` and its gradient for each interface, without tensors. `None` (the
    /// default) has sampling differentiate [`Node::log_density`] with burn instead.
    fn log_density_grad(&self, values: &[&[f64]]) -> Option<(f64, Vec<Vec<f64>>)> {
        let _ = values;
        None
    }

    /// The output of a deterministic node, without tensors. `None` (the default) falls back to
    /// [`Node::forward`].
    fn forward_f64(&self, inputs: &[&[f64]]) -> Option<Vec<f64>> {
        let _ = inputs;
        None
    }

    /// The vector–Jacobian product `adjointᵀ ∂output/∂input` for each input of a deterministic
    /// node. `None` (the default) has sampling differentiate [`Node::forward`] with burn instead.
    fn vjp(&self, inputs: &[&[f64]], adjoint: &[f64]) -> Option<Vec<Vec<f64>>> {
        let _ = (inputs, adjoint);
        None
    }
}
