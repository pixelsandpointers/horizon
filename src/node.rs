//! The factor node interface.

use crate::{dist::Dist, error::Result, tensor::Tensor};

/// The set of values a variable can take, which sampling maps to an unconstrained space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Support {
    Real,
    Positive,
}

/// What a node knows about one of its interfaces when computing a message.
///
/// Which variant an interface receives follows from the node's factorization constraint: an
/// interface in the same cluster as the outgoing one, or in another cluster with several
/// members, contributes its incoming *message* (belief propagation within a cluster); an interface
/// in a cluster of its own contributes its *marginal* from the previous iteration (variational
/// message passing between clusters). Constants and data always arrive as point-mass messages.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> {
    /// The interface the rule computes a message for, with the message last sent out on it, or
    /// `None` before the first update.
    Outgoing(Option<&'a Dist>),
    /// The message arriving at the node from the variable on this interface, as in belief
    /// propagation.
    Message(&'a Dist),
    /// The current marginal of the variable on this interface, as variational rules use in
    /// place of its incoming message.
    Marginal(&'a Dist),
}

impl<'a> Input<'a> {
    pub fn dist(&self) -> Option<&'a Dist> {
        match *self {
            Self::Message(d) | Self::Marginal(d) => Some(d),
            Self::Outgoing(_) => None,
        }
    }

    pub fn is_marginal_like(&self) -> bool {
        match self {
            Self::Marginal(_) => true,
            Self::Message(d) => d.is_point_mass(),
            Self::Outgoing(_) => false,
        }
    }
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

    /// Implements the message passing rule for the node.
    fn rule(&self, interface: usize, inputs: &[Input]) -> Result<Dist> {
        let _ = (interface, inputs);
        unimplemented!("`{}` has no rule attached", self.name())
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
