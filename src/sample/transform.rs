//! Bijections from unconstrained vectors to the supports of variables, as in NumPyro.

use crate::node::Support;
use crate::tensor::Tensor;

/// Maps an unconstrained `y` to a constrained value `x`, elementwise.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Transform {
    Identity,
    /// `x = exp(y)`.
    Exp,
}

impl Transform {
    pub fn for_support(support: Support) -> Self {
        match support {
            Support::Real => Self::Identity,
            Support::Positive => Self::Exp,
        }
    }

    /// The constrained value and `log |det ∂x/∂y|`, if it is not zero.
    pub fn forward(self, y: Tensor<1>) -> (Tensor<1>, Option<Tensor<1>>) {
        match self {
            Self::Identity => (y, None),
            Self::Exp => (y.clone().exp(), Some(y.sum())),
        }
    }

    /// The unconstrained coordinates of a value in the support, or `None` outside it.
    pub fn inverse(self, x: &[f64]) -> Option<Vec<f64>> {
        match self {
            Self::Identity => Some(x.to_vec()),
            Self::Exp => x.iter().map(|&x| (x > 0.0).then(|| x.ln())).collect(),
        }
    }

    /// A typical value in the support, to start samplers.
    pub fn typical_value(self) -> f64 {
        match self {
            Self::Identity => 0.0,
            Self::Exp => 1.0,
        }
    }
}
