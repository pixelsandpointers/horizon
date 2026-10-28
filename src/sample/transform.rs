//! Bijections from unconstrained vectors to the supports of variables, as in NumPyro.

use crate::node::Support;

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

    /// The constrained value `x`, and `log |det ∂x/∂y|`.
    pub fn forward(self, y: &[f64]) -> (Vec<f64>, f64) {
        match self {
            Self::Identity => (y.to_vec(), 0.0),
            Self::Exp => (y.iter().map(|y| y.exp()).collect(), y.iter().sum()),
        }
    }

    /// The gradient of a function of `x` and the log Jacobian with respect to `y`, from its
    /// gradient `grad_x` with respect to `x`.
    pub fn pullback(self, x: &[f64], grad_x: &[f64], grad_y: &mut [f64]) {
        match self {
            Self::Identity => grad_y.copy_from_slice(grad_x),
            Self::Exp => {
                for ((gy, gx), x) in grad_y.iter_mut().zip(grad_x).zip(x) {
                    *gy = gx * x + 1.0;
                }
            }
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
