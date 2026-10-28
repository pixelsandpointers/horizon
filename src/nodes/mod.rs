//! Implements different nodes for distributions and functions to be used
//! in the graphical model/factor graph.

mod function;
mod normal_mean_variance;
mod potential;

pub use function::Function;
pub use normal_mean_variance::NormalMeanVariance;
pub use potential::Potential;
