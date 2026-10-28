//! Sampling with the No-U-Turn Sampler, NumPyro style.
//!
//! Latent variables are mapped to an unconstrained space through transforms chosen from the
//! supports their nodes declare. For each point `q` of that space, the log joint (every
//! stochastic node's log density plus the transforms' log Jacobians) is built as one burn tensor
//! expression of `q`, through deterministic nodes, and its gradient comes from `backward()`.
//! Chains run in parallel.

mod diagnostics;
mod nuts;
mod problem;
mod samples;
mod sampling;
mod summary;
mod transform;

pub use diagnostics::{effective_sample_size, quantile, split_rhat};
pub use samples::Samples;
pub use sampling::Sampling;
pub use summary::{Summary, SummaryRow};
