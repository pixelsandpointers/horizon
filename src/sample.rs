//! Sampling with the No-U-Turn Sampler, NumPyro style.
//!
//! Latent variables are mapped to an unconstrained space through transforms chosen from the
//! supports their nodes declare. For each point `q` of that space, the log joint (every
//! stochastic node's log density plus the transforms' log Jacobians) is evaluated forward
//! through deterministic nodes, and its gradient is carried back through them. Nodes give their
//! gradients in `f64`: analytic for built-in nodes, by central differences or given exactly for
//! scalar closures. Nodes written only with tensors are differentiated by burn, one node at a
//! time. Chains run in parallel.

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
