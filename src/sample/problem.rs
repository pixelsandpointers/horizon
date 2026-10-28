use burn::tensor::Device;

use super::nuts::Target;
use super::transform::Transform;
use crate::error::{Error, Result};
use crate::model::Model;
use crate::node::Support;
use crate::tensor::{Tensor, tensor, to_vec};

/// How a variable is handled during sampling.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// Data, constants and variables attached to no factor.
    Fixed,
    Continuous {
        transform: Transform,
        offset: usize,
    },
    /// The output of a deterministic node.
    Deterministic,
}

/// A model prepared for sampling.
pub(super) struct Problem<'m> {
    model: &'m Model,
    device: Device,
    kinds: Vec<Kind>,
    /// Tensors of data and constants, reused by every evaluation.
    constants: Vec<Option<Tensor<1>>>,
    stochastic: Vec<usize>,
    /// Deterministic factors in an order where inputs come before outputs.
    deterministic: Vec<usize>,
    pub(super) dim: usize,
}

impl<'m> Problem<'m> {
    pub(super) fn new(model: &'m Model, device: Device) -> Result<Self> {
        let n = model.vars.len();
        let name = |v: usize| model.vars[v].name.clone();
        let mut attached = vec![false; n];
        for factor in &model.factors {
            factor.vars.iter().for_each(|&v| attached[v] = true);
        }

        // Deterministic outputs.
        let mut det_out: Vec<Option<usize>> = vec![None; n];
        for (f, factor) in model.factors.iter().enumerate() {
            if !factor.node.is_deterministic() {
                continue;
            }
            let out = factor.vars[0];
            if model.vars[out].fixed.is_some() {
                return Err(Error::InvalidModel(format!(
                    "`{}` is observed but is the output of a deterministic node",
                    name(out)
                )));
            }
            if det_out[out].replace(f).is_some() {
                return Err(Error::InvalidModel(format!(
                    "`{}` is the output of two deterministic nodes",
                    name(out)
                )));
            }
        }

        // Supports, from every node a variable is attached to.
        let mut supports: Vec<Option<Support>> = vec![None; n];
        for factor in &model.factors {
            for (k, &v) in factor.vars.iter().enumerate() {
                let Some(s) = factor.node.support(k) else {
                    continue;
                };
                supports[v] = match (supports[v], s) {
                    (None, s) | (Some(Support::Real), s) => Some(s),
                    (Some(old), _) => Some(old),
                };
            }
        }

        // Kinds, and the layout of the unconstrained parameter vector.
        let mut dim = 0;
        let kinds: Vec<Kind> = (0..n)
            .map(|v| {
                if model.vars[v].fixed.is_some() || !attached[v] {
                    Kind::Fixed
                } else if det_out[v].is_some() {
                    Kind::Deterministic
                } else {
                    let transform = Transform::for_support(supports[v].unwrap_or(Support::Real));
                    dim += model.vars[v].len;
                    Kind::Continuous {
                        transform,
                        offset: dim - model.vars[v].len,
                    }
                }
            })
            .collect();

        // Deterministic factors in dependency order.
        let mut deterministic = Vec::new();
        let mut done: Vec<bool> = (0..n).map(|v| det_out[v].is_none()).collect();
        let mut pending: Vec<usize> = det_out.iter().flatten().copied().collect();
        while !pending.is_empty() {
            let before = pending.len();
            pending.retain(|&f| {
                let vars = &model.factors[f].vars;
                if vars[1..].iter().all(|&v| done[v]) {
                    done[vars[0]] = true;
                    deterministic.push(f);
                    false
                } else {
                    true
                }
            });
            if pending.len() == before {
                return Err(Error::InvalidModel(
                    "deterministic nodes form a cycle".into(),
                ));
            }
        }

        Ok(Self {
            model,
            constants: model
                .vars
                .iter()
                .map(|v| v.fixed.as_ref().map(|x| tensor(x, &device)))
                .collect(),
            device,
            kinds,
            stochastic: (0..model.factors.len())
                .filter(|&f| !model.factors[f].node.is_deterministic())
                .collect(),
            deterministic,
            dim,
        })
    }

    /// Unconstrained coordinates of a typical point of every latent variable's support.
    pub(super) fn typical_params(&self) -> Vec<f64> {
        let mut q = vec![0.0; self.dim];
        for (v, kind) in self.kinds.iter().enumerate() {
            if let Kind::Continuous { transform, offset } = *kind {
                let len = self.model.vars[v].len;
                let x = vec![transform.typical_value(); len];
                let y = transform
                    .inverse(&x)
                    .expect("typical values are in the support");
                q[offset..offset + len].copy_from_slice(&y);
            }
        }
        q
    }

    /// The value of every variable as a tensor of `q`, and the log joint in unconstrained space.
    fn evaluate(&self, q: &Tensor<1>) -> (Vec<Option<Tensor<1>>>, Tensor<1>) {
        let mut values = self.constants.clone();
        let mut total = tensor(&[0.0], &self.device);
        for (v, kind) in self.kinds.iter().enumerate() {
            if let Kind::Continuous { transform, offset } = *kind {
                let y = q.clone().narrow(0, offset, self.model.vars[v].len);
                let (x, log_jac) = transform.forward(y);
                values[v] = Some(x);
                if let Some(lj) = log_jac {
                    total = total + lj;
                }
            }
        }
        let inputs = |values: &[Option<Tensor<1>>], vars: &[usize]| -> Vec<Tensor<1>> {
            vars.iter()
                .map(|&v| values[v].clone().expect("every input has a value"))
                .collect()
        };
        for &f in &self.deterministic {
            let factor = &self.model.factors[f];
            let out = factor.node.forward(&inputs(&values, &factor.vars[1..]));
            values[factor.vars[0]] = Some(out);
        }
        for &f in &self.stochastic {
            let factor = &self.model.factors[f];
            total = total + factor.node.log_density(&inputs(&values, &factor.vars));
        }
        (values, total)
    }

    /// The constrained values of the given variables at `q`.
    pub(super) fn values(&self, q: &[f64], vars: &[usize]) -> Vec<Vec<f64>> {
        let (values, _) = self.evaluate(&tensor(q, &self.device));
        vars.iter()
            .map(|&v| to_vec(values[v].clone().expect("recorded variables have values")))
            .collect()
    }

    /// The variables whose draws are recorded: everything that is not data.
    pub(super) fn recorded(&self) -> Vec<usize> {
        (0..self.kinds.len())
            .filter(|&v| !matches!(self.kinds[v], Kind::Fixed))
            .collect()
    }
}

impl Target for Problem<'_> {
    /// Log density in unconstrained space and its gradient, by automatic differentiation.
    fn log_density_grad(&mut self, q: &[f64]) -> (f64, Vec<f64>) {
        let q = tensor(q, &self.device).require_grad();
        let (_, total) = self.evaluate(&q);
        let value = total.clone().into_scalar::<f64>();
        if !value.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; self.dim]);
        }
        let grads = total.backward();
        let grad = q.grad(&grads).map_or(vec![0.0; self.dim], to_vec);
        (value, grad)
    }
}
