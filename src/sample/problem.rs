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
    /// The values of data and constants, and empty for other variables.
    constants: Vec<Vec<f64>>,
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
                .map(|v| v.fixed.clone().unwrap_or_default())
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

    /// The value of every variable at `q`, and the transforms' log Jacobian.
    fn evaluate(&self, q: &[f64]) -> (Vec<Vec<f64>>, f64) {
        let mut values = self.constants.clone();
        let mut log_jac = 0.0;
        for (v, kind) in self.kinds.iter().enumerate() {
            if let Kind::Continuous { transform, offset } = *kind {
                let (x, lj) = transform.forward(&q[offset..offset + self.model.vars[v].len]);
                values[v] = x;
                log_jac += lj;
            }
        }
        for &f in &self.deterministic {
            let factor = &self.model.factors[f];
            let inputs = inputs(&values, &factor.vars[1..]);
            let out = factor.node.forward_f64(&inputs).unwrap_or_else(|| {
                let inputs: Vec<_> = inputs.iter().map(|x| tensor(x, &self.device)).collect();
                to_vec(factor.node.forward(&inputs))
            });
            debug_assert_eq!(out.len(), self.model.vars[factor.vars[0]].len);
            values[factor.vars[0]] = out;
        }
        (values, log_jac)
    }

    /// `f(inputs)` and its gradient for each input, by burn's automatic differentiation; for
    /// nodes written only with tensors.
    fn autodiff(
        &self,
        inputs: &[&[f64]],
        f: impl FnOnce(&[Tensor<1>]) -> Tensor<1>,
    ) -> (f64, Vec<Vec<f64>>) {
        let tensors: Vec<_> = inputs
            .iter()
            .map(|x| tensor(x, &self.device).require_grad())
            .collect();
        let out = f(&tensors);
        let value = out.clone().into_scalar::<f64>();
        let grads = out.backward();
        let grads = tensors
            .iter()
            .zip(inputs)
            .map(|(t, x)| t.grad(&grads).map_or(vec![0.0; x.len()], to_vec))
            .collect();
        (value, grads)
    }

    /// The constrained values of the given variables at `q`.
    pub(super) fn values(&self, q: &[f64], vars: &[usize]) -> Vec<Vec<f64>> {
        let (values, _) = self.evaluate(q);
        vars.iter().map(|&v| values[v].clone()).collect()
    }

    /// The variables whose draws are recorded: everything that is not data.
    pub(super) fn recorded(&self) -> Vec<usize> {
        (0..self.kinds.len())
            .filter(|&v| !matches!(self.kinds[v], Kind::Fixed))
            .collect()
    }
}

/// The values of `vars`.
fn inputs<'a>(values: &'a [Vec<f64>], vars: &[usize]) -> Vec<&'a [f64]> {
    vars.iter().map(|&v| values[v].as_slice()).collect()
}

impl Target for Problem<'_> {
    /// Log density in unconstrained space and its gradient: every node's log density and
    /// gradient, carried back through deterministic nodes and transforms.
    fn log_density_grad(&mut self, q: &[f64]) -> (f64, Vec<f64>) {
        let (values, mut total) = self.evaluate(q);
        let mut adjoints: Vec<Vec<f64>> = values.iter().map(|x| vec![0.0; x.len()]).collect();
        let accumulate = |adjoints: &mut [Vec<f64>], vars: &[usize], grads: Vec<Vec<f64>>| {
            for (&v, g) in vars.iter().zip(grads) {
                adjoints[v].iter_mut().zip(g).for_each(|(a, g)| *a += g);
            }
        };
        for &f in &self.stochastic {
            let factor = &self.model.factors[f];
            let inputs = inputs(&values, &factor.vars);
            let (value, grads) = factor
                .node
                .log_density_grad(&inputs)
                .unwrap_or_else(|| self.autodiff(&inputs, |x| factor.node.log_density(x)));
            total += value;
            accumulate(&mut adjoints, &factor.vars, grads);
        }
        if !total.is_finite() {
            return (f64::NEG_INFINITY, vec![0.0; self.dim]);
        }
        for &f in self.deterministic.iter().rev() {
            let factor = &self.model.factors[f];
            let adjoint = std::mem::take(&mut adjoints[factor.vars[0]]);
            let inputs = inputs(&values, &factor.vars[1..]);
            let grads = factor.node.vjp(&inputs, &adjoint).unwrap_or_else(|| {
                self.autodiff(&inputs, |x| {
                    (factor.node.forward(x) * tensor(&adjoint, &self.device)).sum()
                })
                .1
            });
            accumulate(&mut adjoints, &factor.vars[1..], grads);
        }
        let mut grad = vec![0.0; self.dim];
        for (v, kind) in self.kinds.iter().enumerate() {
            if let Kind::Continuous { transform, offset } = *kind {
                let len = self.model.vars[v].len;
                transform.pullback(&values[v], &adjoints[v], &mut grad[offset..offset + len]);
            }
        }
        (total, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::{Function, Potential};
    use crate::tensor::default_device;

    /// Eight schools, with its closures written with tensors or as scalars.
    fn eight_schools(scalar: bool) -> Model {
        let y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
        let variance: Vec<f64> = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0f64]
            .iter()
            .map(|s| s * s)
            .collect();
        let mut m = Model::new();
        let mu = m.random("mu");
        m.normal_mean_variance(mu, 0.0, 25.0);
        let tau = m.random("tau");
        let half_cauchy = if scalar {
            Potential::scalar(1, |x| -(1.0 + (x[0] / 5.0).powi(2)).ln())
        } else {
            Potential::new(1, |x| {
                let z = x[0].clone().div_scalar(5.0);
                (z.clone() * z).add_scalar(1.0).log().neg()
            })
        };
        m.potential(half_cauchy.with_support(0, Support::Positive), [tau.into()]);
        let theta_raw = m.random_vec("theta_raw", y.len());
        m.normal_mean_variance(theta_raw, 0.0, 1.0);
        let theta = m.random_vec("theta", y.len());
        let effect = if scalar {
            Function::scalar(3, |x| x[0] + x[1] * x[2])
        } else {
            Function::new(3, |x| x[0].clone() + x[1].clone() * x[2].clone())
        };
        m.function(theta, effect, [mu.into(), tau.into(), theta_raw.into()]);
        let obs = m.data("y", &y[..]);
        m.normal_mean_variance(obs, theta, variance);
        m
    }

    #[test]
    fn gradients_match_finite_differences() {
        for scalar in [false, true] {
            let model = eight_schools(scalar);
            let mut problem = Problem::new(&model, default_device()).unwrap();
            let q: Vec<f64> = (0..problem.dim).map(|i| 0.3 * i as f64 - 1.0).collect();
            let (value, grad) = problem.log_density_grad(&q);
            assert!(value.is_finite());
            for i in 0..problem.dim {
                let h = 1e-5;
                let mut shifted = q.clone();
                shifted[i] += h;
                let plus = problem.log_density_grad(&shifted).0;
                shifted[i] -= 2.0 * h;
                let minus = problem.log_density_grad(&shifted).0;
                let expected = (plus - minus) / (2.0 * h);
                assert!(
                    (grad[i] - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                    "scalar = {scalar}, coordinate {i}: {} vs {expected}",
                    grad[i]
                );
            }
        }
    }

    #[test]
    fn scalar_and_tensor_closures_agree() {
        let (tensor_model, scalar_model) = (eight_schools(false), eight_schools(true));
        let mut a = Problem::new(&tensor_model, default_device()).unwrap();
        let mut b = Problem::new(&scalar_model, default_device()).unwrap();
        let q: Vec<f64> = (0..a.dim).map(|i| 0.5 - 0.1 * i as f64).collect();
        let ((va, ga), (vb, gb)) = (a.log_density_grad(&q), b.log_density_grad(&q));
        assert!((va - vb).abs() < 1e-9);
        for (ga, gb) in ga.iter().zip(&gb) {
            assert!((ga - gb).abs() < 1e-6, "{ga} vs {gb}");
        }
    }
}
