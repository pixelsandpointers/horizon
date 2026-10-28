//! Python bindings: the model builder, scalar `Potential`s and `Function`s written as Python
//! callables, and NUTS sampling with the GIL released.

use std::sync::{Arc, Mutex};

use hrzn::model::{Arg, Model, Value, VarId};
use hrzn::node::Support;
use hrzn::nodes::{Function, Potential};
use hrzn::sample::{Samples, Summary, SummaryRow};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// The first exception raised by a Python callback, kept until sampling returns.
type ErrorSlot = Arc<Mutex<Option<PyErr>>>;

fn record(errors: &ErrorSlot, err: PyErr) {
    errors.lock().unwrap().get_or_insert(err);
}

fn failed(errors: &ErrorSlot) -> bool {
    errors.lock().unwrap().is_some()
}

/// `f(*x)` as a float. An exception is recorded and gives NaN, which the sampler rejects; once
/// one is recorded, `f` is no longer called.
//
// TODO: vectorized callbacks, taking one NumPy array per argument and called once per factor
// rather than once per entry. Each call takes the GIL and builds a tuple, so eight schools with
// Python callbacks took ~10 s against 0.11 s in Rust, nearly all of it in ~3M scalar calls.
// This needs a `Node` written against whole interfaces in f64 (`log_density_grad`,
// `forward_f64` and `vjp` already are), plus the `numpy` crate. With whole arrays, callbacks
// could also be JAX or PyTorch functions differentiated by `jax.value_and_grad` or
// `torch.autograd` in the same call, replacing central differences when no gradient is given.
fn scalar_callback(f: Py<PyAny>, errors: ErrorSlot) -> impl Fn(&[f64]) -> f64 + Send + Sync {
    move |x| {
        if failed(&errors) {
            return f64::NAN;
        }
        Python::attach(|py| {
            let args = PyTuple::new(py, x)?;
            f.call1(py, args)?.extract::<f64>(py)
        })
        .unwrap_or_else(|e| {
            record(&errors, e);
            f64::NAN
        })
    }
}

/// `gradient(*x)` as one float per argument, failing like [`scalar_callback`].
fn gradient_callback(
    gradient: Py<PyAny>,
    errors: ErrorSlot,
) -> impl Fn(&[f64]) -> Vec<f64> + Send + Sync {
    move |x| {
        let nan = || vec![f64::NAN; x.len()];
        if failed(&errors) {
            return nan();
        }
        Python::attach(|py| {
            let args = PyTuple::new(py, x)?;
            let partials = gradient.call1(py, args)?.extract::<Vec<f64>>(py)?;
            if partials.len() != x.len() {
                return Err(PyValueError::new_err(format!(
                    "the gradient gave {} partial derivatives for {} arguments",
                    partials.len(),
                    x.len()
                )));
            }
            Ok(partials)
        })
        .unwrap_or_else(|e| {
            record(&errors, e);
            nan()
        })
    }
}

/// The set of values a variable can take.
#[pyclass(name = "Support", eq, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PySupport {
    Real,
    Positive,
}

impl From<PySupport> for Support {
    fn from(value: PySupport) -> Self {
        match value {
            PySupport::Real => Self::Real,
            PySupport::Positive => Self::Positive,
        }
    }
}

/// A handle to a variable of a `Model`.
#[pyclass(name = "Var", frozen, from_py_object)]
#[derive(Clone)]
struct PyVar {
    id: VarId,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    len: usize,
    /// Whether the variable is data, which has no draws.
    #[pyo3(get)]
    is_data: bool,
}

#[pymethods]
impl PyVar {
    fn __repr__(&self) -> String {
        let kind = if self.is_data { "data" } else { "random" };
        format!("Var({:?}, len={}, {kind})", self.name, self.len)
    }
}

/// An argument of a factor: a variable, or a constant with one or more entries.
#[derive(FromPyObject)]
enum PyArg {
    Var(PyVar),
    Scalar(f64),
    Vector(Vec<f64>),
}

impl From<PyArg> for Arg {
    fn from(value: PyArg) -> Self {
        match value {
            PyArg::Var(v) => Arg::Var(v.id),
            PyArg::Scalar(x) => Arg::Value(x.into()),
            PyArg::Vector(x) => Arg::Value(x.into()),
        }
    }
}

/// An arbitrary log-density term over scalar arguments, like NumPyro's `factor`.
#[pyclass(name = "Potential", frozen)]
struct PyPotential {
    node: Potential,
    #[pyo3(get)]
    arity: usize,
    errors: ErrorSlot,
}

#[pymethods]
impl PyPotential {
    #[new]
    #[pyo3(signature = (arity, density, *, gradient = None))]
    fn new(arity: usize, density: Py<PyAny>, gradient: Option<Py<PyAny>>) -> Self {
        let errors = ErrorSlot::default();
        let mut node = Potential::scalar(arity, scalar_callback(density, errors.clone()));
        if let Some(g) = gradient {
            node = node.with_gradient(gradient_callback(g, errors.clone()));
        }
        Self {
            node,
            arity,
            errors,
        }
    }

    /// A copy of this term that constrains argument `k` to `support`.
    fn with_support(&self, k: usize, support: PySupport) -> PyResult<Self> {
        if k >= self.arity {
            return Err(PyValueError::new_err(format!(
                "argument {k} of a potential with {} arguments",
                self.arity
            )));
        }
        Ok(Self {
            node: self.node.clone().with_support(k, support.into()),
            arity: self.arity,
            errors: self.errors.clone(),
        })
    }
}

/// A deterministic variable computed from scalar inputs, like NumPyro's `deterministic`.
#[pyclass(name = "Function", frozen)]
struct PyFunction {
    node: Function,
    #[pyo3(get)]
    arity: usize,
    errors: ErrorSlot,
}

#[pymethods]
impl PyFunction {
    #[new]
    #[pyo3(signature = (arity, func, *, gradient = None))]
    fn new(arity: usize, func: Py<PyAny>, gradient: Option<Py<PyAny>>) -> Self {
        let errors = ErrorSlot::default();
        let mut node = Function::scalar(arity, scalar_callback(func, errors.clone()));
        if let Some(g) = gradient {
            node = node.with_gradient(gradient_callback(g, errors.clone()));
        }
        Self {
            node,
            arity,
            errors,
        }
    }
}

fn check_arity(node: &str, arity: usize, given: usize) -> PyResult<()> {
    if arity == given {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{node} takes {arity} arguments, got {given}"
    )))
}

/// A factor graph with variables and the factors attached to them.
#[pyclass(name = "Model")]
#[derive(Default)]
struct PyModel {
    model: Model,
    /// The error slots of the callbacks in the model's factors.
    errors: Vec<ErrorSlot>,
}

impl PyModel {
    fn var(&self, id: VarId, len: usize, is_data: bool) -> PyVar {
        PyVar {
            id,
            name: self.model.name(id).to_owned(),
            len,
            is_data,
        }
    }
}

#[pymethods]
impl PyModel {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// A latent variable with `len` entries.
    #[pyo3(signature = (name, len = 1))]
    fn random(&mut self, name: String, len: usize) -> PyResult<PyVar> {
        if len == 0 {
            return Err(PyValueError::new_err("a variable needs at least one entry"));
        }
        let id = self.model.random_vec(name, len);
        Ok(self.var(id, len, false))
    }

    /// Observed data.
    fn data(&mut self, name: String, value: PyArg) -> PyResult<PyVar> {
        let value = match value {
            PyArg::Var(_) => return Err(PyValueError::new_err("data must be numbers")),
            PyArg::Scalar(x) => Value::from(x),
            PyArg::Vector(x) => Value::from(x),
        };
        let len = value.0.len();
        let id = self.model.data(name, value);
        Ok(self.var(id, len, true))
    }

    /// `out ~ Normal(mean, variance)`; elementwise.
    fn normal_mean_variance(&mut self, out: PyArg, mean: PyArg, variance: PyArg) {
        self.model
            .normal_mean_variance(Arg::from(out), Arg::from(mean), Arg::from(variance));
    }

    /// Adds the log-density term `node` over `args`.
    fn potential(&mut self, node: &PyPotential, args: Vec<PyArg>) -> PyResult<()> {
        check_arity("the potential", node.arity, args.len())?;
        self.model
            .potential(node.node.clone(), args.into_iter().map(Arg::from));
        self.errors.push(node.errors.clone());
        Ok(())
    }

    /// `out = node(*inputs)`.
    fn function(&mut self, out: PyArg, node: &PyFunction, inputs: Vec<PyArg>) -> PyResult<()> {
        check_arity("the function", node.arity, inputs.len())?;
        self.model
            .function(out, node.node.clone(), inputs.into_iter().map(Arg::from));
        self.errors.push(node.errors.clone());
        Ok(())
    }

    /// Draws from the posterior with NUTS, running chains in parallel.
    #[pyo3(signature = (
        *, chains = 4, warmup = 1000, draws = 1000, seed = 0, max_tree_depth = 10,
        target_accept = 0.8, threads = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        py: Python<'_>,
        chains: usize,
        warmup: usize,
        draws: usize,
        seed: u64,
        max_tree_depth: usize,
        target_accept: f64,
        threads: Option<usize>,
    ) -> PyResult<PySamples> {
        for slot in &self.errors {
            slot.lock().unwrap().take();
        }
        let samples = py.detach(|| {
            let mut sampling = self
                .model
                .sample()
                .chains(chains)
                .warmup(warmup)
                .draws(draws)
                .seed(seed)
                .max_tree_depth(max_tree_depth)
                .target_accept(target_accept);
            // Python callbacks take the GIL, so worker threads would only contend for it.
            match threads {
                Some(n) => sampling = sampling.threads(n),
                None if !self.errors.is_empty() => sampling = sampling.threads(1),
                None => {}
            }
            sampling.run()
        });
        // An exception from a callback explains a failed run better than the sampler's error.
        for slot in &self.errors {
            if let Some(e) = slot.lock().unwrap().take() {
                return Err(e);
            }
        }
        let samples = samples.map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySamples { samples })
    }
}

/// Draws from the posterior, chain by chain.
#[pyclass(name = "Samples", frozen)]
struct PySamples {
    samples: Samples,
}

fn check_random(v: &PyVar) -> PyResult<()> {
    if v.is_data {
        return Err(PyValueError::new_err(format!(
            "`{}` is data and has no draws",
            v.name
        )));
    }
    Ok(())
}

#[pymethods]
impl PySamples {
    /// Divergent transitions after warmup, over all chains.
    #[getter]
    fn divergences(&self) -> usize {
        self.samples.divergences
    }

    /// Entry `i` of a variable's draws, chain by chain.
    #[pyo3(signature = (var, i = 0))]
    fn component(&self, var: &PyVar, i: usize) -> PyResult<Vec<Vec<f64>>> {
        check_random(var)?;
        if i >= var.len {
            return Err(PyValueError::new_err(format!(
                "entry {i} of `{}`, which has {}",
                var.name, var.len
            )));
        }
        Ok(self.samples.component(var.id, i))
    }

    /// The posterior mean of every entry of a variable.
    fn mean(&self, var: &PyVar) -> PyResult<Vec<f64>> {
        check_random(var)?;
        Ok(self.samples.mean(var.id))
    }

    /// NumPyro-style summary statistics for every entry of every variable.
    fn summary(&self) -> PySummary {
        PySummary(self.samples.summary())
    }
}

/// One line of a `Summary`.
#[pyclass(name = "SummaryRow", get_all, frozen, from_py_object)]
#[derive(Clone)]
struct PySummaryRow {
    name: String,
    mean: f64,
    std: f64,
    median: f64,
    q05: f64,
    q95: f64,
    ess: f64,
    rhat: f64,
}

#[pymethods]
impl PySummaryRow {
    fn __repr__(&self) -> String {
        format!(
            "SummaryRow({:?}, mean={}, std={}, ess={}, rhat={})",
            self.name, self.mean, self.std, self.ess, self.rhat
        )
    }
}

impl From<&SummaryRow> for PySummaryRow {
    fn from(r: &SummaryRow) -> Self {
        Self {
            name: r.name.clone(),
            mean: r.mean,
            std: r.std,
            median: r.median,
            q05: r.q05,
            q95: r.q95,
            ess: r.ess,
            rhat: r.rhat,
        }
    }
}

impl From<PySummaryRow> for SummaryRow {
    fn from(r: PySummaryRow) -> Self {
        Self {
            name: r.name,
            mean: r.mean,
            std: r.std,
            median: r.median,
            q05: r.q05,
            q95: r.q95,
            ess: r.ess,
            rhat: r.rhat,
        }
    }
}

/// Posterior summary statistics with convergence diagnostics; printing it gives a table.
#[pyclass(name = "Summary")]
struct PySummary(Summary);

#[pymethods]
impl PySummary {
    #[getter]
    fn rows(&self) -> Vec<PySummaryRow> {
        self.0.rows.iter().map(PySummaryRow::from).collect()
    }

    #[setter]
    fn set_rows(&mut self, rows: Vec<PySummaryRow>) {
        self.0.rows = rows.into_iter().map(SummaryRow::from).collect();
    }

    #[getter]
    fn divergences(&self) -> usize {
        self.0.divergences
    }

    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

#[pymodule]
fn horizon(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyVar>()?;
    m.add_class::<PySupport>()?;
    m.add_class::<PyPotential>()?;
    m.add_class::<PyFunction>()?;
    m.add_class::<PySamples>()?;
    m.add_class::<PySummary>()?;
    m.add_class::<PySummaryRow>()?;
    Ok(())
}
