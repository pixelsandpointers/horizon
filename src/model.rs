//! Building factor graphs with builder calls; each `x ~ Node(...)` becomes a method call that
//! attaches a factor to existing variables.

use crate::node::Node;
use crate::nodes::{Function, NormalMeanVariance, Potential};

/// Handles to variables of the `Model`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VarId(pub(crate) usize);

/// Entries of the data: value or constant
#[derive(Clone, Debug, PartialEq)]
pub struct Value(pub Vec<f64>);

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self(vec![value])
    }
}

impl From<Vec<f64>> for Value {
    fn from(value: Vec<f64>) -> Self {
        Self(value)
    }
}

impl From<&[f64]> for Value {
    fn from(value: &[f64]) -> Self {
        Self(value.to_vec())
    }
}

impl<const N: usize> From<[f64; N]> for Value {
    fn from(value: [f64; N]) -> Self {
        Self(value.to_vec())
    }
}

/// An arg of a factor.
#[derive(Clone, Debug)]
pub enum Arg {
    Var(VarId),
    Value(Value),
}

impl From<VarId> for Arg {
    fn from(value: VarId) -> Self {
        Self::Var(value)
    }
}

impl<T: Into<Value>> From<T> for Arg {
    fn from(value: T) -> Self {
        Self::Value(value.into())
    }
}

pub(crate) struct Variable {
    pub name: String,
    /// Number of entries.
    pub len: usize,
    /// The value of data and constants, will be `None` for random variables.
    pub fixed: Option<Vec<f64>>,
}

pub(crate) struct Factor {
    pub node: Box<dyn Node>,
    /// Variable index per interface.
    pub vars: Vec<usize>,
}

/// Factor graph with variables and the factors attached to them.
#[derive(Default)]
pub struct Model {
    pub(crate) vars: Vec<Variable>,
    pub(crate) factors: Vec<Factor>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    fn add_var(&mut self, name: String, len: usize, fixed: Option<Vec<f64>>) -> VarId {
        self.vars.push(Variable { name, len, fixed });
        VarId(self.vars.len() - 1)
    }

    /// A latent variable with `len` entries.
    pub fn random_vec(&mut self, name: impl Into<String>, len: usize) -> VarId {
        self.add_var(name.into(), len, None)
    }

    /// Scalar latent variable.
    pub fn random(&mut self, name: impl Into<String>) -> VarId {
        self.random_vec(name, 1)
    }

    pub fn data(&mut self, name: impl Into<String>, value: impl Into<Value>) -> VarId {
        let v = value.into().0;
        self.add_var(name.into(), v.len(), Some(v))
    }

    pub fn name(&self, v: VarId) -> &str {
        &self.vars[v.0].name
    }

    /// Attaches a factor to the given variables, one per interface of `node`.
    /// # Panics
    /// Panics if the number of arguments does not match the node's arity.
    pub fn factor(&mut self, node: impl Node + 'static, args: impl IntoIterator<Item = Arg>) {
        let vars: Vec<usize> = args
            .into_iter()
            .map(|a| match a {
                Arg::Var(v) => v.0,
                Arg::Value(Value(x)) => {
                    let name = format!("const#{}", self.vars.len());
                    self.add_var(name, x.len(), Some(x)).0
                }
            })
            .collect();
        assert_eq!(vars.len(), node.arity(), "node `{}`", node.name());
        self.factors.push(Factor {
            node: Box::new(node),
            vars,
        });
    }

    /// `out ~ Normal(mean, variance)`; elementwise.
    pub fn normal_mean_variance(
        &mut self,
        out: impl Into<Arg>,
        mean: impl Into<Arg>,
        variance: impl Into<Arg>,
    ) {
        self.factor(
            NormalMeanVariance,
            [out.into(), mean.into(), variance.into()],
        );
    }

    /// Adds an arbitrary log-density term over `args`; similar to NumPyro's `factor`
    pub fn potential(&mut self, node: Potential, args: impl IntoIterator<Item = Arg>) {
        self.factor(node, args);
    }

    /// `out = f(inputs ...)` for a closure `f`; similar to NumPyro's `deterministic`
    pub fn function(
        &mut self,
        out: impl Into<Arg>,
        node: Function,
        inputs: impl IntoIterator<Item = Arg>,
    ) {
        let args: Vec<Arg> = std::iter::once(out.into()).chain(inputs).collect();
        self.factor(node, args);
    }
}
