"""NUTS sampling on factor graphs, with Rust doing the work.

Scalar potentials and functions are Python callables taking one float per argument. They are
called with the GIL held from the sampler's worker threads, so they run one at a time; prefer
the built-in nodes where they fit.
"""

from collections.abc import Callable, Sequence
from typing import Final, TypeAlias, final

__all__ = [
    "Function",
    "Model",
    "Potential",
    "Samples",
    "Summary",
    "SummaryRow",
    "Support",
    "Var",
]

_Arg: TypeAlias = Var | float | Sequence[float]
"""An argument of a factor: a variable, or a constant with one or more entries."""

_ScalarFn: TypeAlias = Callable[..., float]
"""Called as `f(*x)` with one float per argument."""

_GradientFn: TypeAlias = Callable[..., Sequence[float]]
"""Called as `gradient(*x)`; gives the partial derivative for each argument."""

@final
class Support:
    """The set of values a variable can take, which sampling maps to an unconstrained space."""

    Real: Final[Support]
    Positive: Final[Support]

@final
class Var:
    """A handle to a variable of a `Model`."""

    @property
    def name(self) -> str: ...
    @property
    def len(self) -> int:
        """The number of entries."""
    @property
    def is_data(self) -> bool:
        """Whether the variable is data, which has no draws."""

@final
class Potential:
    """An arbitrary log-density term over scalar arguments, like NumPyro's `factor`.

    The term is summed over the entries of arguments broadcast against each other: arguments
    with one entry are shared by every entry. Its gradient comes from central differences
    unless `gradient` is given.
    """

    def __new__(
        cls, arity: int, density: _ScalarFn, *, gradient: _GradientFn | None = None
    ) -> Potential: ...
    @property
    def arity(self) -> int: ...
    def with_support(self, k: int, support: Support) -> Potential:
        """A copy of this term that constrains argument `k` to `support`."""

@final
class Function:
    """A deterministic variable computed from scalar inputs, like NumPyro's `deterministic`.

    Applied to each entry of inputs broadcast against each other. Its gradient comes from
    central differences unless `gradient` is given.
    """

    def __new__(
        cls, arity: int, func: _ScalarFn, *, gradient: _GradientFn | None = None
    ) -> Function: ...
    @property
    def arity(self) -> int:
        """The number of inputs."""

@final
class Model:
    """A factor graph with variables and the factors attached to them."""

    def __new__(cls) -> Model: ...
    def random(self, name: str, len: int = 1) -> Var:
        """A latent variable with `len` entries."""
    def data(self, name: str, value: float | Sequence[float]) -> Var:
        """Observed data."""
    def normal_mean_variance(self, out: _Arg, mean: _Arg, variance: _Arg) -> None:
        """`out ~ Normal(mean, variance)`; elementwise."""
    def potential(self, node: Potential, args: Sequence[_Arg]) -> None:
        """Adds the log-density term `node` over `args`."""
    def function(self, out: _Arg, node: Function, inputs: Sequence[_Arg]) -> None:
        """`out = node(*inputs)`."""
    def sample(
        self,
        *,
        chains: int = 4,
        warmup: int = 1000,
        draws: int = 1000,
        seed: int = 0,
        max_tree_depth: int = 10,
        target_accept: float = 0.8,
        threads: int | None = None,
    ) -> Samples:
        """Draws from the posterior with NUTS, running chains in parallel.

        `threads` defaults to the available parallelism, or to 1 when the model has Python
        potentials or functions, since they hold the GIL. An exception raised by a potential or
        function stops it being called and is re-raised here once sampling returns.
        """

@final
class Samples:
    """Draws from the posterior, chain by chain."""

    @property
    def divergences(self) -> int:
        """Divergent transitions after warmup, over all chains."""
    def component(self, var: Var, i: int = 0) -> list[list[float]]:
        """Entry `i` of a variable's draws, chain by chain."""
    def mean(self, var: Var) -> list[float]:
        """The posterior mean of every entry of a variable."""
    def summary(self) -> Summary:
        """NumPyro-style summary statistics for every entry of every variable."""

@final
class SummaryRow:
    """One line of a `Summary`."""

    @property
    def name(self) -> str: ...
    @property
    def mean(self) -> float: ...
    @property
    def std(self) -> float: ...
    @property
    def median(self) -> float: ...
    @property
    def q05(self) -> float: ...
    @property
    def q95(self) -> float: ...
    @property
    def ess(self) -> float: ...
    @property
    def rhat(self) -> float: ...

@final
class Summary:
    """Posterior summary statistics with convergence diagnostics; printing it gives a table."""

    rows: list[SummaryRow]
    @property
    def divergences(self) -> int: ...
