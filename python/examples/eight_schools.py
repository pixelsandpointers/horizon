"""NumPyro's introductory example: eight schools, sampled with NUTS.

The Python port of `examples/eight_schools.rs`. The half-Cauchy prior on the spread between
schools is a `Potential`, and the non-centred school effects are a deterministic `Function`,
both written as Python callables with exact gradients.

Run with `uv run python examples/eight_schools.py` from `python/`.
"""

import math
import time

import horizon as hz

y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

# mu ~ Normal(0, 5²); tau ~ HalfCauchy(5)
# theta_raw ~ Normal(0, 1); theta = mu + tau * theta_raw; y ~ Normal(theta, sigma²)
m = hz.Model()
mu = m.random("mu")
m.normal_mean_variance(mu, 0.0, 25.0)

tau = m.random("tau")
half_cauchy = hz.Potential(
    1,
    lambda t: -math.log1p((t / 5.0) ** 2),
    gradient=lambda t: [-2.0 * t / (25.0 + t * t)],
).with_support(0, hz.Support.Positive)
m.potential(half_cauchy, [tau])

theta_raw = m.random("theta_raw", len(y))
m.normal_mean_variance(theta_raw, 0.0, 1.0)
theta = m.random("theta", len(y))
effect = hz.Function(
    3, lambda mu, tau, raw: mu + tau * raw, gradient=lambda mu, tau, raw: [1.0, raw, tau]
)
m.function(theta, effect, [mu, tau, theta_raw])

obs = m.data("y", y)
m.normal_mean_variance(obs, theta, [s * s for s in sigma])

start = time.perf_counter()
samples = m.sample(chains=4, warmup=1000, draws=1000, target_accept=0.9)
elapsed = time.perf_counter() - start

summary = samples.summary()
summary.rows = [r for r in summary.rows if not r.name.startswith("theta_raw")]
print(summary)
print(f"\n4 chains × 2000 iterations in {elapsed:.2f} s")
