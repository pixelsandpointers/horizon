import math

import pytest

import horizon as hz


def normal_mean_model():
    m = hz.Model()
    mu = m.random("mu")
    m.normal_mean_variance(mu, 0.0, 100.0)
    y = m.data("y", [4.9, 5.3, 4.6, 5.1])
    m.normal_mean_variance(y, mu, 1.0)
    return m, mu, y


def test_conjugate_normal_mean():
    m, mu, _ = normal_mean_model()
    samples = m.sample(chains=2, warmup=500, draws=500, seed=1)
    # Posterior precision 1/100 + 4, mean 19.9 / precision.
    precision = 0.01 + 4.0
    [mean] = samples.mean(mu)
    assert mean == pytest.approx(19.9 / precision, abs=0.1)
    draws = samples.component(mu)
    assert [len(c) for c in draws] == [500, 500]
    [row] = samples.summary().rows
    assert row.name == "mu"
    assert row.std == pytest.approx(math.sqrt(1.0 / precision), rel=0.2)
    assert row.rhat < 1.05


def test_seed_makes_draws_reproducible():
    m, mu, _ = normal_mean_model()
    a = m.sample(chains=2, warmup=100, draws=50, seed=7, threads=1)
    b = m.sample(chains=2, warmup=100, draws=50, seed=7, threads=2)
    assert a.component(mu) == b.component(mu)


def test_potential_with_support_and_function():
    m = hz.Model()
    scale = m.random("scale")
    # Exponential(1) prior on a positive variable.
    exponential = hz.Potential(1, lambda s: -s, gradient=lambda s: [-1.0]).with_support(
        0, hz.Support.Positive
    )
    m.potential(exponential, [scale])
    double = m.random("double", 3)
    m.function(double, hz.Function(2, lambda s, k: k * s), [scale, [1.0, 2.0, 3.0]])

    samples = m.sample(chains=2, warmup=500, draws=1000)
    [s] = samples.mean(scale)
    assert s == pytest.approx(1.0, abs=0.15)
    assert samples.mean(double) == pytest.approx([s, 2 * s, 3 * s])
    names = [r.name for r in samples.summary().rows]
    assert names == ["scale", "double[0]", "double[1]", "double[2]"]


def test_callback_exception_is_raised():
    m = hz.Model()
    x = m.random("x")

    def density(x):
        raise RuntimeError("boom")

    m.potential(hz.Potential(1, density), [x])
    with pytest.raises(RuntimeError, match="boom"):
        m.sample(chains=1, warmup=10, draws=10)


def test_summary_rows_can_be_filtered():
    m, _, _ = normal_mean_model()
    summary = m.sample(chains=1, warmup=50, draws=50).summary()
    summary.rows = []
    assert "mu" not in str(summary)
    assert "Number of divergences" in str(summary)


def test_invalid_arguments():
    m, mu, y = normal_mean_model()
    with pytest.raises(ValueError, match="takes 2 arguments"):
        m.potential(hz.Potential(2, lambda a, b: 0.0), [mu])
    with pytest.raises(ValueError):
        hz.Potential(1, lambda a: 0.0).with_support(1, hz.Support.Positive)
    samples = m.sample(chains=1, warmup=10, draws=10)
    with pytest.raises(ValueError, match="is data"):
        samples.mean(y)
    with pytest.raises(ValueError, match="entry 1"):
        samples.component(mu, 1)
