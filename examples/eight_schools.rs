//! NumPyro's introductory example: eight schools, sampled with NUTS.
//!
//! Eight schools ran coaching programmes for a test; each reports an estimated effect and its
//! standard error. A hierarchical model shares strength between schools. The half-Cauchy prior
//! on the spread between schools has no built-in node, so it is written as a `Potential`, and
//! the non-centred school effects are a deterministic `Function`.
//!
//! Run with `cargo run --release --example eight_schools`.

use horizon::model::Model;
use horizon::node::Support;
use horizon::nodes::{Function, Potential};

fn main() {
    let y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
    let sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];
    let variance: Vec<f64> = sigma.iter().map(|s| s * s).collect();

    // mu ~ Normal(0, 5²); tau ~ HalfCauchy(5)
    // theta_raw ~ Normal(0, 1); theta = mu + tau * theta_raw; y ~ Normal(theta, sigma²)
    let mut m = Model::new();
    let mu = m.random("mu");
    m.normal_mean_variance(mu, 0.0, 25.0);

    let tau = m.random("tau");
    let half_cauchy = Potential::scalar(1, |x| -(1.0 + (x[0] / 5.0).powi(2)).ln())
        .with_support(0, Support::Positive);
    m.potential(half_cauchy, [tau.into()]);

    let theta_raw = m.random_vec("theta_raw", y.len());
    m.normal_mean_variance(theta_raw, 0.0, 1.0);
    let theta = m.random_vec("theta", y.len());
    let effect = Function::scalar(3, |x| x[0] + x[1] * x[2]);
    m.function(theta, effect, [mu.into(), tau.into(), theta_raw.into()]);

    let obs = m.data("y", &y[..]);
    m.normal_mean_variance(obs, theta, variance);

    let start = std::time::Instant::now();
    let samples = m
        .sample()
        .chains(4)
        .warmup(1000)
        .draws(1000)
        .target_accept(0.9)
        .run()
        .expect("sampling failed");
    let elapsed = start.elapsed();

    let mut summary = samples.summary();
    summary.rows.retain(|r| !r.name.starts_with("theta_raw"));
    println!("{summary}");
    println!(
        "\n4 chains × 2000 iterations in {:.2} s",
        elapsed.as_secs_f64()
    );
}
