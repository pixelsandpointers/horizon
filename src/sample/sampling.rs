use std::sync::atomic::{AtomicUsize, Ordering};

use burn::tensor::Device;
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::nuts::{DualAveraging, Nuts, State, Target, Welford, adaptation_windows};
use super::problem::Problem;
use super::samples::Samples;
use crate::error::{Error, Result};
use crate::model::{Model, VarId};
use crate::tensor::default_device;

/// Options for one sampling run, built with [`Model::sample`].
pub struct Sampling<'m> {
    model: &'m Model,
    device: Device,
    chains: usize,
    threads: usize,
    warmup: usize,
    draws: usize,
    seed: u64,
    max_tree_depth: usize,
    target_accept: f64,
}

impl Model {
    /// Starts configuring sampling with NUTS; finish with [`Sampling::run`].
    pub fn sample(&self) -> Sampling<'_> {
        Sampling {
            model: self,
            device: default_device(),
            chains: 4,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            warmup: 1000,
            draws: 1000,
            seed: 0,
            max_tree_depth: 10,
            target_accept: 0.8,
        }
    }
}

struct ChainOutput {
    draws: Vec<Vec<Vec<f64>>>,
    divergences: usize,
}

impl Sampling<'_> {
    /// Number of chains (default 4), run in parallel.
    pub fn chains(mut self, n: usize) -> Self {
        self.chains = n.max(1);
        self
    }

    /// Worker threads that chains are spread over (default: the available parallelism).
    pub fn threads(mut self, n: usize) -> Self {
        self.threads = n.max(1);
        self
    }

    /// Warmup iterations per chain, for adaptation (default 1000).
    pub fn warmup(mut self, n: usize) -> Self {
        self.warmup = n;
        self
    }

    /// Kept draws per chain (default 1000).
    pub fn draws(mut self, n: usize) -> Self {
        self.draws = n.max(1);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Maximum NUTS tree depth (default 10).
    pub fn max_tree_depth(mut self, depth: usize) -> Self {
        self.max_tree_depth = depth.max(1);
        self
    }

    /// Target acceptance probability for step-size adaptation (default 0.8).
    pub fn target_accept(mut self, p: f64) -> Self {
        self.target_accept = p;
        self
    }

    /// Evaluates on another burn device, such as a GPU (default: burn's CPU backend).
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn run(self) -> Result<Samples> {
        let recorded = Problem::new(self.model, self.device.clone())?.recorded();
        // Workers take the next chain until none are left. Seeds depend only on the chain index,
        // so draws do not depend on which worker ran a chain.
        let next = AtomicUsize::new(0);
        let mut outputs: Vec<(usize, Result<ChainOutput>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..self.threads.min(self.chains))
                .map(|_| {
                    let (this, recorded, next) = (&self, &recorded, &next);
                    scope.spawn(move || {
                        let mut done = Vec::new();
                        loop {
                            let c = next.fetch_add(1, Ordering::Relaxed);
                            if c >= this.chains {
                                break done;
                            }
                            done.push((c, this.run_chain(c, recorded)));
                        }
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("chain panicked"))
                .collect()
        });
        outputs.sort_by_key(|&(c, _)| c);
        let outputs = outputs
            .into_iter()
            .map(|(_, o)| o)
            .collect::<Result<Vec<_>>>()?;
        Ok(Samples {
            names: recorded
                .iter()
                .map(|&v| self.model.vars[v].name.clone())
                .collect(),
            vars: recorded.iter().map(|&v| VarId(v)).collect(),
            divergences: outputs.iter().map(|o| o.divergences).sum(),
            chains: outputs.into_iter().map(|o| o.draws).collect(),
        })
    }

    fn run_chain(&self, chain: usize, recorded: &[usize]) -> Result<ChainOutput> {
        // Each chain builds its own tensors, on its worker's thread.
        let mut problem = Problem::new(self.model, self.device.clone())?;
        let mut rng = StdRng::seed_from_u64(
            self.seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(chain as u64),
        );
        let dim = problem.dim;
        let mut nuts = Nuts::new(dim);
        nuts.max_depth = self.max_tree_depth;

        // Start near typical values, jittered so that chains differ.
        let base = problem.typical_params();
        let mut start = None;
        for attempt in 0..100 {
            let scale = 0.5 / (1.0 + attempt as f64);
            let q: Vec<f64> = base
                .iter()
                .map(|y| y + scale * (2.0 * rng.random::<f64>() - 1.0))
                .collect();
            if problem.log_density_grad(&q).0.is_finite() {
                start = Some(q);
                break;
            }
        }
        let q = start.ok_or_else(|| {
            Error::InvalidModel("could not find a starting point with finite log density".into())
        })?;

        let mut state = State::new(q, &mut problem);
        nuts.find_reasonable_step_size(&state, &mut problem, &mut rng);
        let windows = adaptation_windows(self.warmup);
        let mut adapt = DualAveraging::new(nuts.step_size, self.target_accept);
        let mut welford = Welford::new(dim);
        let mut divergences = 0;
        let mut draws = Vec::with_capacity(self.draws);

        for i in 0..self.warmup + self.draws {
            let (next, info) = nuts.transition(&state, &mut problem, &mut rng);
            state = next;
            if i < self.warmup {
                nuts.step_size = adapt.update(info.accept_prob);
                welford.add(&state.q);
                if windows.contains(&(i + 1)) {
                    nuts.inv_mass = welford.regularized_variance();
                    welford = Welford::new(dim);
                    nuts.find_reasonable_step_size(&state, &mut problem, &mut rng);
                    adapt = DualAveraging::new(nuts.step_size, self.target_accept);
                }
                if i + 1 == self.warmup {
                    nuts.step_size = adapt.final_step_size();
                }
            } else {
                divergences += info.divergent as usize;
                draws.push(problem.values(&state.q, recorded));
            }
        }
        Ok(ChainOutput { draws, divergences })
    }
}
