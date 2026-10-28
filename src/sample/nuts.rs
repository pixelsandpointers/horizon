//! The No-U-Turn Sampler with multinomial trajectory sampling, the generalised U-turn criterion
//! and Stan-style warmup adaptation of the step size and a diagonal mass matrix.

use rand::Rng;

/// Evaluates the log density and its gradient at a point.
pub(crate) trait Target {
    fn log_density_grad(&mut self, q: &[f64]) -> (f64, Vec<f64>);
}

pub(crate) fn standard_normal(rng: &mut impl Rng) -> f64 {
    let (u, v): (f64, f64) = (rng.random::<f64>().max(1e-300), rng.random());
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

#[derive(Clone, Debug)]
pub(crate) struct State {
    pub q: Vec<f64>,
    p: Vec<f64>,
    pub log_density: f64,
    pub grad: Vec<f64>,
}

impl State {
    pub fn new(q: Vec<f64>, target: &mut dyn Target) -> Self {
        let (log_density, grad) = target.log_density_grad(&q);
        let p = vec![0.0; q.len()];
        Self {
            q,
            p,
            log_density,
            grad,
        }
    }
}

/// What happened during one transition.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Transition {
    pub accept_prob: f64,
    pub divergent: bool,
    pub tree_depth: usize,
}

/// A subtree of the trajectory: its two ends (the one next to where it started and the far one),
/// the state it proposes, and the bookkeeping for multinomial sampling and U-turn checks.
struct Subtree {
    first: State,
    last: State,
    proposal: State,
    log_sum_weight: f64,
    rho: Vec<f64>,
    valid: bool,
    divergent: bool,
    n_leapfrog: usize,
    sum_accept: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Nuts {
    pub step_size: f64,
    pub inv_mass: Vec<f64>,
    pub max_depth: usize,
}

impl Nuts {
    pub fn new(dim: usize) -> Self {
        Self {
            step_size: 1.0,
            inv_mass: vec![1.0; dim],
            max_depth: 10,
        }
    }

    fn hamiltonian(&self, s: &State) -> f64 {
        let kinetic: f64 = s.p.iter().zip(&self.inv_mass).map(|(p, m)| p * p * m).sum();
        -s.log_density + 0.5 * kinetic
    }

    /// `p ⊙ M⁻¹`, the velocity used by the U-turn criterion.
    fn sharp(&self, p: &[f64]) -> Vec<f64> {
        p.iter().zip(&self.inv_mass).map(|(p, m)| p * m).collect()
    }

    fn no_u_turn(&self, a: &[f64], b: &[f64], rho: &[f64]) -> bool {
        let dot = |x: &[f64]| {
            self.sharp(x)
                .iter()
                .zip(rho)
                .map(|(x, r)| x * r)
                .sum::<f64>()
        };
        dot(a) > 0.0 && dot(b) > 0.0
    }

    fn leapfrog(&self, s: &State, eps: f64, target: &mut dyn Target) -> State {
        let p_half: Vec<f64> =
            s.p.iter()
                .zip(&s.grad)
                .map(|(p, g)| p + 0.5 * eps * g)
                .collect();
        let q: Vec<f64> =
            s.q.iter()
                .zip(&p_half)
                .zip(&self.inv_mass)
                .map(|((q, p), m)| q + eps * m * p)
                .collect();
        let (log_density, grad) = target.log_density_grad(&q);
        let p = p_half
            .iter()
            .zip(&grad)
            .map(|(p, g)| p + 0.5 * eps * g)
            .collect();
        State {
            q,
            p,
            log_density,
            grad,
        }
    }

    fn sample_momentum(&self, rng: &mut impl Rng) -> Vec<f64> {
        self.inv_mass
            .iter()
            .map(|m| standard_normal(rng) / m.sqrt())
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn build_tree(
        &self,
        start: &State,
        eps: f64,
        depth: usize,
        h0: f64,
        target: &mut dyn Target,
        rng: &mut impl Rng,
    ) -> Subtree {
        if depth == 0 {
            let s = self.leapfrog(start, eps, target);
            let h = self.hamiltonian(&s);
            let divergent = !h.is_finite() || h - h0 > 1000.0;
            let log_weight = if h.is_finite() {
                h0 - h
            } else {
                f64::NEG_INFINITY
            };
            return Subtree {
                rho: s.p.clone(),
                first: s.clone(),
                last: s.clone(),
                proposal: s,
                log_sum_weight: log_weight,
                valid: !divergent,
                divergent,
                n_leapfrog: 1,
                sum_accept: if h.is_finite() {
                    (h0 - h).exp().min(1.0)
                } else {
                    0.0
                },
            };
        }
        let a = self.build_tree(start, eps, depth - 1, h0, target, rng);
        if !a.valid {
            return a;
        }
        let b = self.build_tree(&a.last, eps, depth - 1, h0, target, rng);
        let (n_leapfrog, sum_accept) = (a.n_leapfrog + b.n_leapfrog, a.sum_accept + b.sum_accept);
        if !b.valid {
            return Subtree {
                n_leapfrog,
                sum_accept,
                ..b
            };
        }
        let log_sum_weight = log_add_exp(a.log_sum_weight, b.log_sum_weight);
        let take_b = rng.random::<f64>() < (b.log_sum_weight - log_sum_weight).exp();
        let rho: Vec<f64> = a.rho.iter().zip(&b.rho).map(|(x, y)| x + y).collect();
        // The generalised criterion on the merged tree, plus Stan's checks across the junction.
        let rho_a_ext: Vec<f64> = a.rho.iter().zip(&b.first.p).map(|(x, y)| x + y).collect();
        let rho_b_ext: Vec<f64> = a.last.p.iter().zip(&b.rho).map(|(x, y)| x + y).collect();
        let valid = self.no_u_turn(&a.first.p, &b.last.p, &rho)
            && self.no_u_turn(&a.first.p, &b.first.p, &rho_a_ext)
            && self.no_u_turn(&a.last.p, &b.last.p, &rho_b_ext);
        Subtree {
            first: a.first,
            last: b.last,
            proposal: if take_b { b.proposal } else { a.proposal },
            log_sum_weight,
            rho,
            valid,
            divergent: false,
            n_leapfrog,
            sum_accept,
        }
    }

    /// One NUTS transition from `current`.
    pub fn transition(
        &self,
        current: &State,
        target: &mut dyn Target,
        rng: &mut impl Rng,
    ) -> (State, Transition) {
        let mut start = current.clone();
        start.p = self.sample_momentum(rng);
        let h0 = self.hamiltonian(&start);
        let (mut back, mut front) = (start.clone(), start.clone());
        let mut proposal = start.clone();
        let mut rho = start.p.clone();
        let mut log_sum_weight = 0.0;
        let mut info = Transition::default();
        let (mut n_leapfrog, mut sum_accept) = (0usize, 0.0);

        for depth in 0..self.max_depth {
            info.tree_depth = depth + 1;
            let forward = rng.random::<bool>();
            let (near, far) = if forward {
                (&front, &back)
            } else {
                (&back, &front)
            };
            let eps = if forward {
                self.step_size
            } else {
                -self.step_size
            };
            let new = self.build_tree(near, eps, depth, h0, target, rng);
            n_leapfrog += new.n_leapfrog;
            sum_accept += new.sum_accept;
            if new.divergent {
                info.divergent = true;
            }
            if !new.valid {
                break;
            }
            // Biased progressive sampling favours the new subtree.
            if rng.random::<f64>() < (new.log_sum_weight - log_sum_weight).exp() {
                proposal = new.proposal.clone();
            }
            log_sum_weight = log_add_exp(log_sum_weight, new.log_sum_weight);
            let rho_old = rho.clone();
            rho = rho.iter().zip(&new.rho).map(|(x, y)| x + y).collect();
            let rho_old_ext: Vec<f64> = rho_old
                .iter()
                .zip(&new.first.p)
                .map(|(x, y)| x + y)
                .collect();
            let rho_new_ext: Vec<f64> = near.p.iter().zip(&new.rho).map(|(x, y)| x + y).collect();
            let keep_going = self.no_u_turn(&far.p, &new.last.p, &rho)
                && self.no_u_turn(&far.p, &new.first.p, &rho_old_ext)
                && self.no_u_turn(&near.p, &new.last.p, &rho_new_ext);
            if forward {
                front = new.last;
            } else {
                back = new.last;
            }
            if !keep_going {
                break;
            }
        }
        info.accept_prob = if n_leapfrog > 0 {
            sum_accept / n_leapfrog as f64
        } else {
            0.0
        };
        (proposal, info)
    }

    /// Hoffman and Gelman's heuristic for a first step size.
    pub fn find_reasonable_step_size(
        &mut self,
        current: &State,
        target: &mut dyn Target,
        rng: &mut impl Rng,
    ) {
        let mut start = current.clone();
        start.p = self.sample_momentum(rng);
        let h0 = self.hamiltonian(&start);
        let log_accept = |nuts: &Self, target: &mut dyn Target| {
            let h = nuts.hamiltonian(&nuts.leapfrog(&start, nuts.step_size, target));
            if h.is_finite() {
                h0 - h
            } else {
                f64::NEG_INFINITY
            }
        };
        let direction = if log_accept(self, target) > 0.5f64.ln() {
            1.0
        } else {
            -1.0
        };
        for _ in 0..100 {
            let la = log_accept(self, target);
            if direction * la <= -direction * 2f64.ln() {
                break;
            }
            self.step_size *= 2f64.powf(direction);
        }
    }
}

/// Nesterov dual averaging of the log step size towards a target acceptance rate.
#[derive(Clone, Debug)]
pub(crate) struct DualAveraging {
    mu: f64,
    target: f64,
    counter: f64,
    s_bar: f64,
    x_bar: f64,
}

impl DualAveraging {
    pub fn new(step_size: f64, target: f64) -> Self {
        Self {
            mu: (10.0 * step_size).ln(),
            target,
            counter: 0.0,
            s_bar: 0.0,
            x_bar: 0.0,
        }
    }

    /// Updates with an acceptance statistic and returns the next step size.
    pub fn update(&mut self, accept_prob: f64) -> f64 {
        let (gamma, t0, kappa) = (0.05, 10.0, 0.75);
        self.counter += 1.0;
        let eta = 1.0 / (self.counter + t0);
        self.s_bar = (1.0 - eta) * self.s_bar + eta * (self.target - accept_prob);
        let x = self.mu - self.s_bar * self.counter.sqrt() / gamma;
        let weight = self.counter.powf(-kappa);
        self.x_bar = (1.0 - weight) * self.x_bar + weight * x;
        x.exp()
    }

    /// The averaged step size to use after warmup.
    pub fn final_step_size(&self) -> f64 {
        self.x_bar.exp()
    }
}

/// Stan's warmup schedule: a fast initial buffer, doubling slow windows that estimate the mass
/// matrix, and a terminal buffer for the step size alone. Returns the window ends.
pub(crate) fn adaptation_windows(warmup: usize) -> Vec<usize> {
    if warmup < 20 {
        return Vec::new();
    }
    let (init, term, mut window) = if warmup < 150 {
        let init = warmup * 15 / 100;
        let term = warmup / 10;
        (init, term, warmup - init - term)
    } else {
        (75, 50, 25)
    };
    let end_of_slow = warmup - term;
    let mut ends = Vec::new();
    let mut start = init;
    while start < end_of_slow {
        let mut end = start + window;
        // Stretch the last window to the end of the slow phase.
        if end + 2 * window > end_of_slow {
            end = end_of_slow;
        }
        ends.push(end);
        start = end;
        window *= 2;
    }
    ends
}

/// Running mean and variance.
#[derive(Clone, Debug)]
pub(crate) struct Welford {
    n: usize,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Welford {
    pub fn new(dim: usize) -> Self {
        Self {
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }

    pub fn add(&mut self, x: &[f64]) {
        self.n += 1;
        let n = self.n as f64;
        for ((&x, mean), m2) in x.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let delta = x - *mean;
            *mean += delta / n;
            *m2 += delta * (x - *mean);
        }
    }

    /// The variance, shrunk towards a small constant as Stan does.
    pub fn regularized_variance(&self) -> Vec<f64> {
        let n = self.n as f64;
        self.m2
            .iter()
            .map(|m2| {
                let var = if self.n > 1 { m2 / (n - 1.0) } else { 1.0 };
                (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            })
            .collect()
    }
}
