use std::sync::Arc;

type ScalarFn = Arc<dyn Fn(&[f64]) -> f64 + Send + Sync>;
type PartialsFn = Arc<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

/// A closure over scalar arguments, applied entry by entry to arguments that are broadcast
/// against each other: arguments with one entry are shared by every entry.
#[derive(Clone)]
pub(crate) struct Elementwise {
    f: ScalarFn,
    partials: Option<PartialsFn>,
}

impl Elementwise {
    pub fn new(f: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            f: Arc::new(f),
            partials: None,
        }
    }

    pub fn with_partials(
        mut self,
        partials: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        self.partials = Some(Arc::new(partials));
        self
    }

    /// The number of entries after broadcasting.
    fn len(args: &[&[f64]]) -> usize {
        let n = args.iter().map(|x| x.len()).max().unwrap_or(1);
        assert!(
            args.iter().all(|x| x.len() == 1 || x.len() == n),
            "arguments of {n} entries and of other lengths cannot be broadcast"
        );
        n
    }

    fn entry(args: &[&[f64]], i: usize, x: &mut [f64]) {
        for (xk, a) in x.iter_mut().zip(args) {
            *xk = if a.len() == 1 { a[0] } else { a[i] };
        }
    }

    /// `f` at every entry.
    pub fn values(&self, args: &[&[f64]]) -> Vec<f64> {
        let mut x = vec![0.0; args.len()];
        (0..Self::len(args))
            .map(|i| {
                Self::entry(args, i, &mut x);
                (self.f)(&x)
            })
            .collect()
    }

    /// `Σᵢ weight(i) ∂f(entry i)/∂argument`, for each argument.
    pub fn weighted_grad(&self, args: &[&[f64]], weight: impl Fn(usize) -> f64) -> Vec<Vec<f64>> {
        let mut grads: Vec<Vec<f64>> = args.iter().map(|a| vec![0.0; a.len()]).collect();
        let mut x = vec![0.0; args.len()];
        for i in 0..Self::len(args) {
            Self::entry(args, i, &mut x);
            let w = weight(i);
            let partials = match &self.partials {
                Some(p) => p(&x),
                None => central_differences(&*self.f, &mut x),
            };
            for (k, dk) in partials.into_iter().enumerate() {
                let j = if args[k].len() == 1 { 0 } else { i };
                grads[k][j] += w * dk;
            }
        }
        grads
    }
}

/// The partial derivatives of `f` at `x` by central differences, one-sided at the edge of the
/// support.
fn central_differences(f: &dyn Fn(&[f64]) -> f64, x: &mut [f64]) -> Vec<f64> {
    (0..x.len())
        .map(|k| {
            let xk = x[k];
            let h = 1e-6 * (1.0 + xk.abs());
            let mut at = |delta: f64| {
                x[k] = xk + delta;
                let v = f(x);
                x[k] = xk;
                v
            };
            let (plus, minus) = (at(h), at(-h));
            match (plus.is_finite(), minus.is_finite()) {
                (true, true) => (plus - minus) / (2.0 * h),
                (true, false) => (plus - at(0.0)) / h,
                (false, true) => (at(0.0) - minus) / h,
                (false, false) => 0.0,
            }
        })
        .collect()
}
