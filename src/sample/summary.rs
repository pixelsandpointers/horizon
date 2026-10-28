use std::fmt;

/// One line of a [`Summary`].
#[derive(Clone, Debug)]
pub struct SummaryRow {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    pub median: f64,
    pub q05: f64,
    pub q95: f64,
    pub ess: f64,
    pub rhat: f64,
}

/// Posterior summary statistics with convergence diagnostics.
#[derive(Clone, Debug)]
pub struct Summary {
    pub rows: Vec<SummaryRow>,
    pub divergences: usize,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self
            .rows
            .iter()
            .map(|r| r.name.chars().count())
            .max()
            .unwrap_or(4)
            .max(4);
        writeln!(
            f,
            "{:>width$} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}",
            "", "mean", "std", "median", "5.0%", "95.0%", "n_eff", "r_hat"
        )?;
        for r in &self.rows {
            writeln!(
                f,
                "{:>width$} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.1} {:>7.2}",
                r.name, r.mean, r.std, r.median, r.q05, r.q95, r.ess, r.rhat
            )?;
        }
        write!(f, "\nNumber of divergences: {}", self.divergences)
    }
}
