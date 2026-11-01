use crate::{
    dist::Dist::{Normal, PointMass, Uninformative},
    error::{Error, Result},
};
use std::fmt;

/// A probability distribution used as a message or marginal.
///
/// `Normal` is parameterised by mean and variance.
#[derive(Clone, Debug, PartialEq)]
pub enum Dist {
    Uninformative,
    PointMass(Vec<f64>),
    Normal { mean: f64, variance: f64 },
}

impl Dist {
    pub fn normal(mean: f64, variance: f64) -> Self {
        assert!(
            variance > 0.,
            "Normal: Variance must be positive, got {variance}."
        );
        Self::Normal { mean, variance }
    }

    pub(crate) fn family(&self) -> &'static str {
        match self {
            Self::Uninformative => "Uninformative",
            Self::PointMass(_) => "PointMass",
            Self::Normal { .. } => "Normal",
        }
    }

    pub fn is_point_mass(&self) -> bool {
        matches!(self, Self::PointMass(_))
    }

    /// Normalised product of two densities: how messages arrive at a variable and combine.
    pub fn prod(&self, other: &Dist) -> Result<Dist> {
        let incompatible = || Error::IncompatibleProduct {
            left: self.family().to_string(),
            right: other.family().to_string(),
        };
        Ok(match (self, other) {
            (Uninformative, d) | (d, Uninformative) => d.clone(),
            (PointMass(x), PointMass(y)) => {
                if x != y {
                    return Err(Error::ImpossibleEvidence {
                        context: "two different point masses on one variable".into(),
                    });
                }
                PointMass(x.clone())
            }
            (PointMass(x), _) | (_, PointMass(x)) => PointMass(x.clone()),
            (
                Normal {
                    mean: m1,
                    variance: v1,
                },
                Normal {
                    mean: m2,
                    variance: v2,
                },
            ) => {
                let (w1, w2) = (1.0 / v1, 1.0 / v2);
                let w = w1 + w2;
                Normal {
                    mean: (m1 * w1 + m2 * w2) / w,
                    variance: 1.0 / w,
                }
            }
            #[allow(unreachable_patterns)]
            _ => return Err(incompatible()),
        })
    }

    /// Largest absolute difference between the parameters of two distributions of the same family.
    /// Detects convergence.
    pub fn distance(&self, other: &Dist) -> f64 {
        match (self, other) {
            (Uninformative, Uninformative) => 0.0,
            (PointMass(x), PointMass(y)) if x == y => 0.0,
            (
                Normal {
                    mean: m1,
                    variance: v1,
                },
                Normal {
                    mean: m2,
                    variance: v2,
                },
            ) => (m1 - m2).abs().max((v1 - v2).abs()),
            _ => f64::INFINITY,
        }
    }
}

impl fmt::Display for Dist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uninformative => write!(f, "Uninformative()"),
            Self::PointMass(x) => write!(f, "PointMass({x:?})"),
            Self::Normal { mean, variance } => write!(f, "Normal(mean={mean}, var={variance})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_product() {
        let d = Dist::normal(0.0, 1.0)
            .prod(&Dist::normal(2.0, 1.0))
            .unwrap();
        assert_eq!(d, Dist::normal(1.0, 0.5));
    }

    #[test]
    fn uninformative_is_neutral() {
        let n = Dist::normal(1.0, 2.0);
        assert_eq!(Dist::Uninformative.prod(&n).unwrap(), n);
    }
}
