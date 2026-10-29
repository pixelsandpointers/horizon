use std::fmt;

/// Errors raised while building or sampling a model.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Model was built incorrectly.
    InvalidModel(String),
    /// Messages could not be multiplied.
    IncompatibleProduct { left: String, right: String },
    /// The evidence has zero probability under the model.
    ImpossibleEvidence { context: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(msg) => write!(f, "invalid model: {msg}"),
            Self::IncompatibleProduct { left, right } => {
                write!(f, "cannot multiply messages {left} and {right}.")
            }
            Self::ImpossibleEvidence { context } => {
                write!(f, "evidence has zero probability {context}.")
            }
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
