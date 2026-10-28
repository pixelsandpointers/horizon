use std::fmt;

/// Errors raised while building or sampling a model.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Model was built incorrectly.
    InvalidModel(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModel(msg) => write!(f, "invalid model: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
