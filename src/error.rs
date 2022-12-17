use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("error parsing JSON for fixed_key argument: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("invalid support for vector or key: {0}")]
    DataError(#[from] crate::vectors::InvalidSupport),
    #[error("argument outside of valid range: {0}")]
    RangeError(String),
    #[error("broken argument dependency: {0}")]
    DependencyError(String),
    #[error("error writing to file: {0}")]
    IOError(#[from] std::io::Error),
    #[error("miscellaneous error: {0}")]
    MiscError(#[from] anyhow::Error),
}