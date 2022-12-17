use std::{io, sync::mpsc};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("error parsing JSON for fixed_key argument: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("invalid support for vector or key: {0}")]
    DataError(#[from] crate::vectors::InvalidSupport),
    #[error("argument outside of valid range: {0}")]
    RangeError(String),
    #[error("threshold computation error: {0}")]
    ThresholdError(#[from] crate::threshold::ThresholdError),
    #[error("broken argument dependency: {0}")]
    DependencyError(String),
    #[error("error writing to file: {0}")]
    IOError(#[from] io::Error),
    #[error("thread pool error: {0}")]
    ThreadPoolError(#[from] rayon::ThreadPoolBuildError),
    #[error("transmission error: {0}")]
    SendProgressError(#[from] mpsc::SendError<(usize, usize)>),
    #[error(transparent)]
    SeedError(#[from] crate::random::SeedFromHexError),
}
