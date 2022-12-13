#![forbid(unsafe_code)]

pub mod atls;
pub mod cli;
pub mod decoder;
//pub mod graphs;
pub mod keys;
pub mod parameters;
pub mod random;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::cli::{Args, Settings, RuntimeError};
use clap::Parser;

fn main() -> Result<(), RuntimeError> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    cli::run_cli(settings)
}
