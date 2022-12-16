#![forbid(unsafe_code)]

pub mod cli;
pub mod decoder;
pub mod error;
//pub mod graphs;
pub mod keys;
pub mod ncw;
pub mod parameters;
pub mod random;
pub mod record;
pub mod settings;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::{
    settings::{Args, Settings},
    error::RuntimeError,
};
use clap::Parser;

fn main() -> Result<(), RuntimeError> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    cli::run_cli(settings)
}
