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

use clap::Parser;

fn main() -> Result<(), cli::UserInputError> {
    let args = cli::Args::parse();
    cli::run_cli(args)
}
