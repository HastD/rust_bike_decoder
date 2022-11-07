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

fn main() {
    let args = cli::Args::parse();
    if let Err(message) = cli::run_cli(args) {
        eprintln!("Usage error: {}", message);
        std::process::exit(2);
    }
}
