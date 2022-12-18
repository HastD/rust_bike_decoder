pub mod cli;
pub mod decoder;
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

use crate::settings::{Args, Settings};
use anyhow::Result;
use clap::Parser;

fn main() -> Result<()> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    cli::run_cli(settings)?;
    Ok(())
}
