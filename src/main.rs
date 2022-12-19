pub mod application;
pub mod decoder;
//pub mod graphs;
pub mod keys;
pub mod ncw;
pub mod parallel;
pub mod parameters;
pub mod random;
pub mod record;
pub mod settings;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::settings::{Args, Settings};
use crate::record::DataRecord;
use anyhow::Result;
use clap::Parser;

pub fn run_application(settings: Settings) -> Result<DataRecord> {
    if settings.parallel() {
        parallel::run_parallel(settings)
    } else {
        application::run(settings)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    run_application(settings)?;
    Ok(())
}
