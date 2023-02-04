mod application;
mod output;
mod parallel;
mod record;
mod settings;

use crate::record::DataRecord;
use crate::settings::{Args, Settings};
use anyhow::Result;
use clap::Parser;

pub fn run_application(settings: &Settings) -> Result<DataRecord> {
    if settings.parallel() {
        parallel::run_parallel(settings)
    } else {
        application::run(settings)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    run_application(&settings)?;
    Ok(())
}
