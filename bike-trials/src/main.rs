mod application;
mod output;
mod parallel;
mod record;
mod settings;

use crate::record::DataRecord;
use crate::settings::{Args, Settings};
use clap::Parser;

pub fn run_application(settings: &Settings) -> Result<DataRecord, anyhow::Error> {
    if settings.parallel() {
        parallel::run_parallel(settings)
    } else {
        application::run(settings)
    }
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();
    let settings = Settings::from_args(args)?;
    run_application(&settings)?;
    Ok(())
}
