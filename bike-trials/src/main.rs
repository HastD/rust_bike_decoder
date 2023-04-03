use bike_trials::{
    application, parallel,
    settings::{Args, Settings},
};
use clap::Parser;

fn main() -> anyhow::Result<()> {
    let settings = Settings::try_from(Args::parse())?;
    if settings.parallel() {
        parallel::run_parallel(&settings)?;
    } else {
        application::run(&settings)?;
    }
    Ok(())
}
