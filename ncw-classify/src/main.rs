use anyhow::Context;
use bike_decoder::{
    decoder::DecodingFailure,
    keys::{Key, QuasiCyclic},
    ncw::ClassifiedVector,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT},
    random::custom_thread_rng,
};
use clap::{Parser, Subcommand};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Deserializer;
use std::io::{self, Write};

#[derive(Debug, Parser)]
#[command(author, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(long, help = "Run in parallel using multiple threads")]
    parallel: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Classifies decoding failures received on stdin
    Process,
    /// Generates and classifies random vectors in near-codeword sets
    Sample {
        #[arg(
            long,
            help = "Use the specified key (in JSON format) [default: random]"
        )]
        key: Option<String>,
        #[arg(short, long, help = "Weight of absorbing sets")]
        weight: usize,
        #[arg(short = 'N', long, help = "Number of samples")]
        number: f64,
    },
}

/// Writes data in JSON format to stdout
fn write_json(data: &impl Serialize) -> Result<(), anyhow::Error> {
    let mut writer = io::stdout();
    serde_json::to_writer(&mut writer, data).context("data should be writable as JSON")?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn process_input(parallel: bool) -> Result<(), anyhow::Error> {
    let mut de = Deserializer::from_reader(io::stdin());
    let decoding_failures = <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")?;
    let classified: Vec<ClassifiedVector<BLOCK_WEIGHT, BLOCK_LENGTH>> = if parallel {
        decoding_failures
            .into_par_iter()
            .map(|df| ClassifiedVector::new(df.key().clone(), df.vector().vector().support()))
            .collect()
    } else {
        decoding_failures
            .into_iter()
            .map(|df| ClassifiedVector::new(df.key().clone(), df.vector().vector().support()))
            .collect()
    };
    write_json(&classified)
}

fn collect_sample<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    supp_weight: usize,
    samples: usize,
    parallel: bool,
) -> Vec<ClassifiedVector<WT, LEN>> {
    if parallel {
        (0..samples)
            .into_par_iter()
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    } else {
        (0..samples)
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    }
}

fn sample<const WT: usize, const LEN: usize>(
    key: Option<QuasiCyclic<WT, LEN>>,
    supp_weight: usize,
    samples: usize,
    parallel: bool,
) -> Result<(), anyhow::Error> {
    let key = key.unwrap_or_else(|| QuasiCyclic::random(&mut custom_thread_rng()));
    let classified = collect_sample(&key, supp_weight, samples, parallel);
    write_json(&classified)?;
    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    match cli.command {
        Command::Process => process_input(cli.parallel),
        Command::Sample {
            key,
            weight,
            number,
        } => {
            let key: Option<Key> = key
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .context("--key should be valid JSON representing a key")?;
            sample(key, weight, number as usize, cli.parallel)
        }
    }
}
