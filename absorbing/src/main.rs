pub mod graphs;

use crate::graphs::AbsorbingDecodingFailure;
use anyhow::{Context, Result};
use bike_decoder::{decoder::DecodingFailure, keys::QuasiCyclic};
use clap::{Parser, Subcommand};
use num_integer::binomial;
use serde::{Deserialize, Serialize};
use serde_json::Deserializer;
use std::{
    io::{self, Write},
    time::Instant,
};

// Key constants used for `enumerate` and `sample` commands
const BLOCK_WEIGHT: usize = 5;
const BLOCK_LENGTH: usize = 19;
type EnumKey = QuasiCyclic<BLOCK_WEIGHT, BLOCK_LENGTH>;

#[derive(Debug, Parser)]
#[command(author, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Filters decoding failures received on stdin to find absorbing sets
    Filter {
        #[arg(
            long,
            help = "Compute maximum overlaps with near-codeword sets C, N, and 2N"
        )]
        ncw: bool,
    },
    /// Enumerates all absorbing sets of a given weight
    Enumerate {
        #[arg(
            long,
            help = "Use the specified key (in JSON format) [default: random]"
        )]
        key: Option<String>,
        #[arg(help = "Weight of absorbing sets")]
        weight: usize,
        #[arg(short = 'v', long, help = "Verbose output")]
        verbose: bool,
        #[arg(long, help = "Run in parallel using multiple threads")]
        parallel: bool,
    },
    /// Samples absorbing sets of a given weight
    Sample {
        #[arg(long, help = "Use the specified key (in JSON format)")]
        key: Option<String>,
        #[arg(help = "Weight of absorbing sets")]
        weight: usize,
        #[arg(short = 'N', long, help = "Number of samples")]
        number: f64,
        #[arg(short = 'v', long, help = "Verbose output")]
        verbose: bool,
        #[arg(long, help = "Run in parallel using multiple threads")]
        parallel: bool,
    },
}

/// Writes data in JSON format to stdout
fn write_json(data: &impl Serialize) -> Result<()> {
    let mut writer = io::stdout();
    serde_json::to_writer(&mut writer, data).context("data should be writable as JSON")?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn filter(overlaps: bool) -> Result<()> {
    let mut de = Deserializer::from_reader(io::stdin());
    let decoding_failures = <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")?;
    let absorbing: Vec<AbsorbingDecodingFailure> = decoding_failures
        .into_iter()
        .filter_map(|df| AbsorbingDecodingFailure::new(df, overlaps))
        .collect();
    write_json(&absorbing)
}

fn enumerate(
    key: Option<EnumKey>,
    error_weight: usize,
    verbose: bool,
    parallel: bool,
) -> Result<()> {
    let key = key.unwrap_or_else(|| EnumKey::random(&mut rand::thread_rng()));
    let time = Instant::now();
    let absorbing = graphs::enumerate_absorbing_sets(&key, error_weight, parallel);
    if verbose {
        eprintln!("Key: {}", serde_json::to_string(&key)?);
        eprintln!("Runtime: {:?}", time.elapsed());
        eprintln!(
            "There are exactly {} absorbing sets of weight {}.",
            absorbing.len(),
            error_weight,
        );
        let total = binomial(2 * BLOCK_LENGTH, error_weight);
        if !absorbing.is_empty() {
            eprintln!(
                "(1 in {} error vectors of weight {} are absorbing.)",
                (total as f64 / absorbing.len() as f64).round() as usize,
                error_weight
            );
        }
    }
    write_json(&key)?;
    write_json(&absorbing)?;
    Ok(())
}

fn sample(
    key: Option<EnumKey>,
    error_weight: usize,
    samples: usize,
    verbose: bool,
    parallel: bool,
) -> Result<()> {
    if samples >= binomial(2 * BLOCK_LENGTH, error_weight) {
        eprintln!("Number of samples >= total number of candidates; enumerating instead.");
        return enumerate(key, error_weight, verbose, parallel);
    }
    let key = key.unwrap_or_else(|| EnumKey::random(&mut rand::thread_rng()));
    let time = Instant::now();
    let absorbing = graphs::sample_absorbing_sets(&key, error_weight, samples, parallel);
    if verbose {
        eprintln!("Key: {}", serde_json::to_string(&key)?);
        eprintln!("Runtime: {:?}", time.elapsed());
        eprintln!(
            "Found {} absorbing sets (out of {} sampled) of weight {}.",
            absorbing.len(),
            samples,
            error_weight,
        );
    }
    write_json(&key)?;
    write_json(&absorbing)?;
    Ok(())
}

fn parse_key(s: String) -> Result<EnumKey> {
    let key: EnumKey = serde_json::from_str::<EnumKey>(&s)
        .context("--key should be valid JSON representing a key")?;
    Ok(key)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Filter { ncw } => filter(ncw),
        Command::Enumerate {
            key,
            weight,
            verbose,
            parallel,
        } => {
            let key = key.map(parse_key).transpose()?;
            enumerate(key, weight, verbose, parallel)
        }
        Command::Sample {
            key,
            weight,
            number,
            verbose,
            parallel,
        } => {
            let key = key.map(parse_key).transpose()?;
            sample(key, weight, number as usize, verbose, parallel)
        }
    }
}
