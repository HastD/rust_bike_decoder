use anyhow::{Context, Result};
use bike_decoder::{
    decoder::DecodingFailure,
    keys::Key,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT},
};
use clap::{Parser, Subcommand};
use ncw_classify::ClassifiedVector;
use serde::{Deserialize, Serialize};
use serde_json::Deserializer;
use std::io::{self, Write};

#[derive(Debug, Parser)]
#[command(author, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
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
        #[arg(help = "Weight of absorbing sets")]
        weight: usize,
        #[arg(short = 'N', long, help = "Number of samples")]
        number: f64,
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

fn process_input() -> Result<()> {
    let mut de = Deserializer::from_reader(io::stdin());
    let decoding_failures = <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")?;
    let classified: Vec<ClassifiedVector<BLOCK_WEIGHT, BLOCK_LENGTH>> = decoding_failures
        .into_iter()
        .map(|df| ClassifiedVector::new(df.key().clone(), df.vector().vector().support()))
        .collect();
    write_json(&classified)
}

fn sample(key: Option<Key>, error_weight: usize, samples: usize, parallel: bool) -> Result<()> {
    let key = key.unwrap_or_else(|| Key::random(&mut rand::thread_rng()));
    let classified = ClassifiedVector::sample(&key, error_weight, samples, parallel);
    write_json(&classified)?;
    Ok(())
}

fn parse_key(s: String) -> Result<Key> {
    let key: Key =
        serde_json::from_str::<Key>(&s).context("--key should be valid JSON representing a key")?;
    Ok(key)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Process => process_input(),
        Command::Sample {
            key,
            weight,
            number,
            parallel,
        } => {
            let key = key.map(parse_key).transpose()?;
            sample(key, weight, number as usize, parallel)
        }
    }
}
