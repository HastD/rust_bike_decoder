use anyhow::Context;
use bike_analysis::{
    absorbing::find_absorbing, classify::ncw_classify, cycles::find_cycles, output::write_json,
};
use bike_decoder::{
    decoder::DecodingFailure,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT},
};
use clap::{Parser, Subcommand};
use serde::Deserialize;
use serde_json::Deserializer;
use std::{io, time::Instant};

type AnalysisRecord = bike_analysis::record::AnalysisRecord<BLOCK_WEIGHT, BLOCK_LENGTH>;
type AnalysisResults = bike_analysis::record::AnalysisResults<BLOCK_WEIGHT, BLOCK_LENGTH>;

#[derive(Debug, Parser)]
#[command(about = "Filters and analyzes decoding failures received on stdin", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
    #[arg(short, long, help = "Run in parallel using multiple threads")]
    parallel: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Finds which decoding failures yield absorbing sets
    Absorbing {
        #[arg(
            long,
            help = "Classify absorbing decoding failures into near-codeword sets"
        )]
        ncw: bool,
    },
    /// Searches for BGF decoder cycles
    Cycles {
        #[arg(help = "Number of iterations to search for cycles")]
        iters: usize,
        #[arg(long, help = "Classify e_in - e_out into near-codeword sets")]
        ncw: bool,
    },
    /// Classifies decoding failures into near-codeword sets
    Ncw,
}

fn decoding_failures_from_stdin() -> Result<Vec<DecodingFailure>, anyhow::Error> {
    let mut de = Deserializer::from_reader(io::stdin());
    <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")
}

fn run(cli: Cli, decoding_failures: Vec<DecodingFailure>) -> AnalysisRecord {
    let count = decoding_failures.len();
    let start_time = Instant::now();
    let results = match cli.command {
        Command::Absorbing { ncw } => {
            let data = find_absorbing(decoding_failures, cli.parallel, ncw);
            AnalysisResults::AbsorbingDecodingFailures { data }
        }
        Command::Cycles { iters, ncw } => {
            let data = find_cycles(decoding_failures, cli.parallel, iters, ncw);
            AnalysisResults::DecoderCycles {
                data,
                iterations: iters,
            }
        }
        Command::Ncw => {
            let data = ncw_classify(decoding_failures, cli.parallel);
            AnalysisResults::NcwClassified { data }
        }
    };
    AnalysisRecord::new(None, ERROR_WEIGHT, count, results, start_time.elapsed())
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    let decoding_failures = decoding_failures_from_stdin()?;
    let record = run(cli, decoding_failures);
    write_json(&record)
}
