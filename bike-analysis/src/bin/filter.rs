use anyhow::Context;
use bike_analysis::{
    output::write_json,
    record::{AnalysisResult, DecodingFailureAnalysis},
};
use bike_decoder::{
    decoder::DecodingFailure,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT},
};
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::Deserializer;
use std::{io, time::Instant};

type AnalysisRecord = bike_analysis::record::AnalysisRecord<BLOCK_WEIGHT, BLOCK_LENGTH>;

#[derive(Debug, Parser)]
#[command(about = "Filters and analyzes decoding failures received on stdin", long_about = None)]
struct Cli {
    #[arg(short, long, help = "Identify absorbing sets")]
    absorbing: bool,
    #[arg(
        short,
        long,
        default_value_t = 100_000,
        help = "Maximum number of iterations to search for cycles"
    )]
    iters: usize,
    #[arg(long, help = "Classify e_in - e_out into near-codeword sets")]
    ncw: bool,
    #[arg(long, help = "Classify e_in into near-codeword sets")]
    ncw_in: bool,
    #[arg(short, long, help = "Run in parallel using multiple threads")]
    parallel: bool,
}

fn decoding_failures_from_stdin() -> Result<Vec<DecodingFailure>, anyhow::Error> {
    let mut de = Deserializer::from_reader(io::stdin());
    <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")
}

fn run(cli: Cli, decoding_failures: Vec<DecodingFailure>) -> AnalysisRecord {
    let count = decoding_failures.len();
    let start_time = Instant::now();
    let mut results = decoding_failures
        .iter()
        .map(|df| DecodingFailureAnalysis::new(df, cli.iters))
        .collect::<Vec<_>>();
    if cli.parallel {
        results.par_iter_mut().for_each(|df_analysis| {
            if cli.ncw {
                df_analysis.compute_overlaps_diff();
            }
            if cli.ncw_in {
                df_analysis.compute_overlaps_e_in();
            }
            if cli.absorbing {
                df_analysis.compute_absorbing();
            }
        });
    } else {
        results.iter_mut().for_each(|df_analysis| {
            if cli.ncw {
                df_analysis.compute_overlaps_diff();
            }
            if cli.ncw_in {
                df_analysis.compute_overlaps_e_in();
            }
            if cli.absorbing {
                df_analysis.compute_absorbing();
            }
        });
    }
    let results = results
        .into_iter()
        .map(AnalysisResult::DecodingFailure)
        .collect();
    AnalysisRecord::new(None, ERROR_WEIGHT, count, results, start_time.elapsed())
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    let decoding_failures = decoding_failures_from_stdin()?;
    let record = run(cli, decoding_failures);
    write_json(&record)
}
