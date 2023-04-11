use anyhow::{anyhow, Context};
use bike_analysis::{
    output::write_json,
    record::{AnalysisResult, SampleAnalysis},
};
use bike_decoder::{
    env_or_usize, graphs::TannerGraphEdges, keys::QuasiCyclic, random::custom_thread_rng,
    vectors::Index,
};
use clap::{ArgGroup, Parser};
use itertools::Itertools;
use malachite::num::arithmetic::traits::CheckedBinomialCoefficient;
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use std::time::Instant;

// Key constants used for sampling
const SAMPLE_BLOCK_WEIGHT: usize = env_or_usize!("SAMPLE_BLOCK_WEIGHT", 5);
const SAMPLE_BLOCK_LENGTH: usize = env_or_usize!("SAMPLE_BLOCK_LENGTH", 19);
type SampleKey = QuasiCyclic<SAMPLE_BLOCK_WEIGHT, SAMPLE_BLOCK_LENGTH>;
type AnalysisRecord =
    bike_analysis::record::AnalysisRecord<SAMPLE_BLOCK_WEIGHT, SAMPLE_BLOCK_LENGTH>;

#[derive(Clone, Debug, Parser)]
#[command(about = "Generates and analyzes support vectors", long_about = None)]
#[command(group(ArgGroup::new("num").required(true).args(["number", "enumerate"])))]
struct Cli {
    #[arg(
        short,
        long,
        help = "Use the specified key (in JSON format) [default: random]"
    )]
    key: Option<String>,
    #[arg(
        short = 'E',
        long,
        help = "Exhaustively enumerate vectors of given weight"
    )]
    enumerate: bool,
    #[arg(short = 'N', long, help = "Number of samples")]
    number: Option<f64>,
    #[arg(short, long, help = "Search for absorbing sets")]
    absorbing: bool,
    #[arg(long, help = "Classify vectors in near-codeword sets")]
    ncw: bool,
    #[arg(short, long, help = "Run in parallel using multiple threads")]
    parallel: bool,
    #[arg(short, long, help = "Weight of absorbing sets")]
    weight: usize,
}

#[derive(Clone, Debug)]
struct Settings {
    key: SampleKey,
    sample_method: SampleMethod,
    ncw: bool,
    absorbing: bool,
    parallel: bool,
    weight: usize,
}

impl TryFrom<Cli> for Settings {
    type Error = anyhow::Error;
    fn try_from(cli: Cli) -> Result<Self, Self::Error> {
        let mut settings = Self {
            key: parse_key_or_random(cli.key.as_deref())?,
            sample_method: SampleMethod::new(cli.number, cli.enumerate)?,
            ncw: cli.ncw,
            absorbing: cli.absorbing,
            parallel: cli.parallel,
            weight: cli.weight,
        };
        if let SampleMethod::Sample(samples) = settings.sample_method {
            if samples >= settings.enum_count() {
                eprintln!("Number of samples >= total number of candidates; enumerating instead.");
                settings.sample_method = SampleMethod::Enumerate;
            }
        }
        Ok(settings)
    }
}

impl Settings {
    fn count(&self) -> usize {
        match self.sample_method {
            SampleMethod::Sample(samples) => samples,
            SampleMethod::Enumerate => self.enum_count(),
        }
    }

    fn enum_count(&self) -> usize {
        usize::checked_binomial_coefficient(2 * SAMPLE_BLOCK_LENGTH, self.weight)
            .unwrap_or(usize::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SampleMethod {
    Sample(usize),
    Enumerate,
}

impl SampleMethod {
    fn new(number: Option<f64>, enumerate: bool) -> Result<Self, anyhow::Error> {
        match number {
            Some(samples) if !enumerate => Ok(SampleMethod::Sample(samples as usize)),
            None if enumerate => Ok(SampleMethod::Enumerate),
            _ => Err(anyhow!(
                "Invalid sample method: must either enumerate or specify number of samples"
            )),
        }
    }
}

fn parse_key_or_random(s: Option<&str>) -> Result<SampleKey, anyhow::Error> {
    let key = s
        .map(serde_json::from_str)
        .transpose()
        .context("--key should be valid JSON representing a key")?
        .unwrap_or_else(|| SampleKey::random(&mut custom_thread_rng()));
    Ok(key)
}

fn sample_set<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    weight: usize,
    num_samples: usize,
    parallel: bool,
) -> Vec<SampleAnalysis<WT, LEN>> {
    let edges = TannerGraphEdges::new(key);
    let n = 2 * LEN as Index;
    if parallel {
        (0..num_samples)
            .into_par_iter()
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), weight))
            .map(|supp| SampleAnalysis::with_edges(key.clone(), supp, edges.clone()))
            .collect()
    } else {
        (0..num_samples)
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), weight))
            .map(|supp| SampleAnalysis::with_edges(key.clone(), supp, edges.clone()))
            .collect()
    }
}

fn enumerate_set<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    weight: usize,
    parallel: bool,
) -> Vec<SampleAnalysis<WT, LEN>> {
    let edges = TannerGraphEdges::new(key);
    let n = 2 * LEN as Index;
    let combinations = (0..n).combinations(weight);
    if parallel {
        combinations
            .par_bridge()
            .map(|supp| SampleAnalysis::with_edges(key.clone(), supp, edges.clone()))
            .collect()
    } else {
        combinations
            .map(|supp| SampleAnalysis::with_edges(key.clone(), supp, edges.clone()))
            .collect()
    }
}

fn run(settings: Settings) -> AnalysisRecord {
    let start_time = Instant::now();
    let num_processed = settings.count();
    let Settings {
        key,
        sample_method,
        weight,
        ncw,
        absorbing,
        parallel,
    } = settings;
    let mut results = match sample_method {
        SampleMethod::Sample(num_samples) => sample_set(&key, weight, num_samples, parallel),
        SampleMethod::Enumerate => enumerate_set(&key, weight, parallel),
    };
    if parallel {
        results.par_iter_mut().for_each(|sample| {
            if ncw {
                sample.compute_overlaps();
            }
            if absorbing {
                sample.compute_absorbing();
            }
        });
    } else {
        results.iter_mut().for_each(|sample| {
            if ncw {
                sample.compute_overlaps();
            }
            if absorbing {
                sample.compute_absorbing();
            }
        });
    }
    let results = results.into_iter().map(AnalysisResult::Sample).collect();
    AnalysisRecord::new(
        Some(key),
        weight,
        num_processed,
        results,
        start_time.elapsed(),
    )
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    let settings = Settings::try_from(cli)?;
    let record = run(settings);
    write_json(&record)
}

#[allow(dead_code, clippy::all)]
const fn _compile_time_assertions() {
    const _: () = assert!(
        0 < SAMPLE_BLOCK_WEIGHT && SAMPLE_BLOCK_WEIGHT < SAMPLE_BLOCK_LENGTH,
        "SAMPLE_BLOCK_WEIGHT must be positive and less than SAMPLE_BLOCK_LENGTH"
    );
    const _: () = assert!(
        SAMPLE_BLOCK_WEIGHT <= u8::MAX as usize,
        "SAMPLE_BLOCK_WEIGHT > 255 not supported"
    );
    const _: () = assert!(2 * SAMPLE_BLOCK_LENGTH <= u32::MAX as usize);
}
