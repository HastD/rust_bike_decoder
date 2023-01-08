pub mod graphs;

use crate::graphs::AbsorbingDecodingFailure;
use anyhow::{Context, Result};
use bike_decoder::decoder::DecodingFailure;
use serde::Deserialize;
use serde_json::Deserializer;
use std::io::{self, Write};

fn filter() -> Result<()> {
    let mut de = Deserializer::from_reader(io::stdin());
    let decoding_failures = <Vec<DecodingFailure>>::deserialize(&mut de)
        .context("Failed to parse JSON input as Vec<DecodingFailure>")?;
    let absorbing: Vec<AbsorbingDecodingFailure> = decoding_failures
        .into_iter()
        .filter_map(AbsorbingDecodingFailure::new)
        .collect();
    let mut out = io::stdout();
    serde_json::to_writer(&mut out, &absorbing).context("Failed to write output list as JSON")?;
    out.write_all(b"\n")?;
    Ok(())
}

fn main() -> Result<()> {
    filter()
}
