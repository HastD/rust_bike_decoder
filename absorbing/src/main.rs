use bike_decoder::{decoder::DecodingFailure, graphs::is_decoding_failure_absorbing};
use serde::Deserialize;
use serde_json::Deserializer;
use std::io::{self, Write};

fn main() -> io::Result<()> {
    let mut de = Deserializer::from_reader(io::stdin());
    let decoding_failures = <Vec<DecodingFailure>>::deserialize(&mut de)?;
    let absorbing: Vec<DecodingFailure> = decoding_failures
        .into_iter()
        .filter(is_decoding_failure_absorbing)
        .collect();
    serde_json::to_writer(io::stdout(), &absorbing)?;
    io::stdout().write_all(b"\n")?;
    Ok(())
}
