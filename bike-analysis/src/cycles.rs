use bike_decoder::decoder::{self, DecoderCycle, DecodingFailure};
use rayon::prelude::*;

pub fn find_cycles(
    decoding_failures: Vec<DecodingFailure>,
    parallel: bool,
    max_iters: usize,
) -> Vec<DecoderCycle> {
    if parallel {
        decoding_failures
            .into_par_iter()
            .map(|df| decoder::find_bgf_cycle(df.key(), df.vector().vector(), max_iters))
            .collect()
    } else {
        decoding_failures
            .into_iter()
            .map(|df| decoder::find_bgf_cycle(df.key(), df.vector().vector(), max_iters))
            .collect()
    }
}
