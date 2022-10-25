pub mod constants;
pub mod decoder;
pub mod graphs;
pub mod keys;
pub mod random;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::constants::*;
use crate::keys::Key;
use crate::threshold::ThresholdCache;
use crate::vectors::SparseErrorVector;
use crate::syndrome::Syndrome;
use std::time::Instant;

const TRIALS: u32 = 1_000_000;

fn main() {
    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let key_dist = crate::random::get_key_dist();
    let err_dist = crate::random::get_err_dist();
    let mut rng = crate::random::get_rng();
    let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
    let start_time = Instant::now();
    let mut success_count = 0;
    for _ in 0..TRIALS {
        let key = Key::random_non_weak(&mut rng, &key_dist);
        let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
        let mut syn = Syndrome::from_sparse(&key, &e_supp);
        let (_e_out, success) = decoder::bgf_decoder(&key, &mut syn, &mut threshold_cache);
        if success {
            success_count += 1;
        }
        //let _graph = graphs::tanner_graph(&key);
        //println!("Nodes: {}; edges: {}", graph.node_count(), graph.edge_count());
    }
    let duration = start_time.elapsed();
    println!("Trials: {}", TRIALS);
    println!("Successes: {}", success_count);
    let dfr = (TRIALS as f64 - success_count as f64) / TRIALS as f64;
    println!("log2(DFR): {:.2}", dfr.log2());
    println!("Runtime: {}.{:0width$} s", duration.as_secs(), duration.subsec_millis(), width=3);
    let avg_nanos = duration.as_nanos() / TRIALS as u128;
    let (avg_mcs, ns_rem) = (avg_nanos / 1000, avg_nanos % 1000);
    if avg_mcs >= 100 {
        println!("Average: {} μs", avg_mcs);
    } else if avg_mcs >= 10 {
        println!("Average: {}.{} μs", avg_mcs, ns_rem / 100);
    } else if avg_mcs >= 1 {
        println!("Average: {}.{:0width$} μs", avg_mcs, ns_rem / 10, width=2);
    } else {
        println!("Average: {}.{:0width$} μs", avg_mcs, ns_rem, width=3);
    }
}
