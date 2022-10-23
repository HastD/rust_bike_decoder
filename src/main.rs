pub mod constants;
//pub mod graphs;
pub mod keys;
pub mod vectors;
pub mod threshold;

use crate::keys::{Key, ErrorVector};
use crate::constants::*;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

const TRIALS: u32 = 50_000;

fn main() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
    let threshold_cache = threshold::threshold_cache(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
    println!("Thresholds: {:?}", threshold_cache);
    let start_time = Instant::now();
    for _ in 0..TRIALS {
        let key = Key::random_non_weak(&mut rng);
        let e_supp = ErrorVector::random(&mut rng);
        let _syn = keys::syndrome(&key, &e_supp);
        //let _graph = graphs::tanner_graph(&key);
        //println!("Nodes: {}; edges: {}", graph.node_count(), graph.edge_count());
        //let e_out = bgf_decoder(&key, &s);
    }
    let duration = start_time.elapsed();
    println!("Trials: {}", TRIALS);
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
