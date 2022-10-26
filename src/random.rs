use crate::parameters::*;
use crate::vectors::Index;
use rand::distributions::Uniform;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn get_rng() -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(0)
}

pub fn get_key_dist() -> Uniform<Index> {
    Uniform::new(0, BLOCK_LENGTH as Index)
}

pub fn get_err_dist() -> Uniform<Index> {
    Uniform::new(0, ROW_LENGTH as Index)
}
