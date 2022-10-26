use crate::parameters::*;
use crate::vectors::Index;
use rand::distributions::Uniform;
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn get_rng() -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::from_entropy()
}

pub fn get_key_dist() -> Uniform<Index> {
    Uniform::new(0, BLOCK_LENGTH as Index)
}

pub fn get_err_dist() -> Uniform<Index> {
    Uniform::new(0, ROW_LENGTH as Index)
}
