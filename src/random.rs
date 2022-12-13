use rand::{RngCore, SeedableRng, rngs::OsRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub type Seed = <Xoshiro256PlusPlus as SeedableRng>::Seed;

pub fn get_rng(seed: Option<Seed>) -> (Xoshiro256PlusPlus, Seed) {
    let seed = seed.unwrap_or_else(|| {
        let mut seed = Seed::default();
        OsRng.fill_bytes(&mut seed);
        seed
    });
    (Xoshiro256PlusPlus::from_seed(seed), seed)
}
