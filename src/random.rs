use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn get_rng() -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::from_entropy()
}
