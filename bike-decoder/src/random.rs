//! Thread-local, fast, non-cryptographic random number generator with public seed

// This is a modified version of the implementation of rand::ThreadRng,
// suitable for applications where reproducibility of the results is desired.

use hex::{FromHex, ToHex};
use once_cell::sync::OnceCell;
use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::{
    cell::UnsafeCell,
    fmt,
    rc::Rc,
    sync::atomic::{AtomicU32, Ordering},
    thread_local,
};
use thiserror::Error;

static GLOBAL_SEED: OnceCell<Seed> = OnceCell::new();
static GLOBAL_THREAD_COUNT: AtomicU32 = AtomicU32::new(0);

thread_local! {
    // Relaxed ordering is appropriate since we just need the IDs to be unique;
    // it doesn't matter what order they're assigned in.
    static CURRENT_THREAD_ID: u32 = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::Relaxed);
    static CUSTOM_THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256PlusPlus>> = {
        let seed = get_or_insert_global_seed(None);
        let rng = get_rng_from_seed(seed, current_thread_id());
        Rc::new(UnsafeCell::new(rng))
    }
}

pub fn get_rng_from_seed(seed: Seed, jumps: u32) -> Xoshiro256PlusPlus {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed.into());
    for _ in 0..jumps {
        rng.jump();
    }
    rng
}

pub fn global_seed() -> Option<Seed> {
    GLOBAL_SEED.get().copied()
}

pub fn get_or_insert_global_seed(seed: Option<Seed>) -> Seed {
    *GLOBAL_SEED.get_or_init(|| seed.unwrap_or_else(Seed::from_entropy))
}

pub fn try_insert_global_seed(seed: Option<Seed>) -> Result<Seed, TryInsertGlobalSeedError> {
    let global_seed = get_or_insert_global_seed(seed);
    match seed {
        None => Ok(global_seed),
        Some(seed) if seed == global_seed => Ok(global_seed),
        Some(_) => Err(TryInsertGlobalSeedError(global_seed)),
    }
}

#[derive(Clone, Debug, Error)]
#[error("try_insert_global_seed failed, GLOBAL_SEED already set to value: {0}")]
pub struct TryInsertGlobalSeedError(Seed);

pub fn global_thread_count() -> u32 {
    GLOBAL_THREAD_COUNT.load(Ordering::Relaxed)
}

pub fn current_thread_id() -> u32 {
    CURRENT_THREAD_ID.with(|x| *x)
}

/// Generates a thread-local PRNG that uses Xoshiro256PlusPlus as the core,
/// seeded with GLOBAL_SEED, with a number of jumps equal to CURRENT_THREAD_ID.
/// This allows for fast pseudorandom number generation across multiple threads
/// with fully reproducible results given GLOBAL_SEED.
pub fn custom_thread_rng() -> CustomThreadRng {
    CustomThreadRng {
        rng: CUSTOM_THREAD_RNG_KEY.with(|t| t.clone()),
    }
}

// Note: Debug implementation intentionally leaks internal state.
#[derive(Clone, Debug)]
pub struct CustomThreadRng {
    rng: Rc<UnsafeCell<Xoshiro256PlusPlus>>,
}

impl Default for CustomThreadRng {
    fn default() -> Self {
        custom_thread_rng()
    }
}

impl RngCore for CustomThreadRng {
    #[inline(always)]
    fn next_u32(&mut self) -> u32 {
        // SAFETY: self.rng is !Sync, hence can't be concurrently mutated. No
        // other references to self.rng exist because we never give any out.
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        // SAFETY: self.rng is !Sync, hence can't be concurrently mutated. No
        // other references to self.rng exist because we never give any out.
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // SAFETY: self.rng is !Sync, hence can't be concurrently mutated. No
        // other references to self.rng exist because we never give any out.
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        // SAFETY: self.rng is !Sync, hence can't be concurrently mutated. No
        // other references to self.rng exist because we never give any out.
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}

type SeedInner = [u8; 32];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Seed(#[serde(with = "hex::serde")] SeedInner);

impl Seed {
    pub fn new(arr: SeedInner) -> Self {
        Self(arr)
    }

    pub fn from_entropy() -> Self {
        let mut buf = SeedInner::default();
        OsRng.fill_bytes(&mut buf);
        Seed(buf)
    }
}

impl From<Seed> for SeedInner {
    #[inline]
    fn from(seed: Seed) -> Self {
        seed.0
    }
}

impl FromHex for Seed {
    type Error = SeedFromHexError;
    fn from_hex<T: AsRef<[u8]>>(hex: T) -> Result<Self, Self::Error> {
        Ok(Self(SeedInner::from_hex(hex)?))
    }
}

impl ToHex for Seed {
    fn encode_hex<T: std::iter::FromIterator<char>>(&self) -> T {
        self.0.encode_hex()
    }

    fn encode_hex_upper<T: std::iter::FromIterator<char>>(&self) -> T {
        self.0.encode_hex_upper()
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.encode_hex::<String>())
    }
}

#[derive(Debug, Error)]
#[error("PRNG seed must be given by 256-bit hex string")]
pub struct SeedFromHexError(#[from] pub hex::FromHexError);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_rng_seeds() {
        let mut rng = custom_thread_rng();
        {
            // SAFETY: See safety comments for CustomThreadRng methods.
            let rng_inner = unsafe { &mut *rng.rng.get() };
            rng_inner.jump();
        }
        let x = rng.next_u64();
        let (y, other_thread_id) = std::thread::spawn(|| {
            let mut rng = custom_thread_rng();
            (rng.next_u64(), current_thread_id())
        })
        .join()
        .unwrap();
        assert_eq!(current_thread_id() + 1, other_thread_id);
        assert_eq!(x, y);
        assert_eq!(global_thread_count(), 2);
    }
}
