//! Thread-local, fast, non-cryptographic random number generator with public seed

// This is a modified version of the implementation of rand::ThreadRng,
// suitable for applications where reproducibility of the results is desired.

use rand::{rngs::OsRng, Error, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::{
    cell::UnsafeCell,
    fmt,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    thread_local,
};
use thiserror::Error;

static GLOBAL_SEED: Mutex<Option<Seed>> = Mutex::new(None);
static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static CURRENT_THREAD_ID: usize = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::AcqRel);
    static CUSTOM_THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256PlusPlus>> = {
        let seed = get_or_insert_global_seed(None);
        let rng = get_rng_from_seed(seed, current_thread_id());
        Rc::new(UnsafeCell::new(rng))
    }
}

pub fn get_rng_from_seed(seed: Seed, jumps: usize) -> Xoshiro256PlusPlus {
    let mut rng = Xoshiro256PlusPlus::from_seed(seed.into());
    for _ in 0..jumps {
        rng.jump();
    }
    rng
}

pub fn global_seed() -> Option<Seed> {
    *GLOBAL_SEED
        .lock()
        .expect("GLOBAL_SEED should not be poisoned")
}

pub fn get_or_insert_global_seed(seed: Option<Seed>) -> Seed {
    let mut global_seed = GLOBAL_SEED
        .lock()
        .expect("GLOBAL_SEED should not be poisoned");
    *global_seed.get_or_insert(seed.unwrap_or_else(Seed::from_entropy))
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

pub fn global_thread_count() -> usize {
    GLOBAL_THREAD_COUNT.load(Ordering::Acquire)
}

pub fn current_thread_id() -> usize {
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

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
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

impl TryFrom<&str> for Seed {
    type Error = SeedFromHexError;

    fn try_from(value: &str) -> Result<Self, SeedFromHexError> {
        let bytes = hex::decode(value)?;
        let arr = SeedInner::try_from(&bytes[..])?;
        Ok(Self(arr))
    }
}

impl fmt::Display for Seed {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", hex::encode(self.0))
    }
}

#[derive(Debug, Error)]
pub enum SeedFromHexError {
    #[error("failed to decode hex string: {0}")]
    HexDecodeError(#[from] hex::FromHexError),
    #[error("PRNG seed must be 256 bits: {0}")]
    SizeError(#[from] std::array::TryFromSliceError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_rng_seeds() {
        let mut rng = custom_thread_rng();
        {
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
