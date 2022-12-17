//! Thread-local, fast, non-cryptographic random number generator with public seed

// This is a modified version of the implementation of rand::ThreadRng,
// suitable for applications where reproducibility of the results is desired.

use std::{
    cell::UnsafeCell,
    convert::TryFrom,
    rc::Rc,
    sync::{Mutex, atomic::{AtomicUsize, Ordering}},
    thread_local,
};
use lazy_static::lazy_static;
use rand::{RngCore, Error, SeedableRng, rngs::OsRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

lazy_static! {
    static ref GLOBAL_SEED: Mutex<Option<Seed>> = Mutex::new(None);
    static ref GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);
}

pub fn global_seed() -> Option<Seed> {
    *GLOBAL_SEED.lock().expect("Must be able to access global seed")
}

pub fn get_or_insert_global_seed(seed: Option<Seed>) -> Seed {
    let mut global_seed = GLOBAL_SEED.lock().expect("Must be able to access global seed");
    *global_seed.get_or_insert(seed.unwrap_or_else(|| Seed::from_entropy()))
}

pub fn global_thread_count() -> usize {
    GLOBAL_THREAD_COUNT.load(Ordering::SeqCst)
}

thread_local! {
    pub static CURRENT_THREAD_ID: usize = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
    static CUSTOM_THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256PlusPlus>> = {
        let seed = get_or_insert_global_seed(None);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed.0);
        for _ in 0..CURRENT_THREAD_ID.with(|x| *x) {
            rng.jump();
        }
        Rc::new(UnsafeCell::new(rng))
    }
}

/// Generates a thread-local PRNG that uses Xoshiro256PlusPlus as the core,
/// seeded with GLOBAL_SEED, with a number of jumps equal to CURRENT_THREAD_ID.
/// This allows for fast pseudorandom number generation across multiple threads
/// with fully reproducible results given GLOBAL_SEED.
pub fn custom_thread_rng() -> CustomThreadRng {
    CustomThreadRng { rng: CUSTOM_THREAD_RNG_KEY.with(|t| t.clone()) }
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
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u32()
    }

    #[inline(always)]
    fn next_u64(&mut self) -> u64 {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.next_u64()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.fill_bytes(dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        // SAFETY: We must make sure to stop using `rng` before anyone else
        // creates another mutable reference
        let rng = unsafe { &mut *self.rng.get() };
        rng.try_fill_bytes(dest)
    }
}

type SeedInner = <Xoshiro256PlusPlus as SeedableRng>::Seed;

#[derive(Clone, Copy, Debug)]
pub struct Seed(SeedInner);

impl Seed {
    pub fn from_entropy() -> Self {
        let mut buf = SeedInner::default();
        OsRng.fill_bytes(&mut buf);
        Seed(buf)
    }
}

impl TryFrom<String> for Seed {
    type Error = SeedFromHexError;

    fn try_from(value: String) -> Result<Self, SeedFromHexError> {
        let bytes = hex::decode(value)?;
        let arr = SeedInner::try_from(&bytes[..])?;
        Ok(Self(arr))
    }
}

impl<'de> Deserialize<'de> for Seed {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where D: Deserializer<'de>
    {
        Ok(Seed(hex::serde::deserialize(deserializer)?))
    }
}

impl Serialize for Seed {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer
    {
        hex::serde::serialize(self.0, serializer)
    }
}

#[derive(Debug, Error)]
pub enum SeedFromHexError {
    #[error("failed to decode hex string: {0}")]
    HexDecodeError(#[from] hex::FromHexError),
    #[error("PRNG seed must be 256 bits: {0}")]
    SizeError(#[from] std::array::TryFromSliceError),
}
