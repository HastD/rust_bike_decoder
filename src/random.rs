//! Thread-local, fast, non-cryptographic random number generator with public seed

// This is a modified version of the implementation of rand::ThreadRng,
// suitable for applications where reproducibility of the results is desired.

use std::{
    cell::UnsafeCell,
    convert::TryFrom,
    rc::Rc,
    sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}},
    thread_local,
};
use anyhow::Context;
use lazy_static::lazy_static;
use rand::{RngCore, Error, SeedableRng, rngs::OsRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

lazy_static! {
    static ref SEED: Arc<Mutex<Option<Seed>>> = Arc::new(Mutex::new(None));
    static ref JUMP_COUNTER: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
}

pub fn global_seed() -> Option<Seed> {
    *SEED.lock().expect("Must be able to access global seed")
}

pub fn get_or_insert_global_seed(seed: Option<Seed>) -> Seed {
    let mut global_seed = SEED.lock().expect("Must be able to access global seed");
    *global_seed.get_or_insert(seed.unwrap_or_else(|| Seed::from_entropy()))
}

thread_local! {
    pub static JUMPS: usize = JUMP_COUNTER.fetch_add(1, Ordering::SeqCst);
    static CUSTOM_THREAD_RNG_KEY: Rc<UnsafeCell<Xoshiro256PlusPlus>> = {
        let seed = get_or_insert_global_seed(None);
        let mut rng = Xoshiro256PlusPlus::from_seed(seed.0);
        for _ in 0..JUMPS.with(|x| *x) {
            rng.jump();
        }
        Rc::new(UnsafeCell::new(rng))
    }
}

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
    type Error = anyhow::Error;

    fn try_from(value: String) -> anyhow::Result<Self> {
        let bytes = hex::decode(value).context("Failed to decode hex string into byte array")?;
        let arr = SeedInner::try_from(&bytes[..]).context("PRNG seed must be 256 bits")?;
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
