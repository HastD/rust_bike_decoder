use std::convert::TryFrom;
use anyhow::Context;
use rand::{RngCore, SeedableRng, rngs::OsRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub fn get_rng(seed: Option<Seed>) -> (Xoshiro256PlusPlus, Seed) {
    let seed = seed.unwrap_or_else(|| {
        let mut seed_data = SeedInner::default();
        OsRng.fill_bytes(&mut seed_data);
        Seed(seed_data)
    });
    (Xoshiro256PlusPlus::from_seed(seed.0), seed)
}

type SeedInner = <Xoshiro256PlusPlus as SeedableRng>::Seed;

#[derive(Clone, Copy, Debug)]
pub struct Seed(SeedInner);

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
