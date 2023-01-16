// This is a stripped-down version of the `counter` crate, modified to allow
// the use of hash functions other than the default.

use ahash::AHasher;
use std::{
    collections::HashMap,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    ops::{Deref, DerefMut},
};

pub struct Counter<T, S = BuildHasherDefault<AHasher>> {
    map: HashMap<T, usize, S>,
}

impl<T, S> FromIterator<T> for Counter<T, S>
where
    T: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut map = HashMap::default();
        for item in iter {
            let entry = map.entry(item).or_insert(0);
            *entry += 1;
        }
        Self { map }
    }
}

impl<T, S> Deref for Counter<T, S>
where
    T: Hash + Eq,
{
    type Target = HashMap<T, usize, S>;
    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl<T, S> DerefMut for Counter<T, S>
where
    T: Hash + Eq,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

impl<T, S> Counter<T, S>
where
    T: Hash + Eq,
    S: BuildHasher,
{
    pub fn max_count(&self) -> usize {
        self.values().max().copied().unwrap_or(0)
    }

    pub fn count(&self, value: &T) -> usize {
        *self.get(value).unwrap_or(&0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_counter() {
        let list = [1, 2, 3, 1, 1, 3, 1, -1, -1];
        let counter: Counter<_> = list.into_iter().collect();
        assert_eq!(counter.count(&1), 4);
        assert_eq!(counter.count(&2), 1);
        assert_eq!(counter.count(&3), 2);
        assert_eq!(counter.count(&-1), 2);
        assert_eq!(counter.count(&6), 0);
        assert_eq!(counter.len(), 4);
        assert_eq!(counter.max_count(), 4);
    }
}
