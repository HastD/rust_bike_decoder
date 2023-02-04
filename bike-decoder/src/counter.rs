/// A vector-based counter for collections of type convertible to usize.
/// Typically faster than hash-based solutions when the values range over a
/// relatively small set of nonnegative integers. Very inefficient for larger
/// values: uses O(N) memory, where N is the maximum value in the collection,
/// regardless of the size of the collection.
pub struct IndexCounter {
    inner: Vec<usize>,
}

impl<T> FromIterator<T> for IndexCounter
where
    usize: TryFrom<T>,
    <usize as TryFrom<T>>::Error: std::fmt::Debug,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut inner = vec![0; iter.size_hint().0];
        for item in iter {
            let idx = usize::try_from(item).expect("items must be convertible to usize");
            if idx >= inner.len() {
                inner.resize(idx + 1, 0);
            }
            inner[idx] += 1;
        }
        Self { inner }
    }
}

impl std::ops::Deref for IndexCounter {
    type Target = Vec<usize>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl IndexCounter {
    pub fn max_count(&self) -> usize {
        self.inner.iter().max().copied().unwrap_or(0)
    }

    pub fn count<T>(&self, value: T) -> usize
    where
        usize: TryFrom<T>,
    {
        if let Ok(idx) = usize::try_from(value) {
            *self.inner.get(idx).unwrap_or(&0)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_counter() {
        let list = [1, 2, 3, 1, 1, 3, 1, 11, 11];
        let counter: IndexCounter = list.into_iter().collect();
        assert_eq!(counter.count(1), 4);
        assert_eq!(counter.count(2), 1);
        assert_eq!(counter.count(3), 2);
        assert_eq!(counter.count(11), 2);
        assert_eq!(counter.count(6), 0);
        assert_eq!(counter.max_count(), 4);
    }

    #[test]
    #[should_panic]
    fn failed_counter() {
        let list = [1, 2, -5];
        let _: IndexCounter = list.into_iter().collect();
    }
}
