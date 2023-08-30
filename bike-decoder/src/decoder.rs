use crate::{
    keys::Key,
    ncw::TaggedErrorVector,
    parameters::*,
    syndrome::Syndrome,
    threshold::{bf_masked_threshold, build_threshold_cache},
    vectors::{ErrorVector, Index, SparseErrorVector},
};
use getset::{CopyGetters, Getters};
use once_cell::sync::Lazy;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

static THRESHOLD_CACHE: Lazy<Vec<u8>> = Lazy::new(|| {
    build_threshold_cache(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT)
        .expect("Must be able to initialize threshold cache")
});

#[derive(Clone, CopyGetters, Debug, Getters, Serialize, Deserialize)]
pub struct DecodingResult {
    #[getset(get = "pub")]
    key: Key,
    #[getset(get = "pub")]
    vector: TaggedErrorVector,
    #[getset(get_copy = "pub")]
    success: bool,
}

impl DecodingResult {
    pub fn from_key_vector(key: Key, vector: TaggedErrorVector) -> Self {
        let e_supp = vector.vector();
        let e_in = e_supp.dense();
        let mut syn = Syndrome::from_sparse(&key, e_supp);
        let (e_out, same_syndrome) = bgf_decoder(&key, &mut syn);
        let success = e_in == e_out;
        assert!(same_syndrome || !success);
        Self {
            key,
            vector,
            success,
        }
    }

    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, Debug, Getters, Serialize, Deserialize)]
pub struct DecodingFailure {
    #[serde(flatten)]
    #[getset(get = "pub")]
    key: Key,
    #[serde(flatten)]
    #[getset(get = "pub")]
    vector: TaggedErrorVector,
    pub thread: Option<u32>,
}

impl From<DecodingFailure> for DecodingResult {
    fn from(df: DecodingFailure) -> Self {
        let DecodingFailure { key, vector, .. } = df;
        Self {
            key,
            vector,
            success: false,
        }
    }
}

impl TryFrom<DecodingResult> for DecodingFailure {
    type Error = NotFailureError;

    fn try_from(result: DecodingResult) -> Result<Self, NotFailureError> {
        if !result.success() {
            let DecodingResult { key, vector, .. } = result;
            Ok(Self {
                key: key.sorted(),
                vector: vector.sorted(),
                thread: None,
            })
        } else {
            Err(NotFailureError)
        }
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("not a decoding failure, so can't convert to DecodingFailure")]
pub struct NotFailureError;

impl DecodingFailure {
    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, CopyGetters, Debug, Getters, Serialize, Deserialize, PartialEq, Eq)]
pub struct DecoderCycle {
    #[getset(get = "pub")]
    key: Key,
    #[getset(get = "pub")]
    e_in: SparseErrorVector,
    #[getset(get = "pub")]
    e_out: Vec<Index>,
    #[getset(get_copy = "pub")]
    cycle: Option<CycleData>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CycleData {
    pub start: usize,
    pub length: usize,
    pub weight: usize,
    pub syndrome_weight: usize,
    pub threshold: u8,
    pub max_upc: u8,
}

impl DecoderCycle {
    pub fn diff(&self) -> Vec<Index> {
        let mut diff = self.e_out.clone();
        for entry in self.e_in.support() {
            if let Some(index) = diff.iter().position(|item| *item == *entry) {
                diff.swap_remove(index);
            } else {
                diff.push(*entry);
            }
        }
        diff.sort_unstable();
        diff
    }
}

impl From<DecoderCycle> for DecodingResult {
    fn from(cycle: DecoderCycle) -> Self {
        let success = cycle.diff().is_empty();
        Self {
            key: cycle.key,
            vector: cycle.e_in.into(),
            success,
        }
    }
}

pub fn bgf_decoder(key: &Key, s: &mut Syndrome) -> (ErrorVector, bool) {
    const BF_MASKED_THRESHOLD: u8 = bf_masked_threshold(BLOCK_WEIGHT);
    let mut e_out = ErrorVector::zero();
    let mut ws = s.hamming_weight();
    // Iteration 0
    let thr = THRESHOLD_CACHE[ws];
    let (black, gray) = bf_iter(key, s, &mut e_out, thr);
    bf_masked_iter(key, s, &mut e_out, black, BF_MASKED_THRESHOLD);
    bf_masked_iter(key, s, &mut e_out, gray, BF_MASKED_THRESHOLD);
    ws = s.hamming_weight();
    if ws == 0 {
        return (e_out, true);
    }
    for _ in 1..NB_ITER {
        let thr = THRESHOLD_CACHE[ws];
        bf_iter_no_mask(key, s, &mut e_out, thr);
        ws = s.hamming_weight();
        if ws == 0 {
            break;
        }
    }
    (e_out, ws == 0)
}

/// Runs step-by-step decoder (Algorithm 7.1 in Vasseur's thesis) on key `(h0, h1)` and syndrome
/// `s` for up to `max_steps` iterations. Returns the resulting error vector and the number of
/// iterations actually carried out.
pub fn step_by_step_bitflip<R>(
    key: &Key,
    s: &mut Syndrome,
    max_steps: usize,
    rng: &mut R,
) -> (ErrorVector, usize)
where
    R: Rng + ?Sized,
{
    let h_supp = [key.h0().support(), key.h1().support()];
    let mut e_out = ErrorVector::zero();
    let mut ws = s.hamming_weight();
    s.duplicate_contents();
    for step in 0..max_steps {
        let k = rng.gen_range(0..2);
        let j = rng.gen_range(0..BLOCK_LENGTH);
        let upc = h_supp[k]
            .iter()
            // If i + j >= BLOCK_LENGTH, this wraps around because we duplicated s
            .map(|i| u8::from(s.get(*i as usize + j)))
            .sum::<u8>();
        if upc >= THRESHOLD_CACHE[ws] {
            e_out.flip(j + k * BLOCK_LENGTH);
            s.recompute_flipped_bit(key, k, j);
            ws = s.hamming_weight();
            if ws == 0 {
                return (e_out, step + 1);
            }
            s.duplicate_contents();
        }
    }
    (e_out, max_steps)
}

pub fn find_bgf_cycle(key: &Key, e_in: &SparseErrorVector, max_iters: usize) -> DecoderCycle {
    const BF_MASKED_THRESHOLD: u8 = bf_masked_threshold(BLOCK_WEIGHT);
    let mut s = Syndrome::from_sparse(key, e_in);
    let mut e_out = ErrorVector::zero();
    let thr = THRESHOLD_CACHE[s.hamming_weight()];
    let (black, gray) = bf_iter(key, &mut s, &mut e_out, thr);
    bf_masked_iter(key, &mut s, &mut e_out, black, BF_MASKED_THRESHOLD);
    bf_masked_iter(key, &mut s, &mut e_out, gray, BF_MASKED_THRESHOLD);
    let mut e_out_cache = vec![e_out.support()];
    for current_iter in 1..max_iters {
        let thr = THRESHOLD_CACHE[s.hamming_weight()];
        bf_iter_no_mask(key, &mut s, &mut e_out, thr);
        let e_out_supp = e_out.support();
        if let Some(start_iter) = e_out_cache.iter().position(|x| x == &e_out_supp) {
            // diff = support of e_in - e_out
            let diff: Vec<Index> = e_out
                .contents()
                .iter()
                .zip(e_in.dense().contents())
                .enumerate()
                .filter_map(|(idx, (&a, &b))| (a ^ b).then_some(idx as Index))
                .collect();
            let weight = diff.len();
            let syndrome_weight = s.hamming_weight();
            let max_upc = unsatisfied_parity_checks(key, &mut s)
                .into_iter()
                .flatten()
                .max()
                .unwrap_or(0);
            return DecoderCycle {
                key: key.clone(),
                e_in: e_in.clone(),
                e_out: e_out_supp,
                cycle: Some(CycleData {
                    start: start_iter,
                    length: current_iter.abs_diff(start_iter),
                    weight,
                    syndrome_weight,
                    threshold: THRESHOLD_CACHE[syndrome_weight],
                    max_upc,
                }),
            };
        } else {
            e_out_cache.push(e_out_supp);
        }
    }
    DecoderCycle {
        key: key.clone(),
        e_in: e_in.clone(),
        e_out: e_out.support(),
        cycle: None,
    }
}

pub fn unsatisfied_parity_checks(key: &Key, s: &mut Syndrome) -> [[u8; BLOCK_LENGTH]; 2] {
    // Duplicate the syndrome to precompute cyclic shifts and avoid modulo operations
    s.duplicate_contents();
    let h_supp = [key.h0().support(), key.h1().support()];
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[inline]
            fn truncate_buffer(buf: [u8; SIZE_AVX]) -> [u8; BLOCK_LENGTH] {
                (&buf[..BLOCK_LENGTH])
                    .try_into()
                    .expect("Must ensure BLOCK_LENGTH <= SIZE_AVX")
            }
            let mut upc = [[0u8; SIZE_AVX]; 2];
            multiply_avx2(&mut upc[0], h_supp[0], s.contents_with_buffer());
            multiply_avx2(&mut upc[1], h_supp[1], s.contents_with_buffer());
            return [truncate_buffer(upc[0]), truncate_buffer(upc[1])];
        }
    }
    let mut upc = [[0u8; BLOCK_LENGTH]; 2];
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            for &j in h_supp[k] {
                // If i + j >= BLOCK_LENGTH, this wraps around because we duplicated s
                upc[k][i] += u8::from(s.get(i + j as usize));
            }
        }
    }
    upc
}

// the compiler seems to make some bad optimization choices if allowed to inline this
#[inline(never)]
pub fn bf_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    thr: u8,
) -> ([Vec<usize>; 2], [Vec<usize>; 2]) {
    let upc = unsatisfied_parity_checks(key, s);
    let gray_thr = thr - GRAY_THRESHOLD_DIFF;
    let mut black = [
        Vec::with_capacity(BLOCK_LENGTH),
        Vec::with_capacity(BLOCK_LENGTH),
    ];
    let mut gray = [
        Vec::with_capacity(BLOCK_LENGTH),
        Vec::with_capacity(BLOCK_LENGTH),
    ];
    for (k, upc_k) in upc.iter().enumerate() {
        for (i, upc_ki) in upc_k
            .iter()
            .enumerate()
            .filter(|&(_, upc_ki)| *upc_ki >= gray_thr)
        {
            if *upc_ki >= thr {
                e_out.flip(i + k * BLOCK_LENGTH);
                s.recompute_flipped_bit(key, k, i);
                black[k].push(i);
            } else {
                gray[k].push(i);
            }
        }
    }
    (black, gray)
}

#[inline(never)]
pub fn bf_iter_no_mask(key: &Key, s: &mut Syndrome, e_out: &mut ErrorVector, thr: u8) {
    let upc = unsatisfied_parity_checks(key, s);
    for (k, upc_k) in upc.iter().enumerate() {
        for (i, _) in upc_k
            .iter()
            .enumerate()
            .filter(|&(_, upc_ki)| *upc_ki >= thr)
        {
            e_out.flip(i + k * BLOCK_LENGTH);
            s.recompute_flipped_bit(key, k, i);
        }
    }
}

pub fn bf_masked_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    mask: [Vec<usize>; 2],
    thr: u8,
) {
    let upc = unsatisfied_parity_checks(key, s);
    for k in 0..2 {
        for &i in mask[k].iter() {
            if upc[k][i] >= thr {
                e_out.flip(i + k * BLOCK_LENGTH);
                s.recompute_flipped_bit(key, k, i);
            }
        }
    }
}

/// Multiplies a sparse vector by a dense vector. The dense vector should be
/// duplicated in memory to precompute cyclic shifts. The results are stored in
/// the provided output buffer.
/// Adapted from Valentin Vasseur's QC-MDPC decoder implementation.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
fn multiply_avx2(output: &mut [u8], sparse: &[Index], dense: &[bool]) {
    use safe_arch::{add_i8_m256i, zeroed_m256i};
    const AVX_BUFF_LEN: usize = 8;
    let dense: &[u8] = bytemuck::cast_slice(dense);
    let block_length = dense.len() / 2;
    // assertions to eliminate the need for bounds checking in the inner loop
    assert_eq!(dense.len() % (64 * AVX_BUFF_LEN), 0);
    assert!(output.len() >= block_length);
    assert!(sparse.iter().all(|idx| (*idx as usize) < block_length));
    // initialize buffer array of 256-bit integers
    let mut buffer = [zeroed_m256i(); AVX_BUFF_LEN];
    for i in (0..block_length / 32).step_by(AVX_BUFF_LEN) {
        // reset buffer to zero
        buffer.iter_mut().for_each(|x| *x = zeroed_m256i());
        for offset in sparse.iter().map(|idx| *idx as usize + 32 * i) {
            // SAFETY: upper bound <= idx + 32*(i + AVX_BUFF_LEN) <= idx + block_length
            // < dense.len() due to the above assertions. Also, 0 <= lower bound < upper bound.
            let dense_slice = unsafe { dense.get_unchecked(offset..offset + 32 * AVX_BUFF_LEN) };
            for k in 0..AVX_BUFF_LEN {
                let addend = bytemuck::pod_read_unaligned(&dense_slice[32 * k..32 * k + 32]);
                // add offset block of dense vector to buffer
                buffer[k] = add_i8_m256i(buffer[k], addend);
            }
        }
        // SAFETY: upper bound = 32*(i + AVX_BUFF_LEN) <= block_length <= output.len()
        // due to the above assertions. Also, 0 <= lower bound < upper bound.
        let output_slice = unsafe { output.get_unchecked_mut(32 * i..32 * i + 32 * AVX_BUFF_LEN) };
        // copy buffer contents to output slice
        output_slice.copy_from_slice(bytemuck::cast_slice(&buffer[..]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIALS: usize = 1000;

    #[test]
    fn upc_all_ones() {
        let mut rng = rand::thread_rng();
        for _ in 0..TRIALS {
            let key = Key::random(&mut rng);
            let mut syn = Syndrome::new([true; BLOCK_LENGTH]);
            for upc in unsatisfied_parity_checks(&key, &mut syn) {
                assert_eq!(&upc[..BLOCK_LENGTH], &[BLOCK_WEIGHT as u8; BLOCK_LENGTH]);
            }
        }
    }

    #[test]
    fn bgf_cycle_example() {
        assert_eq!((BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT), (587, 15, 18));
        let key = Key::from_support(
            [
                93, 99, 105, 121, 126, 141, 156, 193, 194, 197, 264, 301, 360, 400, 429,
            ],
            [
                100, 117, 189, 191, 211, 325, 340, 386, 440, 461, 465, 474, 534, 565, 578,
            ],
        )
        .unwrap();
        let e_in = SparseErrorVector::from_support([
            16, 73, 89, 201, 346, 522, 547, 553, 574, 575, 613, 619, 637, 713, 955, 960, 983, 1008,
        ])
        .unwrap();
        let cycle = find_bgf_cycle(&key, &e_in, 100);
        assert_eq!(cycle.key, key);
        assert_eq!(cycle.e_in, e_in);
        assert_eq!(
            cycle.e_out,
            vec![67, 73, 201, 242, 459, 481, 501, 507, 547, 575, 637, 759, 922, 955, 1008]
        );
        assert_eq!(
            cycle.cycle,
            Some(CycleData {
                start: 25,
                length: 2,
                weight: 19,
                syndrome_weight: 101,
                threshold: 8,
                max_upc: 11,
            })
        );
    }
}
