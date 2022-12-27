use crate::{
    keys::Key,
    ncw::TaggedErrorVector,
    parameters::*,
    syndrome::Syndrome,
    threshold::THRESHOLD_CACHE,
    vectors::ErrorVector,
};
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct DecodingResult {
    key: Key,
    vector: TaggedErrorVector,
    success: bool,
}

impl DecodingResult {
    pub fn from(key: Key, vector: TaggedErrorVector) -> Self {
        let e_supp = vector.vector();
        let e_in = e_supp.dense();
        let mut syn = Syndrome::from_sparse(&key, vector.vector());
        let (e_out, same_syndrome) = bgf_decoder(&key, &mut syn);
        let success = e_in == e_out;
        assert!(same_syndrome || !success);
        Self { key, vector, success }
    }

    #[inline]
    pub fn key(&self) -> &Key {
        &self.key
    }

    #[inline]
    pub fn vector(&self) -> &TaggedErrorVector {
        &self.vector
    }

    #[inline]
    pub fn success(&self) -> bool {
        self.success
    }

    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, Debug)]
pub struct DecodingFailure {
    key: Key,
    vector: TaggedErrorVector,
}

impl TryFrom<DecodingResult> for DecodingFailure {
    type Error = NotFailureError;

    fn try_from(result: DecodingResult) -> Result<Self, NotFailureError> {
        if !result.success() {
            let (key, vector) = result.take_key_vector();
            Ok(Self { key, vector })
        } else {
            Err(NotFailureError)
        }
    }
}

impl DecodingFailure {
    #[inline]
    pub fn key(&self) -> &Key {
        &self.key
    }

    #[inline]
    pub fn vector(&self) -> &TaggedErrorVector {
        &self.vector
    }

    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("not a decoding failure, so can't convert to DecodingFailure")]
pub struct NotFailureError;

pub fn bgf_decoder(key: &Key, s: &mut Syndrome) -> (ErrorVector, bool) {
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

pub fn unsatisfied_parity_checks(key: &Key, s: &mut Syndrome) -> [[u8; BLOCK_LENGTH]; 2] {
    // Duplicate the syndrome to precompute cyclic shifts and avoid modulo operations
    s.duplicate_up_to(BLOCK_LENGTH);
    let h_supp = [key.h0().support(), key.h1().support()];
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            #[inline]
            fn truncate_buffer(buf: [u8; 2*SIZE_AVX]) -> [u8; BLOCK_LENGTH] {
                (&buf[..BLOCK_LENGTH]).try_into().unwrap()
            }
            let mut upc = [[0u8; 2*SIZE_AVX]; 2];
            multiply_avx2(&mut upc[0], h_supp[0], s.contents_with_buffer(), SIZE_AVX);
            multiply_avx2(&mut upc[1], h_supp[1], s.contents_with_buffer(), SIZE_AVX);
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
    thr: u8
) -> ([Vec<usize>; 2], [Vec<usize>; 2]) {
    let upc = unsatisfied_parity_checks(key, s);
    let gray_thr = thr - GRAY_THRESHOLD_DIFF;
    let mut black = [Vec::with_capacity(BLOCK_LENGTH), Vec::with_capacity(BLOCK_LENGTH)];
    let mut gray = [Vec::with_capacity(BLOCK_LENGTH), Vec::with_capacity(BLOCK_LENGTH)];
    for (k, upc_k) in upc.iter().enumerate() {
        for (i, upc_ki) in upc_k.iter().enumerate()
            .filter(|&(_, upc_ki)| *upc_ki >= gray_thr)
        {
            if *upc_ki >= thr {
                e_out.flip(i + k*BLOCK_LENGTH);
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
pub fn bf_iter_no_mask(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    thr: u8
) {
    let upc = unsatisfied_parity_checks(key, s);
    for (k, upc_k) in upc.iter().enumerate() {
        for (i, _) in upc_k.iter().enumerate()
            .filter(|&(_, upc_ki)| *upc_ki >= thr)
        {
            e_out.flip(i + k*BLOCK_LENGTH);
            s.recompute_flipped_bit(key, k, i);
        }
    }
}

pub fn bf_masked_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    mask: [Vec<usize>; 2],
    thr: u8
) {
    let upc = unsatisfied_parity_checks(key, s);
    for k in 0..2 {
        for &i in mask[k].iter() {
            if upc[k][i] >= thr {
                e_out.flip(i + k*BLOCK_LENGTH);
                s.recompute_flipped_bit(key, k, i);
            }
        }
    }
}

// Adapted from Valentin Vasseur's QC-MDPC decoder implementation
// Multiplies a sparse vector of the given weight by a dense vector of the given length.
// The dense vector should be duplicated in memory to precompute cyclic shifts.
// Stores result in the provided output buffer.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
fn multiply_avx2(
    output: &mut [u8],
    sparse: &[u32],
    dense: &[bool],
    block_length: usize
) {
    use safe_arch::{zeroed_m256i, add_i8_m256i};
    const AVX_BUFF_LEN: usize = 8;
    let dense = bytemuck::cast_slice::<bool, u8>(dense);
    // initialize buffer array of 256-bit integers
    let mut buffer = [zeroed_m256i(); AVX_BUFF_LEN];
    for i in (0 .. block_length / 32).step_by(AVX_BUFF_LEN) {
        // reset buffer to zero
        buffer.iter_mut().for_each(|x| *x = zeroed_m256i());
        for offset in sparse.iter().map(|idx| *idx as usize + 32*i) {
            for k in 0..AVX_BUFF_LEN {
                // add offset block of dense vector to buffer
                let dense_slice = &dense[offset+32*k..offset+32*k+32];
                buffer[k] = add_i8_m256i(
                    buffer[k],
                    <[u8; 32]>::try_from(dense_slice).unwrap().into()
                );
            }
        }
        for k in 0..AVX_BUFF_LEN {
            // copy buffer contents to appropriate address in output vector
            let output_slice = &mut output[32*(i+k)..32*(i+k)+32];
            output_slice.copy_from_slice(&<[u8; 32]>::from(buffer[k]));
        }
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
}
