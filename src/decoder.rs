use crate::parameters::*;
use crate::keys::Key;
use crate::vectors::ErrorVector;
use crate::syndrome::Syndrome;
use crate::threshold::THRESHOLD_CACHE;

pub fn bgf_decoder(key: &Key, s: &mut Syndrome) -> (ErrorVector, bool) {
    let mut e_out = ErrorVector::zero();
    let mut ws = s.hamming_weight();
    let mut black = [[false; BLOCK_LENGTH]; 2];
    let mut gray = [[false; BLOCK_LENGTH]; 2];
    for iter_index in 0..NB_ITER {
        let thr = THRESHOLD_CACHE[ws];
        bf_iter(key, s, &mut e_out, &mut black, &mut gray, thr);
        if iter_index == 0 {
            bf_masked_iter(key, s, &mut e_out, black, BF_MASKED_THRESHOLD);
            bf_masked_iter(key, s, &mut e_out, gray, BF_MASKED_THRESHOLD);
        }
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
            fn truncate_buffer(buf: [u8; DOUBLE_SIZE_AVX]) -> [u8; BLOCK_LENGTH] {
                <[u8; BLOCK_LENGTH]>::try_from(&buf[..BLOCK_LENGTH]).unwrap()
            }
            let mut upc = [[0u8; DOUBLE_SIZE_AVX]; 2];
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
                upc[k][i] += s.get(i + j as usize);
            }
        }
    }
    upc
}

// for some reason allowing the compiler to inline this function slows things down a lot
#[inline(never)]
pub fn bf_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    black: &mut [[bool; BLOCK_LENGTH]; 2],
    gray: &mut [[bool; BLOCK_LENGTH]; 2],
    thr: u8
) {
    let upc = unsatisfied_parity_checks(key, s);
    let gray_thr = thr - GRAY_THRESHOLD_DIFF;
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            if upc[k][i] >= thr {
                e_out.flip(i + k*BLOCK_LENGTH);
                s.recompute_flipped_bit(key, k, i);
                black[k][i] = true;
            } else if upc[k][i] >= gray_thr {
                gray[k][i] = true;
            }
        }
    }
}

pub fn bf_masked_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    mask: [[bool; BLOCK_LENGTH]; 2],
    thr: u8
) {
    let upc = unsatisfied_parity_checks(key, s);
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            if mask[k][i] && upc[k][i] >= thr {
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
    dense: &[u8],
    block_length: usize
) {
    use safe_arch::{zeroed_m256i, add_i8_m256i};
    const AVX_BUFF_LEN: usize = 8;
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
            let mut syn = Syndrome::from([1; BLOCK_LENGTH]);
            let upc = unsatisfied_parity_checks(&key, &mut syn);
            for k in 0..2 {
                assert_eq!(&upc[k][..BLOCK_LENGTH], &[BLOCK_WEIGHT as u8; BLOCK_LENGTH]);
            }
        }
    }
}
