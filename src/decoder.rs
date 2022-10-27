use crate::parameters::*;
use crate::keys::Key;
use crate::vectors::ErrorVector;
use crate::syndrome::Syndrome;
use crate::threshold::ThresholdCache;

pub fn bgf_decoder(
    key: &Key,
    s: &mut Syndrome,
    threshold_cache: &mut ThresholdCache
) -> (ErrorVector, bool) {
    let mut e_out = ErrorVector::zero();
    let mut ws = s.hamming_weight();
    let mut black = [[false; BLOCK_LENGTH]; 2];
    let mut gray = [[false; BLOCK_LENGTH]; 2];
    for iter_index in 0..NB_ITER {
        let thr = threshold_cache.get(ws).expect("threshold should not be NaN");
        bf_iter(key, s, &mut e_out, &mut black, &mut gray, thr);
        if iter_index == 0 {
            let masked_thr = (BLOCK_WEIGHT as u32 + 1) / 2 + 1;
            bf_masked_iter(key, s, &mut e_out, black, masked_thr);
            bf_masked_iter(key, s, &mut e_out, gray, masked_thr);
        }
        ws = s.hamming_weight();
        if ws == 0 {
            break;
        }
    }
    (e_out, ws == 0)
}

pub fn unsatisfied_parity_checks(key: &Key, s: &mut Syndrome) -> [[u8; DOUBLE_SIZE_AVX]; 2] {
    s.duplicate_up_to(BLOCK_LENGTH);
    let h_supp = [key.h0().support(), key.h1().support()];
    let mut upc = [[0u8; DOUBLE_SIZE_AVX]; 2];
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            multiply_avx2(&mut upc[0], h_supp[0], s.contents(), BLOCK_WEIGHT, SIZE_AVX);
            multiply_avx2(&mut upc[1], h_supp[1], s.contents(), BLOCK_WEIGHT, SIZE_AVX);
            return upc;
        }
    }
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            let upc_k_i = &mut upc[k][i];
            for &j in h_supp[k] {
                let mut idx = i.wrapping_add(j as usize);
                if idx >= BLOCK_LENGTH {
                    idx = idx.wrapping_sub(BLOCK_LENGTH);
                }
                // idx is (i + j) % BLOCK_LENGTH
                *upc_k_i = upc_k_i.wrapping_add(s.get(idx));
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
    thr: u32
) {
    let upc = unsatisfied_parity_checks(key, s);
    let gray_thr = thr - BGF_THRESHOLD;
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            if upc[k][i] as u32 >= thr {
                e_out.flip(i + k*BLOCK_LENGTH);
                s.recompute_flipped_bit(key, k, i);
                black[k][i] = true;
            } else if upc[0][i] as u32 >= gray_thr {
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
    thr: u32
) {
    let upc = unsatisfied_parity_checks(key, s);
    let mut flipped_positions = [[0u8; BLOCK_LENGTH]; 2];
    for k in 0..2 {
        for i in 0..BLOCK_LENGTH {
            if mask[k][i] && upc[k][i] as u32 >= thr {
                e_out.flip(i + k*BLOCK_LENGTH);
                flipped_positions[k][i] = 1;
            }
        }
    }
    // Recompute syndrome according to flipped bits
    for k in 0..2 {
        for pos in 0..BLOCK_LENGTH {
            if flipped_positions[k][pos] == 1 {
                s.recompute_flipped_bit(key, k, pos);
            }
        }
    }
}


// Here be dragons...

// Adapted from Valentin Vasseur's QC-MDPC decoder implementation
// Multiplies a sparse vector of the given weight by a dense vector of the given length.
// Stores result in the provided output buffer.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
fn multiply_avx2(
    output: &mut [u8],
    sparse: &[u32],
    dense: &[u8],
    block_weight: usize,
    block_length: usize
) {
    use safe_arch::{m256i, zeroed_m256i, add_i8_m256i};
    const AVX_BUFF_LEN: usize = 8;
    // initialize buffer array of 256-bit integers
    let mut buffer = [zeroed_m256i(); AVX_BUFF_LEN];
    for i in (0 .. block_length / 32).step_by(AVX_BUFF_LEN) {
        // reset buffer to zero
        for k in 0..AVX_BUFF_LEN {
            buffer[k] = zeroed_m256i();
        }
        for j in 0..block_weight {
            // current location in sparse vector
            let offset = sparse[j] as usize + 32*i;
            for k in 0..AVX_BUFF_LEN {
                // add offset block of dense vector to buffer
                let dense_slice = &dense[offset+32*k..offset+32*k+32];
                buffer[k] = add_i8_m256i(
                    buffer[k],
                    m256i::from(*<&[u8; 32]>::try_from(dense_slice).unwrap())
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
