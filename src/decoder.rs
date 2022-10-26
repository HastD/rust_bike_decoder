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
    let mut black = [false; ROW_LENGTH];
    let mut gray = [false; ROW_LENGTH];
    for iter_index in 0..NB_ITER {
        let thr = threshold_cache.get(ws).expect("threshold should not be NaN");
        bf_iter(key, s, &mut e_out, &mut black, &mut gray, thr);
        if iter_index == 0 {
            let thr = (BLOCK_WEIGHT as u32 + 1) / 2;
            bf_masked_iter(key, s, &mut e_out, black, thr);
            bf_masked_iter(key, s, &mut e_out, gray, thr);
        }
        ws = s.hamming_weight();
        if ws == 0 {
            break;
        }
    }
    (e_out, ws == 0)
}

#[inline(never)]
pub fn unsatisfied_parity_checks(key: &Key, s: &Syndrome) -> [u32; ROW_LENGTH] {
    let mut upc = [0; ROW_LENGTH];
    for i in 0..BLOCK_LENGTH {
        let upc_i = &mut upc[i];
        for j in key.h0().support() {
            let mut idx = i.wrapping_add(j as usize);
            if idx >= BLOCK_LENGTH {
                idx = idx.wrapping_sub(BLOCK_LENGTH);
            }
            *upc_i += s.get(idx) as u32;
        }
    }
    for i in 0..BLOCK_LENGTH {
        let upc_r_plus_i = &mut upc[BLOCK_LENGTH + i];
        for j in key.h1().support() {
            let mut idx = i.wrapping_add(j as usize);
            if idx >= BLOCK_LENGTH {
                idx = idx.wrapping_sub(BLOCK_LENGTH);
            }
            *upc_r_plus_i += s.get(idx) as u32;
        }
    }
    upc
}

pub fn bf_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    black: &mut [bool; ROW_LENGTH],
    gray: &mut [bool; ROW_LENGTH],
    thr: u32
) {
    let upc = unsatisfied_parity_checks(key, &*s);
    let gray_thr = thr - BGF_THRESHOLD;
    for i in 0..ROW_LENGTH {
        if upc[i] >= thr {
            e_out.flip(i);
            s.recompute_flipped_bit(key, i);
            black[i] = true;
        } else if upc[i] >= gray_thr {
            gray[i] = true;
        }
    }
}

pub fn bf_masked_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    mask: [bool; ROW_LENGTH],
    thr: u32
) {
    let mut flipped_positions = [0u8; ROW_LENGTH];
    for i in 0..BLOCK_LENGTH {
        if mask[i] {
            let mut upc = 0;
            for j in key.h0().support() {
                upc += s.get((i + j as usize) % BLOCK_LENGTH) as u32;
            }
            if upc >= thr {
                e_out.flip(i);
                flipped_positions[i] = 1;
            }
        }
    }
    for i in BLOCK_LENGTH..ROW_LENGTH {
        if mask[i] {
            let mut upc = 0;
            for j in key.h1().support() {
                upc += s.get((i + j as usize) % BLOCK_LENGTH) as u32;
            }
            if upc >= thr {
                e_out.flip(i);
                flipped_positions[i] = 1;
            }
        }
    }
    // Recompute syndrome according to flipped bits
    for pos in 0..ROW_LENGTH {
        if flipped_positions[pos] == 1 {
            s.recompute_flipped_bit(key, pos);
        }
    }
}
