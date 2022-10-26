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

pub fn bf_iter(
    key: &Key,
    s: &mut Syndrome,
    e_out: &mut ErrorVector,
    black: &mut [bool; ROW_LENGTH],
    gray: &mut [bool; ROW_LENGTH],
    thr: u32
) {
    let gray_thr = thr - BGF_THRESHOLD;
    let mut flipped_positions = [0u8; ROW_LENGTH];
    for i in 0..BLOCK_LENGTH {
        let mut upc = 0;
        for j in key.h0().support() {
            upc += s.get((i + j as usize) % BLOCK_LENGTH) as u32;
        }
        if upc >= thr {
            e_out.flip(i);
            black[i] = true;
            flipped_positions[i] = 1;
        } else if upc >= gray_thr {
            gray[i] = true;
        }
    }
    for i in 0..BLOCK_LENGTH {
        let mut upc = 0;
        for j in key.h1().support() {
            upc += s.get((i + j as usize) % BLOCK_LENGTH) as u32;
        }
        if upc >= thr {
            e_out.flip(BLOCK_LENGTH + i);
            black[BLOCK_LENGTH + i] = true;
            flipped_positions[BLOCK_LENGTH + i] = 1;
        } else if upc >= gray_thr {
            gray[BLOCK_LENGTH + i] = true;
        }
    }
    // Recompute syndrome according to flipped bits
    for pos in 0..2*BLOCK_LENGTH {
        if flipped_positions[pos] == 1 {
            s.recompute_flipped_bit(key, pos);
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
