use crate::parameters::BF_THRESHOLD_MIN;
use num::{BigInt, BigRational, ToPrimitive};
use num_integer::binomial;
use rustc_hash::FxHashMap;
use std::cmp;

fn big_binomial(n: u32, k: u32) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

fn threshold_constants(ws: u32, r: u32, d: u32, t: u32) -> (f64, f64) {
    let n = 2*r;
    let w = 2*d;
    let n_minus_w = n - w;
    let mut x_part = BigInt::from(0);
    for l in (3..cmp::min(t, w)).step_by(2) {
        x_part += (l - 1) * big_binomial(w, l) * big_binomial(n_minus_w, t - l);
    }
    let x = BigRational::new(r * x_part, big_binomial(n, t)).to_f64();
    let x = x.expect("Threshold computation should not overflow");
    let pi1 = (ws as f64 + x) / (t * d) as f64;
    let pi0 = ((w * ws) as f64 - x) / ((n - t) * d) as f64;
    (pi0, pi1)
}

pub fn exact_threshold_ineq(ws: u32, r: u32, d: u32, t: u32) -> Option<u8> {
    let bf_threshold_min = <u8>::try_from(BF_THRESHOLD_MIN).expect("Weight >= 509 not supported");
    if ws == 0 {
        return Some(bf_threshold_min);
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t);
    let mut threshold: i32 = 1;
    let d = d as i32;
    while threshold <= d && t as f64 * pi1.powi(threshold) * (1.0 - pi1).powi(d - threshold)
                    < (n - t) as f64 * pi0.powi(threshold) * (1.0 - pi0).powi(d - threshold) {
        if threshold < u8::MAX as i32 {
            threshold = threshold.wrapping_add(1);
        } else {
            return None;
        }
    }
    let threshold = cmp::max(threshold as u8, bf_threshold_min);
    Some(threshold)
}

pub fn exact_threshold(ws: u32, r: u32, d: u32, t: u32) -> Option<u8> {
    if ws == 0 {
        return Some(1);
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t);

    let log_frac = ((1.0 - pi0) / (1.0 - pi1)).log2();
    let thresh_num = (((n - t) / t) as f64).log2() + d as f64 * log_frac;
    let thresh_den = (pi1 / pi0).log2() + log_frac;
    let threshold = (thresh_num / thresh_den).ceil();
    if threshold.is_nan() { None } else {
        let threshold = <u8>::try_from(threshold as u32).expect("Thresholds >= 256 not supported");
        let bf_threshold_min = <u8>::try_from(BF_THRESHOLD_MIN).expect("Weight >= 509 not supported");
        // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
        let threshold = cmp::max(threshold, bf_threshold_min);
        Some(threshold)
    }
}

#[derive(Debug)]
pub struct ThresholdCache {
    cache: FxHashMap<u32, Option<u8>>,
    pub r: u32,
    pub d: u32,
    pub t: u32
}

impl ThresholdCache {
    pub fn with_parameters(r: u32, d: u32, t: u32) -> Self {
        Self {
            cache: FxHashMap::default(),
            r, d, t
        }
    }

    pub fn get(&mut self, ws: u32) -> Option<u8> {
        *self.cache.entry(ws).or_insert_with(|| exact_threshold_ineq(ws, self.r, self.d, self.t))
    }

    pub fn is_computed(&self, ws: &u32) -> bool {
        self.cache.contains_key(ws)
    }

    pub fn precompute_all(&mut self) {
        for ws in 0..self.r {
            self.get(ws);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_thresholds() {
        let (r, d, t) = (587, 15, 18);
        let bf_threshold_min = <u8>::try_from(BF_THRESHOLD_MIN).unwrap();
        let thresholds_no_min = [
            1,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,
            6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
            8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,
            9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,
            10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
            11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,13,13,13,13,13,
            13,13,13,13,13,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,15,15,15,15,15,15,15,13,13,13,13,13,13,13,13,13,13,13,13,13,
            13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,
            11,11,11,11,11,11,11,11,11,11,11,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,
            7,7,7,5,5,5,5,5,5,5,5,5,3,3,3,3,3,3,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
        ];
        for ws in 0..r {
            let thr = exact_threshold_ineq(ws, r, d, t).unwrap();
            assert_eq!(thr, cmp::max(thresholds_no_min[ws as usize], bf_threshold_min));
        }
    }
}
