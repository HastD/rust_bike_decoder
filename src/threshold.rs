use crate::parameters::BF_THRESHOLD_MIN;
use num::{BigInt, BigRational, ToPrimitive};
use num_integer::binomial;
use rustc_hash::FxHashMap;
use std::cmp;

fn big_binomial(n: u32, k: u32) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

pub fn exact_threshold(ws: u32, r: u32, d: u32, t: u32) -> Option<u32> {
    if ws == 0 {
        return Some(1);
    }
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

    let log_frac = ((1.0 - pi0) / (1.0 - pi1)).log2();
    let thresh_num = (((n - t) / t) as f64).log2() + d as f64 * log_frac;
    let thresh_den = (pi1 / pi0).log2() + log_frac;
    let threshold = (thresh_num / thresh_den).ceil();
    if threshold.is_nan() { None } else {
        // modification to threshold according to Vasseur's thesis, section 6.1.3.1
        let threshold = cmp::max(threshold as u32, BF_THRESHOLD_MIN as u32);
        Some(threshold)
    }
}

pub struct ThresholdCache {
    cache: FxHashMap<u32, Option<u32>>,
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

    pub fn get(&mut self, ws: u32) -> Option<u32> {
        *self.cache.entry(ws).or_insert_with(|| exact_threshold(ws, self.r, self.d, self.t))
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
