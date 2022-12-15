use crate::parameters::BF_THRESHOLD_MIN;
use num::{BigInt, BigRational, ToPrimitive};
use num_integer::binomial;
use rustc_hash::FxHashMap;
use std::cmp;

fn big_binomial(n: usize, k: usize) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

fn compute_x(r: usize, d: usize, t: usize) -> f64 {
    let n = 2*r;
    let w = 2*d;
    let n_minus_w = n - w;
    let mut x_part = BigInt::from(0);
    for l in (3..cmp::min(t, w)).step_by(2) {
        x_part += (l - 1) * big_binomial(w, l) * big_binomial(n_minus_w, t - l);
    }
    let x = BigRational::new(r * x_part, big_binomial(n, t)).to_f64();
    x.expect("Threshold computation should not overflow")
}

fn threshold_constants(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>) -> (f64, f64) {
    let n = 2*r;
    let w = 2*d;
    let x = x.unwrap_or_else(|| compute_x(r, d, t));
    let pi1 = (ws as f64 + x) / (t * d) as f64;
    let pi0 = ((w as usize * ws) as f64 - x) / ((n - t) * d) as f64;
    (pi0, pi1)
}

pub fn exact_threshold_ineq(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>) -> Result<u8, &'static str> {
    if ws == 0 {
        return Ok(BF_THRESHOLD_MIN);
    } else if ws > r as usize {
        return Err("Syndrome weight cannot be greater than block length");
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x);
    let mut threshold: i32 = 1;
    let d = d as i32;
    while threshold <= d && t as f64 * pi1.powi(threshold) * (1.0 - pi1).powi(d - threshold)
                    < (n - t) as f64 * pi0.powi(threshold) * (1.0 - pi0).powi(d - threshold) {
        if threshold < u8::MAX as i32 {
            threshold = threshold.wrapping_add(1);
        } else {
            return Err("Threshold exceeds maximum supported value");
        }
    }
    let threshold = cmp::max(threshold as u8, BF_THRESHOLD_MIN);
    Ok(threshold)
}

pub fn exact_threshold(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>) -> Result<u8, &'static str> {
    if ws == 0 {
        return Ok(BF_THRESHOLD_MIN);
    } else if ws > r as usize {
        return Err("Syndrome weight cannot be greater than block length");
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x);

    let log_frac = ((1.0 - pi0) / (1.0 - pi1)).log2();
    let thresh_num = (((n - t) / t) as f64).log2() + d as f64 * log_frac;
    let thresh_den = (pi1 / pi0).log2() + log_frac;
    let threshold = (thresh_num / thresh_den).ceil();
    if threshold.is_nan() { Err("Invalid threshold (NaN) computed") } else {
        let threshold = <u8>::try_from(threshold as u32)
            .or(Err("Threshold exceeds maximum supported value"))?;
        // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
        let threshold = cmp::max(threshold, BF_THRESHOLD_MIN);
        Ok(threshold)
    }
}

#[derive(Debug)]
pub struct ThresholdCache {
    cache: FxHashMap<usize, Result<u8, &'static str>>,
    pub r: usize,
    pub d: usize,
    pub t: usize,
    x: Option<f64>,
}

impl ThresholdCache {
    pub fn with_parameters(r: usize, d: usize, t: usize) -> Self {
        assert!(d < r && t < 2*r);
        Self {
            cache: FxHashMap::default(),
            r, d, t,
            x: None,
        }
    }

    pub fn threshold(&mut self, ws: usize) -> Result<u8, &'static str> {
        self.x.get_or_insert_with(|| compute_x(self.r, self.d, self.t));
        *self.cache.entry(ws).or_insert_with(|| exact_threshold_ineq(ws, self.r, self.d, self.t, self.x))
    }

    pub fn is_computed(&self, ws: &usize) -> bool {
        self.cache.contains_key(ws)
    }

    pub fn precompute_all(&mut self) -> Result<(), &'static str> {
        for ws in 0..self.r {
            self.threshold(ws)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieve_cached() {
        let (r, d, t) = (587, 15, 18);
        let mut cache = ThresholdCache::with_parameters(r, d, t);
        cache.cache.insert(42, Ok(127));
        assert_eq!(cache.threshold(42).unwrap(), 127);
    }

    #[test]
    fn known_x() {
        let (r, d, t) = (587, 15, 18);
        let x_known: f64 = 10.2859814049302;
        let x_computed = compute_x(r, d, t);
        assert!((x_known - x_computed).abs() < 1e-9);
    }

    #[test]
    fn known_thresholds() {
        let (r, d, t) = (587, 15, 18);
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
        let mut cache = ThresholdCache::with_parameters(r, d, t);
        for ws in 0..r as usize {
            let thresh = cache.threshold(ws).unwrap();
            assert_eq!(thresh, cmp::max(thresholds_no_min[ws as usize], BF_THRESHOLD_MIN));
        }
    }

    #[test]
    fn invalid_threshold() {
        let (r, d, t) = (587, 15, 18);
        let mut cache = ThresholdCache::with_parameters(r, d, t);
        assert!(cache.threshold(600).is_err());
    }
}
