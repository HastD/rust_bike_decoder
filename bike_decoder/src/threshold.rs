use num::{integer::binomial, BigInt, BigRational, ToPrimitive};
use once_cell::sync::Lazy;
use thiserror::Error;

use crate::parameters::{BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT};

pub static THRESHOLD_CACHE: Lazy<Vec<u8>> = Lazy::new(|| {
    build_threshold_cache(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT)
        .expect("Must be able to initialize threshold cache")
});

pub fn build_threshold_cache(r: usize, d: usize, t: usize) -> Result<Vec<u8>, ThresholdError> {
    let x = compute_x(r, d, t)?;
    let mut threshold_cache: Vec<u8> = Vec::with_capacity(r + 1);
    for ws in 0..=r {
        threshold_cache.push(exact_threshold_ineq(ws, r, d, t, Some(x))?);
    }
    Ok(threshold_cache)
}

pub const fn bf_threshold_min(block_weight: usize) -> u8 {
    assert!(block_weight <= 507, "Block weight > 507 not supported");
    ((block_weight + 1) / 2) as u8
}

pub const fn bf_masked_threshold(block_weight: usize) -> u8 {
    assert!(block_weight <= 507, "Block weight > 507 not supported");
    ((block_weight + 1) / 2 + 1) as u8
}

fn big_binomial(n: usize, k: usize) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

pub fn compute_x(r: usize, d: usize, t: usize) -> Result<f64, ThresholdError> {
    let n = 2 * r;
    let w = 2 * d;
    let n_minus_w = n - w;
    let mut x_part = BigInt::from(0);
    for l in (3..t.min(w)).step_by(2) {
        x_part += (l - 1) * big_binomial(w, l) * big_binomial(n_minus_w, t - l);
    }
    let x = BigRational::new(r * x_part, big_binomial(n, t)).to_f64();
    let err = ThresholdError::XError;
    x.ok_or(err)
        .and_then(|x| if x.is_finite() { Ok(x) } else { Err(err) })
}

fn threshold_constants(
    ws: usize,
    r: usize,
    d: usize,
    t: usize,
    x: Option<f64>,
) -> Result<(f64, f64), ThresholdError> {
    let n = 2 * r;
    let w = 2 * d;
    let x = x.map_or_else(|| compute_x(r, d, t), Ok)?;
    let pi1 = (ws as f64 + x) / (t * d) as f64;
    let pi0 = ((w * ws) as f64 - x) / ((n - t) * d) as f64;
    Ok((pi0, pi1))
}

pub fn exact_threshold_ineq(
    ws: usize,
    r: usize,
    d: usize,
    t: usize,
    x: Option<f64>,
) -> Result<u8, ThresholdError> {
    if ws == 0 {
        return Ok(bf_threshold_min(d));
    } else if ws > r {
        return Err(ThresholdError::WeightError(ws, r));
    }
    let n = 2 * r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x)?;
    let mut threshold: i32 = 1;
    let di = d as i32;
    while threshold <= di
        && t as f64 * pi1.powi(threshold) * (1.0 - pi1).powi(di - threshold)
            < (n - t) as f64 * pi0.powi(threshold) * (1.0 - pi0).powi(di - threshold)
    {
        threshold += 1;
    }
    let threshold = u8::try_from(threshold).or(Err(ThresholdError::OverflowError))?;
    // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
    let threshold = threshold.max(bf_threshold_min(d));
    Ok(threshold)
}

pub fn exact_threshold(
    ws: usize,
    r: usize,
    d: usize,
    t: usize,
    x: Option<f64>,
) -> Result<u8, ThresholdError> {
    if ws == 0 {
        return Ok(bf_threshold_min(d));
    } else if ws > r {
        return Err(ThresholdError::WeightError(ws, r));
    }
    let n = 2 * r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x)?;

    let log_frac = ((1.0 - pi0) / (1.0 - pi1)).log2();
    let thresh_num = (((n - t) / t) as f64).log2() + d as f64 * log_frac;
    let thresh_den = (pi1 / pi0).log2() + log_frac;
    let threshold = (thresh_num / thresh_den).ceil();
    if threshold.is_finite() {
        let threshold = u8::try_from(threshold as u32).or(Err(ThresholdError::OverflowError))?;
        // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
        let threshold = threshold.max(bf_threshold_min(d));
        Ok(threshold)
    } else {
        Err(ThresholdError::Infinite)
    }
}

#[derive(Copy, Clone, Debug, Error)]
pub enum ThresholdError {
    #[error("Threshold constant X must be finite")]
    XError,
    #[error("Syndrome weight ({0}) cannot be greater than block length ({1})")]
    WeightError(usize, usize),
    #[error("Computed threshold exceeds maximum supported value {}", u8::MAX)]
    OverflowError,
    #[error("Computed threshold was infinite or NaN")]
    Infinite,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_x() {
        let (r, d, t) = (587, 15, 18);
        let x_known: f64 = 10.2859814049302;
        let x_computed = compute_x(r, d, t).unwrap();
        assert!((x_known - x_computed).abs() < 1e-9);
    }

    #[test]
    fn known_thresholds() {
        let (r, d, t) = (587, 15, 18);
        let thresholds_no_min = [
            1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8,
            8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
            10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
            12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14,
            14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
            11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3, 3, 3,
            3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
        ];
        let x = compute_x(r, d, t).unwrap();
        for ws in 0..=r {
            let thresh = exact_threshold_ineq(ws, r, d, t, Some(x)).unwrap();
            assert_eq!(thresh, thresholds_no_min[ws].max(bf_threshold_min(d)));
            assert_eq!(thresh, THRESHOLD_CACHE[ws]);
        }
    }
}
