use malachite::{
    num::{arithmetic::traits::BinomialCoefficient, conversion::traits::RoundingInto},
    rounding_modes::RoundingMode,
    Natural, Rational,
};
use once_cell::sync::Lazy;
use thiserror::Error;

use crate::parameters::{BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT};

pub static THRESHOLD_CACHE: Lazy<Vec<u8>> = Lazy::new(|| {
    build_threshold_cache(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT)
        .expect("Must be able to initialize threshold cache")
});

pub fn build_threshold_cache(r: usize, d: usize, t: usize) -> Result<Vec<u8>, ThresholdError> {
    let x = compute_x(r, d, t);
    let mut threshold_cache: Vec<u8> = Vec::with_capacity(r + 1);
    for ws in 0..=r {
        threshold_cache.push(exact_threshold_ineq(ws, r, d, t, Some(x))?);
    }
    Ok(threshold_cache)
}

pub const fn bf_threshold_min(block_weight: usize) -> u8 {
    assert!(
        block_weight <= u8::MAX as usize,
        "Block weight > 255 not supported"
    );
    (block_weight as u8 + 1) / 2
}

pub const fn bf_masked_threshold(block_weight: usize) -> u8 {
    assert!(
        block_weight <= u8::MAX as usize,
        "Block weight > 255 not supported"
    );
    (block_weight as u8 + 1) / 2 + 1
}

fn big_binomial(n: usize, k: usize) -> Natural {
    Natural::binomial_coefficient(Natural::from(n), Natural::from(k))
}

pub fn compute_x(r: usize, d: usize, t: usize) -> f64 {
    let n = 2 * r;
    let w = 2 * d;
    let n_minus_w = n - w;
    let mut x_part = Natural::default();
    for l in (3..t.min(w)).step_by(2) {
        x_part += Natural::from(l - 1) * big_binomial(w, l) * big_binomial(n_minus_w, t - l);
    }
    Rational::from_naturals(Natural::from(r) * x_part, big_binomial(n, t))
        .rounding_into(RoundingMode::Nearest)
}

fn threshold_constants(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>) -> (f64, f64) {
    let n = 2 * r;
    let w = 2 * d;
    let x = x.unwrap_or_else(|| compute_x(r, d, t));
    let pi1 = (ws as f64 + x) / (t * d) as f64;
    let pi0 = ((w * ws) as f64 - x) / ((n - t) * d) as f64;
    (pi0, pi1)
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
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x);
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
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x);

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
        let x_computed = compute_x(r, d, t);
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
        let x = compute_x(r, d, t);
        for ws in 0..=r {
            let thresh = exact_threshold_ineq(ws, r, d, t, Some(x)).unwrap();
            assert_eq!(thresh, thresholds_no_min[ws].max(bf_threshold_min(d)));
            assert_eq!(thresh, THRESHOLD_CACHE[ws]);
        }
    }

    #[test]
    fn big_threshold_cache() {
        // BIKE security level 5 parameters
        let (r, d, t) = (40_973, 137, 264);
        let mut big_cache = build_threshold_cache(r, d, t).unwrap();
        assert_eq!(big_cache.iter().filter(|&thr| *thr == 69).count(), 19_884);
        assert_eq!(
            &big_cache[31_950..32_050],
            &[
                132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                132, 132, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 134,
                134, 134, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 135, 135,
                135, 135, 135, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133, 133,
                133, 133, 133, 133
            ]
        );
        big_cache.dedup();
        assert_eq!(
            big_cache,
            vec![
                69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107,
                108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
                124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 133, 131, 129, 127,
                125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93,
                91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69
            ]
        );
    }
}
