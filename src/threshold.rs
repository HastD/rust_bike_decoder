use crate::parameters::*;
use lazy_static::lazy_static;
use num::{BigInt, BigRational, ToPrimitive};
use num_integer::binomial;
use thiserror::Error;

lazy_static! {
    pub static ref THRESHOLD_CACHE: Vec<u8> = {
        let (r, d, t) = (BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
        let x = compute_x(r, d, t).expect("Must be able to compute threshold constant X");
        (0..=BLOCK_LENGTH).map(|ws|
            exact_threshold_ineq(ws, r, d, t, Some(x))
                .expect("Must be able to compute thresholds")
        ).collect()
    };
}

fn big_binomial(n: usize, k: usize) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

pub fn compute_x(r: usize, d: usize, t: usize) -> Result<f64, ThresholdError> {
    let n = 2*r;
    let w = 2*d;
    let n_minus_w = n - w;
    let mut x_part = BigInt::from(0);
    for l in (3 .. t.min(w)).step_by(2) {
        x_part += (l - 1) * big_binomial(w, l) * big_binomial(n_minus_w, t - l);
    }
    let x = BigRational::new(r * x_part, big_binomial(n, t)).to_f64();
    let err = ThresholdError::XError;
    x.ok_or(err).and_then(|x| if x.is_finite() { Ok(x) } else { Err(err) })
}

fn threshold_constants(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>)
-> Result<(f64, f64), ThresholdError> {
    let n = 2*r;
    let w = 2*d;
    let x = x.map_or_else(|| compute_x(r, d, t), Ok)?;
    let pi1 = (ws as f64 + x) / (t * d) as f64;
    let pi0 = ((w * ws) as f64 - x) / ((n - t) * d) as f64;
    Ok((pi0, pi1))
}

pub fn exact_threshold_ineq(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>)
-> Result<u8, ThresholdError> {
    if ws == 0 {
        return Ok(BF_THRESHOLD_MIN);
    } else if ws > r {
        return Err(ThresholdError::WeightError(ws, r));
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x)?;
    let mut threshold: i32 = 1;
    let d = d as i32;
    while threshold <= d && t as f64 * pi1.powi(threshold) * (1.0 - pi1).powi(d - threshold)
                    < (n - t) as f64 * pi0.powi(threshold) * (1.0 - pi0).powi(d - threshold) {
        threshold += 1;
    }
    let threshold = u8::try_from(threshold).or(Err(ThresholdError::OverflowError))?;
    // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
    let threshold = threshold.max(BF_THRESHOLD_MIN);
    Ok(threshold)
}

pub fn exact_threshold(ws: usize, r: usize, d: usize, t: usize, x: Option<f64>)
-> Result<u8, ThresholdError> {
    if ws == 0 {
        return Ok(BF_THRESHOLD_MIN);
    } else if ws > r {
        return Err(ThresholdError::WeightError(ws, r));
    }
    let n = 2*r;
    let (pi0, pi1) = threshold_constants(ws, r, d, t, x)?;

    let log_frac = ((1.0 - pi0) / (1.0 - pi1)).log2();
    let thresh_num = (((n - t) / t) as f64).log2() + d as f64 * log_frac;
    let thresh_den = (pi1 / pi0).log2() + log_frac;
    let threshold = (thresh_num / thresh_den).ceil();
    if threshold.is_finite() {
        let threshold = u8::try_from(threshold as u32).or(Err(ThresholdError::OverflowError))?;
        // modification to threshold mentioned in Vasseur's thesis, section 6.1.3.1
        let threshold = threshold.max(BF_THRESHOLD_MIN);
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
        assert_eq!((r, d, t), (BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT));
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
            assert_eq!(thresh, thresholds_no_min[ws].max(BF_THRESHOLD_MIN));
            assert_eq!(thresh, THRESHOLD_CACHE[ws]);
        }
    }
}
