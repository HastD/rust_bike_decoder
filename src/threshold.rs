use num::{BigInt, BigRational, ToPrimitive};
use num_integer::binomial;
use std::cmp;

fn big_binomial(n: u32, k: u32) -> BigInt {
    binomial(BigInt::from(n), BigInt::from(k))
}

pub fn exact_threshold(ws: u32, r: u32, d: u32, t: u32) -> (u32, bool) {
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
    (threshold as u32, threshold.is_nan())
}

pub fn threshold_cache(r: usize, d: usize, t: usize) -> Vec<u32> {
    let (r, d, t) = (r as u32, d as u32, t as u32);
    let mut cache = vec![0];
    let mut ws = 1;
    loop {
        let (thresh, is_nan) = exact_threshold(ws, r, d, t);
        if is_nan {
            break;
        } else {
            cache.push(thresh);
            ws += 1;
        }
    }
    cache
}
