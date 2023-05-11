use bike_decoder::{
    decoder,
    keys::Key,
    ncw::NcwOverlaps,
    parameters::*,
    syndrome::Syndrome,
    vectors::{ErrorVector, SparseErrorVector},
};
use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

/// Optimized, non-cryptographic Rust implementation of BGF decoder used in BIKE.
#[pymodule]
fn bike_decoder_pyo3(_py: Python, m: &PyModule) -> PyResult<()> {
    // Constants from bike_decoder::parameters
    m.add("BLOCK_LENGTH", BLOCK_LENGTH)?;
    m.add("BLOCK_WEIGHT", BLOCK_WEIGHT)?;
    m.add("ERROR_WEIGHT", ERROR_WEIGHT)?;
    m.add("NB_ITER", NB_ITER)?;
    m.add("GRAY_THRESHOLD_DIFF", GRAY_THRESHOLD_DIFF)?;
    m.add("ROW_LENGTH", ROW_LENGTH)?;
    m.add("ROW_WEIGHT", ROW_WEIGHT)?;
    // Functions providing a Python interface to the BGF decoder and related utilities
    m.add_function(wrap_pyfunction!(bgf_decoder, m)?)?;
    m.add_function(wrap_pyfunction!(random_key, m)?)?;
    m.add_function(wrap_pyfunction!(random_non_weak_key, m)?)?;
    m.add_function(wrap_pyfunction!(random_error_support, m)?)?;
    m.add_function(wrap_pyfunction!(syndrome, m)?)?;
    m.add_function(wrap_pyfunction!(bf_iter, m)?)?;
    m.add_function(wrap_pyfunction!(bf_iter_no_mask, m)?)?;
    m.add_function(wrap_pyfunction!(bf_masked_iter, m)?)?;
    m.add_function(wrap_pyfunction!(unsatisfied_parity_checks, m)?)?;
    m.add_function(wrap_pyfunction!(exact_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(ncw_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(is_absorbing, m)?)?;
    Ok(())
}

/// Runs BGF decoder on key `(h0, h1)` and syndrome `s`. Returns the resulting error vector and
/// a boolean indicating decoding success or failure.
#[pyfunction]
fn bgf_decoder(h0: Vec<u32>, h1: Vec<u32>, s: Vec<bool>) -> PyResult<(Vec<bool>, bool)> {
    let key = key_from_vec_supp(h0, h1)?;
    let mut s = syndrome_from_vec(s)?;
    let (e_out, success) = decoder::bgf_decoder(&key, &mut s);
    Ok((e_out.contents().to_vec(), success))
}

/// Generates a random key
#[pyfunction]
fn random_key() -> (Vec<u32>, Vec<u32>) {
    let mut rng = bike_decoder::random::custom_thread_rng();
    let key = Key::random(&mut rng).sorted();
    (key.h0().support().to_vec(), key.h1().support().to_vec())
}

/// Generates a random key that is not weak with respect to the given threshold
#[pyfunction]
fn random_non_weak_key(threshold: u8) -> PyResult<(Vec<u32>, Vec<u32>)> {
    if threshold < 3 {
        return Err(PyValueError::new_err("weak key threshold must be >= 3"));
    }
    let mut rng = bike_decoder::random::custom_thread_rng();
    let key = Key::random_non_weak(threshold, &mut rng).sorted();
    Ok((key.h0().support().to_vec(), key.h1().support().to_vec()))
}

/// Generates a random error vector
#[pyfunction]
fn random_error_support() -> Vec<u32> {
    let mut rng = bike_decoder::random::custom_thread_rng();
    let e_supp = SparseErrorVector::random(&mut rng).sorted();
    e_supp.support().to_vec()
}

/// Computes the syndrome associated to an error vector.
#[pyfunction]
fn syndrome(h0: Vec<u32>, h1: Vec<u32>, e_supp: Vec<u32>) -> PyResult<Vec<bool>> {
    let key = key_from_vec_supp(h0, h1)?;
    let Ok(err) = ErrorVector::try_from(&*e_supp) else {
        return Err(PyValueError::new_err(format!("indices must be <= {}", 2 * BLOCK_LENGTH)))
    };
    Ok(Syndrome::from_dense(&key, &err).contents().to_vec())
}

/// Runs one iteration of the BF (bit-flipping) algorithm with the given threshold. Returns the
/// modified syndrome, modified error vector, and black/gray masks.
#[allow(clippy::type_complexity)]
#[pyfunction]
fn bf_iter(
    h0: Vec<u32>,
    h1: Vec<u32>,
    s: Vec<bool>,
    e_out: Vec<bool>,
    thr: u8,
) -> PyResult<(
    Vec<bool>,
    Vec<bool>,
    (Vec<usize>, Vec<usize>),
    (Vec<usize>, Vec<usize>),
)> {
    let key = key_from_vec_supp(h0, h1)?;
    let mut s = syndrome_from_vec(s)?;
    let mut e_out = err_from_vec(e_out)?;
    let ([black0, black1], [gray0, gray1]) = decoder::bf_iter(&key, &mut s, &mut e_out, thr);
    Ok((
        s.contents().to_vec(),
        e_out.contents().to_vec(),
        (black0, black1),
        (gray0, gray1),
    ))
}

/// Runs one iteration of the BF (bit-flipping) algorithm with the given threshold. Returns the
/// modified syndrome and modified error vector.
#[pyfunction]
fn bf_iter_no_mask(
    h0: Vec<u32>,
    h1: Vec<u32>,
    s: Vec<bool>,
    e_out: Vec<bool>,
    thr: u8,
) -> PyResult<(Vec<bool>, Vec<bool>)> {
    let key = key_from_vec_supp(h0, h1)?;
    let mut s = syndrome_from_vec(s)?;
    let mut e_out = err_from_vec(e_out)?;
    decoder::bf_iter_no_mask(&key, &mut s, &mut e_out, thr);
    Ok((s.contents().to_vec(), e_out.contents().to_vec()))
}

#[pyfunction]
fn bf_masked_iter(
    h0: Vec<u32>,
    h1: Vec<u32>,
    s: Vec<bool>,
    e_out: Vec<bool>,
    mask: (Vec<usize>, Vec<usize>),
    thr: u8,
) -> PyResult<(Vec<bool>, Vec<bool>)> {
    let key = key_from_vec_supp(h0, h1)?;
    let mut s = syndrome_from_vec(s)?;
    let mut e_out = err_from_vec(e_out)?;
    let mask = mask_from_vec_pair(mask)?;
    decoder::bf_masked_iter(&key, &mut s, &mut e_out, mask, thr);
    Ok((s.contents().to_vec(), e_out.contents().to_vec()))
}

/// Computes the number of unsatisfied parity checks for the given syndrome and key.
#[pyfunction]
fn unsatisfied_parity_checks(
    h0: Vec<u32>,
    h1: Vec<u32>,
    s: Vec<bool>,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let key = key_from_vec_supp(h0, h1)?;
    let mut s = syndrome_from_vec(s)?;
    let [upc0, upc1] = decoder::unsatisfied_parity_checks(&key, &mut s);
    Ok((Vec::from(upc0), Vec::from(upc1)))
}

/// Computes the exact threshold used in the bit-flipping algorithm for syndrome weight `ws`,
/// block length `r`, block weight `d`, and error weight `t`. The parameters `(r, d, t)` default
/// to the values `(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT)` set at compile-time.
#[pyfunction]
#[pyo3(signature = (ws, r=BLOCK_LENGTH, d=BLOCK_WEIGHT, t=ERROR_WEIGHT))]
fn exact_threshold(ws: usize, r: usize, d: usize, t: usize) -> PyResult<u8> {
    bike_decoder::threshold::exact_threshold_ineq(ws, r, d, t, None)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Computes the maximum overlap of the vector with support `supp` with each of the near-codeword
/// sets C, N, and 2N associated to the key `(h0, h1)`.
#[pyfunction]
fn ncw_overlaps(
    h0: Vec<u32>,
    h1: Vec<u32>,
    supp: Vec<u32>,
) -> PyResult<HashMap<&'static str, usize>> {
    let key = key_from_vec_supp(h0, h1)?;
    let overlaps = NcwOverlaps::new(&key, &supp);
    Ok(HashMap::from([
        ("C", overlaps.c),
        ("N", overlaps.n),
        ("2N", overlaps.two_n),
    ]))
}

/// Determines whether `supp` is an absorbing set for the key `(h0, h1)`.
#[pyfunction]
fn is_absorbing(h0: Vec<u32>, h1: Vec<u32>, supp: Vec<u32>) -> PyResult<bool> {
    let key = key_from_vec_supp(h0, h1)?;
    Ok(bike_decoder::graphs::is_absorbing(&key, &supp))
}

fn key_from_vec_supp(h0: Vec<u32>, h1: Vec<u32>) -> PyResult<Key> {
    let Ok(h0) = <[u32; BLOCK_WEIGHT]>::try_from(h0) else {
        return Err(PyValueError::new_err(format!("h0 must have length {BLOCK_WEIGHT}")))
    };
    let Ok(h1) = <[u32; BLOCK_WEIGHT]>::try_from(h1) else {
        return Err(PyValueError::new_err(format!("h1 must have length {BLOCK_WEIGHT}")))
    };
    Key::from_support(h0, h1).map_err(|e| {
        let msg = format!("(h0, h1) was not a valid key: {e}");
        PyValueError::new_err(msg)
    })
}

fn syndrome_from_vec(s: Vec<bool>) -> PyResult<Syndrome> {
    if let Ok(s) = <[bool; BLOCK_LENGTH]>::try_from(s) {
        Ok(Syndrome::new(s))
    } else {
        Err(PyValueError::new_err(format!(
            "syndrome must have length {BLOCK_LENGTH}"
        )))
    }
}

fn err_from_vec(e: Vec<bool>) -> PyResult<ErrorVector> {
    if let Ok(e) = <[bool; ROW_LENGTH]>::try_from(e) {
        Ok(ErrorVector::new(e))
    } else {
        Err(PyValueError::new_err(format!(
            "error vector must have length {ROW_LENGTH}"
        )))
    }
}

fn mask_from_vec_pair((mask0, mask1): (Vec<usize>, Vec<usize>)) -> PyResult<[Vec<usize>; 2]> {
    if mask0.iter().chain(&mask1).all(|idx| *idx < BLOCK_LENGTH) {
        Ok([mask0, mask1])
    } else {
        Err(PyValueError::new_err(format!(
            "mask entries must be < {BLOCK_LENGTH}"
        )))
    }
}
