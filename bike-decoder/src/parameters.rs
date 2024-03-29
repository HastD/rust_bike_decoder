// Can change these parameters at compile-time using environment variables
pub const BLOCK_LENGTH: usize = env_or_usize!("BIKE_BLOCK_LENGTH", 587);
pub const BLOCK_WEIGHT: usize = env_or_usize!("BIKE_BLOCK_WEIGHT", 15);
pub const ERROR_WEIGHT: usize = env_or_usize!("BIKE_ERROR_WEIGHT", 18);
pub const NB_ITER: usize = env_or_usize!("BIKE_NB_ITER", 7);

pub const GRAY_THRESHOLD_DIFF: u8 = 3;

// Computed constants, don't change these
pub const ROW_LENGTH: usize = 2 * BLOCK_LENGTH;
pub const ROW_WEIGHT: usize = 2 * BLOCK_WEIGHT;
pub const TANNER_GRAPH_EDGES: usize = BLOCK_WEIGHT * ROW_LENGTH;
// Padding of certain arrays for efficient use of 256-bit AVX2 registers
pub const SIZE_AVX: usize = ((BLOCK_LENGTH * 8) + (256 * 16 - 1)) / (256 * 16) * (256 * 16) / 8;

#[allow(dead_code, clippy::all)]
const fn _compile_time_assertions() {
    const _: () = assert!(usize::BITS >= 32, "16-bit systems not supported");
    const _: () = assert!(
        0 < BLOCK_WEIGHT && BLOCK_WEIGHT < BLOCK_LENGTH,
        "BLOCK_WEIGHT must be positive and less than BLOCK_LENGTH"
    );
    const _: () = assert!(
        0 < ERROR_WEIGHT && ERROR_WEIGHT < ROW_LENGTH,
        "ERROR_WEIGHT must be positive and less than ROW_LENGTH"
    );
    const _: () = assert!(
        BLOCK_WEIGHT <= u8::MAX as usize,
        "BLOCK_WEIGHT > 255 not supported"
    );
    const _: () = assert!(BLOCK_LENGTH <= SIZE_AVX);
    const _: () = assert!(2 * SIZE_AVX <= u32::MAX as usize);
    const _: () = assert!(NB_ITER >= 1, "NB_ITER must be positive");
}

#[doc(hidden)]
pub mod __macro {
    pub use ::core::option::Option;
    pub use ::core::option_env;
    pub use ::core::primitive::usize;
    pub use ::core::result::Result;
    pub use ::konst::primitive::parse_usize;
}

/// Parses environment variable at compile-time as `usize` if defined,
/// otherwise yields the given `usize` value. Fails to compile if the
/// environment variable is defined but cannot be parsed.
#[macro_export]
macro_rules! env_or_usize {
    ( $name:expr, $default:expr $(,)? ) => {{
        use $crate::parameters::__macro::{option_env, parse_usize, usize, Option, Result};
        if let Option::Some(s) = option_env!($name) {
            match parse_usize(s) {
                Result::<usize, _>::Ok(value) => value,
                Result::Err(err) => err.panic(),
            }
        } else {
            $default
        }
    }};
}

use env_or_usize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_usize_default_value() {
        const N: usize = env_or_usize!("ENV_USIZE_TEST_VAR", 42);
        assert_eq!(N, 42);
    }
}
