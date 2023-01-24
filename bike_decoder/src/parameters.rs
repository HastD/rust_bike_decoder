// Can change these parameters at compile-time using environment variables
pub const BLOCK_LENGTH: usize = env_usize_or!("BIKE_BLOCK_LENGTH", 587);
pub const BLOCK_WEIGHT: usize = env_usize_or!("BIKE_BLOCK_WEIGHT", 15);
pub const ERROR_WEIGHT: usize = env_usize_or!("BIKE_ERROR_WEIGHT", 18);
pub const NB_ITER: usize = env_usize_or!("BIKE_NB_ITER", 7);

pub const GRAY_THRESHOLD_DIFF: u8 = 3;

// Computed constants, don't change these
pub const ROW_LENGTH: usize = 2 * BLOCK_LENGTH;
pub const ROW_WEIGHT: usize = 2 * BLOCK_WEIGHT;
pub const TANNER_GRAPH_EDGES: usize = BLOCK_WEIGHT * ROW_LENGTH;
// Padding of certain arrays for efficient use of 256-bit AVX2 registers
pub const SIZE_AVX: usize = ((BLOCK_LENGTH * 8) + (256 * 16 - 1)) / (256 * 16) * (256 * 16) / 8;

#[allow(dead_code, clippy::all)]
const fn compile_time_assertions() {
    const _: () = assert!(usize::BITS >= 32);
    const _: () = assert!(0 < BLOCK_WEIGHT && BLOCK_WEIGHT < BLOCK_LENGTH);
    const _: () = assert!(0 < ERROR_WEIGHT && ERROR_WEIGHT < BLOCK_LENGTH);
    const _: () = assert!((BLOCK_WEIGHT + 1) / 2 + 1 <= u8::MAX as usize);
    const _: () = assert!(BLOCK_LENGTH <= SIZE_AVX);
    const _: () = assert!(2 * SIZE_AVX <= u32::MAX as usize);
    const _: () = assert!(NB_ITER >= 1);
}

/// Parses environment variable as a usize if defined, otherwise yields the
/// given default value. Causes a compilation error if the environment variable
/// is defined but cannot be parsed.
macro_rules! env_usize_or {
    ( $name:literal, $default:literal $(,)? ) => {{
        ::konst::result::unwrap_ctx!(::konst::option::unwrap_or!(
            ::konst::option::map!(::core::option_env!($name), ::konst::primitive::parse_usize),
            ::core::result::Result::Ok::<
                ::core::primitive::usize,
                ::konst::primitive::ParseIntError,
            >($default)
        ))
    }};
}

use env_usize_or;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_usize_default_value() {
        const N: usize = env_usize_or!("ENV_USIZE_TEST_VAR", 42);
        assert_eq!(N, 42);
    }
}
