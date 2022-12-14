pub const BLOCK_LENGTH: usize = 587;
pub const BLOCK_WEIGHT: usize = 15;
pub const ERROR_WEIGHT: usize = 18;
pub const NB_ITER: usize = 7;
pub const GRAY_THRESHOLD_DIFF: u8 = 3;

// computed constants, don't change these
pub const ROW_LENGTH: usize = 2 * BLOCK_LENGTH;
pub const ROW_WEIGHT: usize = 2 * BLOCK_WEIGHT;
pub const TANNER_GRAPH_EDGES: usize = BLOCK_WEIGHT * ROW_LENGTH;
// Padding of certain arrays for efficient use of 256-bit AVX2 registers
pub const SIZE_AVX: usize = ((BLOCK_LENGTH * 8) + (256 * 16 - 1)) / (256 * 16) * (256 * 16) / 8;

#[allow(dead_code, clippy::all)]
const fn compile_time_assertions() {
    const _: () = assert!(BLOCK_WEIGHT < BLOCK_LENGTH);
    const _: () = assert!(ERROR_WEIGHT < BLOCK_LENGTH);
    const _: () = assert!((BLOCK_WEIGHT + 1) / 2 + 1 <= u8::MAX as usize);
    const _: () = assert!(BLOCK_LENGTH < SIZE_AVX);
    const _: () = assert!(2 * SIZE_AVX < usize::MAX);
}
