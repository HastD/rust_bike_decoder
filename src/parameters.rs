pub const BLOCK_LENGTH: usize = 587;
pub const BLOCK_WEIGHT: usize = 15;
pub const ERROR_WEIGHT: usize = 18;
pub const BGF_THRESHOLD: u32 = 3;
pub const WEAK_KEY_THRESHOLD: u8 = 4;
pub const NB_ITER: usize = 7;
// threshold function takes max(threshold, BF_THRESHOLD_MIN)
pub const BF_THRESHOLD_MIN: usize = (BLOCK_WEIGHT + 1)/2;

// computed constants, don't change these
pub const ROW_LENGTH: usize = 2*BLOCK_LENGTH;
pub const ROW_WEIGHT: usize = 2*BLOCK_WEIGHT;
pub const TANNER_GRAPH_EDGES: usize = ROW_WEIGHT*BLOCK_LENGTH;
// Padding of certain arrays for optimal use of 256-bit AVX2 registers
pub const SIZE_AVX: usize = ((((BLOCK_LENGTH*8) + (256*16 - 1)) / (256 * 16)) * (256 * 16)) / 8;
pub const DOUBLE_SIZE_AVX: usize = 2*SIZE_AVX;
