pub const BLOCK_LENGTH: usize = 587;
pub const ROW_LENGTH: usize = 2*BLOCK_LENGTH;
pub const BLOCK_WEIGHT: usize = 15;
pub const ROW_WEIGHT: usize = 2*BLOCK_WEIGHT;
pub const TANNER_GRAPH_EDGES: usize = ROW_WEIGHT*BLOCK_LENGTH;
pub const ERROR_WEIGHT: usize = 18;
pub const BGF_THRESHOLD: u32 = 3;
pub const WEAK_KEY_THRESHOLD: u8 = 3;
pub const NB_ITER: usize = 7;
// threshold function takes max(threshold, BF_THRESHOLD_MIN)
pub const BF_THRESHOLD_MIN: u32 = (BLOCK_LENGTH + 1)/2;
