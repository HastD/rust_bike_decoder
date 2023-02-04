use bike_decoder::random::{get_or_insert_global_seed, global_seed, Seed};

#[test]
fn global_seed_init() {
    assert_eq!(global_seed(), None);
    let seed1 = Seed::from_entropy();
    get_or_insert_global_seed(Some(seed1));
    assert_eq!(global_seed(), Some(seed1));
    let seed2 = Seed::from_entropy();
    assert_ne!(seed1, seed2);
    get_or_insert_global_seed(Some(seed2));
    assert_eq!(global_seed(), Some(seed1));
}
