use crate::keys::Key;
use crate::parameters::*;
use petgraph::graph::UnGraph;

pub fn tanner_graph(key: &Key) -> UnGraph<(), ()> {
    let r = BLOCK_LENGTH as u32;
    let mut edges = [(0u32, 0u32); TANNER_GRAPH_EDGES];
    let mut max_i = 0;
    for k in 0..r {
        for b in key.h0().support() {
            edges[max_i] = ((b as u32 + k) % r, k + 2*r);
            max_i += 1;
        }
        for b in key.h1().support() {
            edges[max_i] = (((b as u32 + k) % r) + r, k + 2*r);
            max_i += 1;
        }
    }
    UnGraph::<(), ()>::from_edges(&edges)
}
