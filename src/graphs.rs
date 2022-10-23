use crate::keys::Key;
use crate::constants::*;
use petgraph::graph::UnGraph;
//use std::collections::HashSet;
//use std::collections::VecDeque;

/*
// Modified version of breadth_first_search
// Source: https://github.com/TheAlgorithms/Rust/blob/master/src/graph/breadth_first_search.rs
pub fn breadth_first_cycle_search(graph: &Graph, root: u32, bound: usize) -> Option<Vec<u32>> {
    let mut visited: HashSet<Node> = HashSet::new();
    let mut history: Vec<u32> = Vec::new();
    let mut queue = VecDeque::new();

    visited.insert(root);
    history.push(root);
    for neighbor in graph.neighbors(root) {
        queue.push_back(neighbor);
    }
    while let Some(currentnode) = queue.pop_front() {
        history.push(currentnode.value());

        // If we reach the goal, return our travel history.
        if currentnode == root {
            return Some(history);
        }

        // Check the neighboring nodes for any that we've not visited yet.
        for neighbor in currentnode.neighbors(graph) {
            if !visited.contains(&neighbor) {
                visited.insert(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    // All nodes were visited, yet the target was not found.
    None
}
*/

pub fn tanner_graph(key: &Key) -> UnGraph<(), ()> {
    let r = BLOCK_LENGTH as u32;
    let mut edges = [(0u32, 0u32); TANNER_GRAPH_EDGES];
    let mut max_i = 0;
    for k in 0..r {
        for b in key.h0.support() {
            edges[max_i] = ((b as u32 + k) % r, k + 2*r);
            max_i += 1;
        }
        for b in key.h1.support() {
            edges[max_i] = (((b as u32 + k) % r) + r, k + 2*r);
            max_i += 1;
        }
    }
    let graph = UnGraph::<(), ()>::from_edges(&edges);
    graph
}
