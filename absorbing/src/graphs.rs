use bike_decoder::{
    decoder::{bgf_decoder, DecodingFailure},
    keys::QuasiCyclic,
    random::custom_thread_rng,
    syndrome::Syndrome,
    vectors::Index,
};
use counter::Counter;
use itertools::Itertools;
use petgraph::graph::{NodeIndex, UnGraph};
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Node {
    Variable(VariableNode),
    Check(CheckNode),
}

impl From<VariableNode> for Node {
    fn from(var: VariableNode) -> Self {
        Self::Variable(var)
    }
}

impl From<CheckNode> for Node {
    fn from(check: CheckNode) -> Self {
        Self::Check(check)
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VariableNode(pub Index);

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CheckNode(pub Index);

impl From<Node> for NodeIndex {
    fn from(node: Node) -> Self {
        Self::new(match node {
            Node::Check(CheckNode(idx)) => idx as usize,
            Node::Variable(VariableNode(idx)) => idx as usize,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TannerGraphEdges<const WEIGHT: usize, const LENGTH: usize>(
    Vec<[(VariableNode, CheckNode); WEIGHT]>,
);

impl<const WEIGHT: usize, const LENGTH: usize> TannerGraphEdges<WEIGHT, LENGTH> {
    #[inline]
    pub fn new(key: &QuasiCyclic<WEIGHT, LENGTH>) -> Self {
        Self(tanner_graph_edges(key))
    }
}

pub fn tanner_graph_edges<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
) -> Vec<[(VariableNode, CheckNode); WEIGHT]> {
    let mut edges = vec![[(VariableNode(0), CheckNode(0)); WEIGHT]; 2 * LENGTH];
    for k in 0..LENGTH {
        for (i, &b) in key.h0().support().iter().enumerate() {
            let var = (b as usize + k) % LENGTH;
            edges[var][i] = (VariableNode(var as Index), CheckNode(k as Index));
        }
        for (i, &b) in key.h1().support().iter().enumerate() {
            let var = (b as usize + k) % LENGTH + LENGTH;
            edges[var][i] = (VariableNode(var as Index), CheckNode(k as Index));
        }
    }
    edges
}

pub fn tanner_graph<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
) -> UnGraph<(), ()> {
    let edges = tanner_graph_edges(key)
        .into_iter()
        .flatten()
        .map(|(var, check)| (Node::from(var), Node::from(check)));
    UnGraph::<(), ()>::from_edges(edges)
}

#[inline]
fn subgraph_from_support<const WEIGHT: usize, const LENGTH: usize>(
    edges: &TannerGraphEdges<WEIGHT, LENGTH>,
    supp: &[Index],
) -> Vec<(VariableNode, CheckNode)> {
    supp.iter()
        .flat_map(|&idx| &edges.0[idx as usize])
        .copied()
        .collect()
}

#[inline]
fn check_node_degrees(subgraph: &[(VariableNode, CheckNode)]) -> Counter<CheckNode> {
    subgraph.iter().map(|&(_, check)| check).collect()
}

pub fn odd_check_node_neighbors<const WEIGHT: usize, const LENGTH: usize>(
    edges: &TannerGraphEdges<WEIGHT, LENGTH>,
    supp: &[Index],
) -> (
    Vec<(VariableNode, CheckNode)>,
    Counter<CheckNode>,
    Vec<CheckNode>,
) {
    let subgraph = subgraph_from_support(edges, supp);
    let check_node_degrees = check_node_degrees(&subgraph);
    let mut odd_check_nodes: Vec<CheckNode> = check_node_degrees
        .iter()
        .filter_map(|(&check, &count)| (count % 2 == 1).then_some(check))
        .collect();
    odd_check_nodes.sort();
    (subgraph, check_node_degrees, odd_check_nodes)
}

pub fn is_absorbing_subgraph<const WEIGHT: usize, const LENGTH: usize>(
    edges: &TannerGraphEdges<WEIGHT, LENGTH>,
    supp: &[Index],
) -> bool {
    let subgraph = subgraph_from_support(edges, supp);
    let check_node_degrees = check_node_degrees(&subgraph);
    for &var in supp {
        let even_count = 1 + &edges.0[var as usize]
            .iter()
            .filter(|(_, check)| *check_node_degrees.get(check).unwrap_or(&0) % 2 == 0)
            .count();
        if even_count <= (WEIGHT + 1) / 2 {
            return false;
        }
    }
    true
}

/// Given an r-by-2r parity check matrix specified by circulant blocks, and a
/// vector D specified by the support of the vector, this function determines
/// if the vector D defines an absorbing set.
#[inline]
pub fn is_absorbing<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    supp: &[Index],
) -> bool {
    let edges = TannerGraphEdges::new(key);
    is_absorbing_subgraph(&edges, supp)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsorbingDecodingFailure {
    df: DecodingFailure,
    supp: Vec<Index>,
    odd_check_nodes: Vec<CheckNode>,
}

impl AbsorbingDecodingFailure {
    pub fn new(df: DecodingFailure) -> Option<Self> {
        let key = df.key();
        let edges = TannerGraphEdges::new(key);
        let e_supp = df.vector().vector();
        let e_in = e_supp.dense();
        let mut syn = Syndrome::from_sparse(key, e_supp);
        let (e_out, _) = bgf_decoder(key, &mut syn);
        let supp = (e_in - e_out).support();
        if is_absorbing_subgraph(&edges, &supp) {
            let (_, _, odd_check_nodes) = odd_check_node_neighbors(&edges, &supp);
            Some(Self {
                df,
                supp,
                odd_check_nodes,
            })
        } else {
            None
        }
    }
}

/// Enumerates all absorbing sets of a given weight for `key`.
/// The `parallel` argument uses Rayon to run the computation in parallel.
pub fn enumerate_absorbing_sets<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    supp_weight: usize,
    parallel: bool,
) -> Vec<Vec<Index>> {
    let n = 2 * LENGTH as Index;
    let edges = TannerGraphEdges::new(key);
    let combinations = (0..n).combinations(supp_weight);
    if parallel {
        combinations
            .par_bridge()
            .filter(|supp| is_absorbing_subgraph(&edges, supp))
            .collect()
    } else {
        combinations
            .filter(|supp| is_absorbing_subgraph(&edges, supp))
            .collect()
    }
}

/// Searches for absorbing sets for `key`.
pub fn sample_absorbing_sets<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    supp_weight: usize,
    samples: usize,
    parallel: bool,
) -> Vec<Vec<Index>> {
    let n = 2 * LENGTH as Index;
    let edges = TannerGraphEdges::new(key);
    if parallel {
        (0..samples)
            .into_par_iter()
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight))
            .filter(|supp| is_absorbing_subgraph(&edges, supp))
            .collect()
    } else {
        (0..samples)
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight))
            .filter(|supp| is_absorbing_subgraph(&edges, supp))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorbing_example() {
        let df: DecodingFailure = serde_json::from_str(
            r#"{"h0":[0,11,14,53,69,134,190,213,218,245,378,408,411,480,545],"h1":[26,104,110,137,
            207,252,258,310,326,351,367,459,461,506,570],"e_supp":[16,37,83,130,186,289,351,460,
            481,527,558,662,724,772,1008,1011,1038,1072]}"#,
        )
        .unwrap();
        AbsorbingDecodingFailure::new(df).unwrap();
    }

    #[test]
    fn absorbing_non_example() {
        let df: DecodingFailure = serde_json::from_str(
            r#"{"h0":[337,180,234,163,573,63,276,451,428,57,213,41,158,194,485],"h1":[260,528,580,
            127,537,84,404,218,374,394,509,194,382,55,185],"e_supp":[1078,283,10,62,460,806,715,
            157,1096,849,503,996,533,1004,564,991,858,916]}"#,
        )
        .unwrap();
        assert!(AbsorbingDecodingFailure::new(df).is_none());
    }
}
