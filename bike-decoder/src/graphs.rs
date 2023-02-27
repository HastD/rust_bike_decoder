use crate::{
    counter::IndexCounter,
    decoder::{find_bgf_cycle, DecodingFailure},
    keys::QuasiCyclic,
    ncw::NcwOverlaps,
    vectors::Index,
};
use getset::Getters;
use petgraph::graph::{NodeIndex, UnGraph};
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

impl From<CheckNode> for usize {
    fn from(value: CheckNode) -> usize {
        usize::try_from(value.0).expect("Node value should not overflow usize")
    }
}

impl From<Node> for NodeIndex {
    fn from(node: Node) -> Self {
        Self::new(match node {
            Node::Check(CheckNode(idx)) => idx as usize,
            // Bitwise negate the indices for variable nodes so they don't overlap
            // with check node indices.
            Node::Variable(VariableNode(idx)) => !(idx as usize),
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
            let chk = (b as usize + k) % LENGTH;
            edges[k][i] = (VariableNode(k as Index), CheckNode(chk as Index));
        }
        for (i, &b) in key.h1().support().iter().enumerate() {
            let chk = (b as usize + k) % LENGTH;
            edges[k + LENGTH][i] = (VariableNode((k + LENGTH) as Index), CheckNode(chk as Index));
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
fn check_node_degrees(subgraph: &[(VariableNode, CheckNode)]) -> IndexCounter {
    subgraph.iter().map(|&(_, check)| check).collect()
}

pub fn odd_check_node_neighbors<const WEIGHT: usize, const LENGTH: usize>(
    edges: &TannerGraphEdges<WEIGHT, LENGTH>,
    supp: &[Index],
) -> (Vec<(VariableNode, CheckNode)>, IndexCounter, Vec<CheckNode>) {
    let subgraph = subgraph_from_support(edges, supp);
    let check_node_degrees = check_node_degrees(&subgraph);
    let mut odd_check_nodes: Vec<CheckNode> = check_node_degrees
        .iter()
        .enumerate()
        .filter_map(|(check, &count)| (count % 2 == 1).then_some(CheckNode(check as u32)))
        .collect();
    odd_check_nodes.sort_unstable();
    (subgraph, check_node_degrees, odd_check_nodes)
}

pub fn is_absorbing_subgraph<const WEIGHT: usize, const LENGTH: usize>(
    edges: &TannerGraphEdges<WEIGHT, LENGTH>,
    supp: &[Index],
) -> bool {
    let subgraph = subgraph_from_support(edges, supp);
    let check_node_degrees = check_node_degrees(&subgraph);
    for &var in supp {
        let odd_count = edges.0[var as usize]
            .iter()
            .filter(|(_, check)| check_node_degrees.count(*check) % 2 == 1)
            .count();
        if odd_count >= (WEIGHT + 1) / 2 {
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

#[derive(Debug, Clone, Getters, Serialize, Deserialize)]
#[getset(get = "pub")]
pub struct AbsorbingDecodingFailure {
    df: DecodingFailure,
    supp: Vec<Index>,
    odd_check_nodes: Vec<CheckNode>,
    overlaps: Option<NcwOverlaps>,
}

impl AbsorbingDecodingFailure {
    pub fn new(df: DecodingFailure, compute_overlaps: bool) -> Option<Self> {
        let key = df.key();
        let edges = TannerGraphEdges::new(key);
        let e_supp = df.vector().vector();
        // Compute stable output of decoder
        let cycle = find_bgf_cycle(key, e_supp, usize::MAX, compute_overlaps);
        // diff = support of e_in - e_out
        let diff = cycle.diff();
        if is_absorbing_subgraph(&edges, &diff) {
            let (_, _, odd_check_nodes) = odd_check_node_neighbors(&edges, &diff);
            let overlaps = compute_overlaps.then(|| NcwOverlaps::new(key, &diff));
            Some(Self {
                df,
                supp: diff,
                odd_check_nodes,
                overlaps,
            })
        } else {
            None
        }
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
        AbsorbingDecodingFailure::new(df, false).unwrap();
    }

    #[test]
    fn absorbing_non_example() {
        let df: DecodingFailure = serde_json::from_str(
            r#"{"h0":[93,99,105,121,126,141,156,193,194,197,264,301,360,400,429],"h1":[100,117,
            189,191,211,325,340,386,440,461,465,474,534,565,578],"e_supp":[30,91,310,337,487,597,
            616,712,766,816,923,933,956,1062,1069,1131,1134,1152]}"#,
        )
        .unwrap();
        assert!(AbsorbingDecodingFailure::new(df, false).is_none());
    }
}
