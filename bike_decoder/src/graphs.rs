use crate::{
    decoder::{bgf_decoder, DecodingFailure},
    keys::Key,
    parameters::*,
    syndrome::Syndrome,
    vectors::Index,
};
use counter::Counter;
use petgraph::graph::{NodeIndex, UnGraph};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Node {
    Check(Index),
    Variable(Index),
}

impl From<Node> for NodeIndex {
    fn from(node: Node) -> Self {
        Self::new(match node {
            Node::Check(idx) => idx as usize,
            Node::Variable(idx) => idx as usize,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariableNode(pub Index);

pub fn tanner_graph_edges(key: &Key) -> [(Index, Index); TANNER_GRAPH_EDGES] {
    let mut edges = [(0, 0); TANNER_GRAPH_EDGES];
    for k in 0..BLOCK_LENGTH {
        for (i, &b) in key.h0().support().iter().enumerate() {
            let variable_idx = (b as usize + k) % BLOCK_LENGTH;
            edges[BLOCK_WEIGHT * variable_idx + i] = (variable_idx as Index, k as Index);
        }
        for (i, &b) in key.h1().support().iter().enumerate() {
            let variable_idx = (b as usize + k) % BLOCK_LENGTH + BLOCK_LENGTH;
            edges[BLOCK_WEIGHT * variable_idx + i] = (variable_idx as Index, k as Index);
        }
    }
    edges
}

pub fn tanner_graph(key: &Key) -> UnGraph<(), ()> {
    let raw_edges = tanner_graph_edges(key);
    let edges = raw_edges
        .iter()
        .map(|&(var, check)| (Node::Variable(var), Node::Check(check)));
    UnGraph::<(), ()>::from_edges(edges)
}

/// Given an r-by-2r parity check matrix specified by circulant block, and a
/// vector D specified by the support of the vector, this function determines
/// if the vector D defines an absorbing set.
pub fn is_absorbing(key: &Key, supp: &[Index]) -> bool {
    let edges = tanner_graph_edges(key);
    let subgraph: Vec<(Index, Index)> = supp
        .iter()
        .flat_map(|&idx| &edges[BLOCK_WEIGHT * idx as usize..BLOCK_WEIGHT * (idx as usize + 1)])
        .copied()
        .collect();
    let check_node_counts: Counter<_> = subgraph.iter().map(|&(_, check)| check).collect();
    for &idx in supp {
        let even_count = 1 + &edges[BLOCK_WEIGHT * idx as usize..BLOCK_WEIGHT * (idx as usize + 1)]
            .iter()
            .filter(|(_, check)| *check_node_counts.get(check).unwrap_or(&0) % 2 == 0)
            .count();
        if even_count <= (BLOCK_WEIGHT + 1) / 2 {
            return false;
        }
    }
    true
}

pub fn is_decoding_failure_absorbing(df: &DecodingFailure) -> bool {
    let key = df.key();
    let e_supp = df.vector().vector();
    let e_in = e_supp.dense();
    let mut syn = Syndrome::from_sparse(key, e_supp);
    let (e_out, _) = bgf_decoder(key, &mut syn);
    is_absorbing(key, &(e_in - e_out).support())
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
        assert!(is_decoding_failure_absorbing(&df));
    }

    #[test]
    fn absorbing_non_example() {
        let df: DecodingFailure = serde_json::from_str(
            r#"{"h0":[337,180,234,163,573,63,276,451,428,57,213,41,158,194,485],"h1":[260,528,580,
            127,537,84,404,218,374,394,509,194,382,55,185],"e_supp":[1078,283,10,62,460,806,715,
            157,1096,849,503,996,533,1004,564,991,858,916]}"#,
        )
        .unwrap();
        assert!(!is_decoding_failure_absorbing(&df));
    }
}
