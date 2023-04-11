use bike_decoder::{
    decoder::{find_bgf_cycle, DecoderCycle, DecodingFailure, DecodingResult},
    graphs::{
        is_absorbing_subgraph, odd_check_node_neighbors, AbsorbingDecodingResult, CheckNode,
        TannerGraphEdges,
    },
    keys::QuasiCyclic,
    ncw::NcwOverlaps,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT, GRAY_THRESHOLD_DIFF},
    random::{global_seed, Seed},
    threshold::{bf_masked_threshold, bf_threshold_min},
    vectors::Index,
};
use serde::{Deserialize, Serialize};
use serde_with::{formats::Flexible, serde_as, skip_serializing_none, DurationSecondsWithFrac};
use std::time::Duration;

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalysisRecord<const WT: usize, const LEN: usize> {
    r: usize,
    d: usize,
    weight: usize,
    gray_threshold_diff: u8,
    bf_threshold_min: u8,
    bf_masked_threshold: u8,
    fixed_key: Option<QuasiCyclic<WT, LEN>>,
    num_processed: usize,
    data: Vec<AnalysisResult<WT, LEN>>,
    seed: Option<Seed>,
    #[serde_as(as = "DurationSecondsWithFrac<f64, Flexible>")]
    runtime: Duration,
}

impl<const WT: usize, const LEN: usize> AnalysisRecord<WT, LEN> {
    pub fn new(
        fixed_key: Option<QuasiCyclic<WT, LEN>>,
        weight: usize,
        num_processed: usize,
        data: Vec<AnalysisResult<WT, LEN>>,
        runtime: Duration,
    ) -> Self {
        Self {
            r: LEN,
            d: WT,
            weight,
            gray_threshold_diff: GRAY_THRESHOLD_DIFF,
            bf_threshold_min: bf_threshold_min(WT),
            bf_masked_threshold: bf_masked_threshold(WT),
            fixed_key,
            num_processed,
            data,
            seed: global_seed(),
            runtime,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnalysisResult<const WT: usize = BLOCK_WEIGHT, const LEN: usize = BLOCK_LENGTH> {
    DecodingFailure(DecodingFailureAnalysis),
    Sample(SampleAnalysis<WT, LEN>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodingFailureAnalysis {
    #[serde(flatten)]
    decoder_cycle: DecoderCycle,
    diff: RawSupportAnalysis,
    e_in_data: RawSupportAnalysis,
}

impl DecodingFailureAnalysis {
    pub fn new(df: &DecodingFailure, max_iters: usize) -> Self {
        let decoder_cycle = find_bgf_cycle(df.key(), df.vector().vector(), max_iters);
        let diff = RawSupportAnalysis::new(decoder_cycle.diff());
        let e_in_data = RawSupportAnalysis::new(decoder_cycle.e_in().support().to_vec());
        Self {
            decoder_cycle,
            diff,
            e_in_data,
        }
    }

    pub fn compute_overlaps_diff(&mut self) {
        self.diff
            .ncw_overlaps
            .get_or_insert_with(|| NcwOverlaps::new(self.decoder_cycle.key(), &self.diff.supp));
    }

    pub fn compute_overlaps_e_in(&mut self) {
        self.e_in_data.ncw_overlaps.get_or_insert_with(|| {
            NcwOverlaps::new(self.decoder_cycle.key(), &self.e_in_data.supp)
        });
    }

    pub fn compute_absorbing(&mut self) {
        if self.diff.is_absorbing.is_some() {
            return;
        }
        let decoder_result = DecodingResult::from(self.decoder_cycle.clone());
        if let Some(absorbing_data) = AbsorbingDecodingResult::new(decoder_result) {
            self.diff.is_absorbing = Some(true);
            let odd_check_nodes = absorbing_data.take_odd_check_nodes();
            let b = odd_check_nodes.len();
            self.diff.odd_check_nodes = Some(odd_check_nodes);
            self.diff.absorbing = Some(AbsorbingParameters {
                a: self.diff.supp.len(),
                b,
            });
        } else {
            self.diff.is_absorbing = Some(false);
            self.diff.absorbing = None;
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleAnalysis<const WT: usize, const LEN: usize> {
    key: QuasiCyclic<WT, LEN>,
    #[serde(flatten)]
    sample_data: RawSupportAnalysis,
    #[serde(skip)]
    graph_edges: Option<TannerGraphEdges<WT, LEN>>,
}

impl<const WT: usize, const LEN: usize> SampleAnalysis<WT, LEN> {
    pub fn new(key: QuasiCyclic<WT, LEN>, supp: Vec<Index>) -> Self {
        Self {
            key,
            sample_data: RawSupportAnalysis {
                supp,
                ncw_overlaps: None,
                is_absorbing: None,
                absorbing: None,
                odd_check_nodes: None,
            },
            graph_edges: None,
        }
    }

    pub fn with_edges(
        key: QuasiCyclic<WT, LEN>,
        supp: Vec<Index>,
        edges: TannerGraphEdges<WT, LEN>,
    ) -> Self {
        let mut data = Self::new(key, supp);
        data.graph_edges = Some(edges);
        data
    }

    pub fn compute_overlaps(&mut self) {
        self.sample_data
            .ncw_overlaps
            .get_or_insert_with(|| NcwOverlaps::new(&self.key, &self.sample_data.supp));
    }

    pub fn compute_absorbing(&mut self) {
        if self.sample_data.is_absorbing.is_some() {
            return;
        }
        self.graph_edges
            .get_or_insert_with(|| TannerGraphEdges::new(&self.key));
        let edges = self.graph_edges.as_ref().unwrap();
        let supp = &self.sample_data.supp;
        if is_absorbing_subgraph(edges, supp) {
            self.sample_data.is_absorbing = Some(true);
            let (_, _, odd_check_nodes) = odd_check_node_neighbors(edges, supp);
            let b = odd_check_nodes.len();
            self.sample_data.odd_check_nodes = Some(odd_check_nodes);
            self.sample_data.absorbing = Some(AbsorbingParameters { a: supp.len(), b });
        } else {
            self.sample_data.is_absorbing = Some(false);
            self.sample_data.absorbing = None;
        }
    }
}

#[skip_serializing_none]
#[serde_with::apply(Option => #[serde(default)])]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RawSupportAnalysis {
    supp: Vec<Index>,
    ncw_overlaps: Option<NcwOverlaps>,
    is_absorbing: Option<bool>,
    absorbing: Option<AbsorbingParameters>,
    odd_check_nodes: Option<Vec<CheckNode>>,
}

impl RawSupportAnalysis {
    pub fn new(supp: Vec<Index>) -> Self {
        Self {
            supp,
            ncw_overlaps: None,
            is_absorbing: None,
            absorbing: None,
            odd_check_nodes: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AbsorbingParameters {
    a: usize,
    b: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serde_raw_support_analysis() {
        let value = json!({
            "supp": [1, 2, 3, 4],
            "ncw_overlaps": {"c": 4, "n": 8, "2n": 11},
            "is_absorbing": true,
            "absorbing": {"a": 4, "b": 5},
            "odd_check_nodes": [10, 11, 12, 13, 14]
        });
        let data: RawSupportAnalysis = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(data).unwrap(), value);
    }
}
