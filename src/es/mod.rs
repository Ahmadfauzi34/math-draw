pub mod optimizer;
pub mod search;
pub mod solve_gd;

pub use optimizer::*;
pub use search::*;
pub use solve_gd::*;

use serde::Serialize;
use std::collections::HashSet;

use crate::engine::get_layer_weight_len;
use crate::spec::SpecV1;

#[derive(Debug, Clone, Serialize)]
pub struct ActiveLayerSlice {
    pub layer_index: usize,
    pub constructor: String,
    pub offset: usize,
    pub length: usize,
}

fn default_weights_for_layer(spec: &SpecV1, layer_idx: usize) -> Vec<f32> {
    let decl = &spec.layers[layer_idx];
    let len = get_layer_weight_len(decl);
    if let Some(existing) = spec.weights.get(&layer_idx.to_string())
        && existing.len() == len
    {
        return existing.clone();
    }
    match decl.constructor.as_str() {
        "batchNorm" | "groupNorm" | "instanceNorm" | "layerNorm" => {
            let mut v = vec![0.0f32; len];
            for x in v.iter_mut().take(len / 2) {
                *x = 1.0;
            }
            v
        }
        "rmsNorm" => vec![1.0f32; len],
        "prelu" => {
            let alpha = decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.25) as f32;
            vec![alpha; len]
        }
        _ => vec![0.0f32; len],
    }
}

pub fn discover_trainable_slices(
    spec: &SpecV1,
    layer_filter: Option<usize>,
) -> Result<(Vec<ActiveLayerSlice>, Vec<f32>), String> {
    let used_layers: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    let mut slices = Vec::new();
    let mut flat = Vec::new();

    for li in 0..spec.layers.len() {
        if let Some(target_li) = layer_filter
            && li != target_li
        {
            continue;
        }
        if !used_layers.contains(&li) && layer_filter.is_none() {
            continue;
        }
        let decl = &spec.layers[li];
        let len = get_layer_weight_len(decl);
        if len == 0 || decl.constructor == "embedding" {
            continue;
        }
        let w = default_weights_for_layer(spec, li);
        slices.push(ActiveLayerSlice {
            layer_index: li,
            constructor: decl.constructor.clone(),
            offset: flat.len(),
            length: len,
        });
        flat.extend(w);
    }

    if slices.is_empty() {
        if let Some(idx) = layer_filter {
            return Err(format!("layer {idx} has no differentiable trainable weights"));
        }
        return Err("spec has no active trainable weighted layers".into());
    }
    Ok((slices, flat))
}

pub fn apply_flat_weights(spec: &SpecV1, slices: &[ActiveLayerSlice], flat: &[f32]) -> SpecV1 {
    let mut s = spec.clone();
    for sl in slices {
        s.weights.insert(
            sl.layer_index.to_string(),
            flat[sl.offset..sl.offset + sl.length].to_vec(),
        );
    }
    s
}
