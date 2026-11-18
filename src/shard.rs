use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

use crate::spec::{LayerDecl, SpecV1, StepDecl, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{default_cases_from_spec, verify_specs};

#[derive(Debug, Clone, Serialize)]
pub struct ShardLayerSummary {
    pub original_layer: usize,
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: bool,
    pub original_weights: usize,
    pub num_shards: usize,
    pub shard_output_slices: Vec<[usize; 2]>,
    pub reassembly_exact: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ShardReport {
    pub ok: bool,
    pub sharded: bool,
    pub max_weights: usize,
    pub shards: Vec<ShardLayerSummary>,
    pub reassembly_exact: bool,
    pub verified: bool,
    pub max_abs_error: f64,
    pub sharded_sha256: String,
    pub spec: SpecV1,
}

struct ShardPlan {
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    blocks: Vec<[usize; 2]>,
}

fn slice_block_weights(
    w: &[f32],
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    r0: usize,
    r1: usize,
) -> Vec<f32> {
    let block_out = r1 - r0;
    let mut out = Vec::with_capacity(block_out * (in_dim + if bias { 1 } else { 0 }));
    for k in 0..in_dim {
        for i in r0..r1 {
            out.push(w[k * out_dim + i]);
        }
    }
    if bias {
        for i in r0..r1 {
            out.push(w[in_dim * out_dim + i]);
        }
    }
    out
}

fn verify_reassembly(
    orig: &[f32],
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    blocks: &[(usize, usize, Vec<f32>)],
) -> bool {
    let mut rebuilt = vec![0.0f32; orig.len()];
    for (r0, r1, bw) in blocks {
        let block_out = r1 - r0;
        for k in 0..in_dim {
            for li in 0..block_out {
                rebuilt[k * out_dim + (r0 + li)] = bw[k * block_out + li];
            }
        }
        if bias {
            for li in 0..block_out {
                rebuilt[in_dim * out_dim + (r0 + li)] = bw[in_dim * block_out + li];
            }
        }
    }
    rebuilt == orig
}

pub fn shard_spec(spec: &SpecV1, max_weights: usize) -> Result<ShardReport, String> {
    if max_weights == 0 {
        return Err("shard: max_weights must be >= 1".into());
    }
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }

    let mut plans: HashMap<usize, ShardPlan> = HashMap::new();
    for (li, decl) in spec.layers.iter().enumerate() {
        if decl.constructor != "linear" {
            continue;
        }
        let in_dim = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let out_dim = decl.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let bias = decl.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
        let expected_len = in_dim * out_dim + if bias { out_dim } else { 0 };
        let actual_len = spec
            .weights
            .get(&li.to_string())
            .map(|w| w.len())
            .unwrap_or(expected_len);

        if actual_len <= max_weights {
            continue;
        }
        let per_row = in_dim + if bias { 1 } else { 0 };
        let block_out = max_weights / per_row.max(1);
        if block_out == 0 {
            return Err(format!(
                "layer {li} unshardable: input dim {in_dim} (per_row={per_row}) exceeds max_weights={max_weights}"
            ));
        }
        let mut blocks = Vec::new();
        let mut r = 0usize;
        while r < out_dim {
            let r1 = (r + block_out).min(out_dim);
            blocks.push([r, r1]);
            r = r1;
        }
        if blocks.len() > 1 {
            plans.insert(
                li,
                ShardPlan {
                    in_dim,
                    out_dim,
                    bias,
                    blocks,
                },
            );
        }
    }

    if plans.is_empty() {
        return Ok(ShardReport {
            ok: true,
            sharded: false,
            max_weights,
            shards: vec![],
            reassembly_exact: true,
            verified: true,
            max_abs_error: 0.0,
            sharded_sha256: sha256_json(spec),
            spec: spec.clone(),
        });
    }

    let mut next_reg_id = spec
        .layers
        .iter()
        .filter_map(|l| l.args.first().and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut next_slot = spec.num_slots;
    let mut new_layers = Vec::new();
    let mut new_weights = std::collections::BTreeMap::new();
    let mut layer_remap = HashMap::new();

    for (li, decl) in spec.layers.iter().enumerate() {
        if plans.contains_key(&li) {
            continue;
        }
        let new_idx = new_layers.len();
        layer_remap.insert(li, new_idx);
        new_layers.push(decl.clone());
        if let Some(w) = spec.weights.get(&li.to_string()) {
            new_weights.insert(new_idx.to_string(), w.clone());
        }
    }

    let mut concat_layer_idx: Option<usize> = None;
    let mut shard_summaries = Vec::new();
    let mut all_reassembly_exact = true;
    let mut new_steps = Vec::new();

    for st in &spec.steps {
        let Some(plan) = plans.get(&st.layer) else {
            new_steps.push(StepDecl {
                kind: st.kind.clone(),
                layer: layer_remap[&st.layer],
                slots: st.slots.clone(),
            });
            continue;
        };

        let in_slot = st.slots[0];
        let out_slot = st.slots[1];
        let orig_w = spec
            .weights
            .get(&st.layer.to_string())
            .cloned()
            .unwrap_or_else(|| {
                vec![0.0f32; plan.in_dim * plan.out_dim + if plan.bias { plan.out_dim } else { 0 }]
            });

        let mut block_slots = Vec::with_capacity(plan.blocks.len());
        let mut reassembly_blocks = Vec::with_capacity(plan.blocks.len());

        for &[r0, r1] in &plan.blocks {
            let idx = new_layers.len();
            new_layers.push(LayerDecl {
                constructor: "linear".into(),
                args: vec![
                    Value::from(next_reg_id),
                    Value::from(plan.in_dim),
                    Value::from(r1 - r0),
                    Value::from(plan.bias),
                ],
            });
            next_reg_id += 1;

            let bw = slice_block_weights(&orig_w, plan.in_dim, plan.out_dim, plan.bias, r0, r1);
            new_weights.insert(idx.to_string(), bw.clone());
            reassembly_blocks.push((r0, r1, bw));

            let bslot = next_slot;
            next_slot += 1;
            block_slots.push(bslot);
            new_steps.push(StepDecl {
                kind: "unary".into(),
                layer: idx,
                slots: vec![in_slot, bslot],
            });
        }

        let cat_li = *concat_layer_idx.get_or_insert_with(|| {
            let idx = new_layers.len();
            new_layers.push(LayerDecl {
                constructor: "concat".into(),
                args: vec![Value::from(next_reg_id), Value::from(1)],
            });
            next_reg_id += 1;
            idx
        });

        let mut acc_slot = block_slots[0];
        for (bi, &next_block_slot) in block_slots.iter().enumerate().skip(1) {
            let is_last = bi + 1 == block_slots.len();
            let dst = if is_last {
                out_slot
            } else {
                let s = next_slot;
                next_slot += 1;
                s
            };
            new_steps.push(StepDecl {
                kind: "binary".into(),
                layer: cat_li,
                slots: vec![acc_slot, next_block_slot, dst],
            });
            acc_slot = dst;
        }

        let exact = verify_reassembly(&orig_w, plan.in_dim, plan.out_dim, plan.bias, &reassembly_blocks);
        if !exact {
            all_reassembly_exact = false;
        }
        shard_summaries.push(ShardLayerSummary {
            original_layer: st.layer,
            in_dim: plan.in_dim,
            out_dim: plan.out_dim,
            bias: plan.bias,
            original_weights: orig_w.len(),
            num_shards: plan.blocks.len(),
            shard_output_slices: plan.blocks.clone(),
            reassembly_exact: exact,
        });
    }

    let sharded = SpecV1 {
        num_slots: next_slot.max(spec.num_slots),
        layers: new_layers,
        steps: new_steps,
        output_slot: spec.output_slot,
        ports: spec.ports.clone(),
        weights: new_weights,
        inputs: spec.inputs.clone(),
    };

    let s_val = validate_spec(&sharded, false);
    if !s_val.valid {
        return Err(format!("sharded spec failed validation: {:?}", s_val.code));
    }

    let cases = default_cases_from_spec(spec);
    let ver = verify_specs(spec, &sharded, &cases, 1e-4, 1e-4);

    Ok(ShardReport {
        ok: all_reassembly_exact && ver.equivalent,
        sharded: true,
        max_weights,
        shards: shard_summaries,
        reassembly_exact: all_reassembly_exact,
        verified: ver.equivalent,
        max_abs_error: ver.max_abs_diff,
        sharded_sha256: sha256_json(&sharded),
        spec: sharded,
    })
}
