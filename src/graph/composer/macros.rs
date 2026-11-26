use serde_json::{Map, Value};
use std::collections::BTreeMap;

use crate::spec::{LayerDecl, StepDecl, parse_f32_values};

use super::{CompileState, ExpansionEvent};

pub(super) fn is_builtin_macro(name: &str) -> bool {
    matches!(
        name,
        "mlp"
            | "gated_unit"
            | "residual_mlp"
            | "transformer_ffn"
            | "recurrent_cell"
            | "attention_head"
    )
}

fn need_usize(params: &Map<String, Value>, key: &str) -> Result<usize, String> {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("macro param '{key}' (positive integer) is required"))
}

pub(super) fn expand_builtin_macro(
    macro_name: &str,
    params: &Map<String, Value>,
    step_weights: Option<&Value>,
    in_slots: &[usize],
    out_slot: usize,
    state: &mut CompileState,
    prefix: &str,
) -> Result<(), String> {
    let base_layer_idx = state.layers.len();
    let mut local_layers = Vec::new();
    let mut local_steps = Vec::new();

    match macro_name {
        "mlp" => {
            if in_slots.is_empty() {
                return Err("macro 'mlp' requires 1 input".into());
            }
            let in_dim = need_usize(params, "in_dim")?;
            let hid = need_usize(params, "hidden_dim")?;
            let out_dim = need_usize(params, "out_dim")?;
            let bias = params.get("bias").and_then(|v| v.as_bool()).unwrap_or(true);
            let s_in = in_slots[0];
            let s_h1 = state.alloc.fresh_slot();
            let s_h2 = state.alloc.fresh_slot();
            let l0 = state.alloc.fresh_layer_id();
            let l1 = state.alloc.fresh_layer_id();
            let l2 = state.alloc.fresh_layer_id();
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l0.into(), in_dim.into(), hid.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "relu".into(), args: vec![l1.into()] });
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l2.into(), hid.into(), out_dim.into(), bias.into()] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_in, s_h1] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_h1, s_h2] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_h2, out_slot] });
        }
        "gated_unit" => {
            if in_slots.is_empty() {
                return Err("macro 'gated_unit' requires 1 input".into());
            }
            let in_dim = need_usize(params, "in_dim")?;
            let dim = need_usize(params, "dim")?;
            let bias = params.get("bias").and_then(|v| v.as_bool()).unwrap_or(true);
            let s_in = in_slots[0];
            let s_a1 = state.alloc.fresh_slot();
            let s_a2 = state.alloc.fresh_slot();
            let s_b = state.alloc.fresh_slot();
            let l0 = state.alloc.fresh_layer_id();
            let l1 = state.alloc.fresh_layer_id();
            let l2 = state.alloc.fresh_layer_id();
            let l3 = state.alloc.fresh_layer_id();
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l0.into(), in_dim.into(), dim.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "sigmoid".into(), args: vec![l1.into()] });
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l2.into(), in_dim.into(), dim.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "mul".into(), args: vec![l3.into()] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_in, s_a1] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_a1, s_a2] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_in, s_b] });
            local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 3, slots: vec![s_a2, s_b, out_slot] });
        }
        "residual_mlp" => {
            if in_slots.is_empty() {
                return Err("macro 'residual_mlp' requires 1 input".into());
            }
            let in_dim = need_usize(params, "in_dim")?;
            let hid = need_usize(params, "hidden_dim")?;
            let bias = params.get("bias").and_then(|v| v.as_bool()).unwrap_or(true);
            let s_in = in_slots[0];
            let s_h1 = state.alloc.fresh_slot();
            let s_h2 = state.alloc.fresh_slot();
            let s_h3 = state.alloc.fresh_slot();
            let l0 = state.alloc.fresh_layer_id();
            let l1 = state.alloc.fresh_layer_id();
            let l2 = state.alloc.fresh_layer_id();
            let l3 = state.alloc.fresh_layer_id();
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l0.into(), in_dim.into(), hid.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "relu".into(), args: vec![l1.into()] });
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l2.into(), hid.into(), in_dim.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "add".into(), args: vec![l3.into()] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_in, s_h1] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_h1, s_h2] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_h2, s_h3] });
            local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 3, slots: vec![s_h3, s_in, out_slot] });
        }
        "transformer_ffn" => {
            if in_slots.is_empty() {
                return Err("macro 'transformer_ffn' requires 1 input".into());
            }
            let d = need_usize(params, "d_model")?;
            let d_ff = params.get("d_ff").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(4 * d);
            let bias = params.get("bias").and_then(|v| v.as_bool()).unwrap_or(true);
            let s_in = in_slots[0];
            let s_h1 = state.alloc.fresh_slot();
            let s_h2 = state.alloc.fresh_slot();
            let s_h3 = state.alloc.fresh_slot();
            let l0 = state.alloc.fresh_layer_id();
            let l1 = state.alloc.fresh_layer_id();
            let l2 = state.alloc.fresh_layer_id();
            let l3 = state.alloc.fresh_layer_id();
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l0.into(), d.into(), d_ff.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "gelu".into(), args: vec![l1.into()] });
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l2.into(), d_ff.into(), d.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "add".into(), args: vec![l3.into()] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_in, s_h1] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_h1, s_h2] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_h2, s_h3] });
            local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 3, slots: vec![s_h3, s_in, out_slot] });
        }
        "recurrent_cell" => {
            if in_slots.len() < 2 {
                return Err("macro 'recurrent_cell' requires 2 inputs (state, input)".into());
            }
            let in_dim = need_usize(params, "in_dim")?;
            let state_dim = need_usize(params, "state_dim")?;
            let bias = params.get("bias").and_then(|v| v.as_bool()).unwrap_or(true);
            let s_state = in_slots[0];
            let s_in = in_slots[1];
            let s_ws = state.alloc.fresh_slot();
            let s_wx = state.alloc.fresh_slot();
            let s_sum = state.alloc.fresh_slot();
            let s_act = state.alloc.fresh_slot();
            let l0 = state.alloc.fresh_layer_id();
            let l1 = state.alloc.fresh_layer_id();
            let l2 = state.alloc.fresh_layer_id();
            let l3 = state.alloc.fresh_layer_id();
            let l4 = state.alloc.fresh_layer_id();
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l0.into(), state_dim.into(), state_dim.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "linear".into(), args: vec![l1.into(), in_dim.into(), state_dim.into(), bias.into()] });
            local_layers.push(LayerDecl { constructor: "add".into(), args: vec![l2.into()] });
            local_layers.push(LayerDecl { constructor: "tanh".into(), args: vec![l3.into()] });
            local_layers.push(LayerDecl { constructor: "add".into(), args: vec![l4.into()] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_state, s_ws] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_in, s_wx] });
            local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 2, slots: vec![s_ws, s_wx, s_sum] });
            local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 3, slots: vec![s_sum, s_act] });
            local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 4, slots: vec![s_act, s_state, out_slot] });
        }
        "attention_head" => {
            if in_slots.len() < 3 {
                return Err("macro 'attention_head' requires 3 inputs (Q, K, V)".into());
            }
            let _d_model = need_usize(params, "d_model")?;
            let layout = params
                .get("layout")
                .and_then(|v| v.as_str())
                .unwrap_or("seq_axis1");
            let (s_q, s_k, s_v) = (in_slots[0], in_slots[1], in_slots[2]);
            match layout {
                "heads_axis1" => {
                    // [B, H, S, D]: K^T = transpose(2,3)(K); scores = Q @ K^T; W = softmax(scores, dim=3); out = W @ V
                    let s_tk = state.alloc.fresh_slot();
                    let s_scores = state.alloc.fresh_slot();
                    let s_w = state.alloc.fresh_slot();
                    let l0 = state.alloc.fresh_layer_id();
                    let l1 = state.alloc.fresh_layer_id();
                    let l2 = state.alloc.fresh_layer_id();
                    let l3 = state.alloc.fresh_layer_id();
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![l0.into(), 2usize.into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "matmul".into(), args: vec![l1.into()] });
                    local_layers.push(LayerDecl { constructor: "softmax".into(), args: vec![l2.into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "matmul".into(), args: vec![l3.into()] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_k, s_tk] });
                    local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 1, slots: vec![s_q, s_tk, s_scores] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_scores, s_w] });
                    local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 3, slots: vec![s_w, s_v, out_slot] });
                }
                "seq_axis1" => {
                    // [B, S, D, 1] -> [B, 1, S, D], attention over axis 3, transpose back to [B, Sq, Dv, 1]
                    let s_tq1 = state.alloc.fresh_slot();
                    let s_tq2 = state.alloc.fresh_slot();
                    let s_tk = state.alloc.fresh_slot();
                    let s_scores = state.alloc.fresh_slot();
                    let s_w = state.alloc.fresh_slot();
                    let s_tv1 = state.alloc.fresh_slot();
                    let s_tv2 = state.alloc.fresh_slot();
                    let s_o = state.alloc.fresh_slot();
                    let s_to = state.alloc.fresh_slot();
                    let lids: Vec<usize> = (0..10).map(|_| state.alloc.fresh_layer_id()).collect();
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[0].into(), 2usize.into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[1].into(), 1usize.into(), 2usize.into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[2].into(), 1usize.into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "matmul".into(), args: vec![lids[3].into()] });
                    local_layers.push(LayerDecl { constructor: "softmax".into(), args: vec![lids[4].into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[5].into(), 2usize.into(), 3usize.into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[6].into(), 1usize.into(), 2usize.into()] });
                    local_layers.push(LayerDecl { constructor: "matmul".into(), args: vec![lids[7].into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[8].into(), 1usize.into(), 2usize.into()] });
                    local_layers.push(LayerDecl { constructor: "transpose".into(), args: vec![lids[9].into(), 2usize.into(), 3usize.into()] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx, slots: vec![s_q, s_tq1] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 1, slots: vec![s_tq1, s_tq2] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 2, slots: vec![s_k, s_tk] });
                    local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 3, slots: vec![s_tq2, s_tk, s_scores] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 4, slots: vec![s_scores, s_w] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 5, slots: vec![s_v, s_tv1] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 6, slots: vec![s_tv1, s_tv2] });
                    local_steps.push(StepDecl { kind: "binary".into(), layer: base_layer_idx + 7, slots: vec![s_w, s_tv2, s_o] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 8, slots: vec![s_o, s_to] });
                    local_steps.push(StepDecl { kind: "unary".into(), layer: base_layer_idx + 9, slots: vec![s_to, out_slot] });
                }
                other => {
                    return Err(format!(
                        "attention_head layout must be 'seq_axis1' or 'heads_axis1', got '{other}'"
                    ));
                }
            }
        }
        _ => {
            return Err(format!("unknown built-in macro '{macro_name}'"));
        }
    }

    state.layers.extend(local_layers);
    state.steps.extend(local_steps);

    if let Some(obj) = step_weights.and_then(|v| v.as_object()) {
        let mut parsed: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (k, v) in obj {
            if let Ok(local_idx) = k.parse::<usize>()
                && let Some(arr) = v.as_array()
            {
                let floats = parse_f32_values(arr, &format!("{prefix}:{macro_name}.weights[{k}]"))?;
                parsed.insert((base_layer_idx + local_idx).to_string(), floats);
            }
        }
        state.weights.extend(parsed);
    }

    state.events.push(ExpansionEvent {
        kind: "builtin_macro".to_string(),
        target: macro_name.to_string(),
        prefix: prefix.to_string(),
        depth: None,
        tie_weights: None,
    });
    Ok(())
}
