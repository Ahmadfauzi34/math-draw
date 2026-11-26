use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

use crate::graph::composer::ExpansionReceipt;
use crate::graph::liveness::{auto_repair_spec, compact_slots_by_liveness, ssa_normalize_spec};
use crate::graph::validate::validate_spec;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl, is_canonical_uint_key};

#[derive(Debug, Clone)]
pub struct GradSpecOptions {
    pub wrt: String,
    pub order: usize,
    pub dout: Option<Vec<f32>>,
    pub hvp_vec: Option<Vec<f32>>,
    pub compact_slots: bool,
}

impl Default for GradSpecOptions {
    fn default() -> Self {
        Self {
            wrt: "0".to_string(),
            order: 1,
            dout: None,
            hvp_vec: None,
            compact_slots: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GradSpecReport {
    pub ok: bool,
    pub kind: &'static str,
    pub order: usize,
    pub wrt_slot: usize,
    pub dout_slot: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hvp_slot: Option<usize>,
    pub forward_output_slot: usize,
    pub layers_count: usize,
    pub steps_count: usize,
    pub num_slots: usize,
    pub output_slot: usize,
    pub spec: SpecV1,
}

fn transpose_col_major(weights: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; in_dim * out_dim];
    for j in 0..in_dim {
        for i in 0..out_dim {
            out[i * in_dim + j] = weights[j * out_dim + i];
        }
    }
    out
}

fn one_minus_weights(dim: usize) -> Vec<f32> {
    let mut w = Vec::with_capacity(dim * dim + dim);
    for j in 0..dim {
        for i in 0..dim {
            w.push(if i == j { -1.0 } else { 0.0 });
        }
    }
    w.extend(std::iter::repeat_n(1.0, dim));
    w
}

fn scaled_identity_weights(dim: usize, scale: f32) -> Vec<f32> {
    let mut w = Vec::with_capacity(dim * dim);
    for j in 0..dim {
        for i in 0..dim {
            w.push(if i == j { scale } else { 0.0 });
        }
    }
    w
}

fn affine_identity_weights(dim: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut w = Vec::with_capacity(dim * dim + dim);
    for j in 0..dim {
        for i in 0..dim {
            w.push(if i == j { scale } else { 0.0 });
        }
    }
    w.extend(std::iter::repeat_n(bias, dim));
    w
}

fn resolve_wrt_slot(
    spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    wrt_ref: &str,
) -> Result<usize, String> {
    if is_canonical_uint_key(wrt_ref) {
        let slot: usize = wrt_ref
            .parse()
            .map_err(|_| format!("invalid --wrt slot '{wrt_ref}'"))?;
        if slot >= spec.num_slots {
            return Err(format!(
                "unresolved_wrt_slot: slot {slot} out of range 0..{}",
                spec.num_slots.saturating_sub(1)
            ));
        }
        return Ok(slot);
    }
    if let Some(exp) = expansion
        && let Some(&slot) = exp.port_names.get(wrt_ref)
    {
        return Ok(slot);
    }
    Err(format!(
        "unresolved_wrt_slot: cannot resolve --wrt '{wrt_ref}' to a slot index or named port"
    ))
}

fn f32_slice_to_json(vals: &[f32]) -> Vec<Value> {
    vals.iter().map(|&v| Value::from(v as f64)).collect()
}

fn validate_seed_vec(name: &str, vals: &[f32], expected_len: usize) -> Result<(), String> {
    if vals.len() != expected_len {
        return Err(format!(
            "{name} length mismatch: expected {expected_len} floats, got {}",
            vals.len()
        ));
    }
    for (i, &v) in vals.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{name}[{i}] must be finite, got {v}"));
        }
    }
    Ok(())
}

/// Compile a forward `SpecV1` into a standalone runnable 1st-order VJP `SpecV1`
/// computing `d(forward_output · dout) / d(wrt_slot)`.
pub fn compile_backward_spec(
    forward_spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    wrt_ref: &str,
    dout_values: Option<&[f32]>,
) -> Result<GradSpecReport, String> {
    let val_in = validate_spec(forward_spec, false);
    if !val_in.valid {
        return Err(format!(
            "cannot differentiate invalid forward spec ({})",
            val_in.code
        ));
    }
    let resolved_wrt = resolve_wrt_slot(forward_spec, expansion, wrt_ref)?;

    // SSA-normalize first so compacted/reused slots never clobber forward activations.
    let ssa_rep = ssa_normalize_spec(forward_spec);
    let s = ssa_rep.ssa_spec;
    let final_slot_map = ssa_rep.final_slot_map;

    let explain_rep = validate_spec(&s, true);
    if !explain_rep.valid {
        return Err(format!(
            "cannot differentiate SSA-normalized spec ({})",
            explain_rep.code
        ));
    }
    let slot_shapes: BTreeMap<usize, [usize; 4]> = explain_rep
        .explain
        .as_ref()
        .map(|e| e.slot_shapes.clone())
        .unwrap_or_default();
    let Some(&out_shape) = slot_shapes.get(&s.output_slot) else {
        return Err(format!(
            "unknown_output_shape: cannot infer shape for outputSlot {}",
            s.output_slot
        ));
    };
    let out_numel: usize = out_shape.iter().product();
    if let Some(dv) = dout_values {
        validate_seed_vec("dout", dv, out_numel)?;
    }

    let mut next_reg_id = s
        .layers
        .iter()
        .filter_map(|l| l.args.first().and_then(|v| v.as_u64()))
        .max()
        .map(|m| (m as usize) + 1)
        .unwrap_or(0);
    let mut next_slot = s.num_slots;

    let mut layers = s.layers.clone();
    let mut steps = s.steps.clone();
    let mut weights = s.weights.clone();
    let mut ports = s.ports.clone();
    let mut inputs = s.inputs.clone();

    let dout_slot = next_slot;
    next_slot += 1;
    let dout_layout = if out_shape[2] == 1 && out_shape[3] == 1 {
        "feature_axis1_singleton"
    } else {
        "any_rank4"
    };
    ports.push(PortDecl {
        slot: dout_slot,
        role: "context".to_string(),
        shape: out_shape,
        layout: dout_layout.to_string(),
    });
    let dout_vec: Vec<f32> = match dout_values {
        Some(v) => v.to_vec(),
        None => vec![1.0f32; out_numel],
    };
    inputs.insert(dout_slot.to_string(), f32_slice_to_json(&dout_vec));

    let mut grad_slot_of: HashMap<usize, usize> = HashMap::new();
    let mut saved_grad_slot_of: HashMap<usize, usize> = HashMap::new();
    grad_slot_of.insert(s.output_slot, dout_slot);
    saved_grad_slot_of.insert(s.output_slot, dout_slot);

    let push_layer = |layers: &mut Vec<LayerDecl>,
                      next_reg_id: &mut usize,
                      ctor: &str,
                      extra_args: Vec<Value>|
     -> usize {
        let idx = layers.len();
        let rid = *next_reg_id;
        *next_reg_id += 1;
        let mut args = Vec::with_capacity(1 + extra_args.len());
        args.push(Value::from(rid));
        args.extend(extra_args);
        layers.push(LayerDecl {
            constructor: ctor.to_string(),
            args,
        });
        idx
    };

    let accumulate_grad = |target_fwd: usize,
                           contrib: usize,
                           grad_slot_of: &mut HashMap<usize, usize>,
                           saved_grad_slot_of: &mut HashMap<usize, usize>,
                           layers: &mut Vec<LayerDecl>,
                           steps: &mut Vec<StepDecl>,
                           next_reg_id: &mut usize,
                           next_slot: &mut usize| {
        if let Some(&prev) = grad_slot_of.get(&target_fwd) {
            let sum_slot = *next_slot;
            *next_slot += 1;
            let add_idx = push_layer(layers, next_reg_id, "add", vec![]);
            steps.push(StepDecl {
                kind: "binary".to_string(),
                layer: add_idx,
                slots: vec![prev, contrib, sum_slot],
            });
            grad_slot_of.insert(target_fwd, sum_slot);
            saved_grad_slot_of.insert(target_fwd, sum_slot);
        } else {
            grad_slot_of.insert(target_fwd, contrib);
            saved_grad_slot_of.insert(target_fwd, contrib);
        }
    };

    let emit_neg = |in_slot: usize,
                    layers: &mut Vec<LayerDecl>,
                    steps: &mut Vec<StepDecl>,
                    next_reg_id: &mut usize,
                    next_slot: &mut usize|
     -> usize {
        let sub_idx = push_layer(layers, next_reg_id, "sub", vec![]);
        let zero_slot = *next_slot;
        *next_slot += 1;
        let neg_slot = *next_slot;
        *next_slot += 1;
        steps.push(StepDecl {
            kind: "binary".to_string(),
            layer: sub_idx,
            slots: vec![in_slot, in_slot, zero_slot],
        });
        steps.push(StepDecl {
            kind: "binary".to_string(),
            layer: sub_idx,
            slots: vec![zero_slot, in_slot, neg_slot],
        });
        neg_slot
    };

    let emit_one_minus = |in_slot: usize,
                          sh: [usize; 4],
                          op_name: &str,
                          layers: &mut Vec<LayerDecl>,
                          steps: &mut Vec<StepDecl>,
                          weights: &mut BTreeMap<String, Vec<f32>>,
                          next_reg_id: &mut usize,
                          next_slot: &mut usize|
     -> Result<usize, String> {
        if sh[2] != 1 || sh[3] != 1 {
            return Err(format!(
                "unsupported_shape_for_polynomial_vjp: in-graph {op_name} VJP requires [B, D, 1, 1], got {sh:?}"
            ));
        }
        let dim = sh[1];
        let lin_idx = push_layer(
            layers,
            next_reg_id,
            "linear",
            vec![Value::from(dim), Value::from(dim), Value::from(true)],
        );
        weights.insert(lin_idx.to_string(), one_minus_weights(dim));
        let res_slot = *next_slot;
        *next_slot += 1;
        steps.push(StepDecl {
            kind: "unary".to_string(),
            layer: lin_idx,
            slots: vec![in_slot, res_slot],
        });
        Ok(res_slot)
    };

    let emit_scale = |in_slot: usize,
                      sh: [usize; 4],
                      scale: f32,
                      op_name: &str,
                      layers: &mut Vec<LayerDecl>,
                      steps: &mut Vec<StepDecl>,
                      weights: &mut BTreeMap<String, Vec<f32>>,
                      next_reg_id: &mut usize,
                      next_slot: &mut usize|
     -> Result<usize, String> {
        if sh[2] != 1 || sh[3] != 1 {
            return Err(format!(
                "unsupported_shape_for_polynomial_vjp: in-graph {op_name} VJP requires [B, D, 1, 1], got {sh:?}"
            ));
        }
        let dim = sh[1];
        let lin_idx = push_layer(
            layers,
            next_reg_id,
            "linear",
            vec![Value::from(dim), Value::from(dim), Value::from(false)],
        );
        weights.insert(lin_idx.to_string(), scaled_identity_weights(dim, scale));
        let res_slot = *next_slot;
        *next_slot += 1;
        steps.push(StepDecl {
            kind: "unary".to_string(),
            layer: lin_idx,
            slots: vec![in_slot, res_slot],
        });
        Ok(res_slot)
    };

    let emit_affine = |in_slot: usize,
                       sh: [usize; 4],
                       scale: f32,
                       bias: f32,
                       op_name: &str,
                       layers: &mut Vec<LayerDecl>,
                       steps: &mut Vec<StepDecl>,
                       weights: &mut BTreeMap<String, Vec<f32>>,
                       next_reg_id: &mut usize,
                       next_slot: &mut usize|
     -> Result<usize, String> {
        if sh[2] != 1 || sh[3] != 1 {
            return Err(format!(
                "unsupported_shape_for_polynomial_vjp: in-graph {op_name} VJP requires [B, D, 1, 1], got {sh:?}"
            ));
        }
        let dim = sh[1];
        let lin_idx = push_layer(
            layers,
            next_reg_id,
            "linear",
            vec![Value::from(dim), Value::from(dim), Value::from(true)],
        );
        weights.insert(lin_idx.to_string(), affine_identity_weights(dim, scale, bias));
        let res_slot = *next_slot;
        *next_slot += 1;
        steps.push(StepDecl {
            kind: "unary".to_string(),
            layer: lin_idx,
            slots: vec![in_slot, res_slot],
        });
        Ok(res_slot)
    };

    for si in (0..s.steps.len()).rev() {
        let st = &s.steps[si];
        let decl = &s.layers[st.layer];
        let ctor = decl.constructor.as_str();
        let Some(out_slot) = st.output_slot() else {
            continue;
        };
        let Some(up_grad) = grad_slot_of.remove(&out_slot) else {
            continue;
        };
        saved_grad_slot_of.insert(out_slot, up_grad);

        match ctor {
            "linear" => {
                let in_dim = decl.args[1].as_u64().unwrap_or(0) as usize;
                let out_dim = decl.args[2].as_u64().unwrap_or(0) as usize;
                let bias = decl.args[3].as_bool().unwrap_or(false);
                let exp_len = in_dim * out_dim + if bias { out_dim } else { 0 };
                let w = s
                    .weights
                    .get(&st.layer.to_string())
                    .cloned()
                    .unwrap_or_else(|| vec![0.0f32; exp_len]);
                let w_t = transpose_col_major(&w, in_dim, out_dim);
                let lin_t_idx = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "linear",
                    vec![Value::from(out_dim), Value::from(in_dim), Value::from(false)],
                );
                weights.insert(lin_t_idx.to_string(), w_t);
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: lin_t_idx,
                    slots: vec![up_grad, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "add" => {
                accumulate_grad(
                    st.slots[0],
                    up_grad,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    st.slots[1],
                    up_grad,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "sub" => {
                accumulate_grad(
                    st.slots[0],
                    up_grad,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                let neg = emit_neg(
                    up_grad,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    st.slots[1],
                    neg,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "mul" => {
                let (a_slot, b_slot) = (st.slots[0], st.slots[1]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let da_slot = next_slot;
                next_slot += 1;
                let db_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, b_slot, da_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, a_slot, db_slot],
                });
                accumulate_grad(
                    a_slot,
                    da_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    b_slot,
                    db_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "fma" => {
                // y = a * b + c => dA = G * b, dB = G * a, dC = G
                let (a_slot, b_slot, c_slot) = (st.slots[0], st.slots[1], st.slots[2]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let da_slot = next_slot;
                next_slot += 1;
                let db_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, b_slot, da_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, a_slot, db_slot],
                });
                accumulate_grad(
                    a_slot,
                    da_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    b_slot,
                    db_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    c_slot,
                    up_grad,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "div" => {
                // y = a / b => dA = G / b, dB = -(G * y) / b
                let (a_slot, b_slot) = (st.slots[0], st.slots[1]);
                let div_idx = push_layer(&mut layers, &mut next_reg_id, "div", vec![]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let da_slot = next_slot;
                next_slot += 1;
                let gy_slot = next_slot;
                next_slot += 1;
                let pos_db_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: div_idx,
                    slots: vec![up_grad, b_slot, da_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, out_slot, gy_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: div_idx,
                    slots: vec![gy_slot, b_slot, pos_db_slot],
                });
                let db_slot = emit_neg(
                    pos_db_slot,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    a_slot,
                    da_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    b_slot,
                    db_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "sqrt" => {
                // y = sqrt(x) => dX = G / (y + y)
                let add_idx = push_layer(&mut layers, &mut next_reg_id, "add", vec![]);
                let div_idx = push_layer(&mut layers, &mut next_reg_id, "div", vec![]);
                let two_y = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![out_slot, out_slot, two_y],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: div_idx,
                    slots: vec![up_grad, two_y, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "exp" => {
                // y = exp(x) => dX = G * y
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, out_slot, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "log" => {
                // y = log(x) => dX = G / x
                let div_idx = push_layer(&mut layers, &mut next_reg_id, "div", vec![]);
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: div_idx,
                    slots: vec![up_grad, st.slots[0], dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "sin" => {
                // y = sin(x) => dX = G * cos(x)
                let cos_idx = push_layer(&mut layers, &mut next_reg_id, "cos", vec![]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let cos_x = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: cos_idx,
                    slots: vec![st.slots[0], cos_x],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, cos_x, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "cos" => {
                // y = cos(x) => dX = -(G * sin(x))
                let sin_idx = push_layer(&mut layers, &mut next_reg_id, "sin", vec![]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let sin_x = next_slot;
                next_slot += 1;
                let pos_dx = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sin_idx,
                    slots: vec![st.slots[0], sin_x],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, sin_x, pos_dx],
                });
                let dx_slot = emit_neg(
                    pos_dx,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "sigmoid" => {
                let sh = *slot_shapes
                    .get(&out_slot)
                    .ok_or_else(|| format!("missing inferred shape for slot {out_slot}"))?;
                let one_minus_y = emit_one_minus(
                    out_slot,
                    sh,
                    "sigmoid",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let y_om = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![out_slot, one_minus_y, y_om],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, y_om, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "tanh" => {
                let sh = *slot_shapes
                    .get(&out_slot)
                    .ok_or_else(|| format!("missing inferred shape for slot {out_slot}"))?;
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let y_sq = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![out_slot, out_slot, y_sq],
                });
                let one_minus_ysq = emit_one_minus(
                    y_sq,
                    sh,
                    "tanh",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, one_minus_ysq, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "mish" => {
                // y = x * tanh(softplus(x))
                // dy/dx = tanh(sp) + x * (1 - tanh^2(sp)) * sigmoid(x)
                let sh = *slot_shapes
                    .get(&out_slot)
                    .ok_or_else(|| format!("missing inferred shape for slot {out_slot}"))?;
                let x_slot = st.slots[0];
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let sp_idx =
                    push_layer(&mut layers, &mut next_reg_id, "softplus", vec![Value::from(1.0)]);
                let th_idx = push_layer(&mut layers, &mut next_reg_id, "tanh", vec![]);
                let sig_idx = push_layer(&mut layers, &mut next_reg_id, "sigmoid", vec![]);
                let add_idx = push_layer(&mut layers, &mut next_reg_id, "add", vec![]);
                let sp_slot = next_slot;
                next_slot += 1;
                let th_slot = next_slot;
                next_slot += 1;
                let th2_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sp_idx,
                    slots: vec![x_slot, sp_slot],
                });
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: th_idx,
                    slots: vec![sp_slot, th_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![th_slot, th_slot, th2_slot],
                });
                let one_minus_th2 = emit_one_minus(
                    th2_slot,
                    sh,
                    "mish",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let sig_slot = next_slot;
                next_slot += 1;
                let x_sig = next_slot;
                next_slot += 1;
                let x_sig_om = next_slot;
                next_slot += 1;
                let dydx = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sig_idx,
                    slots: vec![x_slot, sig_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![x_slot, sig_slot, x_sig],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![x_sig, one_minus_th2, x_sig_om],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![th_slot, x_sig_om, dydx],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, dydx, dx_slot],
                });
                accumulate_grad(
                    x_slot,
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "gelu" => {
                // y = x * Phi(x); dy/dx = Phi(x) + x * phi(x)
                // phi(x) = (1/sqrt(2*pi)) * exp(-0.5 * x^2)
                // Compute Phi(x) smoothly without gelu(x)/x (which divides by zero at x = 0):
                //   Phi(x) ~= 0.5 * (1 + tanh((c0 * x + p3 * x^3) / (1 + q2 * x^2)))
                // where c0 = sqrt(2/pi), p3 = 0.04305, q2 = 0.00806.
                // Denominator 1 + q2*x^2 >= 1 everywhere; exact at x = 0 (Phi(0) = 0.5,
                // gelu'(0) = 0.5, gelu''(0) = sqrt(2/pi)) with max |err| < 3.4e-5 on R.
                let sh = *slot_shapes
                    .get(&out_slot)
                    .ok_or_else(|| format!("missing inferred shape for slot {out_slot}"))?;
                let x_slot = st.slots[0];
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let div_idx = push_layer(&mut layers, &mut next_reg_id, "div", vec![]);
                let exp_idx = push_layer(&mut layers, &mut next_reg_id, "exp", vec![]);
                let tanh_idx = push_layer(&mut layers, &mut next_reg_id, "tanh", vec![]);
                let add_idx = push_layer(&mut layers, &mut next_reg_id, "add", vec![]);

                let x2_slot = next_slot;
                next_slot += 1;
                let x3_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![x_slot, x_slot, x2_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![x2_slot, x_slot, x3_slot],
                });

                // Smooth Phi(x) via Padé [3/2] inside tanh
                let c0 = (2.0 / std::f64::consts::PI).sqrt() as f32;
                let c0_x = emit_scale(
                    x_slot,
                    sh,
                    c0,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let p3_x3 = emit_scale(
                    x3_slot,
                    sh,
                    0.04305,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let pade_num = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![c0_x, p3_x3, pade_num],
                });
                let pade_den = emit_affine(
                    x2_slot,
                    sh,
                    0.00806,
                    1.0,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let pade_u = next_slot;
                next_slot += 1;
                let th_u = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: div_idx,
                    slots: vec![pade_num, pade_den, pade_u],
                });
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tanh_idx,
                    slots: vec![pade_u, th_u],
                });
                let big_phi = emit_affine(
                    th_u,
                    sh,
                    0.5,
                    0.5,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;

                // Exact x * phi(x) term
                let neg_half_x2 = emit_scale(
                    x2_slot,
                    sh,
                    -0.5,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let exp_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: exp_idx,
                    slots: vec![neg_half_x2, exp_slot],
                });
                let inv_sqrt_2pi = (1.0 / (2.0 * std::f64::consts::PI).sqrt()) as f32;
                let phi_slot = emit_scale(
                    exp_slot,
                    sh,
                    inv_sqrt_2pi,
                    "gelu",
                    &mut layers,
                    &mut steps,
                    &mut weights,
                    &mut next_reg_id,
                    &mut next_slot,
                )?;
                let x_phi = next_slot;
                next_slot += 1;
                let dydx = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![x_slot, phi_slot, x_phi],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![big_phi, x_phi, dydx],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![up_grad, dydx, dx_slot],
                });
                accumulate_grad(
                    x_slot,
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "swiGlu" => {
                // y = silu(W_i x + b_i) * (W_o x + b_o), feature_last [B, 1, 1, d_in]
                let d_in = decl.args[1].as_u64().unwrap_or(0) as usize;
                let d_out = decl.args[2].as_u64().unwrap_or(0) as usize;
                let bias = decl.args[3].as_bool().unwrap_or(false);
                let x_slot = st.slots[0];
                let x_sh = *slot_shapes
                    .get(&x_slot)
                    .ok_or_else(|| format!("missing inferred shape for slot {x_slot}"))?;
                if x_sh[1] != 1 || x_sh[2] != 1 || x_sh[3] != d_in {
                    return Err(format!(
                        "unsupported_shape_for_polynomial_vjp: in-graph swiGlu VJP requires [B, 1, 1, {d_in}], got {x_sh:?}"
                    ));
                }
                let inner_len = d_in * d_out + if bias { d_out } else { 0 };
                let w = s
                    .weights
                    .get(&st.layer.to_string())
                    .cloned()
                    .unwrap_or_else(|| vec![0.0f32; 2 * inner_len]);
                let inner_seg = w[..inner_len].to_vec();
                let outer_seg = w[inner_len..2 * inner_len].to_vec();

                let tr_idx = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "transpose",
                    vec![Value::from(1), Value::from(3)],
                );
                let x_t = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![x_slot, x_t],
                });

                let lin_i = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "linear",
                    vec![Value::from(d_in), Value::from(d_out), Value::from(bias)],
                );
                weights.insert(lin_i.to_string(), inner_seg.clone());
                let a_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: lin_i,
                    slots: vec![x_t, a_slot],
                });

                let lin_o = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "linear",
                    vec![Value::from(d_in), Value::from(d_out), Value::from(bias)],
                );
                weights.insert(lin_o.to_string(), outer_seg.clone());
                let o_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: lin_o,
                    slots: vec![x_t, o_slot],
                });

                let g_t = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![up_grad, g_t],
                });

                let sig_idx = push_layer(&mut layers, &mut next_reg_id, "sigmoid", vec![]);
                let mul_idx = push_layer(&mut layers, &mut next_reg_id, "mul", vec![]);
                let sub_idx = push_layer(&mut layers, &mut next_reg_id, "sub", vec![]);
                let add_idx = push_layer(&mut layers, &mut next_reg_id, "add", vec![]);

                let sig_slot = next_slot;
                next_slot += 1;
                let s_slot = next_slot;
                next_slot += 1;
                let a_minus_s = next_slot;
                next_slot += 1;
                let term_slot = next_slot;
                next_slot += 1;
                let silu_prime = next_slot;
                next_slot += 1;
                let g1a = next_slot;
                next_slot += 1;
                let g1 = next_slot;
                next_slot += 1;
                let g2 = next_slot;
                next_slot += 1;

                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sig_idx,
                    slots: vec![a_slot, sig_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![a_slot, sig_slot, s_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: sub_idx,
                    slots: vec![a_slot, s_slot, a_minus_s],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![a_minus_s, sig_slot, term_slot],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![sig_slot, term_slot, silu_prime],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![silu_prime, o_slot, g1a],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![g1a, g_t, g1],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mul_idx,
                    slots: vec![s_slot, g_t, g2],
                });

                let lin_ti = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "linear",
                    vec![Value::from(d_out), Value::from(d_in), Value::from(false)],
                );
                weights.insert(
                    lin_ti.to_string(),
                    transpose_col_major(&inner_seg, d_in, d_out),
                );
                let dx1_t = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: lin_ti,
                    slots: vec![g1, dx1_t],
                });

                let lin_to = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "linear",
                    vec![Value::from(d_out), Value::from(d_in), Value::from(false)],
                );
                weights.insert(
                    lin_to.to_string(),
                    transpose_col_major(&outer_seg, d_in, d_out),
                );
                let dx2_t = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: lin_to,
                    slots: vec![g2, dx2_t],
                });

                let dx1 = next_slot;
                next_slot += 1;
                let dx2 = next_slot;
                next_slot += 1;
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![dx1_t, dx1],
                });
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![dx2_t, dx2],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: add_idx,
                    slots: vec![dx1, dx2, dx_slot],
                });
                accumulate_grad(
                    x_slot,
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "transpose" => {
                let d0 = decl.args[1].as_u64().unwrap_or(0) as usize;
                let d1 = decl.args[2].as_u64().unwrap_or(0) as usize;
                let tr_idx = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "transpose",
                    vec![Value::from(d0), Value::from(d1)],
                );
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![up_grad, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "matmul" => {
                let (a_slot, b_slot) = (st.slots[0], st.slots[1]);
                let tr_idx = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "transpose",
                    vec![Value::from(2), Value::from(3)],
                );
                let mm_idx = push_layer(&mut layers, &mut next_reg_id, "matmul", vec![]);
                let b_t = next_slot;
                next_slot += 1;
                let a_t = next_slot;
                next_slot += 1;
                let da_slot = next_slot;
                next_slot += 1;
                let db_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![b_slot, b_t],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mm_idx,
                    slots: vec![up_grad, b_t, da_slot],
                });
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: tr_idx,
                    slots: vec![a_slot, a_t],
                });
                steps.push(StepDecl {
                    kind: "binary".to_string(),
                    layer: mm_idx,
                    slots: vec![a_t, up_grad, db_slot],
                });
                accumulate_grad(
                    a_slot,
                    da_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    b_slot,
                    db_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "shiftUp" | "shiftDown" | "shiftLeft" | "shiftRight" => {
                let inv = match ctor {
                    "shiftUp" => "shiftDown",
                    "shiftDown" => "shiftUp",
                    "shiftLeft" => "shiftRight",
                    _ => "shiftLeft",
                };
                let shift_size = decl.args[1].as_u64().unwrap_or(0) as usize;
                let sh_idx = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    inv,
                    vec![Value::from(shift_size)],
                );
                let dx_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sh_idx,
                    slots: vec![up_grad, dx_slot],
                });
                accumulate_grad(
                    st.slots[0],
                    dx_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            "concat" => {
                let dim = decl.args[1].as_u64().unwrap_or(0) as usize;
                let (a_slot, b_slot) = (st.slots[0], st.slots[1]);
                let a_sh = *slot_shapes
                    .get(&a_slot)
                    .ok_or_else(|| format!("missing inferred shape for concat input slot {a_slot}"))?;
                let out_sh = *slot_shapes
                    .get(&out_slot)
                    .ok_or_else(|| format!("missing inferred shape for concat output slot {out_slot}"))?;
                let starts_a = vec![Value::from(0usize); 4];
                let ends_a: Vec<Value> = a_sh.iter().map(|&d| Value::from(d)).collect();
                let mut starts_b_u = [0usize; 4];
                starts_b_u[dim] = a_sh[dim];
                let starts_b: Vec<Value> = starts_b_u.iter().map(|&d| Value::from(d)).collect();
                let ends_b: Vec<Value> = out_sh.iter().map(|&d| Value::from(d)).collect();
                let sl_a = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "slice",
                    vec![Value::Array(starts_a), Value::Array(ends_a)],
                );
                let sl_b = push_layer(
                    &mut layers,
                    &mut next_reg_id,
                    "slice",
                    vec![Value::Array(starts_b), Value::Array(ends_b)],
                );
                let da_slot = next_slot;
                next_slot += 1;
                let db_slot = next_slot;
                next_slot += 1;
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sl_a,
                    slots: vec![up_grad, da_slot],
                });
                steps.push(StepDecl {
                    kind: "unary".to_string(),
                    layer: sl_b,
                    slots: vec![up_grad, db_slot],
                });
                accumulate_grad(
                    a_slot,
                    da_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
                accumulate_grad(
                    b_slot,
                    db_slot,
                    &mut grad_slot_of,
                    &mut saved_grad_slot_of,
                    &mut layers,
                    &mut steps,
                    &mut next_reg_id,
                    &mut next_slot,
                );
            }
            other => {
                return Err(format!(
                    "unsupported_symbolic_grad_op: symbolic graph differentiation does not support constructor '{other}' at step {si} (supported: linear, add, sub, mul, fma, div, sqrt, exp, log, sin, cos, sigmoid, tanh, transpose, matmul, concat, shift*, gelu, mish, swiGlu)"
                ));
            }
        }
    }

    let is_port = s.ports.iter().any(|p| p.slot == resolved_wrt);
    let ssa_wrt = if is_port {
        resolved_wrt
    } else {
        final_slot_map
            .get(&resolved_wrt)
            .copied()
            .unwrap_or(resolved_wrt)
    };
    let target_grad_slot = if is_port {
        grad_slot_of
            .get(&resolved_wrt)
            .or_else(|| saved_grad_slot_of.get(&resolved_wrt))
            .or_else(|| grad_slot_of.get(&ssa_wrt))
            .or_else(|| saved_grad_slot_of.get(&ssa_wrt))
            .copied()
    } else {
        grad_slot_of
            .get(&ssa_wrt)
            .or_else(|| saved_grad_slot_of.get(&ssa_wrt))
            .or_else(|| grad_slot_of.get(&resolved_wrt))
            .or_else(|| saved_grad_slot_of.get(&resolved_wrt))
            .copied()
    };

    let output_slot = match target_grad_slot {
        Some(slot) => slot,
        None => {
            let zero_ref = if slot_shapes.contains_key(&resolved_wrt) {
                resolved_wrt
            } else if slot_shapes.contains_key(&ssa_wrt) {
                ssa_wrt
            } else {
                0
            };
            let sub_idx = push_layer(&mut layers, &mut next_reg_id, "sub", vec![]);
            let z_slot = next_slot;
            next_slot += 1;
            steps.push(StepDecl {
                kind: "binary".to_string(),
                layer: sub_idx,
                slots: vec![zero_ref, zero_ref, z_slot],
            });
            z_slot
        }
    };

    let raw_compiled = SpecV1 {
        num_slots: next_slot.max(output_slot + 1),
        layers,
        steps,
        output_slot,
        ports,
        weights,
        inputs,
    };
    let repaired = auto_repair_spec(&raw_compiled, 0)?.spec;
    let post_val = validate_spec(&repaired, false);
    if !post_val.valid {
        return Err(format!(
            "compiled backward spec failed static validation ({}): {:?}",
            post_val.code, post_val.issues
        ));
    }

    Ok(GradSpecReport {
        ok: true,
        kind: "symbolic_backward_spec",
        order: 1,
        wrt_slot: resolved_wrt,
        dout_slot,
        hvp_slot: None,
        forward_output_slot: s.output_slot,
        layers_count: repaired.layers.len(),
        steps_count: repaired.steps.len(),
        num_slots: repaired.num_slots,
        output_slot: repaired.output_slot,
        spec: repaired,
    })
}

/// Compile a 1st-order VJP (`order == 1`) or 2nd-order Hessian-Vector Product (`order == 2`)
/// runnable `SpecV1` graph.
pub fn compile_grad_spec(
    forward_spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    opts: &GradSpecOptions,
) -> Result<GradSpecReport, String> {
    if opts.order != 1 && opts.order != 2 {
        return Err(format!(
            "grad-spec: order must be 1 (VJP) or 2 (HVP), got {}",
            opts.order
        ));
    }
    if opts.order == 1 && opts.hvp_vec.is_some() {
        return Err("grad-spec: --hvp-vec is only valid with --order 2".into());
    }

    let first = compile_backward_spec(
        forward_spec,
        expansion,
        &opts.wrt,
        opts.dout.as_deref(),
    )?;

    let mut report = if opts.order == 1 {
        first
    } else {
        let wrt_str = first.wrt_slot.to_string();
        let second = compile_backward_spec(
            &first.spec,
            None,
            &wrt_str,
            opts.hvp_vec.as_deref(),
        )?;
        GradSpecReport {
            ok: true,
            kind: "symbolic_hvp_spec",
            order: 2,
            wrt_slot: first.wrt_slot,
            dout_slot: first.dout_slot,
            hvp_slot: Some(second.dout_slot),
            forward_output_slot: first.forward_output_slot,
            layers_count: second.spec.layers.len(),
            steps_count: second.spec.steps.len(),
            num_slots: second.spec.num_slots,
            output_slot: second.spec.output_slot,
            spec: second.spec,
        }
    };

    if opts.compact_slots {
        let compacted = compact_slots_by_liveness(&report.spec)?;
        let remap = |slot: usize| compacted.slot_remap.get(&slot).copied().unwrap_or(slot);
        report.wrt_slot = remap(report.wrt_slot);
        report.dout_slot = remap(report.dout_slot);
        report.hvp_slot = report.hvp_slot.map(remap);
        report.forward_output_slot = remap(report.forward_output_slot);
        report.spec = compacted.spec;
        report.num_slots = report.spec.num_slots;
        report.output_slot = report.spec.output_slot;
        report.layers_count = report.spec.layers.len();
        report.steps_count = report.spec.steps.len();
    }

    Ok(report)
}
