use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::catalog::{
    ParamType, expected_weight_count, find_constructor, is_int_constructor,
};
use crate::spec::{SpecV1, is_canonical_uint_key, parse_f32_values, parse_i64_values};

#[derive(Debug, Clone, Serialize)]
pub struct ShapeMismatch {
    pub step: usize,
    pub layer: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DataflowReport {
    pub ok: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExplainStep {
    pub step: usize,
    pub constructor: String,
    pub kind: String,
    pub in_slots: Vec<usize>,
    pub out_slot: usize,
    pub in_shapes: Vec<Option<[usize; 4]>>,
    pub out_shape: Option<[usize; 4]>,
    pub dtype_out: String,
    pub param_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExplainReport {
    pub num_slots: usize,
    pub output_slot: usize,
    pub total_params: usize,
    pub steps: Vec<ExplainStep>,
    pub slot_shapes: BTreeMap<usize, [usize; 4]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ValidateReport {
    pub ok: bool,
    pub valid: bool,
    pub stage: &'static str,
    pub code: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unknown_constructors: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mismatches: Vec<ShapeMismatch>,
    pub dataflow: DataflowReport,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub static_warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remediation_hint: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explain: Option<ExplainReport>,
}

fn check_arg_type(label: &str, name: &str, ty: ParamType, v: &Value) -> Option<String> {
    match ty {
        ParamType::U32 => {
            let Some(n) = v.as_u64() else {
                return Some(format!("{label}: arg '{name}' must be u32, got {v}"));
            };
            if n > u32::MAX as u64 {
                return Some(format!("{label}: arg '{name}' exceeds u32::MAX, got {n}"));
            }
            None
        }
        ParamType::Bool => {
            if !v.is_boolean() {
                Some(format!("{label}: arg '{name}' must be bool, got {v}"))
            } else {
                None
            }
        }
        ParamType::F64 => {
            let Some(f) = v.as_f64() else {
                return Some(format!("{label}: arg '{name}' must be finite f64, got {v}"));
            };
            if !f.is_finite() {
                Some(format!("{label}: arg '{name}' must be finite, got {v}"))
            } else {
                None
            }
        }
        ParamType::OptionU32 => {
            if v.is_null() {
                return None;
            }
            let Some(n) = v.as_u64() else {
                return Some(format!("{label}: arg '{name}' must be u32 or null, got {v}"));
            };
            if n > u32::MAX as u64 {
                return Some(format!("{label}: arg '{name}' exceeds u32::MAX, got {n}"));
            }
            None
        }
        ParamType::OptionF64 => {
            if v.is_null() {
                return None;
            }
            let Some(f) = v.as_f64() else {
                return Some(format!("{label}: arg '{name}' must be finite f64 or null, got {v}"));
            };
            if !f.is_finite() {
                Some(format!("{label}: arg '{name}' must be finite, got {v}"))
            } else {
                None
            }
        }
    }
}

fn arg_usize(args: &[Value], idx: usize) -> Option<usize> {
    args.get(idx).and_then(|v| v.as_u64()).map(|n| n as usize)
}

fn arg_opt_usize(args: &[Value], idx: usize) -> Option<usize> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_u64().map(|n| n as usize),
    }
}

fn arg_f64(args: &[Value], idx: usize) -> Option<f64> {
    args.get(idx).and_then(|v| v.as_f64())
}

fn arg_opt_f64(args: &[Value], idx: usize) -> Option<f64> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_f64(),
    }
}

fn infer_step_shape(ctor: &str, in_shapes: &[[usize; 4]], args: &[Value]) -> Result<[usize; 4], String> {
    let s0 = in_shapes[0];
    match ctor {
        "relu" | "gelu" | "sigmoid" | "tanh" | "hardSwish" | "hardSigmoid" | "mish" | "sqrt"
        | "shiftUp" | "shiftDown" | "shiftLeft" | "shiftRight" | "seBlock" | "featureNorm"
        | "intToFloat" | "intFromFloat" => Ok(s0),

        "softmax" | "logSoftmax" => {
            let dim = arg_usize(args, 1).ok_or("missing dim")?;
            if !(1..=3).contains(&dim) {
                return Err(format!("{ctor} dim {dim} out of range 1..3"));
            }
            Ok(s0)
        }
        "leakyRelu" => {
            let slope = arg_f64(args, 1).ok_or("missing negative_slope")?;
            if slope < 0.0 {
                return Err(format!("leakyRelu negative_slope must be >= 0, got {slope}"));
            }
            Ok(s0)
        }
        "softplus" => {
            let beta = arg_f64(args, 1).ok_or("missing beta")?;
            if beta <= 0.0 {
                return Err(format!("softplus beta must be > 0, got {beta}"));
            }
            Ok(s0)
        }
        "prelu" => {
            let np = arg_usize(args, 1).ok_or("missing num_parameters")?;
            if np == 0 || (np != 1 && np != s0[1]) {
                return Err(format!(
                    "prelu num_parameters {np} must be 1 or match channel dim {}",
                    s0[1]
                ));
            }
            Ok(s0)
        }
        "swiGlu" => {
            let d_in = arg_usize(args, 1).ok_or("missing d_input")?;
            let d_out = arg_usize(args, 2).ok_or("missing d_output")?;
            if d_in == 0 || d_out == 0 {
                return Err(format!("swiGlu dims [{d_in},{d_out}] must be > 0"));
            }
            if s0[3] != d_in {
                return Err(format!("swiGlu d_input {d_in} != last-axis feature dim {}", s0[3]));
            }
            Ok([s0[0], s0[1], s0[2], d_out])
        }
        "glu" => {
            let dim = arg_usize(args, 1).ok_or("missing dim")?;
            if !(1..=3).contains(&dim) {
                return Err(format!("glu dim {dim} out of range 1..3"));
            }
            if !s0[dim].is_multiple_of(2) {
                return Err(format!("glu dim {dim} size {} is not divisible by 2", s0[dim]));
            }
            let mut out = s0;
            out[dim] /= 2;
            Ok(out)
        }
        "transpose" => {
            let d0 = arg_usize(args, 1).ok_or("missing dim0")?;
            let d1 = arg_usize(args, 2).ok_or("missing dim1")?;
            if d0 >= 4 || d1 >= 4 {
                return Err(format!("transpose dims [{d0},{d1}] out of range 0..3"));
            }
            let mut out = s0;
            out.swap(d0, d1);
            Ok(out)
        }
        "batchNorm" => {
            let nf = arg_usize(args, 1).ok_or("missing num_features")?;
            if nf == 0 {
                return Err("batchNorm num_features must be > 0".into());
            }
            if let Some(eps) = arg_opt_f64(args, 2)
                && eps <= 0.0
            {
                return Err(format!("batchNorm epsilon must be > 0, got {eps}"));
            }
            if s0[1] != nf {
                return Err(format!("batchNorm num_features {nf} != channel dim {}", s0[1]));
            }
            Ok(s0)
        }
        "groupNorm" => {
            let ng = arg_usize(args, 1).ok_or("missing num_groups")?;
            let nc = arg_usize(args, 2).ok_or("missing num_channels")?;
            if ng == 0 || nc == 0 {
                return Err("groupNorm num_groups and num_channels must be > 0".into());
            }
            if let Some(eps) = arg_opt_f64(args, 3)
                && eps <= 0.0
            {
                return Err(format!("groupNorm epsilon must be > 0, got {eps}"));
            }
            if s0[1] != nc {
                return Err(format!("groupNorm num_channels {nc} != channel dim {}", s0[1]));
            }
            if !nc.is_multiple_of(ng) {
                return Err(format!("groupNorm channels {nc} not divisible by num_groups {ng}"));
            }
            Ok(s0)
        }
        "instanceNorm" => {
            let nc = arg_usize(args, 1).ok_or("missing num_channels")?;
            if nc == 0 {
                return Err("instanceNorm num_channels must be > 0".into());
            }
            if let Some(eps) = arg_opt_f64(args, 2)
                && eps <= 0.0
            {
                return Err(format!("instanceNorm epsilon must be > 0, got {eps}"));
            }
            if s0[1] != nc {
                return Err(format!("instanceNorm num_channels {nc} != channel dim {}", s0[1]));
            }
            Ok(s0)
        }
        "layerNorm" | "rmsNorm" => {
            let sz = arg_usize(args, 1).ok_or("missing size")?;
            if sz == 0 {
                return Err(format!("{ctor} size must be > 0"));
            }
            if let Some(eps) = arg_opt_f64(args, 2)
                && eps <= 0.0
            {
                return Err(format!("{ctor} epsilon must be > 0, got {eps}"));
            }
            if s0[3] != sz {
                return Err(format!("{ctor} size {sz} != last-axis feature dim {}", s0[3]));
            }
            Ok(s0)
        }
        "linear" => {
            let in_dim = arg_usize(args, 1).ok_or("missing in_dim")?;
            let out_dim = arg_usize(args, 2).ok_or("missing out_dim")?;
            if in_dim == 0 || out_dim == 0 {
                return Err(format!("linear dims [{in_dim},{out_dim}] must be > 0"));
            }
            if s0[1] != in_dim {
                return Err(format!("linear in_dim {in_dim} != channel dim {}", s0[1]));
            }
            if s0[2] != 1 || s0[3] != 1 {
                return Err(format!("linear requires [B,F,1,1], got {s0:?}"));
            }
            Ok([s0[0], out_dim, 1, 1])
        }
        "embedding" => {
            let vocab = arg_usize(args, 1).ok_or("missing vocab_size")?;
            let d_model = arg_usize(args, 2).ok_or("missing d_model")?;
            if vocab == 0 || d_model == 0 {
                return Err(format!("embedding dims [{vocab},{d_model}] must be > 0"));
            }
            if s0[2] != 1 || s0[3] != 1 {
                return Err(format!("embedding requires [B,S,1,1], got {s0:?}"));
            }
            Ok([s0[0], s0[1], d_model, 1])
        }
        "add" | "sub" | "mul" | "div" | "intAdd" | "intSub" | "intMul" | "intDiv" | "intRem" => {
            let s1 = in_shapes[1];
            if s0 != s1 {
                return Err(format!("{ctor} shape mismatch {s0:?} vs {s1:?}"));
            }
            Ok(s0)
        }
        "matmul" | "intMatmul" => {
            let s1 = in_shapes[1];
            if s0[0] != s1[0] || s0[1] != s1[1] || s0[3] != s1[2] {
                return Err(format!("{ctor} shape mismatch {s0:?} @ {s1:?}"));
            }
            Ok([s0[0], s0[1], s0[2], s1[3]])
        }
        "concat" => {
            let s1 = in_shapes[1];
            let dim = arg_usize(args, 1).ok_or("missing dim")?;
            if !(1..=3).contains(&dim) {
                return Err(format!("concat dim {dim} out of range 1..3"));
            }
            for d in 0..4 {
                if d != dim && s0[d] != s1[d] {
                    return Err(format!("concat axis {d} mismatch {s0:?} vs {s1:?}"));
                }
            }
            let mut out = s0;
            out[dim] = s0[dim] + s1[dim];
            Ok(out)
        }
        "conv1d" => {
            let in_ch = arg_usize(args, 1).ok_or("missing in_ch")?;
            let out_ch = arg_usize(args, 2).ok_or("missing out_ch")?;
            let k = arg_usize(args, 3).ok_or("missing kernel")?;
            let s = arg_opt_usize(args, 4).unwrap_or(1);
            let p = arg_opt_usize(args, 5).unwrap_or(0);
            if in_ch == 0 || out_ch == 0 || k == 0 || s == 0 {
                return Err("conv1d channels, kernel, stride must be > 0".into());
            }
            if s0[1] != in_ch {
                return Err(format!("conv1d in_ch {in_ch} != channel dim {}", s0[1]));
            }
            if s0[3] != 1 {
                return Err(format!("conv1d requires [B,C,L,1], got {s0:?}"));
            }
            let num = s0[2] + 2 * p;
            if num < k {
                return Err("conv1d output length non-positive".into());
            }
            let ol = (num - k) / s + 1;
            Ok([s0[0], out_ch, ol, 1])
        }
        "conv2d" => {
            let in_ch = arg_usize(args, 1).ok_or("missing in_ch")?;
            let out_ch = arg_usize(args, 2).ok_or("missing out_ch")?;
            let kh = arg_usize(args, 3).ok_or("missing kh")?;
            let kw = arg_usize(args, 4).ok_or("missing kw")?;
            let sh = arg_opt_usize(args, 5).unwrap_or(1);
            let sw = arg_opt_usize(args, 6).unwrap_or(1);
            let ph = arg_opt_usize(args, 7).unwrap_or(0);
            let pw = arg_opt_usize(args, 8).unwrap_or(0);
            if in_ch == 0 || out_ch == 0 || kh == 0 || kw == 0 || sh == 0 || sw == 0 {
                return Err("conv2d channels, kernel, stride must be > 0".into());
            }
            if s0[1] != in_ch {
                return Err(format!("conv2d in_ch {in_ch} != channel dim {}", s0[1]));
            }
            let nh = s0[2] + 2 * ph;
            let nw = s0[3] + 2 * pw;
            if nh < kh || nw < kw {
                return Err("conv2d output spatial dims non-positive".into());
            }
            Ok([s0[0], out_ch, (nh - kh) / sh + 1, (nw - kw) / sw + 1])
        }
        "convTranspose2d" => {
            let in_ch = arg_usize(args, 1).ok_or("missing in_ch")?;
            let out_ch = arg_usize(args, 2).ok_or("missing out_ch")?;
            let kh = arg_usize(args, 3).ok_or("missing kh")?;
            let kw = arg_usize(args, 4).ok_or("missing kw")?;
            let sh = arg_opt_usize(args, 5).unwrap_or(1);
            let sw = arg_opt_usize(args, 6).unwrap_or(1);
            let ph = arg_opt_usize(args, 7).unwrap_or(0);
            let pw = arg_opt_usize(args, 8).unwrap_or(0);
            if in_ch == 0 || out_ch == 0 || kh == 0 || kw == 0 || sh == 0 || sw == 0 {
                return Err("convTranspose2d channels, kernel, stride must be > 0".into());
            }
            if s0[1] != in_ch {
                return Err(format!("convTranspose2d in_ch {in_ch} != channel dim {}", s0[1]));
            }
            let oh = (s0[2] - 1) * sh + kh;
            let ow = (s0[3] - 1) * sw + kw;
            if oh <= 2 * ph || ow <= 2 * pw {
                return Err("convTranspose2d output spatial dims non-positive".into());
            }
            Ok([s0[0], out_ch, oh - 2 * ph, ow - 2 * pw])
        }
        "ghost" => {
            let in_ch = arg_usize(args, 1).ok_or("missing in_ch")?;
            let out_ch = arg_usize(args, 2).ok_or("missing out_ch")?;
            let kh = arg_usize(args, 3).ok_or("missing kh")?;
            let kw = arg_usize(args, 4).ok_or("missing kw")?;
            let ratio = arg_usize(args, 5).ok_or("missing ratio")?;
            let sh = arg_opt_usize(args, 6).unwrap_or(1);
            let sw = arg_opt_usize(args, 7).unwrap_or(1);
            let ph = arg_opt_usize(args, 8).unwrap_or(0);
            let pw = arg_opt_usize(args, 9).unwrap_or(0);
            if in_ch == 0 || out_ch == 0 || kh == 0 || kw == 0 || ratio == 0 || sh == 0 || sw == 0 {
                return Err("ghost dimensions must be > 0".into());
            }
            if !out_ch.is_multiple_of(ratio) {
                return Err(format!("ghost out_ch {out_ch} not divisible by ratio {ratio}"));
            }
            if s0[1] != in_ch {
                return Err(format!("ghost in_ch {in_ch} != channel dim {}", s0[1]));
            }
            let nh = s0[2] + 2 * ph;
            let nw = s0[3] + 2 * pw;
            if nh < kh || nw < kw {
                return Err("ghost output spatial dims non-positive".into());
            }
            Ok([s0[0], 2 * (out_ch / ratio), (nh - kh) / sh + 1, (nw - kw) / sw + 1])
        }
        "maxPool1d" | "avgPool1d" => {
            let k = arg_usize(args, 1).ok_or("missing kernel")?;
            let s = arg_opt_usize(args, 2).unwrap_or(k);
            let p = arg_opt_usize(args, 3).unwrap_or(0);
            if k == 0 || s == 0 {
                return Err(format!("{ctor} kernel and stride must be > 0"));
            }
            if s0[3] != 1 {
                return Err(format!("{ctor} requires [B,C,L,1], got {s0:?}"));
            }
            let num = s0[2] + 2 * p;
            if num < k {
                return Err(format!("{ctor} output length non-positive"));
            }
            Ok([s0[0], s0[1], (num - k) / s + 1, 1])
        }
        "maxPool2d" | "avgPool2d" => {
            let kh = arg_usize(args, 1).ok_or("missing kh")?;
            let kw = arg_usize(args, 2).ok_or("missing kw")?;
            let sh = arg_opt_usize(args, 3).unwrap_or(kh);
            let sw = arg_opt_usize(args, 4).unwrap_or(kw);
            let ph = arg_opt_usize(args, 5).unwrap_or(0);
            let pw = arg_opt_usize(args, 6).unwrap_or(0);
            if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
                return Err(format!("{ctor} kernel and stride must be > 0"));
            }
            let nh = s0[2] + 2 * ph;
            let nw = s0[3] + 2 * pw;
            if nh < kh || nw < kw {
                return Err(format!("{ctor} output spatial dims non-positive"));
            }
            Ok([s0[0], s0[1], (nh - kh) / sh + 1, (nw - kw) / sw + 1])
        }
        "adaptiveAvgPool2d" => {
            let oh = arg_usize(args, 1).ok_or("missing output_h")?;
            let ow = arg_usize(args, 2).ok_or("missing output_w")?;
            if oh == 0 || ow == 0 {
                return Err("adaptiveAvgPool2d output dims must be > 0".into());
            }
            Ok([s0[0], s0[1], oh, ow])
        }
        _ => Err(format!("unknown constructor '{ctor}'")),
    }
}

pub fn validate_spec(spec: &SpecV1, include_explain: bool) -> ValidateReport {
    // 1a. Constructor existence
    let mut unknown = Vec::new();
    for l in &spec.layers {
        if find_constructor(&l.constructor).is_none() && !unknown.contains(&l.constructor) {
            unknown.push(l.constructor.clone());
        }
    }
    if !unknown.is_empty() {
        return ValidateReport {
            ok: false,
            valid: false,
            stage: "static",
            code: "unknown_layer_constructor",
            unknown_constructors: unknown,
            issues: vec![],
            mismatches: vec![],
            dataflow: DataflowReport { ok: true, issues: vec![] },
            static_warnings: vec![],
            remediation_hint: Some("run `math-draw find-op --list` and use only supported constructors"),
            explain: None,
        };
    }

    // 1b/1c. Arg types & unique registry IDs
    let mut arg_issues = Vec::new();
    let mut seen_ids: HashMap<u64, usize> = HashMap::new();
    for (li, decl) in spec.layers.iter().enumerate() {
        let meta = find_constructor(&decl.constructor).unwrap();
        let required = meta.parameters.iter().filter(|p| !p.optional).count();
        if decl.args.len() < required || decl.args.len() > meta.parameters.len() {
            arg_issues.push(format!(
                "layer {li} ({}): expected {required}..={} args, got {}",
                decl.constructor,
                meta.parameters.len(),
                decl.args.len()
            ));
        }
        for (ai, (val, param)) in decl.args.iter().zip(meta.parameters.iter()).enumerate() {
            if let Some(msg) = check_arg_type(&format!("layer {li} ({})", decl.constructor), param.name, param.ty, val) {
                arg_issues.push(format!("arg[{ai}] {msg}"));
            }
        }
        if let Some(id) = decl.args.first().and_then(|v| v.as_u64()) {
            if let Some(prev) = seen_ids.insert(id, li) {
                arg_issues.push(format!(
                    "duplicate registry id {id} (layers {prev} and {li}): registry ids (args[0]) must be unique"
                ));
            }
        }
    }

    for (k, vals) in &spec.weights {
        if !is_canonical_uint_key(k) {
            arg_issues.push(format!("weights key '{k}' must be a canonical non-negative integer"));
        }
        for (vi, v) in vals.iter().enumerate() {
            if !v.is_finite() {
                arg_issues.push(format!("weights[{k}][{vi}] is non-finite"));
            }
        }
    }
    for (k, raw_vals) in &spec.inputs {
        if !is_canonical_uint_key(k) {
            arg_issues.push(format!("inputs key '{k}' must be a canonical non-negative integer"));
        }
        if let Err(e) = parse_f32_values(raw_vals, &format!("inputs[{k}]"))
            && parse_i64_values(raw_vals, &format!("inputs[{k}]")).is_err()
        {
            arg_issues.push(e);
        }
    }

    let mut seen_port_slots = HashSet::new();
    for p in &spec.ports {
        if !seen_port_slots.insert(p.slot) {
            arg_issues.push(format!("duplicate port slot {}", p.slot));
        }
        if p.shape.contains(&0) {
            arg_issues.push(format!("port slot {}: shape dims must all be > 0, got {:?}", p.slot, p.shape));
        }
    }

    if !arg_issues.is_empty() {
        return ValidateReport {
            ok: false,
            valid: false,
            stage: "static",
            code: "invalid_argument",
            unknown_constructors: vec![],
            issues: arg_issues,
            mismatches: vec![],
            dataflow: DataflowReport { ok: true, issues: vec![] },
            static_warnings: vec![],
            remediation_hint: Some("fix layer args, registry id uniqueness, or port shapes"),
            explain: None,
        };
    }

    // Dataflow + Static Shape Inference + Mixed Dtype checks
    let mut dataflow_issues = Vec::new();
    let mut mismatches = Vec::new();
    let mut warnings = Vec::new();

    if spec.num_slots == 0 {
        dataflow_issues.push("numSlots must be > 0".into());
    }
    if spec.output_slot >= spec.num_slots {
        dataflow_issues.push(format!(
            "outputSlot {} out of range 0..{}",
            spec.output_slot,
            spec.num_slots.saturating_sub(1)
        ));
    }

    let mut written_slots = HashSet::new();
    let mut slot_shapes: BTreeMap<usize, [usize; 4]> = BTreeMap::new();
    let mut slot_dtypes: HashMap<usize, &'static str> = HashMap::new();

    for p in &spec.ports {
        if p.slot >= spec.num_slots {
            dataflow_issues.push(format!("port slot {} out of range", p.slot));
        } else {
            written_slots.insert(p.slot);
            slot_shapes.insert(p.slot, p.shape);
            slot_dtypes.insert(p.slot, "f32");
        }
    }

    // Pre-pass: check if any port starts as i64 (feeds directly into an int* op)
    for st in &spec.steps {
        if let Some(layer) = spec.layers.get(st.layer)
            && is_int_constructor(&layer.constructor)
            && layer.constructor != "intFromFloat"
        {
            for &in_s in st.slots.iter().take(st.slots.len().saturating_sub(1)) {
                if spec.port_by_slot(in_s).is_some() {
                    slot_dtypes.insert(in_s, "i64");
                }
            }
        }
    }

    let mut explain_steps = Vec::new();
    let mut total_params = 0usize;

    for (si, st) in spec.steps.iter().enumerate() {
        let Some(layer) = spec.layers.get(st.layer) else {
            mismatches.push(ShapeMismatch {
                step: si,
                layer: format!("!missing:{}", st.layer),
                message: format!("step {si} references missing layer index {}", st.layer),
            });
            continue;
        };
        let meta = find_constructor(&layer.constructor).unwrap();
        let want_kind = if meta.arity == 1 { "unary" } else { "binary" };
        let want_slots = if meta.arity == 1 { 2 } else { 3 };
        if st.kind != want_kind || st.slots.len() != want_slots {
            mismatches.push(ShapeMismatch {
                step: si,
                layer: layer.constructor.clone(),
                message: format!(
                    "constructor '{}' requires {} step with {} slots, got '{}' with {} slots",
                    layer.constructor, want_kind, want_slots, st.kind, st.slots.len()
                ),
            });
            continue;
        }

        let in_slots = &st.slots[..meta.arity];
        let out_slot = st.slots[meta.arity];
        let mut blocked = false;
        for &s in &st.slots {
            if s >= spec.num_slots {
                dataflow_issues.push(format!("step {si}: slot {s} out of range"));
                blocked = true;
            }
        }
        for &s in in_slots {
            if s < spec.num_slots && !written_slots.contains(&s) {
                dataflow_issues.push(format!("step {si}: slot {s} read before written"));
                blocked = true;
            }
        }
        if blocked {
            continue;
        }

        // Check dtype consistency
        let expected_in_dtype = if layer.constructor == "intFromFloat" {
            "f32"
        } else if is_int_constructor(&layer.constructor) {
            "i64"
        } else {
            "f32"
        };
        for &s in in_slots {
            if let Some(&actual) = slot_dtypes.get(&s)
                && actual != expected_in_dtype
            {
                mismatches.push(ShapeMismatch {
                    step: si,
                    layer: layer.constructor.clone(),
                    message: format!(
                        "dtype mismatch on input slot {s}: '{}' expects {expected_in_dtype}, got {actual}",
                        layer.constructor
                    ),
                });
            }
        }
        let out_dtype = if layer.constructor == "intToFloat" {
            "f32"
        } else if is_int_constructor(&layer.constructor) {
            "i64"
        } else {
            "f32"
        };
        slot_dtypes.insert(out_slot, out_dtype);
        written_slots.insert(out_slot);

        let in_shapes: Vec<Option<[usize; 4]>> =
            in_slots.iter().map(|s| slot_shapes.get(s).copied()).collect();
        let mut out_shape = None;
        if in_shapes.iter().all(|s| s.is_some()) {
            let concrete: Vec<[usize; 4]> = in_shapes.iter().map(|s| s.unwrap()).collect();
            match infer_step_shape(&layer.constructor, &concrete, &layer.args) {
                Ok(sh) => {
                    slot_shapes.insert(out_slot, sh);
                    out_shape = Some(sh);
                }
                Err(msg) => {
                    slot_shapes.remove(&out_slot);
                    mismatches.push(ShapeMismatch {
                        step: si,
                        layer: layer.constructor.clone(),
                        message: msg,
                    });
                }
            }
        }

        let pcount = expected_weight_count(&layer.constructor, &layer.args).unwrap_or(0);
        total_params += pcount;
        if include_explain {
            explain_steps.push(ExplainStep {
                step: si,
                constructor: layer.constructor.clone(),
                kind: st.kind.clone(),
                in_slots: in_slots.to_vec(),
                out_slot,
                in_shapes,
                out_shape,
                dtype_out: out_dtype.to_string(),
                param_count: pcount,
            });
        }
    }

    if !written_slots.contains(&spec.output_slot) {
        dataflow_issues.push(format!("outputSlot {} is never written", spec.output_slot));
    } else if slot_dtypes.get(&spec.output_slot).copied() == Some("i64") {
        warnings.push("outputSlot carries i64 dtype; add intToFloat before outputSlot for f32 output".into());
    }

    // Weight count & input length warnings
    for (k, w) in &spec.weights {
        if let Ok(li) = k.parse::<usize>() {
            if let Some(decl) = spec.layers.get(li) {
                if let Some(exp) = expected_weight_count(&decl.constructor, &decl.args)
                    && w.len() != exp
                {
                    warnings.push(format!(
                        "weights[{li}] ({}): expected {exp} floats, got {}",
                        decl.constructor,
                        w.len()
                    ));
                }
            } else {
                warnings.push(format!("weights[{k}] has no matching layer"));
            }
        }
    }
    for (k, vals) in &spec.inputs {
        if let Ok(slot) = k.parse::<usize>() {
            if let Some(p) = spec.port_by_slot(slot) {
                let prod: usize = p.shape.iter().product();
                if vals.len() != prod {
                    warnings.push(format!(
                        "inputs[{slot}]: expected {prod} elements for shape {:?}, got {}",
                        p.shape,
                        vals.len()
                    ));
                }
            } else {
                warnings.push(format!("inputs[{k}] has no matching port"));
            }
        }
    }

    // Engine plan port checks (minimum 1 port + all declared ports consumed)
    if !mismatches.is_empty() || !dataflow_issues.is_empty() {
        return ValidateReport {
            ok: false,
            valid: false,
            stage: "static",
            code: "static_check_failed",
            unknown_constructors: vec![],
            issues: vec![],
            mismatches,
            dataflow: DataflowReport {
                ok: dataflow_issues.is_empty(),
                issues: dataflow_issues,
            },
            static_warnings: warnings,
            remediation_hint: Some("fix the listed steps/slots/shapes and re-run validate"),
            explain: None,
        };
    }

    if spec.ports.is_empty() {
        return ValidateReport {
            ok: false,
            valid: false,
            stage: "engine",
            code: "plan_minimum_ports",
            unknown_constructors: vec![],
            issues: vec!["spec must declare at least 1 input port".into()],
            mismatches: vec![],
            dataflow: DataflowReport { ok: true, issues: vec![] },
            static_warnings: warnings,
            remediation_hint: Some("declare input port(s) in spec.ports"),
            explain: None,
        };
    }

    let mut consumed_ports = HashSet::new();
    for st in &spec.steps {
        let arity = if st.kind == "binary" { 2 } else { 1 };
        for &s in st.slots.iter().take(arity) {
            consumed_ports.insert(s);
        }
    }
    let unused: Vec<usize> = spec
        .ports
        .iter()
        .map(|p| p.slot)
        .filter(|s| !consumed_ports.contains(s))
        .collect();
    if !unused.is_empty() {
        return ValidateReport {
            ok: false,
            valid: false,
            stage: "engine",
            code: "plan_unused_ports",
            unknown_constructors: vec![],
            issues: vec![format!("declared port slots are never consumed by any step: {unused:?}")],
            mismatches: vec![],
            dataflow: DataflowReport { ok: true, issues: vec![] },
            static_warnings: warnings,
            remediation_hint: Some("connect every declared port slot to a step or remove unused ports"),
            explain: None,
        };
    }

    let explain = if include_explain {
        Some(ExplainReport {
            num_slots: spec.num_slots,
            output_slot: spec.output_slot,
            total_params,
            steps: explain_steps,
            slot_shapes,
        })
    } else {
        None
    };

    ValidateReport {
        ok: true,
        valid: true,
        stage: "ok",
        code: "ok",
        unknown_constructors: vec![],
        issues: vec![],
        mismatches: vec![],
        dataflow: DataflowReport { ok: true, issues: vec![] },
        static_warnings: warnings,
        remediation_hint: None,
        explain,
    }
}
