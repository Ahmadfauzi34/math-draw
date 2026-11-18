use serde::Serialize;
use std::collections::{HashMap, HashSet};

use crate::engine::PreparedGraph;
use crate::spec::{LayerDecl, SpecV1, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{EvalCase, VerifyReport, default_cases_from_spec, verify_specs};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantScheme {
    Int8,
    Q8PerBlock,
    Fp16,
}

impl QuantScheme {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "int8" => Ok(Self::Int8),
            "q8-per-block" => Ok(Self::Q8PerBlock),
            "fp16" => Ok(Self::Fp16),
            _ => Err(format!(
                "unknown quant scheme '{s}' (use: int8 | q8-per-block | fp16)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Q8PerBlock => "q8-per-block",
            Self::Fp16 => "fp16",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BudgetMode {
    Bound,
    Measured,
}

impl BudgetMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "bound" => Ok(Self::Bound),
            "measured" => Ok(Self::Measured),
            _ => Err(format!("unknown budget mode '{s}' (use: bound | measured)")),
        }
    }
}

/// Pure-Rust IEEE 754 binary32 -> binary16 -> binary32 round-to-nearest-even simulation.
pub fn f32_to_f16_roundtrip(val: f32) -> f32 {
    if !val.is_finite() {
        return val;
    }
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    let h_bits: u16 = if exp == 0 {
        sign as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 31 {
            (sign | 0x7bff) as u16 // clamp finite f16 max (65504.0)
        } else if new_exp <= 0 {
            let shift = 1 - new_exp;
            if shift > 24 {
                sign as u16
            } else {
                let full_mant = mant | 0x0080_0000;
                let total_shift = shift + 13;
                let half_ulp = 1u32 << (total_shift - 1);
                let lsb = (full_mant >> total_shift) & 1;
                let rem = full_mant & ((1u32 << total_shift) - 1);
                let mut sub = full_mant >> total_shift;
                if rem > half_ulp || (rem == half_ulp && lsb == 1) {
                    sub += 1;
                }
                (sign | sub) as u16
            }
        } else {
            let round_bit = 1u32 << 12;
            let lsb = (mant >> 13) & 1;
            let rem = mant & 0x1fff;
            let mut h_mant = mant >> 13;
            let mut h_exp = new_exp as u32;
            if rem > round_bit || (rem == round_bit && lsb == 1) {
                h_mant += 1;
                if h_mant == 0x0400 {
                    h_mant = 0;
                    h_exp += 1;
                    if h_exp >= 31 {
                        return f32_from_f16_bits((sign | 0x7bff) as u16);
                    }
                }
            }
            (sign | (h_exp << 10) | h_mant) as u16
        }
    };
    f32_from_f16_bits(h_bits)
}

fn f32_from_f16_bits(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1f;
    let mant = (h as u32) & 0x03ff;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let mut m = mant;
            let mut e = -14i32;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03ff;
            let f_exp = ((e + 127) as u32) << 23;
            f32::from_bits(sign | f_exp | (m << 13))
        }
    } else if exp == 31 {
        let f_exp = 0xffu32 << 23;
        f32::from_bits(sign | f_exp | (mant << 13))
    } else {
        let f_exp = (exp + 127 - 15) << 23;
        f32::from_bits(sign | f_exp | (mant << 13))
    }
}

pub fn quantize_weights(w: &[f32], scheme: QuantScheme) -> (Vec<f32>, Option<f64>, usize) {
    match scheme {
        QuantScheme::Int8 => {
            let max_abs = w.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            let q = w
                .iter()
                .map(|&v| {
                    let qi = (v / scale).round().clamp(-127.0, 127.0);
                    qi * scale
                })
                .collect();
            let bytes = w.len() + 4;
            (q, Some(scale as f64), bytes)
        }
        QuantScheme::Q8PerBlock => {
            let block_size = 32usize;
            let num_blocks = w.len().div_ceil(block_size).max(1);
            let mut q = Vec::with_capacity(w.len());
            for chunk in w.chunks(block_size) {
                let max_abs = chunk.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
                let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
                for &v in chunk {
                    let qi = (v / scale).round().clamp(-127.0, 127.0);
                    q.push(qi * scale);
                }
            }
            let bytes = w.len() + num_blocks * 4;
            (q, None, bytes)
        }
        QuantScheme::Fp16 => {
            let q = w.iter().map(|&v| f32_to_f16_roundtrip(v)).collect();
            let bytes = w.len() * 2;
            (q, None, bytes)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantLayerReport {
    pub layer_index: usize,
    pub constructor: String,
    pub weight_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    pub max_abs_weight_error: f64,
    pub bytes_fp32: usize,
    pub bytes_quant: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantTotals {
    pub bytes_fp32: usize,
    pub bytes_quant: usize,
    pub saved_bytes: isize,
    pub compression_ratio: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantVerifyReport {
    pub ok: bool,
    pub scheme: &'static str,
    pub layers: Vec<QuantLayerReport>,
    pub totals: QuantTotals,
    pub verification: VerifyReport,
    pub verdict: &'static str,
    pub quantized_spec: SpecV1,
}

pub fn quant_verify(
    spec: &SpecV1,
    scheme: QuantScheme,
    cases_override: Option<&[EvalCase]>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<QuantVerifyReport, String> {
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }
    if spec.weights.is_empty() {
        return Err("quant verify: spec has no weights to quantize".into());
    }

    let cases = match cases_override {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => default_cases_from_spec(spec),
    };

    let mut q_spec = spec.clone();
    let mut layer_reports = Vec::new();
    let mut bytes_fp32 = 0usize;
    let mut bytes_quant = 0usize;

    for (k, values) in &spec.weights {
        let li: usize = k
            .parse()
            .map_err(|_| format!("invalid weight key '{k}'"))?;
        let ctor = spec
            .layers
            .get(li)
            .map(|l| l.constructor.clone())
            .unwrap_or_else(|| "unknown".into());
        let (q_vals, scale, bq) = quantize_weights(values, scheme);
        let max_err = values
            .iter()
            .zip(q_vals.iter())
            .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
            .fold(0.0f64, f64::max);
        let b32 = values.len() * 4;
        bytes_fp32 += b32;
        bytes_quant += bq;
        q_spec.weights.insert(k.clone(), q_vals);
        layer_reports.push(QuantLayerReport {
            layer_index: li,
            constructor: ctor,
            weight_count: values.len(),
            scale,
            max_abs_weight_error: max_err,
            bytes_fp32: b32,
            bytes_quant: bq,
        });
    }

    let ver = verify_specs(spec, &q_spec, &cases, abs_tol, rel_tol);
    let saved_bytes = (bytes_fp32 as isize) - (bytes_quant as isize);
    let ratio = if bytes_quant > 0 {
        (bytes_fp32 as f64) / (bytes_quant as f64)
    } else {
        1.0
    };

    Ok(QuantVerifyReport {
        ok: ver.equivalent,
        scheme: scheme.as_str(),
        layers: layer_reports,
        totals: QuantTotals {
            bytes_fp32,
            bytes_quant,
            saved_bytes,
            compression_ratio: (ratio * 1000.0).round() / 1000.0,
        },
        verdict: ver.verdict,
        verification: ver,
        quantized_spec: q_spec,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct BudgetLayerDecision {
    pub layer_index: usize,
    pub precision: &'static str,
    pub weight_count: usize,
    pub bytes_fp32: usize,
    pub bytes_after: usize,
    pub saved_bytes: usize,
    pub estimated_output_delta: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantBudgetReport {
    pub ok: bool,
    pub mode: BudgetMode,
    pub scheme: &'static str,
    pub tol: f64,
    pub quantized_layers: Vec<usize>,
    pub kept_fp32_layers: Vec<usize>,
    pub total_saved_bytes: usize,
    pub total_estimated_delta: f64,
    pub verified_delta: f64,
    pub decisions: Vec<BudgetLayerDecision>,
    pub plan_sha256: String,
    pub planned_spec: SpecV1,
}

fn linear_op_norm_inf(w: &[f32], in_dim: usize, out_dim: usize) -> f64 {
    let mut max_row_sum = 0.0f64;
    for i in 0..out_dim {
        let mut s = 0.0f64;
        for k in 0..in_dim {
            s += (w[k * out_dim + i] as f64).abs();
        }
        if s > max_row_sum {
            max_row_sum = s;
        }
    }
    max_row_sum
}

fn linear_bias_max(w: &[f32], in_dim: usize, out_dim: usize, bias: bool) -> f64 {
    if !bias {
        return 0.0;
    }
    w[in_dim * out_dim..]
        .iter()
        .take(out_dim)
        .map(|&v| (v as f64).abs())
        .fold(0.0f64, f64::max)
}

fn norm_gamma_max(spec: &SpecV1, layer_idx: usize, nf: usize) -> f64 {
    if let Some(w) = spec.weights.get(&layer_idx.to_string())
        && w.len() >= nf
    {
        w[..nf].iter().map(|&v| (v as f64).abs()).fold(0.0f64, f64::max)
    } else {
        1.0
    }
}

fn unary_lipschitz_const(
    decl: &LayerDecl,
    spec: &SpecV1,
    layer_idx: usize,
    in_bound: f64,
) -> Result<f64, String> {
    let c = decl.constructor.as_str();
    match c {
        "relu" | "tanh" | "softplus" | "transpose" | "shiftUp" | "shiftDown" | "shiftLeft"
        | "shiftRight" | "avgPool1d" | "avgPool2d" | "maxPool1d" | "maxPool2d"
        | "adaptiveAvgPool2d" | "embedding" | "intToFloat" | "abs" | "clamp" | "reshape"
        | "permute" | "slice" | "selectAxis" | "mean" | "meanAxis" | "min" | "max"
        | "minAxis" | "maxAxis" | "sin" | "cos" => Ok(1.0),
        "fillLike" | "indicesLike" => Ok(0.0),
        "sigmoid" => Ok(0.25),
        "gelu" => Ok(1.13),
        "hardSwish" => Ok(1.5),
        "mish" => Ok(1.11),
        "softmax" => Ok(0.5),
        "logSoftmax" => Ok(2.0),
        "leakyRelu" => {
            let slope = decl.args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.01);
            Ok(1.0f64.max(slope.abs()))
        }
        "prelu" => {
            let alpha_max = if let Some(w) = spec.weights.get(&layer_idx.to_string())
                && !w.is_empty()
            {
                w.iter().map(|&v| (v as f64).abs()).fold(0.0f64, f64::max)
            } else {
                decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.25).abs()
            };
            Ok(1.0f64.max(alpha_max))
        }
        "hardSigmoid" => Ok(decl.args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.2).abs()),
        "glu" => Ok(1.0 + in_bound / 4.0),
        "linear" => {
            let in_dim = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let out_dim = decl.args.get(2).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let Some(w) = spec.weights.get(&layer_idx.to_string()) else {
                return Err(format!("linear layer {layer_idx} missing weights for bound mode"));
            };
            Ok(linear_op_norm_inf(w, in_dim, out_dim))
        }
        "featureNorm" => {
            let eps = decl.args.get(1).and_then(|v| v.as_f64()).unwrap_or(1e-12);
            Ok(2.0 / eps.max(1e-30).sqrt())
        }
        "rmsNorm" | "layerNorm" => {
            let nf = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let eps = decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(1e-5);
            let gm = norm_gamma_max(spec, layer_idx, nf);
            Ok((2.0 * gm) / eps.max(1e-30).sqrt())
        }
        "batchNorm" => {
            let nf = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let eps = decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(1e-5);
            let gm = norm_gamma_max(spec, layer_idx, nf);
            Ok(gm / (1.0 + eps.max(0.0)).sqrt())
        }
        "groupNorm" | "instanceNorm" => {
            let nf = if c == "groupNorm" {
                decl.args.get(2).and_then(|v| v.as_u64()).unwrap_or(1) as usize
            } else {
                decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(1) as usize
            };
            let eps = if c == "groupNorm" {
                decl.args.get(3).and_then(|v| v.as_f64()).unwrap_or(1e-5)
            } else {
                decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(1e-5)
            };
            let gm = norm_gamma_max(spec, layer_idx, nf);
            Ok((2.0 * gm) / eps.max(1e-30).sqrt())
        }
        "div" | "sqrt" | "exp" | "log" | "tan" | "powf" | "floor" | "ceil" | "round" | "allFinite" => Err(format!(
            "bound mode refuses unbounded/discontinuous constructor '{c}'; use --mode measured instead"
        )),
        other => Err(format!(
            "bound mode does not have analytical Lipschitz bound for '{other}'; use --mode measured"
        )),
    }
}

fn input_slot_bounds(spec: &SpecV1, cases: &[EvalCase]) -> Result<HashMap<usize, f64>, String> {
    let mut bounds: HashMap<usize, f64> = HashMap::new();
    for p in &spec.ports {
        bounds.insert(p.slot, 0.0);
    }
    for case in cases {
        for p in &spec.ports {
            let key = p.slot.to_string();
            let raw = case
                .inputs
                .get(&key)
                .or_else(|| spec.inputs.get(&key))
                .ok_or_else(|| format!("missing inputs for slot {key}"))?;
            let vals = crate::spec::parse_f32_values(raw, &format!("inputs[{key}]"))?;
            let max_v = vals.iter().map(|&v| (v as f64).abs()).fold(0.0f64, f64::max);
            let entry = bounds.entry(p.slot).or_insert(0.0);
            if max_v > *entry {
                *entry = max_v;
            }
        }
    }
    Ok(bounds)
}

fn forward_value_bounds(
    spec: &SpecV1,
    init_bounds: &HashMap<usize, f64>,
    slot_shapes: &std::collections::BTreeMap<usize, [usize; 4]>,
) -> Result<HashMap<usize, f64>, String> {
    let mut slot_b = init_bounds.clone();
    for st in &spec.steps {
        let decl = &spec.layers[st.layer];
        if st.kind == "ternary" {
            let ba = *slot_b.get(&st.slots[0]).unwrap_or(&0.0);
            let bb = *slot_b.get(&st.slots[1]).unwrap_or(&0.0);
            let bc = *slot_b.get(&st.slots[2]).unwrap_or(&0.0);
            let out_b = match decl.constructor.as_str() {
                "fma" => ba * bb + bc,
                other => {
                    return Err(format!(
                        "bound mode unsupported ternary op '{other}'; use --mode measured"
                    ))
                }
            };
            slot_b.insert(st.slots[3], out_b);
        } else if st.kind == "binary" {
            let ba = *slot_b.get(&st.slots[0]).unwrap_or(&0.0);
            let bb = *slot_b.get(&st.slots[1]).unwrap_or(&0.0);
            let out_b = match decl.constructor.as_str() {
                "add" | "sub" => ba + bb,
                "mul" => ba * bb,
                "concat" => ba.max(bb),
                "expandLike" => ba,
                "hypot" => ba.hypot(bb),
                "atan2" => std::f64::consts::PI,
                other => {
                    return Err(format!(
                        "bound mode unsupported binary op '{other}'; use --mode measured"
                    ))
                }
            };
            slot_b.insert(st.slots[2], out_b);
        } else {
            let bi = *slot_b.get(&st.slots[0]).unwrap_or(&0.0);
            let out_b = match decl.constructor.as_str() {
                "linear" => {
                    let in_dim = decl.args[1].as_u64().unwrap_or(0) as usize;
                    let out_dim = decl.args[2].as_u64().unwrap_or(0) as usize;
                    let bias = decl.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
                    let w = spec
                        .weights
                        .get(&st.layer.to_string())
                        .ok_or_else(|| format!("linear {} missing weights", st.layer))?;
                    bi * linear_op_norm_inf(w, in_dim, out_dim) + linear_bias_max(w, in_dim, out_dim, bias)
                }
                "sigmoid" => 1.0f64.min(bi * 0.25 + 0.5),
                "tanh" => 1.0f64.min(bi),
                "sin" => bi.clamp(0.0, 1.0),
                "cos" => 1.0,
                "allFinite" => 1.0,
                "relu" | "abs" => bi,
                "clamp" => {
                    let min_v = decl.args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let max_v = decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0);
                    min_v.abs().max(max_v.abs())
                }
                "fillLike" => decl.args.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0).abs(),
                "indicesLike" => {
                    let axis = decl.args.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                    let axis_len = slot_shapes
                        .get(&st.slots[1])
                        .and_then(|sh| sh.get(axis))
                        .copied()
                        .unwrap_or(1);
                    axis_len.saturating_sub(1) as f64
                }
                _ => {
                    let l = unary_lipschitz_const(decl, spec, st.layer, bi)?;
                    bi * l
                }
            };
            slot_b.insert(st.slots[1], out_b);
        }
    }
    Ok(slot_b)
}

fn analytical_layer_contrib(
    spec: &SpecV1,
    candidate_li: usize,
    layer_init_err: f64,
    val_bounds: &HashMap<usize, f64>,
) -> Result<f64, String> {
    let mut delta: HashMap<usize, f64> = HashMap::new();
    for st in &spec.steps {
        let decl = &spec.layers[st.layer];
        if st.kind == "ternary" {
            let da = *delta.get(&st.slots[0]).unwrap_or(&0.0);
            let db = *delta.get(&st.slots[1]).unwrap_or(&0.0);
            let dc = *delta.get(&st.slots[2]).unwrap_or(&0.0);
            let dout = if da == 0.0 && db == 0.0 && dc == 0.0 {
                0.0
            } else {
                match decl.constructor.as_str() {
                    "fma" => {
                        let ba = *val_bounds.get(&st.slots[0]).unwrap_or(&0.0);
                        let bb = *val_bounds.get(&st.slots[1]).unwrap_or(&0.0);
                        ba * db + bb * da + da * db + dc
                    }
                    other => {
                        return Err(format!(
                            "bound mode unsupported downstream ternary op '{other}'"
                        ))
                    }
                }
            };
            delta.insert(st.slots[3], dout);
        } else if st.kind == "binary" {
            let da = *delta.get(&st.slots[0]).unwrap_or(&0.0);
            let db = *delta.get(&st.slots[1]).unwrap_or(&0.0);
            let dout = if da == 0.0 && db == 0.0 {
                0.0
            } else {
                match decl.constructor.as_str() {
                    "add" | "sub" => da + db,
                    "mul" => {
                        let ba = *val_bounds.get(&st.slots[0]).unwrap_or(&0.0);
                        let bb = *val_bounds.get(&st.slots[1]).unwrap_or(&0.0);
                        ba * db + bb * da + da * db
                    }
                    "concat" => da.max(db),
                    "expandLike" => da,
                    "hypot" => da + db,
                    other => {
                        return Err(format!(
                            "bound mode unsupported downstream binary op '{other}'"
                        ))
                    }
                }
            };
            delta.insert(st.slots[2], dout);
        } else {
            let din = *delta.get(&st.slots[0]).unwrap_or(&0.0);
            let dout = if st.layer == candidate_li {
                let in_dim = decl.args[1].as_u64().unwrap_or(0) as usize;
                let out_dim = decl.args[2].as_u64().unwrap_or(0) as usize;
                let w = &spec.weights[&candidate_li.to_string()];
                let op_gain = linear_op_norm_inf(w, in_dim, out_dim);
                din * op_gain + layer_init_err
            } else if din > 0.0 {
                let bi = *val_bounds.get(&st.slots[0]).unwrap_or(&0.0);
                let l = unary_lipschitz_const(decl, spec, st.layer, bi)?;
                din * l
            } else {
                0.0
            };
            delta.insert(st.slots[1], dout);
        }
    }
    Ok(*delta.get(&spec.output_slot).unwrap_or(&0.0))
}

fn measure_cases_max_diff(
    base_graph: &PreparedGraph,
    base: &SpecV1,
    cand: &SpecV1,
    cases: &[EvalCase],
) -> Result<f64, String> {
    let g_cand = PreparedGraph::compile(cand)?;
    let mut max_d = 0.0f64;
    for c in cases {
        let ob = base_graph.run_once(base, Some(&c.inputs))?;
        let oc = g_cand.run_once(cand, Some(&c.inputs))?;
        for (&a, &b) in ob.values.iter().zip(oc.values.iter()) {
            let d = ((a as f64) - (b as f64)).abs();
            if d > max_d {
                max_d = d;
            }
        }
    }
    Ok(max_d)
}

pub fn quant_budget(
    spec: &SpecV1,
    cases_override: Option<&[EvalCase]>,
    tol: f64,
    scheme: QuantScheme,
    mode: BudgetMode,
) -> Result<QuantBudgetReport, String> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!("invalid quant budget tol: {tol}"));
    }
    let val = validate_spec(spec, mode == BudgetMode::Bound);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }

    let cases = match cases_override {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => default_cases_from_spec(spec),
    };

    let active_layers: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    struct CandLayer {
        li: usize,
        w_orig: Vec<f32>,
        w_quant: Vec<f32>,
        saved_bytes: usize,
        bytes_fp32: usize,
        bytes_quant: usize,
        est_delta: f64,
    }

    let mut cands = Vec::new();
    let val_bounds = if mode == BudgetMode::Bound {
        let init_b = input_slot_bounds(spec, &cases)?;
        let empty_shapes = std::collections::BTreeMap::new();
        let slot_shapes = val
            .explain
            .as_ref()
            .map(|e| &e.slot_shapes)
            .unwrap_or(&empty_shapes);
        Some(forward_value_bounds(spec, &init_b, slot_shapes)?)
    } else {
        None
    };

    for (li, decl) in spec.layers.iter().enumerate() {
        if !active_layers.contains(&li) || decl.constructor != "linear" {
            continue;
        }
        let Some(w) = spec.weights.get(&li.to_string()) else {
            continue;
        };
        let (wq, _sc, bq) = quantize_weights(w, scheme);
        let b32 = w.len() * 4;
        if b32 <= bq {
            continue;
        }
        let saved = b32 - bq;

        let est_delta = if let Some(ref vb) = val_bounds {
            let in_dim = decl.args[1].as_u64().unwrap_or(0) as usize;
            let out_dim = decl.args[2].as_u64().unwrap_or(0) as usize;
            let bias = decl.args.get(3).and_then(|v| v.as_bool()).unwrap_or(false);
            let mut in_bound = 0.0f64;
            for st in &spec.steps {
                if st.layer == li {
                    in_bound = in_bound.max(*vb.get(&st.slots[0]).unwrap_or(&0.0));
                }
            }
            let n_w = in_dim * out_dim;
            let dw: Vec<f32> = w[..n_w].iter().zip(wq[..n_w].iter()).map(|(&a, &b)| a - b).collect();
            let err_w = linear_op_norm_inf(&dw, in_dim, out_dim);
            let err_b = if bias {
                w[n_w..]
                    .iter()
                    .zip(wq[n_w..].iter())
                    .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
                    .fold(0.0f64, f64::max)
            } else {
                0.0
            };
            let local_err = err_w * in_bound + err_b;
            analytical_layer_contrib(spec, li, local_err, vb)?
        } else {
            0.0
        };

        cands.push(CandLayer {
            li,
            w_orig: w.clone(),
            w_quant: wq,
            saved_bytes: saved,
            bytes_fp32: b32,
            bytes_quant: bq,
            est_delta,
        });
    }

    let base_graph = PreparedGraph::compile(spec)?;
    let mut planned_spec = spec.clone();
    let mut quantized_set: HashSet<usize> = HashSet::new();
    let mut total_saved = 0usize;
    let mut total_delta = 0.0f64;

    match mode {
        BudgetMode::Bound => {
            cands.sort_by(|a, b| {
                let ra = (a.saved_bytes as f64) / a.est_delta.max(1e-12);
                let rb = (b.saved_bytes as f64) / b.est_delta.max(1e-12);
                rb.total_cmp(&ra)
            });
            for c in &cands {
                if total_delta + c.est_delta <= tol + 1e-12 {
                    quantized_set.insert(c.li);
                    planned_spec.weights.insert(c.li.to_string(), c.w_quant.clone());
                    total_saved += c.saved_bytes;
                    total_delta += c.est_delta;
                }
            }
        }
        BudgetMode::Measured => {
            let mut remaining: Vec<usize> = (0..cands.len()).collect();
            while !remaining.is_empty() {
                let mut best_pos: Option<usize> = None;
                let mut best_ratio = -1.0f64;
                let mut best_new_delta = total_delta;

                for (pos, &ci) in remaining.iter().enumerate() {
                    let c = &cands[ci];
                    let mut trial = planned_spec.clone();
                    trial.weights.insert(c.li.to_string(), c.w_quant.clone());
                    let new_d = measure_cases_max_diff(&base_graph, spec, &trial, &cases)?;
                    if new_d <= tol + 1e-12 {
                        let inc = (new_d - total_delta).max(1e-12);
                        let ratio = (c.saved_bytes as f64) / inc;
                        if ratio > best_ratio {
                            best_ratio = ratio;
                            best_pos = Some(pos);
                            best_new_delta = new_d;
                        }
                    }
                }

                if let Some(pos) = best_pos {
                    let ci = remaining.remove(pos);
                    cands[ci].est_delta = best_new_delta - total_delta;
                    quantized_set.insert(cands[ci].li);
                    planned_spec
                        .weights
                        .insert(cands[ci].li.to_string(), cands[ci].w_quant.clone());
                    total_saved += cands[ci].saved_bytes;
                    total_delta = best_new_delta;
                } else {
                    break;
                }
            }
        }
    }

    let verified_delta = measure_cases_max_diff(&base_graph, spec, &planned_spec, &cases)?;

    let mut decisions = Vec::with_capacity(cands.len());
    let mut quantized_layers = Vec::new();
    let mut kept_fp32_layers = Vec::new();
    for c in &cands {
        let is_q = quantized_set.contains(&c.li);
        if is_q {
            quantized_layers.push(c.li);
        } else {
            kept_fp32_layers.push(c.li);
            planned_spec.weights.insert(c.li.to_string(), c.w_orig.clone());
        }
        decisions.push(BudgetLayerDecision {
            layer_index: c.li,
            precision: if is_q { scheme.as_str() } else { "fp32" },
            weight_count: c.w_orig.len(),
            bytes_fp32: c.bytes_fp32,
            bytes_after: if is_q { c.bytes_quant } else { c.bytes_fp32 },
            saved_bytes: if is_q { c.saved_bytes } else { 0 },
            estimated_output_delta: c.est_delta,
        });
    }
    quantized_layers.sort_unstable();
    kept_fp32_layers.sort_unstable();
    decisions.sort_by_key(|d| d.layer_index);

    Ok(QuantBudgetReport {
        ok: verified_delta <= tol + 1e-9,
        mode,
        scheme: scheme.as_str(),
        tol,
        quantized_layers,
        kept_fp32_layers,
        total_saved_bytes: total_saved,
        total_estimated_delta: total_delta,
        verified_delta,
        decisions,
        plan_sha256: sha256_json(&planned_spec),
        planned_spec,
    })
}
