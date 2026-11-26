use burn::tensor::{Device, Tensor, TensorData};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::layers::{SlotValue, build_layers, exec_layer};
use crate::spec::{LayerDecl, SpecV1};

const F32_EXACT_INT_MAX: i64 = 16_777_216; // 2^24

pub fn parse_f32_vector_str(raw: &str, name: &str) -> Result<(Vec<f32>, Vec<String>), String> {
    if raw.trim().is_empty() {
        return Err(format!("{name}: empty vector"));
    }
    let mut vals = Vec::new();
    let mut warnings = Vec::new();
    for (idx, tok) in raw.split(',').enumerate() {
        let s = tok.trim();
        let v64: f64 = s
            .parse()
            .map_err(|_| format!("{name}[{idx}]: not a finite number: '{s}'"))?;
        if !v64.is_finite() {
            return Err(format!("{name}[{idx}]: must be finite, got '{s}'"));
        }
        let v32 = v64 as f32;
        if !v32.is_finite() {
            return Err(format!("{name}[{idx}]: value '{s}' overflows f32"));
        }
        if v64 != 0.0 && v32 == 0.0 {
            warnings.push(format!("{name}[{idx}] ({s}) underflowed to f32 0.0"));
        } else if (v32 as f64 - v64).abs() > 0.0 {
            warnings.push(format!("{name}[{idx}] ({s}) rounded to f32 {v32}"));
        }
        vals.push(if v32 == 0.0 { 0.0 } else { v32 });
    }
    Ok((vals, warnings))
}

pub fn parse_f32_matrix_str(
    raw: &str,
    name: &str,
) -> Result<(Vec<f32>, usize, usize, Vec<String>), String> {
    if raw.trim().is_empty() {
        return Err(format!("{name}: empty matrix"));
    }
    let mut flat = Vec::new();
    let mut warnings = Vec::new();
    let mut cols: Option<usize> = None;
    let mut rows = 0usize;
    for (ri, row_s) in raw.split(';').enumerate() {
        let (row_vals, mut row_warn) = parse_f32_vector_str(row_s, &format!("{name}[row {ri}]"))?;
        if let Some(expected_c) = cols {
            if row_vals.len() != expected_c {
                return Err(format!(
                    "{name}: non-rectangular matrix (row {ri} has {} cols, expected {expected_c})",
                    row_vals.len()
                ));
            }
        } else {
            cols = Some(row_vals.len());
        }
        flat.extend(row_vals);
        warnings.append(&mut row_warn);
        rows += 1;
    }
    Ok((flat, rows, cols.unwrap_or(0), warnings))
}

pub fn parse_i64_vector_str(raw: &str, name: &str) -> Result<Vec<i64>, String> {
    if raw.trim().is_empty() {
        return Err(format!("{name}: empty integer vector"));
    }
    let mut out = Vec::new();
    for (idx, tok) in raw.split(',').enumerate() {
        let s = tok.trim();
        let v: i64 = s
            .parse()
            .map_err(|_| format!("{name}[{idx}]: invalid i64 integer '{s}'"))?;
        out.push(v);
    }
    Ok(out)
}

pub fn parse_i64_matrix_str(raw: &str, name: &str) -> Result<(Vec<i64>, usize, usize), String> {
    if raw.trim().is_empty() {
        return Err(format!("{name}: empty integer matrix"));
    }
    let mut flat = Vec::new();
    let mut cols: Option<usize> = None;
    let mut rows = 0usize;
    for (ri, row_s) in raw.split(';').enumerate() {
        let row_vals = parse_i64_vector_str(row_s, &format!("{name}[row {ri}]"))?;
        if let Some(expected_c) = cols {
            if row_vals.len() != expected_c {
                return Err(format!(
                    "{name}: non-rectangular integer matrix (row {ri} has {} cols, expected {expected_c})",
                    row_vals.len()
                ));
            }
        } else {
            cols = Some(row_vals.len());
        }
        flat.extend(row_vals);
        rows += 1;
    }
    Ok((flat, rows, cols.unwrap_or(0)))
}

fn make_f32_slot(vals: &[f32], shape: [usize; 4], device: &Device) -> SlotValue {
    SlotValue::F32(Tensor::<4>::from_data(
        TensorData::new(vals.to_vec(), shape),
        device,
    ))
}

fn run_single_layer(
    constructor: &str,
    extra_args: Vec<Value>,
    inputs: &[SlotValue],
) -> Result<SlotValue, String> {
    let device = Device::flex();
    let mut args = Vec::with_capacity(1 + extra_args.len());
    args.push(Value::from(0u64));
    args.extend(extra_args);
    let spec = SpecV1 {
        num_slots: inputs.len() + 1,
        layers: vec![LayerDecl {
            constructor: constructor.to_string(),
            args,
        }],
        steps: vec![],
        output_slot: inputs.len(),
        ports: vec![],
        weights: BTreeMap::new(),
        inputs: BTreeMap::new(),
    };
    let (layers, _) = build_layers(&spec, &device, &[])?;
    let layer = layers
        .first()
        .ok_or_else(|| format!("internal error: failed to construct layer '{constructor}'"))?;
    exec_layer(layer, inputs, &device)
}

fn extract_f32_slot(val: SlotValue, op: &str) -> Result<(Vec<f32>, [usize; 4]), String> {
    let t = val.as_f32(op)?;
    let shape = t.dims();
    let data = t
        .into_data()
        .try_into_vec::<f32>()
        .map_err(|e| format!("{op}: tensor read failed: {e}"))?;
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{op}: non-finite output at index {i}: {v}"));
        }
    }
    let clean: Vec<f32> = data
        .into_iter()
        .map(|v| if v == 0.0 { 0.0 } else { v })
        .collect();
    Ok((clean, shape))
}

type BroadcastPair<T> = (Vec<T>, Vec<T>, Option<&'static str>);

fn broadcast_f32(a: &[f32], b: &[f32]) -> Result<BroadcastPair<f32>, String> {
    if a.len() == b.len() {
        return Ok((a.to_vec(), b.to_vec(), None));
    }
    if a.len() == 1 {
        return Ok((vec![a[0]; b.len()], b.to_vec(), Some("a")));
    }
    if b.len() == 1 {
        return Ok((a.to_vec(), vec![b[0]; a.len()], Some("b")));
    }
    Err(format!(
        "length mismatch: a has {} values, b has {} values",
        a.len(),
        b.len()
    ))
}

fn broadcast_i64(a: &[i64], b: &[i64]) -> Result<BroadcastPair<i64>, String> {
    if a.len() == b.len() {
        return Ok((a.to_vec(), b.to_vec(), None));
    }
    if a.len() == 1 {
        return Ok((vec![a[0]; b.len()], b.to_vec(), Some("a")));
    }
    if b.len() == 1 {
        return Ok((a.to_vec(), vec![b[0]; a.len()], Some("b")));
    }
    Err(format!(
        "length mismatch: a has {} values, b has {} values",
        a.len(),
        b.len()
    ))
}

/// Execute a single spec-free `arith` operation on Burn `Device::flex()`.
pub fn eval_arith_op(
    op: &str,
    a_raw: &str,
    b_raw: Option<&str>,
) -> Result<Value, String> {
    let device = Device::flex();
    match op {
        "add" | "sub" | "mul" | "div" | "dot" | "dist" | "cos" | "xent" | "kl" => {
            let b_str = b_raw.ok_or_else(|| format!("arith {op}: requires two vector arguments"))?;
            let (a_v, mut warn) = parse_f32_vector_str(a_raw, "a")?;
            let (b_v, mut wb) = parse_f32_vector_str(b_str, "b")?;
            warn.append(&mut wb);
            let (a_eff, b_eff, bcast) = broadcast_f32(&a_v, &b_v)?;
            let sh = [1, a_eff.len(), 1, 1];
            let sa = make_f32_slot(&a_eff, sh, &device);
            let sb = make_f32_slot(&b_eff, sh, &device);
            let (ctor, extra) = match op {
                "add" => ("add", vec![]),
                "sub" => ("sub", vec![]),
                "mul" => ("mul", vec![]),
                "div" => ("div", vec![]),
                "dot" => ("dot", vec![]),
                "dist" => ("l2Distance", vec![]),
                "cos" => ("cosineSimilarity", vec![Value::from(1e-8)]),
                "xent" => ("crossEntropy", vec![]),
                "kl" => ("klDivergence", vec![]),
                _ => return Err(format!("unknown arith op '{op}'")),
            };
            let out = run_single_layer(ctor, extra, &[sa, sb])?;
            let (vals, _osh) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": op,
                "values": vals,
            });
            if let Some(bc) = bcast {
                obj["broadcast"] = Value::String(bc.to_string());
            }
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "sqrt" | "exp" | "log" | "abs" | "norm" | "normalize" | "entropy" | "sum" | "mean"
        | "min" | "max" | "var" | "std" | "var_sample" | "std_sample" => {
            if b_raw.is_some() {
                return Err(format!("arith {op}: unexpected second argument"));
            }
            let (a_v, warn) = parse_f32_vector_str(a_raw, "a")?;
            let sh = [1, a_v.len(), 1, 1];
            let sa = make_f32_slot(&a_v, sh, &device);
            let ctor = match op {
                "sqrt" => "sqrt",
                "exp" => "exp",
                "log" => "log",
                "abs" => "abs",
                "norm" => "l2Norm",
                "normalize" => "normalize",
                "entropy" => "entropy",
                "sum" => "sum",
                "mean" => "mean",
                "min" => "min",
                "max" => "max",
                "var" => "variancePopulation",
                "std" => "stdPopulation",
                "var_sample" => "varianceSample",
                "std_sample" => "stdSample",
                _ => return Err(format!("unknown arith op '{op}'")),
            };
            let out = run_single_layer(ctor, vec![], &[sa])?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": op,
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "clamp" => {
            let bounds_s = b_raw
                .ok_or_else(|| "arith clamp: requires <values> <min>,<max>".to_string())?;
            let (a_v, mut warn) = parse_f32_vector_str(a_raw, "a")?;
            let (bounds, mut wb) = parse_f32_vector_str(bounds_s, "bounds")?;
            warn.append(&mut wb);
            if bounds.len() != 2 {
                return Err(format!(
                    "arith clamp: bounds must be '<min>,<max>', got {} values",
                    bounds.len()
                ));
            }
            if bounds[0] > bounds[1] {
                return Err(format!(
                    "arith clamp: min ({}) must be <= max ({})",
                    bounds[0], bounds[1]
                ));
            }
            let sh = [1, a_v.len(), 1, 1];
            let sa = make_f32_slot(&a_v, sh, &device);
            let out = run_single_layer(
                "clamp",
                vec![Value::from(bounds[0] as f64), Value::from(bounds[1] as f64)],
                &[sa],
            )?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": "clamp",
                "min": bounds[0],
                "max": bounds[1],
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "matmul" => {
            let b_str = b_raw.ok_or_else(|| "arith matmul: requires two matrices".to_string())?;
            let (a_flat, m, k, mut warn) = parse_f32_matrix_str(a_raw, "a")?;
            let (b_flat, k2, n, mut wb) = parse_f32_matrix_str(b_str, "b")?;
            warn.append(&mut wb);
            if k != k2 {
                return Err(format!(
                    "matmul shape mismatch: a is {m}x{k}, b is {k2}x{n}"
                ));
            }
            let sa = make_f32_slot(&a_flat, [1, 1, m, k], &device);
            let sb = make_f32_slot(&b_flat, [1, 1, k, n], &device);
            let out = run_single_layer("matmul", vec![], &[sa, sb])?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": "matmul",
                "shape": [m, n],
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "sum_axis" | "mean_axis" | "min_axis" | "max_axis" => {
            let ax_str = b_raw
                .ok_or_else(|| format!("arith {op}: requires matrix and axis (0 or 1)"))?;
            let ax: usize = ax_str
                .trim()
                .parse()
                .map_err(|_| format!("arith {op}: invalid axis '{ax_str}' (expected 0 or 1)"))?;
            if ax > 1 {
                return Err(format!("arith {op}: axis must be 0 or 1, got {ax}"));
            }
            let (a_flat, m, n, warn) = parse_f32_matrix_str(a_raw, "a")?;
            let sa = make_f32_slot(&a_flat, [1, 1, m, n], &device);
            let ctor = match op {
                "sum_axis" => "sumAxis",
                "mean_axis" => "meanAxis",
                "min_axis" => "minAxis",
                "max_axis" => "maxAxis",
                _ => return Err(format!("unknown arith axis op '{op}'")),
            };
            let out = run_single_layer(ctor, vec![Value::from(ax + 2)], &[sa])?;
            let (vals, osh) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": op,
                "axis": ax,
                "shape": [osh[2], osh[3]],
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        other => Err(format!(
            "unknown arith op '{other}' (supported: add, sub, mul, div, dot, dist, cos, xent, kl, sqrt, exp, log, abs, norm, normalize, entropy, sum, mean, min, max, var, std, var_sample, std_sample, clamp, matmul, sum_axis, mean_axis, min_axis, max_axis)"
        )),
    }
}

/// Execute a single spec-free exact `i64` integer operation on Burn `Device::flex()`.
pub fn eval_int_op(
    op: &str,
    a_raw: &str,
    b_raw: Option<&str>,
    wrap: bool,
    allow_inexact: bool,
) -> Result<Value, String> {
    let device = Device::flex();
    match op {
        "add" | "sub" | "mul" | "div" | "rem" => {
            let b_str = b_raw.ok_or_else(|| format!("int {op}: requires two integer vectors"))?;
            let a_v = parse_i64_vector_str(a_raw, "a")?;
            let b_v = parse_i64_vector_str(b_str, "b")?;
            let (a_eff, b_eff, bcast) = broadcast_i64(&a_v, &b_v)?;
            let vals: Vec<i64> = if wrap {
                let mut res = Vec::with_capacity(a_eff.len());
                for (i, (&x, &y)) in a_eff.iter().zip(b_eff.iter()).enumerate() {
                    let v = match op {
                        "add" => x.wrapping_add(y),
                        "sub" => x.wrapping_sub(y),
                        "mul" => x.wrapping_mul(y),
                        "div" => {
                            if y == 0 {
                                return Err(format!("int div: division by zero at index {i}"));
                            }
                            x.wrapping_div(y)
                        }
                        "rem" => {
                            if y == 0 {
                                return Err(format!("int rem: division by zero at index {i}"));
                            }
                            x.wrapping_rem(y)
                        }
                        _ => unreachable!(),
                    };
                    res.push(v);
                }
                res
            } else {
                let sh = [1, a_eff.len(), 1, 1];
                let sa = SlotValue::I64(a_eff, sh);
                let sb = SlotValue::I64(b_eff, sh);
                let ctor = match op {
                    "add" => "intAdd",
                    "sub" => "intSub",
                    "mul" => "intMul",
                    "div" => "intDiv",
                    "rem" => "intRem",
                    _ => unreachable!(),
                };
                let out = run_single_layer(ctor, vec![], &[sa, sb])?;
                let (slice, _) = out.as_i64(op)?;
                slice.to_vec()
            };
            let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            let mut obj = json!({
                "ok": true,
                "op": op,
                "wrap": wrap,
                "values": strs,
            });
            if let Some(bc) = bcast {
                obj["broadcast"] = Value::String(bc.to_string());
            }
            Ok(obj)
        }
        "matmul" => {
            let b_str = b_raw.ok_or_else(|| "int matmul: requires two integer matrices".to_string())?;
            let (a_flat, m, k) = parse_i64_matrix_str(a_raw, "a")?;
            let (b_flat, k2, n) = parse_i64_matrix_str(b_str, "b")?;
            if k != k2 {
                return Err(format!("int matmul shape mismatch: a is {m}x{k}, b is {k2}x{n}"));
            }
            let vals: Vec<i64> = if wrap {
                let mut res = vec![0i64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0i64;
                        for kk in 0..k {
                            acc = acc.wrapping_add(a_flat[i * k + kk].wrapping_mul(b_flat[kk * n + j]));
                        }
                        res[i * n + j] = acc;
                    }
                }
                res
            } else {
                let sa = SlotValue::I64(a_flat, [1, 1, m, k]);
                let sb = SlotValue::I64(b_flat, [1, 1, k, n]);
                let out = run_single_layer("intMatmul", vec![], &[sa, sb])?;
                let (slice, _) = out.as_i64("intMatmul")?;
                slice.to_vec()
            };
            let strs: Vec<String> = vals.iter().map(|v| v.to_string()).collect();
            Ok(json!({
                "ok": true,
                "op": "matmul",
                "wrap": wrap,
                "shape": [m, n],
                "values": strs,
            }))
        }
        "to-f32" => {
            if b_raw.is_some() {
                return Err("int to-f32: unexpected second argument".into());
            }
            let a_v = parse_i64_vector_str(a_raw, "a")?;
            let vals: Vec<f32> = if allow_inexact {
                a_v.iter()
                    .map(|&v| {
                        let f = v as f32;
                        if f == 0.0 { 0.0 } else { f }
                    })
                    .collect()
            } else {
                for (i, &v) in a_v.iter().enumerate() {
                    if v.unsigned_abs() > F32_EXACT_INT_MAX as u64 {
                        return Err(format!(
                            "int to-f32: value {v} at index {i} exceeds exact f32 integer range +/-2^24 (pass --allow-inexact to round)"
                        ));
                    }
                }
                let sh = [1, a_v.len(), 1, 1];
                let sa = SlotValue::I64(a_v, sh);
                let out = run_single_layer("intToFloat", vec![], &[sa])?;
                let (v, _) = extract_f32_slot(out, "to-f32")?;
                v
            };
            Ok(json!({
                "ok": true,
                "op": "to-f32",
                "allow_inexact": allow_inexact,
                "values": vals,
            }))
        }
        "from-f32" => {
            if b_raw.is_some() {
                return Err("int from-f32: unexpected second argument".into());
            }
            let (a_v, warn) = parse_f32_vector_str(a_raw, "a")?;
            let sh = [1, a_v.len(), 1, 1];
            let sa = make_f32_slot(&a_v, sh, &device);
            let out = run_single_layer("intFromFloat", vec![], &[sa])?;
            let (slice, _) = out.as_i64("from-f32")?;
            let strs: Vec<String> = slice.iter().map(|v| v.to_string()).collect();
            let mut obj = json!({
                "ok": true,
                "op": "from-f32",
                "values": strs,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        other => Err(format!(
            "unknown int op '{other}' (supported: add, sub, mul, div, rem, matmul, to-f32, from-f32)"
        )),
    }
}

/// Execute a single Lane-E elementwise `kernel` operation on Burn `Device::flex()`.
/// Strict same-length inputs (no broadcast).
pub fn eval_kernel_op(
    op: &str,
    a_raw: &str,
    b_raw: Option<&str>,
    c_raw: Option<&str>,
) -> Result<Value, String> {
    let device = Device::flex();
    match op {
        "sin" | "cos" | "tan" | "floor" | "ceil" | "round" => {
            if b_raw.is_some() || c_raw.is_some() {
                return Err(format!("kernel {op}: expects exactly 1 vector argument"));
            }
            let (a_v, warn) = parse_f32_vector_str(a_raw, "a")?;
            let sa = make_f32_slot(&a_v, [1, a_v.len(), 1, 1], &device);
            let out = run_single_layer(op, vec![], &[sa])?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": op,
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "atan2" | "powf" | "hypot" => {
            if c_raw.is_some() {
                return Err(format!("kernel {op}: expects exactly 2 vector arguments"));
            }
            let b_str = b_raw.ok_or_else(|| format!("kernel {op}: requires 2 vector arguments"))?;
            let (a_v, mut warn) = parse_f32_vector_str(a_raw, "a")?;
            let (b_v, mut wb) = parse_f32_vector_str(b_str, "b")?;
            warn.append(&mut wb);
            if a_v.len() != b_v.len() {
                return Err(format!(
                    "kernel {op}: length mismatch (a has {}, b has {}; no broadcast)",
                    a_v.len(),
                    b_v.len()
                ));
            }
            let sh = [1, a_v.len(), 1, 1];
            let sa = make_f32_slot(&a_v, sh, &device);
            let sb = make_f32_slot(&b_v, sh, &device);
            let out = run_single_layer(op, vec![], &[sa, sb])?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": op,
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        "fma" => {
            let b_str = b_raw.ok_or_else(|| "kernel fma: requires 3 vector arguments (a, b, c)".to_string())?;
            let c_str = c_raw.ok_or_else(|| "kernel fma: requires 3 vector arguments (a, b, c)".to_string())?;
            let (a_v, mut warn) = parse_f32_vector_str(a_raw, "a")?;
            let (b_v, mut wb) = parse_f32_vector_str(b_str, "b")?;
            let (c_v, mut wc) = parse_f32_vector_str(c_str, "c")?;
            warn.append(&mut wb);
            warn.append(&mut wc);
            if a_v.len() != b_v.len() || a_v.len() != c_v.len() {
                return Err(format!(
                    "kernel fma: length mismatch (a={}, b={}, c={}; no broadcast)",
                    a_v.len(),
                    b_v.len(),
                    c_v.len()
                ));
            }
            let sh = [1, a_v.len(), 1, 1];
            let sa = make_f32_slot(&a_v, sh, &device);
            let sb = make_f32_slot(&b_v, sh, &device);
            let sc = make_f32_slot(&c_v, sh, &device);
            let out = run_single_layer("fma", vec![], &[sa, sb, sc])?;
            let (vals, _) = extract_f32_slot(out, op)?;
            let mut obj = json!({
                "ok": true,
                "op": "fma",
                "values": vals,
            });
            if !warn.is_empty() {
                obj["precision_warnings"] = Value::from(warn);
            }
            Ok(obj)
        }
        other => Err(format!(
            "unknown kernel op '{other}' (supported: sin, cos, tan, floor, ceil, round, atan2, powf, hypot, fma)"
        )),
    }
}

fn value_to_csv_arg(
    val: &Value,
    prev_results: &[Value],
    expect_matrix: bool,
    ctx: &str,
) -> Result<String, String> {
    match val {
        Value::String(s) if s == "$prev" => {
            let last = prev_results
                .last()
                .ok_or_else(|| format!("{ctx}: '$prev' used on first batch entry"))?;
            result_to_csv(last, expect_matrix, ctx)
        }
        Value::String(s) if s.starts_with('$') => {
            let idx: usize = s[1..]
                .parse()
                .map_err(|_| format!("{ctx}: invalid register reference '{s}'"))?;
            let r = prev_results
                .get(idx)
                .ok_or_else(|| {
                    format!(
                        "{ctx}: register '{s}' out of range (only {} prior results)",
                        prev_results.len()
                    )
                })?;
            result_to_csv(r, expect_matrix, ctx)
        }
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err(format!("{ctx}: empty array argument"));
            }
            if arr.first().is_some_and(|v| v.is_array()) {
                let mut rows = Vec::with_capacity(arr.len());
                for (ri, r) in arr.iter().enumerate() {
                    let row_arr = r
                        .as_array()
                        .ok_or_else(|| format!("{ctx}: matrix row {ri} must be array"))?;
                    let cells: Result<Vec<String>, String> = row_arr
                        .iter()
                        .map(|c| scalar_to_str(c, ctx))
                        .collect();
                    rows.push(cells?.join(","));
                }
                Ok(rows.join(";"))
            } else {
                let cells: Result<Vec<String>, String> = arr
                    .iter()
                    .map(|c| scalar_to_str(c, ctx))
                    .collect();
                Ok(cells?.join(","))
            }
        }
        other => Err(format!("{ctx}: unsupported operand value {other}")),
    }
}

fn scalar_to_str(v: &Value, ctx: &str) -> Result<String, String> {
    match v {
        Value::Number(n) => Ok(n.to_string()),
        Value::String(s) => Ok(s.clone()),
        other => Err(format!("{ctx}: expected numeric/string element, got {other}")),
    }
}

fn result_to_csv(res: &Value, expect_matrix: bool, ctx: &str) -> Result<String, String> {
    let vals = res
        .get("values")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("{ctx}: referenced result missing 'values' array"))?;
    let cells: Result<Vec<String>, String> = vals.iter().map(|v| scalar_to_str(v, ctx)).collect();
    let cells = cells?;
    if expect_matrix
        && let Some(shape) = res.get("shape").and_then(|v| v.as_array())
        && shape.len() == 2
        && let (Some(r), Some(c)) = (shape[0].as_u64(), shape[1].as_u64())
        && (r as usize) * (c as usize) == cells.len()
        && c > 0
    {
        let c_u = c as usize;
        let rows: Vec<String> = cells
            .chunks(c_u)
            .map(|chunk| chunk.join(","))
            .collect();
        return Ok(rows.join(";"));
    }
    Ok(cells.join(","))
}

fn is_axis_arith_op(op: &str) -> bool {
    matches!(op, "sum_axis" | "mean_axis" | "min_axis" | "max_axis")
}

fn op_expects_matrix_a(kind: DirectBatchKind, op: &str) -> bool {
    match kind {
        DirectBatchKind::Arith => op == "matmul" || is_axis_arith_op(op),
        DirectBatchKind::Int => op == "matmul",
        DirectBatchKind::Kernel => false,
    }
}

fn op_expects_matrix_b(kind: DirectBatchKind, op: &str) -> bool {
    matches!(kind, DirectBatchKind::Arith | DirectBatchKind::Int) && op == "matmul"
}

fn parse_batch_opt_bool(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    ctx: &str,
) -> Result<bool, String> {
    match obj.get(key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(format!("{ctx}: field '{key}' must be a boolean, got {other}")),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectBatchKind {
    Arith,
    Int,
    Kernel,
}

/// Execute a batch JSON array of `{op, a, b?, c?, min?, max?, axis?, wrap?, allow_inexact?}`
/// supporting `"$prev"` and `"$<idx>"` register chaining.
pub fn eval_direct_batch(kind: DirectBatchKind, doc: &Value) -> Result<Value, String> {
    let items = doc
        .as_array()
        .or_else(|| doc.get("ops").and_then(|v| v.as_array()))
        .ok_or_else(|| "batch JSON must be an array of op objects (or {\"ops\": [...]})".to_string())?;
    if items.is_empty() {
        return Err("batch ops array must not be empty".into());
    }

    let mut results = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let ctx = format!("batch[{idx}]");
        let obj = item
            .as_object()
            .ok_or_else(|| format!("{ctx}: entry must be a JSON object"))?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{ctx}: missing 'op' string"))?;

        // Strict per-kind and per-op field allowlist (H5)
        for key in obj.keys() {
            let allowed = match kind {
                DirectBatchKind::Arith => match key.as_str() {
                    "op" | "a" | "b" => true,
                    "min" | "max" => op == "clamp",
                    "axis" => is_axis_arith_op(op),
                    _ => false,
                },
                DirectBatchKind::Int => {
                    matches!(key.as_str(), "op" | "a" | "b" | "wrap" | "allow_inexact")
                }
                DirectBatchKind::Kernel => match key.as_str() {
                    "op" | "a" | "b" => true,
                    "c" => op == "fma",
                    _ => false,
                },
            };
            if !allowed {
                return Err(format!(
                    "{ctx} ({op}): unknown or unsupported batch field '{key}'"
                ));
            }
        }

        let a_val = obj
            .get("a")
            .ok_or_else(|| format!("{ctx} ({op}): missing 'a' operand"))?;
        let a_csv = value_to_csv_arg(a_val, &results, op_expects_matrix_a(kind, op), &format!("{ctx}.a"))?;

        let b_csv = if kind == DirectBatchKind::Arith && op == "clamp" {
            let has_b = obj.contains_key("b");
            let has_min = obj.contains_key("min");
            let has_max = obj.contains_key("max");
            if has_b && (has_min || has_max) {
                return Err(format!(
                    "{ctx} (clamp): specify either 'b' or ('min' and 'max'), not both"
                ));
            }
            if has_b {
                Some(value_to_csv_arg(&obj["b"], &results, false, &format!("{ctx}.b"))?)
            } else if has_min || has_max {
                let mn = obj
                    .get("min")
                    .ok_or_else(|| format!("{ctx} (clamp): 'min' is required when 'max' is provided"))?;
                let mx = obj
                    .get("max")
                    .ok_or_else(|| format!("{ctx} (clamp): 'max' is required when 'min' is provided"))?;
                Some(format!(
                    "{},{}",
                    scalar_to_str(mn, &format!("{ctx}.min"))?,
                    scalar_to_str(mx, &format!("{ctx}.max"))?
                ))
            } else {
                None
            }
        } else if kind == DirectBatchKind::Arith && is_axis_arith_op(op) {
            if obj.contains_key("b") && obj.contains_key("axis") {
                return Err(format!(
                    "{ctx} ({op}): specify either 'axis' or 'b', not both"
                ));
            }
            if let Some(ax) = obj.get("axis") {
                Some(scalar_to_str(ax, &format!("{ctx}.axis"))?)
            } else if let Some(bv) = obj.get("b") {
                Some(value_to_csv_arg(bv, &results, false, &format!("{ctx}.b"))?)
            } else {
                None
            }
        } else if let Some(bv) = obj.get("b") {
            Some(value_to_csv_arg(bv, &results, op_expects_matrix_b(kind, op), &format!("{ctx}.b"))?)
        } else {
            None
        };

        let c_csv = match obj.get("c") {
            Some(cv) => Some(value_to_csv_arg(cv, &results, false, &format!("{ctx}.c"))?),
            None => None,
        };

        let res = match kind {
            DirectBatchKind::Arith => eval_arith_op(op, &a_csv, b_csv.as_deref())?,
            DirectBatchKind::Int => {
                let wrap = parse_batch_opt_bool(obj, "wrap", &ctx)?;
                let allow_inexact = parse_batch_opt_bool(obj, "allow_inexact", &ctx)?;
                eval_int_op(op, &a_csv, b_csv.as_deref(), wrap, allow_inexact)?
            }
            DirectBatchKind::Kernel => {
                eval_kernel_op(op, &a_csv, b_csv.as_deref(), c_csv.as_deref())?
            }
        };
        results.push(res);
    }

    Ok(json!({
        "ok": true,
        "count": results.len(),
        "results": results,
    }))
}
