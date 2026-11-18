use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Instant;

use crate::catalog::is_int_constructor;
use crate::engine::PreparedGraph;
use crate::profile::profile_spec;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl};
use crate::validate::validate_spec;

#[derive(Debug, Clone, Serialize)]
pub struct OpCostRow {
    pub step: usize,
    pub layer: String,
    pub kind: String,
    pub in_shapes: Vec<[usize; 4]>,
    pub out_shape: [usize; 4],
    pub out_bytes: usize,
    pub ms_median: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share_pct: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub clamped_zero: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HotspotInfo {
    pub step: usize,
    pub layer: String,
    pub ms_median: f64,
    pub share_pct: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CostModelReport {
    pub ok: bool,
    pub iterations: usize,
    pub per_op: Vec<OpCostRow>,
    pub predicted_total_ms: f64,
    pub measured_total_ms: f64,
    pub prediction_error_pct: Option<f64>,
    pub hotspot: Option<HotspotInfo>,
}

fn measure_steady_median_ms(spec: &SpecV1, iterations: usize) -> Result<f64, String> {
    let graph = PreparedGraph::compile(spec)?;
    let _ = graph.run_once(spec, None)?; // cold warmup
    let runs = iterations.saturating_sub(1).max(1);
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = graph.run_once(spec, None)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let mid = times.len() / 2;
    let med = if times.len() % 2 == 1 {
        times[mid]
    } else {
        0.5 * (times[mid - 1] + times[mid])
    };
    Ok(med)
}

fn fill_f32_vals(n: usize, seed_offset: usize) -> Vec<Value> {
    (0..n)
        .map(|i| {
            let v = (((i + seed_offset) % 17) as f64) * 0.1 - 0.5;
            Value::from((v * 1000.0).round() / 1000.0)
        })
        .collect()
}

fn fill_pos_f32_vals(n: usize, seed_offset: usize) -> Vec<Value> {
    (0..n)
        .map(|i| {
            let v = (((i + seed_offset) % 9) as f64) * 0.15 + 0.25;
            Value::from((v * 1000.0).round() / 1000.0)
        })
        .collect()
}

fn fill_prob_vals(shape: [usize; 4]) -> Vec<Value> {
    let b = shape[0].max(1);
    let f = shape[1].max(1);
    let total = shape.iter().product();
    let mut out = Vec::with_capacity(total);
    if f == 1 {
        out.resize(total, Value::from(1.0));
        return out;
    }
    let step = (0.5 / (f as f64)).max(1e-4);
    let tail = 1.0 - step * ((f - 1) as f64);
    for _ in 0..b {
        for _ in 0..(f - 1) {
            out.push(Value::from(step));
        }
        out.push(Value::from(tail));
    }
    out.resize(total, Value::from(1.0 / (f as f64)));
    out
}

fn fill_int_vals(n: usize, max_val: usize) -> Vec<Value> {
    let m = max_val.max(2);
    (0..n).map(|i| Value::from(((i % (m - 1)) + 1) as i64)).collect()
}

pub fn run_cost_model(spec: &SpecV1, iterations: usize) -> Result<CostModelReport, String> {
    if iterations < 2 {
        return Err("cost-model: iterations must be >= 2".into());
    }
    let val = validate_spec(spec, true);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }
    let explain = val
        .explain
        .ok_or_else(|| "cost-model: missing explain report".to_string())?;

    let max_id = spec
        .layers
        .iter()
        .filter_map(|l| l.args.first().and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(0) as usize;

    let mut per_op = Vec::with_capacity(explain.steps.len());

    for (i, ex) in explain.steps.iter().enumerate() {
        let st_decl = &spec.steps[i];
        let layer_decl = &spec.layers[st_decl.layer];
        let s0 = ex
            .in_shapes
            .first()
            .and_then(|o| *o)
            .ok_or_else(|| format!("step {i} missing input shape"))?;
        let out_shape = ex
            .out_shape
            .ok_or_else(|| format!("step {i} missing output shape"))?;
        let elem_bytes = if ex.dtype_out == "i64" { 8 } else { 4 };
        let out_bytes: usize = out_shape.iter().product::<usize>() * elem_bytes;

        let n0: usize = s0.iter().product();
        let n_out: usize = out_shape.iter().product();
        let is_int = is_int_constructor(&layer_decl.constructor);
        let is_emb = layer_decl.constructor == "embedding";

        let mut clamped_zero = false;
        let ms = if st_decl.kind == "unary" {
            let in_vals = if is_emb {
                let vocab = layer_decl
                    .args
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(16) as usize;
                (0..n0).map(|idx| Value::from((idx % vocab.max(1)) as i64)).collect()
            } else if is_int && layer_decl.constructor != "intFromFloat" {
                fill_int_vals(n0, 8)
            } else if matches!(layer_decl.constructor.as_str(), "log" | "sqrt" | "normalize") {
                fill_pos_f32_vals(n0, i)
            } else if layer_decl.constructor == "entropy" {
                fill_prob_vals(s0)
            } else {
                fill_f32_vals(n0, i)
            };

            let mut weights = BTreeMap::new();
            if let Some(w) = spec.weights.get(&st_decl.layer.to_string()) {
                weights.insert("0".to_string(), w.clone());
            }

            if ex.dtype_out == "f32" {
                let mut inputs = BTreeMap::new();
                inputs.insert("0".to_string(), in_vals);
                inputs.insert("1".to_string(), vec![Value::from(0.0); n_out]);
                let micro = SpecV1 {
                    num_slots: 4,
                    layers: vec![
                        layer_decl.clone(),
                        LayerDecl {
                            constructor: "add".into(),
                            args: vec![Value::from(max_id + 10)],
                        },
                    ],
                    steps: vec![
                        StepDecl {
                            kind: "unary".into(),
                            layer: 0,
                            slots: vec![0, 2],
                        },
                        StepDecl {
                            kind: "binary".into(),
                            layer: 1,
                            slots: vec![2, 1, 3],
                        },
                    ],
                    output_slot: 3,
                    ports: vec![
                        PortDecl {
                            slot: 0,
                            role: "observation".into(),
                            shape: s0,
                            layout: "any_rank4".into(),
                        },
                        PortDecl {
                            slot: 1,
                            role: "observation".into(),
                            shape: out_shape,
                            layout: "any_rank4".into(),
                        },
                    ],
                    weights,
                    inputs,
                };

                let mut base_inputs = BTreeMap::new();
                base_inputs.insert("0".to_string(), fill_f32_vals(n_out, i + 1));
                base_inputs.insert("1".to_string(), vec![Value::from(0.0); n_out]);
                let base_add = SpecV1 {
                    num_slots: 3,
                    layers: vec![LayerDecl {
                        constructor: "add".into(),
                        args: vec![Value::from(max_id + 11)],
                    }],
                    steps: vec![StepDecl {
                        kind: "binary".into(),
                        layer: 0,
                        slots: vec![0, 1, 2],
                    }],
                    output_slot: 2,
                    ports: vec![
                        PortDecl {
                            slot: 0,
                            role: "observation".into(),
                            shape: out_shape,
                            layout: "any_rank4".into(),
                        },
                        PortDecl {
                            slot: 1,
                            role: "observation".into(),
                            shape: out_shape,
                            layout: "any_rank4".into(),
                        },
                    ],
                    weights: BTreeMap::new(),
                    inputs: base_inputs,
                };
                let m_ms = measure_steady_median_ms(&micro, iterations)?;
                let b_ms = measure_steady_median_ms(&base_add, iterations)?;
                let diff = m_ms - b_ms;
                if diff < 0.0 {
                    clamped_zero = true;
                    0.0
                } else {
                    diff
                }
            } else {
                // Unary producing i64 (e.g. intFromFloat) -> convert back to f32 for runner output
                let mut inputs = BTreeMap::new();
                inputs.insert("0".to_string(), in_vals);
                let micro = SpecV1 {
                    num_slots: 3,
                    layers: vec![
                        layer_decl.clone(),
                        LayerDecl {
                            constructor: "intToFloat".into(),
                            args: vec![Value::from(max_id + 12)],
                        },
                    ],
                    steps: vec![
                        StepDecl {
                            kind: "unary".into(),
                            layer: 0,
                            slots: vec![0, 1],
                        },
                        StepDecl {
                            kind: "unary".into(),
                            layer: 1,
                            slots: vec![1, 2],
                        },
                    ],
                    output_slot: 2,
                    ports: vec![PortDecl {
                        slot: 0,
                        role: "observation".into(),
                        shape: s0,
                        layout: "any_rank4".into(),
                    }],
                    weights,
                    inputs,
                };
                measure_steady_median_ms(&micro, iterations)?
            }
        } else {
            let s1 = ex.in_shapes.get(1).and_then(|o| *o).unwrap_or(s0);
            let n1: usize = s1.iter().product();
            let mut inputs = BTreeMap::new();
            if is_int {
                inputs.insert("0".to_string(), fill_int_vals(n0, 8));
                inputs.insert("1".to_string(), fill_int_vals(n1, 8));
                let micro = SpecV1 {
                    num_slots: 4,
                    layers: vec![
                        layer_decl.clone(),
                        LayerDecl {
                            constructor: "intToFloat".into(),
                            args: vec![Value::from(max_id + 13)],
                        },
                    ],
                    steps: vec![
                        StepDecl {
                            kind: "binary".into(),
                            layer: 0,
                            slots: vec![0, 1, 2],
                        },
                        StepDecl {
                            kind: "unary".into(),
                            layer: 1,
                            slots: vec![2, 3],
                        },
                    ],
                    output_slot: 3,
                    ports: vec![
                        PortDecl {
                            slot: 0,
                            role: "observation".into(),
                            shape: s0,
                            layout: "any_rank4".into(),
                        },
                        PortDecl {
                            slot: 1,
                            role: "observation".into(),
                            shape: s1,
                            layout: "any_rank4".into(),
                        },
                    ],
                    weights: BTreeMap::new(),
                    inputs,
                };
                measure_steady_median_ms(&micro, iterations)?
            } else {
                let (v0, v1) = if matches!(layer_decl.constructor.as_str(), "crossEntropy" | "klDivergence") {
                    (fill_prob_vals(s0), fill_prob_vals(s1))
                } else {
                    let mut rhs = fill_f32_vals(n1, i + 3);
                    if layer_decl.constructor == "div" {
                        rhs = vec![Value::from(1.5); n1];
                    }
                    (fill_f32_vals(n0, i), rhs)
                };
                inputs.insert("0".to_string(), v0);
                inputs.insert("1".to_string(), v1);
                let micro = SpecV1 {
                    num_slots: 3,
                    layers: vec![layer_decl.clone()],
                    steps: vec![StepDecl {
                        kind: "binary".into(),
                        layer: 0,
                        slots: vec![0, 1, 2],
                    }],
                    output_slot: 2,
                    ports: vec![
                        PortDecl {
                            slot: 0,
                            role: "observation".into(),
                            shape: s0,
                            layout: "any_rank4".into(),
                        },
                        PortDecl {
                            slot: 1,
                            role: "observation".into(),
                            shape: s1,
                            layout: "any_rank4".into(),
                        },
                    ],
                    weights: BTreeMap::new(),
                    inputs,
                };
                measure_steady_median_ms(&micro, iterations)?
            }
        };

        per_op.push(OpCostRow {
            step: i,
            layer: layer_decl.constructor.clone(),
            kind: st_decl.kind.clone(),
            in_shapes: ex.in_shapes.iter().filter_map(|&s| s).collect(),
            out_shape,
            out_bytes,
            ms_median: (ms * 10000.0).round() / 10000.0,
            share_pct: None,
            clamped_zero,
        });
    }

    let prof = profile_spec(spec, 1, iterations.saturating_sub(1).max(1))?;
    let measured = prof.median_ms;
    let predicted: f64 = per_op.iter().map(|r| r.ms_median).sum();
    let predicted_rounded = (predicted * 10000.0).round() / 10000.0;
    let err_pct = if measured > 0.0 {
        Some((((predicted - measured).abs() / measured) * 1000.0).round() / 10.0)
    } else {
        None
    };

    per_op.sort_by(|a, b| b.ms_median.total_cmp(&a.ms_median));
    for row in &mut per_op {
        if predicted > 0.0 {
            row.share_pct = Some(((row.ms_median / predicted) * 1000.0).round() / 10.0);
        }
    }
    let hotspot = per_op.first().map(|r| HotspotInfo {
        step: r.step,
        layer: r.layer.clone(),
        ms_median: r.ms_median,
        share_pct: r.share_pct,
    });

    Ok(CostModelReport {
        ok: true,
        iterations,
        per_op,
        predicted_total_ms: predicted_rounded,
        measured_total_ms: measured,
        prediction_error_pct: err_pct,
        hotspot,
    })
}
