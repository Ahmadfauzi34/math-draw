mod math_ops;
mod nn_ops;

use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::evidence::verify::EvalCase;
use crate::graph::validate::validate_spec;
use crate::runtime::engine::PreparedGraph;
use crate::spec::SpecV1;

/// All 87 `f32` constructors supported by the independent pure-Rust `f64` reference interpreter.
pub const SUPPORTED_REF_OPS: &[&str] = &[
    // Activations (15)
    "relu",
    "gelu",
    "sigmoid",
    "tanh",
    "hardSwish",
    "leakyRelu",
    "prelu",
    "swiGlu",
    "hardSigmoid",
    "softplus",
    "mish",
    "softmax",
    "logSoftmax",
    "glu",
    "sqrt",
    // Linear (1)
    "linear",
    // Norms (5)
    "batchNorm",
    "groupNorm",
    "instanceNorm",
    "layerNorm",
    "rmsNorm",
    // Convs (3)
    "conv1d",
    "conv2d",
    "convTranspose2d",
    // Embedding (1)
    "embedding",
    // Pools (5)
    "maxPool1d",
    "maxPool2d",
    "avgPool1d",
    "avgPool2d",
    "adaptiveAvgPool2d",
    // Custom (7)
    "featureNorm",
    "shiftUp",
    "shiftDown",
    "shiftLeft",
    "shiftRight",
    "ghost",
    "seBlock",
    // Binary f32 (6)
    "add",
    "sub",
    "mul",
    "matmul",
    "div",
    "concat",
    // Transform (1)
    "transpose",
    // Numeric primitives (15)
    "abs",
    "exp",
    "log",
    "clamp",
    "sin",
    "cos",
    "tan",
    "floor",
    "ceil",
    "round",
    "atan2",
    "powf",
    "hypot",
    "fma",
    "allFinite",
    // Tensor transforms (6)
    "reshape",
    "permute",
    "slice",
    "selectAxis",
    "fillLike",
    "expandLike",
    // Linear algebra vectors (4)
    "dot",
    "l2Norm",
    "l2Distance",
    "cosineSimilarity",
    // Statistics (8)
    "sum",
    "mean",
    "variancePopulation",
    "stdPopulation",
    "varianceSample",
    "stdSample",
    "min",
    "max",
    // Rank-4 axis reduction (4)
    "sumAxis",
    "meanAxis",
    "minAxis",
    "maxAxis",
    // Probability (4)
    "normalize",
    "entropy",
    "crossEntropy",
    "klDivergence",
    // Comparison & index source (2)
    "lessEqual01",
    "indicesLike",
];

#[derive(Debug, Clone, Serialize)]
pub struct RefTensor {
    pub values: Vec<f64>,
    pub shape: [usize; 4],
}

#[derive(Debug, Clone, Serialize)]
pub struct DiffRefCaseReport {
    pub case_index: usize,
    pub passed: bool,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub output_len: usize,
    pub output_shape: [usize; 4],
    pub first_diff_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DiffRefReport {
    pub ok: bool,
    pub verdict: &'static str,
    pub reference_source: &'static str,
    pub tol: f64,
    pub cases_checked: usize,
    pub max_abs_error: f64,
    pub max_rel_error: f64,
    pub all_pass: bool,
    pub cases: Vec<DiffRefCaseReport>,
}

pub(crate) fn numel(shape: [usize; 4]) -> usize {
    shape[0] * shape[1] * shape[2] * shape[3]
}

pub(crate) fn coords4(shape: [usize; 4], flat: usize) -> [usize; 4] {
    let s2 = shape[3];
    let s1 = shape[2] * s2;
    let s0 = shape[1] * s1;
    let d0 = flat / s0;
    let r0 = flat % s0;
    let d1 = r0 / s1;
    let r1 = r0 % s1;
    let d2 = r1 / s2;
    let d3 = r1 % s2;
    [d0, d1, d2, d3]
}

pub(crate) fn flat4(shape: [usize; 4], c: [usize; 4]) -> usize {
    ((c[0] * shape[1] + c[1]) * shape[2] + c[2]) * shape[3] + c[3]
}

fn parse_f64_input(raw: &[Value], context: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (idx, val) in raw.iter().enumerate() {
        let num = val
            .as_f64()
            .ok_or_else(|| format!("{context}[{idx}]: expected finite number, got {val}"))?;
        if !num.is_finite() {
            return Err(format!("{context}[{idx}]: non-finite number"));
        }
        out.push(num);
    }
    Ok(out)
}

fn format_validate_err(val: &crate::graph::validate::ValidateReport) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !val.unknown_constructors.is_empty() {
        parts.push(format!(
            "unknown constructors: {}",
            val.unknown_constructors.join(", ")
        ));
    }
    parts.extend(val.issues.iter().cloned());
    parts.extend(val.mismatches.iter().map(|m| m.message.clone()));
    parts.extend(val.dataflow.issues.iter().cloned());
    if parts.is_empty() {
        val.code.to_string()
    } else {
        parts.join("; ")
    }
}

/// Independent pure-Rust `f64` interpreter for `SpecV1` graphs.
/// Executes without touching Burn `Device::flex()` or `src/layers/`.
pub fn run_reference_f64(
    spec: &SpecV1,
    inputs_override: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<RefTensor, String> {
    let val = validate_spec(spec, false);
    if !val.ok {
        return Err(format!(
            "diff-ref: invalid spec: {}",
            format_validate_err(&val)
        ));
    }

    let mut slots: Vec<Option<RefTensor>> = vec![None; spec.num_slots];
    for port in &spec.ports {
        let key = port.slot.to_string();
        let prod = numel(port.shape);
        let raw_opt = inputs_override
            .and_then(|m| m.get(&key))
            .or_else(|| spec.inputs.get(&key));
        let Some(raw) = raw_opt else {
            return Err(format!("diff-ref: missing inputs for port slot {}", port.slot));
        };
        if raw.len() != prod {
            return Err(format!(
                "diff-ref: inputs[{}] length mismatch: expected {prod} for shape {:?}, got {}",
                port.slot,
                port.shape,
                raw.len()
            ));
        }
        let values = parse_f64_input(raw, &format!("diff-ref inputs[{}]", port.slot))?;
        slots[port.slot] = Some(RefTensor {
            values,
            shape: port.shape,
        });
    }

    for (si, st) in spec.steps.iter().enumerate() {
        let layer = spec
            .layers
            .get(st.layer)
            .ok_or_else(|| format!("diff-ref: step {si}: missing layer {}", st.layer))?;
        let arity = st.expected_arity();
        let mut ins: Vec<&RefTensor> = Vec::with_capacity(arity);
        for &s in st.slots.iter().take(arity) {
            let t = slots
                .get(s)
                .and_then(|x| x.as_ref())
                .ok_or_else(|| format!("diff-ref: step {si}: slot {s} read before written"))?;
            ins.push(t);
        }
        let out_slot = st
            .output_slot()
            .ok_or_else(|| format!("diff-ref: step {si}: missing output slot"))?;
        let w_key = st.layer.to_string();
        let weights = spec.weights.get(&w_key).map(|v| v.as_slice());

        let res = if let Some(t) =
            nn_ops::try_eval_nn_op(&layer.constructor, &ins, &layer.args, weights)?
        {
            t
        } else if let Some(t) =
            math_ops::try_eval_math_op(&layer.constructor, &ins, &layer.args)?
        {
            t
        } else {
            return Err(format!(
                "diff-ref: unsupported constructor '{}' in f64 reference oracle (hint: provide --ref <ref.json> for integer/custom specs)",
                layer.constructor
            ));
        };

        if !res.values.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "diff-ref: non-finite output produced at step {si} ({})",
                layer.constructor
            ));
        }
        slots[out_slot] = Some(res);
    }

    slots
        .get_mut(spec.output_slot)
        .and_then(|s| s.take())
        .ok_or_else(|| {
            format!(
                "diff-ref: outputSlot {} was not produced",
                spec.output_slot
            )
        })
}

/// Differential test Burn `Device::flex()` `f32` engine against either the
/// built-in pure-Rust `f64` reference oracle or externally supplied reference vectors.
pub fn diff_ref_spec(
    spec: &SpecV1,
    cases_override: Option<&[EvalCase]>,
    external_ref: Option<&[Vec<f64>]>,
    tol: f64,
) -> Result<DiffRefReport, String> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!("diff-ref: tol must be finite and > 0, got {tol}"));
    }

    let val = validate_spec(spec, false);
    if !val.ok {
        return Err(format!(
            "diff-ref: invalid spec: {}",
            format_validate_err(&val)
        ));
    }

    let default_cases;
    let cases: &[EvalCase] = match cases_override {
        Some(cs) => {
            if cs.is_empty() {
                return Err("diff-ref: cases list must not be empty".into());
            }
            cs
        }
        None => {
            default_cases = vec![EvalCase {
                inputs: spec.inputs.clone(),
                expected: None,
            }];
            &default_cases
        }
    };

    if let Some(refs) = external_ref {
        if refs.len() != cases.len() {
            return Err(format!(
                "diff-ref: external reference vector count ({}) != cases count ({})",
                refs.len(),
                cases.len()
            ));
        }
        for (ci, r) in refs.iter().enumerate() {
            if !r.iter().all(|v| v.is_finite()) {
                return Err(format!(
                    "diff-ref: external reference vector[{ci}] contains non-finite values"
                ));
            }
        }
    }

    let graph = PreparedGraph::compile(spec)?;
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut overall_abs = 0.0f64;
    let mut overall_rel = 0.0f64;
    let mut all_pass = true;

    for (ci, case) in cases.iter().enumerate() {
        let burn_out = graph.run_once(spec, Some(&case.inputs))?;
        if !burn_out.values.iter().all(|v| v.is_finite()) {
            return Err(format!(
                "diff-ref: Burn engine produced non-finite output on case {ci}"
            ));
        }

        let ref_vals: Vec<f64> = match external_ref {
            Some(refs) => refs[ci].clone(),
            None => run_reference_f64(spec, Some(&case.inputs))?.values,
        };

        if burn_out.values.len() != ref_vals.len() {
            return Err(format!(
                "diff-ref: case {ci} output length mismatch: engine={}, reference={}",
                burn_out.values.len(),
                ref_vals.len()
            ));
        }

        let mut c_abs = 0.0f64;
        let mut c_rel = 0.0f64;
        let mut first_diff = None;
        for (i, (&bv, &rv)) in burn_out.values.iter().zip(ref_vals.iter()).enumerate() {
            let a = (bv as f64 - rv).abs();
            let r = a / rv.abs().max(1e-12);
            if a > c_abs {
                c_abs = a;
            }
            if r > c_rel {
                c_rel = r;
            }
            if a > tol && r > tol && first_diff.is_none() {
                first_diff = Some(i);
            }
        }

        let passed = c_abs <= tol || c_rel <= tol;
        if !passed {
            all_pass = false;
        }
        if c_abs > overall_abs {
            overall_abs = c_abs;
        }
        if c_rel > overall_rel {
            overall_rel = c_rel;
        }

        case_reports.push(DiffRefCaseReport {
            case_index: ci,
            passed,
            max_abs_error: c_abs,
            max_rel_error: c_rel,
            output_len: burn_out.values.len(),
            output_shape: burn_out.shape,
            first_diff_index: first_diff,
        });
    }

    Ok(DiffRefReport {
        ok: all_pass,
        verdict: if all_pass { "PASS" } else { "SIGNAL" },
        reference_source: if external_ref.is_some() {
            "external_ref"
        } else {
            "pure_rust_f64"
        },
        tol,
        cases_checked: cases.len(),
        max_abs_error: overall_abs,
        max_rel_error: overall_rel,
        all_pass,
        cases: case_reports,
    })
}
