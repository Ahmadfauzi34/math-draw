use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

use crate::engine::{StepTraceEntry, TensorOutput, execute_forward_traced};
use crate::spec::{SpecV1, sha256_f32_slice, sha256_json};
use crate::validate::validate_spec;

pub const DEFAULT_MAX_TENSOR_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct TraceHashReport {
    pub ok: bool,
    pub mode: &'static str,
    pub spec_sha256: String,
    pub max_tensor_bytes: usize,
    pub steps_count: usize,
    pub captured_steps: usize,
    pub skipped_steps: usize,
    pub steps: Vec<StepTraceEntry>,
    pub output: TensorOutput,
    pub output_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DivergentDetail {
    pub step_index: usize,
    pub reason: &'static str,
    pub left: Option<StepTraceEntry>,
    pub right: Option<StepTraceEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceHashCompareReport {
    pub ok: bool,
    pub equivalent_outputs: bool,
    pub max_output_abs_diff: f64,
    pub hash_chain_equal: bool,
    pub max_tensor_bytes: usize,
    pub first_divergent_step: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub divergent_step_detail: Option<DivergentDetail>,
    pub left_sha256: String,
    pub right_sha256: String,
    pub left_output_sha256: String,
    pub right_output_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left_prefix_subspec: Option<SpecV1>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right_prefix_subspec: Option<SpecV1>,
    pub left_steps: Vec<StepTraceEntry>,
    pub right_steps: Vec<StepTraceEntry>,
}

pub fn trace_hash_spec(spec: &SpecV1, max_tensor_bytes: usize) -> Result<TraceHashReport, String> {
    trace_hash_spec_with_inputs(spec, None, max_tensor_bytes)
}

pub fn trace_hash_spec_with_inputs(
    spec: &SpecV1,
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
    max_tensor_bytes: usize,
) -> Result<TraceHashReport, String> {
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }
    let (output, steps) = execute_forward_traced(spec, input_overrides, max_tensor_bytes)?;
    let captured_steps = steps.iter().filter(|s| s.capture_status == "ok").count();
    let skipped_steps = steps.len().saturating_sub(captured_steps);
    let output_sha256 = sha256_f32_slice(&output.values);
    Ok(TraceHashReport {
        ok: true,
        mode: "native_flex_trace_hash",
        spec_sha256: sha256_json(spec),
        max_tensor_bytes,
        steps_count: steps.len(),
        captured_steps,
        skipped_steps,
        steps,
        output,
        output_sha256,
    })
}

pub fn extract_prefix_subspec(spec: &SpecV1, step_inclusive: usize) -> Option<SpecV1> {
    if spec.steps.is_empty() {
        return None;
    }
    let end = step_inclusive.min(spec.steps.len() - 1);
    let prefix_steps = spec.steps[..=end].to_vec();
    let last_step = prefix_steps.last()?;
    let arity = last_step.expected_arity();
    let out_slot = *last_step.slots.get(arity)?;

    let mut used_in_slots = HashSet::new();
    for st in &prefix_steps {
        let a = st.expected_arity();
        for &s in st.slots.iter().take(a) {
            used_in_slots.insert(s);
        }
    }

    let kept_ports: Vec<_> = spec
        .ports
        .iter()
        .filter(|p| used_in_slots.contains(&p.slot))
        .cloned()
        .collect();
    if kept_ports.is_empty() {
        return None;
    }
    let kept_port_slots: HashSet<usize> = kept_ports.iter().map(|p| p.slot).collect();
    let mut kept_inputs = BTreeMap::new();
    for (k, v) in &spec.inputs {
        if let Ok(slot) = k.parse::<usize>()
            && kept_port_slots.contains(&slot)
        {
            kept_inputs.insert(k.clone(), v.clone());
        }
    }

    let sub = SpecV1 {
        num_slots: spec.num_slots,
        layers: spec.layers.clone(),
        steps: prefix_steps,
        output_slot: out_slot,
        ports: kept_ports,
        weights: spec.weights.clone(),
        inputs: kept_inputs,
    };
    if validate_spec(&sub, false).valid {
        Some(sub)
    } else {
        None
    }
}

pub fn compare_trace_hashes(
    left: &SpecV1,
    right: &SpecV1,
    max_tensor_bytes: usize,
    isolate: bool,
) -> Result<TraceHashCompareReport, String> {
    compare_trace_hashes_with_inputs(left, right, None, max_tensor_bytes, isolate)
}

pub fn compare_trace_hashes_with_inputs(
    left: &SpecV1,
    right: &SpecV1,
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
    max_tensor_bytes: usize,
    isolate: bool,
) -> Result<TraceHashCompareReport, String> {
    let l_rep = trace_hash_spec_with_inputs(left, input_overrides, max_tensor_bytes)?;
    let right_override = if input_overrides.is_some() {
        input_overrides
    } else if right.inputs.is_empty() && !left.inputs.is_empty() {
        Some(&left.inputs)
    } else {
        None
    };
    let r_rep = trace_hash_spec_with_inputs(right, right_override, max_tensor_bytes)?;

    let mut first_div: Option<usize> = None;
    let mut div_detail: Option<DivergentDetail> = None;

    let min_len = l_rep.steps.len().min(r_rep.steps.len());
    for i in 0..min_len {
        let ls = &l_rep.steps[i];
        let rs = &r_rep.steps[i];

        if ls.out_shape != rs.out_shape {
            first_div = Some(i);
            div_detail = Some(DivergentDetail {
                step_index: i,
                reason: "shape_mismatch",
                left: Some(ls.clone()),
                right: Some(rs.clone()),
            });
            break;
        }
        if ls.finite_values != rs.finite_values || !ls.finite_values {
            first_div = Some(i);
            div_detail = Some(DivergentDetail {
                step_index: i,
                reason: "non_finite_values",
                left: Some(ls.clone()),
                right: Some(rs.clone()),
            });
            break;
        }
        if ls.capture_status == "tensor_budget_exceeded"
            && rs.capture_status == "tensor_budget_exceeded"
        {
            continue;
        }
        if ls.capture_status != rs.capture_status {
            first_div = Some(i);
            div_detail = Some(DivergentDetail {
                step_index: i,
                reason: "capture_status_mismatch",
                left: Some(ls.clone()),
                right: Some(rs.clone()),
            });
            break;
        }
        if ls.value_sha256 != rs.value_sha256 {
            let reason = if ls.layer != rs.layer {
                "layer_and_value_hash_mismatch"
            } else {
                "value_hash_mismatch"
            };
            first_div = Some(i);
            div_detail = Some(DivergentDetail {
                step_index: i,
                reason,
                left: Some(ls.clone()),
                right: Some(rs.clone()),
            });
            break;
        }
        if ls.layer != rs.layer {
            first_div = Some(i);
            div_detail = Some(DivergentDetail {
                step_index: i,
                reason: "layer_constructor_mismatch",
                left: Some(ls.clone()),
                right: Some(rs.clone()),
            });
            break;
        }
    }

    if first_div.is_none() && l_rep.steps.len() != r_rep.steps.len() {
        first_div = Some(min_len);
        div_detail = Some(DivergentDetail {
            step_index: min_len,
            reason: "step_count_mismatch",
            left: l_rep.steps.get(min_len).cloned(),
            right: r_rep.steps.get(min_len).cloned(),
        });
    }

    let hash_chain_equal = first_div.is_none();
    let max_output_abs_diff = if l_rep.output.shape == r_rep.output.shape
        && l_rep.output.values.len() == r_rep.output.values.len()
    {
        l_rep
            .output
            .values
            .iter()
            .zip(r_rep.output.values.iter())
            .map(|(&a, &b)| ((a as f64) - (b as f64)).abs())
            .fold(0.0f64, f64::max)
    } else {
        f64::INFINITY
    };
    let equivalent_outputs = l_rep.output.shape == r_rep.output.shape && max_output_abs_diff <= 1e-5;

    let (left_prefix_subspec, right_prefix_subspec) = if isolate {
        if let Some(div_idx) = first_div {
            (
                extract_prefix_subspec(left, div_idx),
                extract_prefix_subspec(right, div_idx),
            )
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(TraceHashCompareReport {
        ok: true,
        equivalent_outputs,
        max_output_abs_diff,
        hash_chain_equal,
        max_tensor_bytes,
        first_divergent_step: first_div,
        divergent_step_detail: div_detail,
        left_sha256: l_rep.spec_sha256,
        right_sha256: r_rep.spec_sha256,
        left_output_sha256: l_rep.output_sha256,
        right_output_sha256: r_rep.output_sha256,
        left_prefix_subspec,
        right_prefix_subspec,
        left_steps: l_rep.steps,
        right_steps: r_rep.steps,
    })
}
