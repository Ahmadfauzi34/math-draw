use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};

use crate::composer::ExpansionReceipt;
use crate::engine::{PreparedGraph, TensorOutput};
use crate::spec::{SpecV1, StepDecl, sha256_f32_slice, sha256_json};
use crate::trace::{DEFAULT_MAX_TENSOR_BYTES, TraceHashCompareReport, compare_trace_hashes_with_inputs};
use crate::validate::validate_spec;

#[derive(Debug, Clone, Deserialize)]
pub struct InputsSequenceWrapper {
    pub steps: Vec<BTreeMap<String, Vec<Value>>>,
}

pub fn parse_inputs_sequence(raw: &str, expansion: Option<&ExpansionReceipt>) -> Result<Vec<BTreeMap<String, Vec<Value>>>, String> {
    let seq: Vec<BTreeMap<String, Vec<Value>>> = if let Ok(w) = serde_json::from_str::<InputsSequenceWrapper>(raw) {
        w.steps
    } else if let Ok(arr) = serde_json::from_str::<Vec<BTreeMap<String, Vec<Value>>>>(raw) {
        arr
    } else {
        return Err("inputs-seq JSON must be an array of slot->values maps or {\"steps\": [...]}".into());
    };

    if let Some(exp) = expansion {
        Ok(seq
            .into_iter()
            .map(|step_map| {
                step_map
                    .into_iter()
                    .map(|(k, v)| {
                        let mapped_k = exp
                            .port_names
                            .get(&k)
                            .map(|slot| slot.to_string())
                            .unwrap_or(k);
                        (mapped_k, v)
                    })
                    .collect()
            })
            .collect())
    } else {
        Ok(seq)
    }
}

pub fn resolve_state_port_slot(
    spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    requested: Option<&str>,
) -> Result<usize, String> {
    let rep = validate_spec(spec, true);
    if !rep.valid {
        return Err(
            serde_json::to_string(&rep).unwrap_or_else(|_| format!("invalid spec: {:?}", rep.code)),
        );
    }

    let port = if let Some(req) = requested {
        if let Ok(slot_idx) = req.parse::<usize>() {
            spec.port_by_slot(slot_idx)
                .ok_or_else(|| format!("state port slot {slot_idx} not declared in spec.ports"))?
        } else if let Some(&slot_idx) = expansion.and_then(|e| e.port_names.get(req)) {
            spec.port_by_slot(slot_idx)
                .ok_or_else(|| format!("state port '{req}' -> slot {slot_idx} missing from spec.ports"))?
        } else {
            spec.ports
                .iter()
                .find(|p| p.role == req)
                .ok_or_else(|| format!("state port '{req}' not found by slot, port name, or role"))?
        }
    } else {
        spec.ports
            .iter()
            .find(|p| p.role == "state")
            .or_else(|| spec.ports.first())
            .ok_or_else(|| "spec has no ports for state feedback".to_string())?
    };

    let explain = rep.explain.ok_or("missing explain report")?;
    let out_shape = explain
        .slot_shapes
        .get(&spec.output_slot)
        .ok_or_else(|| format!("missing shape for outputSlot {}", spec.output_slot))?;
    if &port.shape != out_shape {
        return Err(format!(
            "state port slot {} shape {:?} does not match outputSlot {} shape {:?}",
            port.slot, port.shape, spec.output_slot, out_shape
        ));
    }

    if let Some(out_step) = explain.steps.iter().rfind(|s| s.out_slot == spec.output_slot)
        && out_step.dtype_out != "f32"
    {
        return Err(format!(
            "rollout requires f32 outputSlot dtype, got {}",
            out_step.dtype_out
        ));
    }

    Ok(port.slot)
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutTick {
    pub tick: usize,
    pub in_state_sha256: String,
    pub out_state_sha256: String,
    pub delta_linf: f64,
    pub finite: bool,
    pub out_state_preview: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutScanReport {
    pub ok: bool,
    pub mode: &'static str,
    pub steps: usize,
    pub state_port: usize,
    pub state_shape: [usize; 4],
    pub ticks: Vec<RolloutTick>,
    pub final_state: TensorOutput,
    pub final_state_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutFixpointReport {
    pub ok: bool,
    pub converged: bool,
    pub steps_run: usize,
    pub max_steps: usize,
    pub tol: f64,
    pub state_port: usize,
    pub state_shape: [usize; 4],
    pub final_delta_linf: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contraction_ratio_last: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contraction_ratio_mean: Option<f64>,
    pub ticks: Vec<RolloutTick>,
    pub final_state: TensorOutput,
    pub final_state_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutUnrollReport {
    pub ok: bool,
    pub steps: usize,
    pub state_port: usize,
    pub original_steps: usize,
    pub unrolled_steps: usize,
    pub unrolled_num_slots: usize,
    pub unrolled_sha256: String,
    pub unrolled_spec: SpecV1,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutBisectTick {
    pub tick: usize,
    pub delta_linf: f64,
    pub base_sha256: String,
    pub cand_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RolloutBisectReport {
    pub ok: bool,
    pub equivalent: bool,
    pub steps_checked: usize,
    pub abs_tol: f64,
    pub state_port: usize,
    pub divergent_tick: Option<usize>,
    pub tick_delta_linf: Option<f64>,
    pub first_divergent_step: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_compare: Option<TraceHashCompareReport>,
    pub ticks: Vec<RolloutBisectTick>,
}

fn linf_delta(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| ((x as f64) - (y as f64)).abs())
        .fold(0.0f64, f64::max)
}

fn f32_slice_to_json(vals: &[f32]) -> Vec<Value> {
    vals.iter().map(|&v| Value::from(v as f64)).collect()
}

pub fn rollout_scan(
    spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    steps: usize,
    state_port_req: Option<&str>,
    inputs_seq: Option<&[BTreeMap<String, Vec<Value>>]>,
) -> Result<RolloutScanReport, String> {
    if steps == 0 {
        return Err("rollout scan requires steps >= 1".into());
    }
    let state_port = resolve_state_port_slot(spec, expansion, state_port_req)?;
    let state_shape = spec.port_by_slot(state_port).unwrap().shape;
    let graph = PreparedGraph::compile(spec)?;

    let state_key = state_port.to_string();
    let mut cur_state = spec.parse_f32_input(state_port)?;
    let mut cur_inputs = spec.inputs.clone();
    let mut ticks = Vec::with_capacity(steps);
    let mut final_out = TensorOutput {
        shape: state_shape,
        values: cur_state.clone(),
    };

    for tick in 0..steps {
        if let Some(seq) = inputs_seq
            && let Some(step_ov) = seq.get(tick)
        {
            for (k, v) in step_ov {
                if !(tick > 0 && k == &state_key) {
                    cur_inputs.insert(k.clone(), v.clone());
                }
            }
            if tick == 0
                && let Some(s0) = step_ov.get(&state_key)
            {
                cur_state = crate::spec::parse_f32_values(s0, &format!("inputs_seq[0][{state_key}]"))?;
            }
        }
        cur_inputs.insert(state_key.clone(), f32_slice_to_json(&cur_state));

        let in_sha = sha256_f32_slice(&cur_state);
        let out = graph.run_once(spec, Some(&cur_inputs))?;
        let finite = out.values.iter().all(|v| v.is_finite());
        if !finite {
            return Err(format!("non-finite state output at rollout tick {tick}"));
        }
        let delta = linf_delta(&cur_state, &out.values);
        let out_sha = sha256_f32_slice(&out.values);
        let preview = out.values.iter().copied().take(8).collect();
        ticks.push(RolloutTick {
            tick,
            in_state_sha256: in_sha,
            out_state_sha256: out_sha,
            delta_linf: delta,
            finite,
            out_state_preview: preview,
        });
        cur_state = out.values.clone();
        final_out = out;
    }

    let final_sha = sha256_f32_slice(&final_out.values);
    Ok(RolloutScanReport {
        ok: true,
        mode: "rollout_scan",
        steps,
        state_port,
        state_shape,
        ticks,
        final_state: final_out,
        final_state_sha256: final_sha,
    })
}

pub fn rollout_fixpoint(
    spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    tol: f64,
    max_steps: usize,
    state_port_req: Option<&str>,
) -> Result<RolloutFixpointReport, String> {
    if max_steps == 0 {
        return Err("rollout fixpoint requires max_steps >= 1".into());
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(format!("invalid fixpoint tol: {tol}"));
    }
    let state_port = resolve_state_port_slot(spec, expansion, state_port_req)?;
    let state_shape = spec.port_by_slot(state_port).unwrap().shape;
    let graph = PreparedGraph::compile(spec)?;

    let state_key = state_port.to_string();
    let mut cur_state = spec.parse_f32_input(state_port)?;
    let mut cur_inputs = spec.inputs.clone();
    let mut ticks = Vec::new();
    let mut ratios = Vec::new();
    let mut prev_delta: Option<f64> = None;
    let mut converged = false;
    let mut final_out = TensorOutput {
        shape: state_shape,
        values: cur_state.clone(),
    };
    let mut final_delta = f64::INFINITY;

    for tick in 0..max_steps {
        cur_inputs.insert(state_key.clone(), f32_slice_to_json(&cur_state));
        let in_sha = sha256_f32_slice(&cur_state);
        let out = graph.run_once(spec, Some(&cur_inputs))?;
        let finite = out.values.iter().all(|v| v.is_finite());
        if !finite {
            return Err(format!("non-finite state output at fixpoint tick {tick}"));
        }
        let delta = linf_delta(&cur_state, &out.values);
        if let Some(pd) = prev_delta
            && pd > 1e-12
        {
            ratios.push(delta / pd);
        }
        prev_delta = Some(delta);
        final_delta = delta;

        let out_sha = sha256_f32_slice(&out.values);
        let preview = out.values.iter().copied().take(8).collect();
        ticks.push(RolloutTick {
            tick,
            in_state_sha256: in_sha,
            out_state_sha256: out_sha,
            delta_linf: delta,
            finite,
            out_state_preview: preview,
        });
        cur_state = out.values.clone();
        final_out = out;

        if delta <= tol {
            converged = true;
            break;
        }
    }

    let contraction_ratio_last = ratios.last().copied();
    let contraction_ratio_mean = if ratios.is_empty() {
        None
    } else {
        Some(ratios.iter().sum::<f64>() / (ratios.len() as f64))
    };
    let final_sha = sha256_f32_slice(&final_out.values);

    Ok(RolloutFixpointReport {
        ok: converged,
        converged,
        steps_run: ticks.len(),
        max_steps,
        tol,
        state_port,
        state_shape,
        final_delta_linf: final_delta,
        contraction_ratio_last,
        contraction_ratio_mean,
        ticks,
        final_state: final_out,
        final_state_sha256: final_sha,
    })
}

pub fn rollout_unroll(
    spec: &SpecV1,
    expansion: Option<&ExpansionReceipt>,
    steps: usize,
    state_port_req: Option<&str>,
) -> Result<RolloutUnrollReport, String> {
    if steps == 0 {
        return Err("rollout unroll requires steps >= 1".into());
    }
    let state_port = resolve_state_port_slot(spec, expansion, state_port_req)?;

    let max_port_slot = spec.ports.iter().map(|p| p.slot).max().unwrap_or(0);
    let mut next_slot = max_port_slot + 1;
    let mut cur_state_slot = state_port;
    let mut unrolled_steps_vec = Vec::with_capacity(spec.steps.len() * steps);

    for _t in 0..steps {
        let mut slot_map: HashMap<usize, usize> = HashMap::new();
        for p in &spec.ports {
            slot_map.insert(p.slot, p.slot);
        }
        slot_map.insert(state_port, cur_state_slot);

        for st in &spec.steps {
            let arity = st.expected_arity();
            let mut new_slots = Vec::with_capacity(arity + 1);
            for &s_in in st.slots.iter().take(arity) {
                let mapped_in = *slot_map
                    .get(&s_in)
                    .ok_or_else(|| format!("unroll: slot {s_in} read before written"))?;
                new_slots.push(mapped_in);
            }
            let orig_out = st.slots[arity];
            let new_out = next_slot;
            next_slot += 1;
            slot_map.insert(orig_out, new_out);
            new_slots.push(new_out);

            unrolled_steps_vec.push(StepDecl {
                kind: st.kind.clone(),
                layer: st.layer,
                slots: new_slots,
            });
        }

        cur_state_slot = *slot_map
            .get(&spec.output_slot)
            .ok_or_else(|| format!("unroll: outputSlot {} missing after tick", spec.output_slot))?;
    }

    let unrolled = SpecV1 {
        num_slots: next_slot.max(cur_state_slot + 1),
        layers: spec.layers.clone(),
        steps: unrolled_steps_vec,
        output_slot: cur_state_slot,
        ports: spec.ports.clone(),
        weights: spec.weights.clone(),
        inputs: spec.inputs.clone(),
    };
    let val = validate_spec(&unrolled, false);
    if !val.valid {
        return Err(format!("unrolled spec failed validation: {:?}", val.code));
    }

    Ok(RolloutUnrollReport {
        ok: true,
        steps,
        state_port,
        original_steps: spec.steps.len(),
        unrolled_steps: unrolled.steps.len(),
        unrolled_num_slots: unrolled.num_slots,
        unrolled_sha256: sha256_json(&unrolled),
        unrolled_spec: unrolled,
    })
}

pub fn rollout_bisect(
    base: &SpecV1,
    cand: &SpecV1,
    base_expansion: Option<&ExpansionReceipt>,
    steps: usize,
    state_port_req: Option<&str>,
    inputs_seq: Option<&[BTreeMap<String, Vec<Value>>]>,
    abs_tol: f64,
) -> Result<RolloutBisectReport, String> {
    if steps == 0 {
        return Err("rollout bisect requires steps >= 1".into());
    }
    let state_port = resolve_state_port_slot(base, base_expansion, state_port_req)?;
    let cand_port = resolve_state_port_slot(cand, None, Some(&state_port.to_string()))?;
    if cand_port != state_port {
        return Err(format!(
            "candidate state port {cand_port} does not match base state port {state_port}"
        ));
    }

    let g_base = PreparedGraph::compile(base)?;
    let g_cand = PreparedGraph::compile(cand)?;

    let state_key = state_port.to_string();
    let mut base_state = base.parse_f32_input(state_port)?;
    let mut cand_state = if cand.inputs.contains_key(&state_key) {
        cand.parse_f32_input(state_port)?
    } else {
        base_state.clone()
    };
    let mut base_inputs = base.inputs.clone();
    let mut cand_inputs = if cand.inputs.is_empty() {
        base.inputs.clone()
    } else {
        cand.inputs.clone()
    };

    let mut ticks = Vec::new();
    for tick in 0..steps {
        if let Some(seq) = inputs_seq
            && let Some(step_ov) = seq.get(tick)
        {
            for (k, v) in step_ov {
                if !(tick > 0 && k == &state_key) {
                    base_inputs.insert(k.clone(), v.clone());
                    cand_inputs.insert(k.clone(), v.clone());
                }
            }
            if tick == 0
                && let Some(s0) = step_ov.get(&state_key)
            {
                let parsed = crate::spec::parse_f32_values(s0, &format!("inputs_seq[0][{state_key}]"))?;
                base_state = parsed.clone();
                cand_state = parsed;
            }
        }
        base_inputs.insert(state_key.clone(), f32_slice_to_json(&base_state));
        cand_inputs.insert(state_key.clone(), f32_slice_to_json(&cand_state));

        let out_b = g_base.run_once(base, Some(&base_inputs))?;
        let out_c = g_cand.run_once(cand, Some(&cand_inputs))?;
        if out_b.values.len() != out_c.values.len() {
            let cmp = compare_trace_hashes_with_inputs(
                base,
                cand,
                Some(&base_inputs),
                DEFAULT_MAX_TENSOR_BYTES,
                false,
            )
            .ok();
            let first_step = cmp.as_ref().and_then(|c| c.first_divergent_step);
            return Ok(RolloutBisectReport {
                ok: true,
                equivalent: false,
                steps_checked: tick + 1,
                abs_tol,
                state_port,
                divergent_tick: Some(tick),
                tick_delta_linf: Some(f64::INFINITY),
                first_divergent_step: first_step,
                trace_compare: cmp,
                ticks,
            });
        }

        let delta = linf_delta(&out_b.values, &out_c.values);
        let b_sha = sha256_f32_slice(&out_b.values);
        let c_sha = sha256_f32_slice(&out_c.values);
        ticks.push(RolloutBisectTick {
            tick,
            delta_linf: delta,
            base_sha256: b_sha,
            cand_sha256: c_sha,
        });

        if delta > abs_tol {
            let cmp = compare_trace_hashes_with_inputs(
                base,
                cand,
                Some(&base_inputs),
                DEFAULT_MAX_TENSOR_BYTES,
                false,
            )
            .ok();
            let first_step = cmp.as_ref().and_then(|c| c.first_divergent_step);
            return Ok(RolloutBisectReport {
                ok: true,
                equivalent: false,
                steps_checked: tick + 1,
                abs_tol,
                state_port,
                divergent_tick: Some(tick),
                tick_delta_linf: Some(delta),
                first_divergent_step: first_step,
                trace_compare: cmp,
                ticks,
            });
        }

        base_state = out_b.values;
        cand_state = out_c.values;
    }

    Ok(RolloutBisectReport {
        ok: true,
        equivalent: true,
        steps_checked: steps,
        abs_tol,
        state_port,
        divergent_tick: None,
        tick_delta_linf: None,
        first_divergent_step: None,
        trace_compare: None,
        ticks,
    })
}
