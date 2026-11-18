use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

use crate::catalog::is_int_constructor;
pub use crate::layers::{
    GradParam, GradTensor, RuntimeLayer, SlotValue, build_layers, exec_layer,
    get_layer_weight_len, is_linear_like, layer_has_trainable_weights,
};
use crate::spec::{SpecV1, parse_f32_values, parse_i64_values, sha256_f32_slice};
use crate::validate::validate_spec;

#[derive(Debug, Clone, Serialize)]
pub struct TensorOutput {
    pub shape: [usize; 4],
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunResult {
    pub ok: bool,
    pub mode: &'static str,
    pub outputs: TensorOutput,
    pub trace_sha256: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

pub fn port_is_i64_mask(spec: &SpecV1) -> Vec<bool> {
    let mut i64_ports = vec![false; spec.num_slots];
    for st in &spec.steps {
        if let Some(layer) = spec.layers.get(st.layer)
            && is_int_constructor(&layer.constructor)
            && layer.constructor != "intFromFloat"
        {
            for &in_s in st.slots.iter().take(st.slots.len().saturating_sub(1)) {
                if in_s < spec.num_slots && spec.port_by_slot(in_s).is_some() {
                    i64_ports[in_s] = true;
                }
            }
        }
    }
    i64_ports
}

fn load_port_inputs(
    spec: &SpecV1,
    device: &Device,
    overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<Vec<Option<SlotValue>>, String> {
    let mut slots: Vec<Option<SlotValue>> = vec![None; spec.num_slots];
    let i64_ports = port_is_i64_mask(spec);

    for port in &spec.ports {
        let key = port.slot.to_string();
        let prod: usize = port.shape.iter().product();
        let raw_opt = overrides
            .and_then(|m| m.get(&key))
            .or_else(|| spec.inputs.get(&key));
        let Some(raw) = raw_opt else {
            return Err(format!("missing inputs for port slot {}", port.slot));
        };
        if raw.len() != prod {
            return Err(format!(
                "inputs[{}] length mismatch: expected {prod} for shape {:?}, got {}",
                port.slot,
                port.shape,
                raw.len()
            ));
        }
        if i64_ports[port.slot] {
            let vals = parse_i64_values(raw, &format!("inputs[{}]", port.slot))?;
            slots[port.slot] = Some(SlotValue::I64(vals, port.shape));
        } else {
            let vals = parse_f32_values(raw, &format!("inputs[{}]", port.slot))?;
            let tensor = Tensor::<4>::from_data(TensorData::new(vals, port.shape), device);
            slots[port.slot] = Some(SlotValue::F32(tensor));
        }
    }
    Ok(slots)
}

pub struct PreparedGraph {
    device: Device,
    layers: Vec<RuntimeLayer>,
}

impl PreparedGraph {
    pub fn compile(spec: &SpecV1) -> Result<Self, String> {
        let device = Device::flex();
        let (layers, _) = build_layers(spec, &device, &[])?;
        Ok(Self { device, layers })
    }

    pub fn run_once(
        &self,
        spec: &SpecV1,
        input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
    ) -> Result<TensorOutput, String> {
        let mut slots = load_port_inputs(spec, &self.device, input_overrides)?;
        for (si, st) in spec.steps.iter().enumerate() {
            let layer = self
                .layers
                .get(st.layer)
                .ok_or_else(|| format!("step {si}: missing layer {}", st.layer))?;
            let arity = st.expected_arity();
            let mut ins = Vec::with_capacity(arity);
            for &s in st.slots.iter().take(arity) {
                let val = slots
                    .get(s)
                    .and_then(|x| x.as_ref())
                    .cloned()
                    .ok_or_else(|| format!("step {si}: slot {s} read before written"))?;
                ins.push(val);
            }
            let out_slot = st.slots[arity];
            let res = exec_layer(layer, &ins, &self.device)?;
            slots[out_slot] = Some(res);
        }
        let final_val = slots[spec.output_slot]
            .take()
            .ok_or_else(|| format!("outputSlot {} was not produced", spec.output_slot))?;
        slot_value_to_output(final_val)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepTraceEntry {
    pub index: usize,
    pub layer: String,
    pub out_slot: usize,
    pub out_shape: [usize; 4],
    pub out_bytes: usize,
    pub capture_status: &'static str,
    pub finite_values: bool,
    pub value_sha256: Option<String>,
}

fn slot_value_to_output(val: SlotValue) -> Result<TensorOutput, String> {
    match val {
        SlotValue::F32(t) => {
            let shape = t.dims();
            let td = t.into_data();
            let values = td
                .try_into_vec::<f32>()
                .map_err(|e| format!("failed reading output f32 tensor: {e}"))?;
            for (i, &v) in values.iter().enumerate() {
                if !v.is_finite() {
                    return Err(format!("non-finite output at index {i}: {v}"));
                }
            }
            Ok(TensorOutput { shape, values })
        }
        SlotValue::I64(vi, shape) => {
            let values: Vec<f32> = vi.into_iter().map(|v| v as f32).collect();
            Ok(TensorOutput { shape, values })
        }
    }
}

fn capture_slot_trace(
    index: usize,
    ctor: &str,
    out_slot: usize,
    val: &SlotValue,
    max_tensor_bytes: usize,
) -> StepTraceEntry {
    use sha2::{Digest, Sha256};
    match val {
        SlotValue::F32(t) => {
            let shape = t.dims();
            let numel: usize = shape.iter().product();
            let out_bytes = numel * 4;
            if out_bytes > max_tensor_bytes {
                return StepTraceEntry {
                    index,
                    layer: ctor.to_string(),
                    out_slot,
                    out_shape: shape,
                    out_bytes,
                    capture_status: "tensor_budget_exceeded",
                    finite_values: true,
                    value_sha256: None,
                };
            }
            let vals = t.clone().into_data().try_into_vec::<f32>().unwrap_or_default();
            let finite = vals.iter().all(|v| v.is_finite());
            StepTraceEntry {
                index,
                layer: ctor.to_string(),
                out_slot,
                out_shape: shape,
                out_bytes,
                capture_status: "ok",
                finite_values: finite,
                value_sha256: Some(sha256_f32_slice(&vals)),
            }
        }
        SlotValue::I64(vi, shape) => {
            let out_bytes = vi.len() * 8;
            if out_bytes > max_tensor_bytes {
                return StepTraceEntry {
                    index,
                    layer: ctor.to_string(),
                    out_slot,
                    out_shape: *shape,
                    out_bytes,
                    capture_status: "tensor_budget_exceeded",
                    finite_values: true,
                    value_sha256: None,
                };
            }
            let mut hasher = Sha256::new();
            for &v in vi {
                hasher.update(v.to_le_bytes());
            }
            StepTraceEntry {
                index,
                layer: ctor.to_string(),
                out_slot,
                out_shape: *shape,
                out_bytes,
                capture_status: "ok",
                finite_values: true,
                value_sha256: Some(format!("sha256:{:x}", hasher.finalize())),
            }
        }
    }
}

pub fn execute_forward(
    spec: &SpecV1,
    device: &Device,
    grad_layers: &[usize],
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<(SlotValue, Vec<GradParam>), String> {
    let (layers, grad_params) = build_layers(spec, device, grad_layers)?;
    let mut slots = load_port_inputs(spec, device, input_overrides)?;

    for (si, st) in spec.steps.iter().enumerate() {
        let layer = layers
            .get(st.layer)
            .ok_or_else(|| format!("step {si}: missing layer {}", st.layer))?;
        let arity = st.expected_arity();
        let mut ins = Vec::with_capacity(arity);
        for &s in st.slots.iter().take(arity) {
            let val = slots
                .get(s)
                .and_then(|x| x.as_ref())
                .cloned()
                .ok_or_else(|| format!("step {si}: slot {s} read before written"))?;
            ins.push(val);
        }
        let out_slot = st.slots[arity];
        let res = exec_layer(layer, &ins, device)?;
        slots[out_slot] = Some(res);
    }

    let final_val = slots[spec.output_slot]
        .take()
        .ok_or_else(|| format!("outputSlot {} was not produced", spec.output_slot))?;
    Ok((final_val, grad_params))
}

pub fn execute_forward_traced(
    spec: &SpecV1,
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
    max_tensor_bytes: usize,
) -> Result<(TensorOutput, Vec<StepTraceEntry>), String> {
    let device = Device::flex();
    let (layers, _) = build_layers(spec, &device, &[])?;
    let mut slots = load_port_inputs(spec, &device, input_overrides)?;
    let mut traces = Vec::with_capacity(spec.steps.len());

    for (si, st) in spec.steps.iter().enumerate() {
        let layer = layers
            .get(st.layer)
            .ok_or_else(|| format!("step {si}: missing layer {}", st.layer))?;
        let arity = st.expected_arity();
        let mut ins = Vec::with_capacity(arity);
        for &s in st.slots.iter().take(arity) {
            let val = slots
                .get(s)
                .and_then(|x| x.as_ref())
                .cloned()
                .ok_or_else(|| format!("step {si}: slot {s} read before written"))?;
            ins.push(val);
        }
        let out_slot = st.slots[arity];
        let res = exec_layer(layer, &ins, &device)?;
        let ctor = spec.layers[st.layer].constructor.as_str();
        traces.push(capture_slot_trace(si, ctor, out_slot, &res, max_tensor_bytes));
        slots[out_slot] = Some(res);
    }

    let final_val = slots[spec.output_slot]
        .take()
        .ok_or_else(|| format!("outputSlot {} was not produced", spec.output_slot))?;
    let output = slot_value_to_output(final_val)?;
    Ok((output, traces))
}

pub fn eval_spec_with_inputs(
    spec: &SpecV1,
    input_overrides: Option<&BTreeMap<String, Vec<Value>>>,
) -> Result<TensorOutput, String> {
    let device = Device::flex();
    let (val, _) = execute_forward(spec, &device, &[], input_overrides)?;
    slot_value_to_output(val)
}

pub fn run_spec(spec: &SpecV1) -> Result<RunResult, String> {
    let report = validate_spec(spec, false);
    if !report.valid {
        return Err(serde_json::to_string(&report).unwrap_or_else(|_| "validation failed".into()));
    }
    let outputs = eval_spec_with_inputs(spec, None)?;
    let trace_sha256 = sha256_f32_slice(&outputs.values);
    Ok(RunResult {
        ok: true,
        mode: "native_flex",
        outputs,
        trace_sha256,
        warnings: report.static_warnings,
    })
}
