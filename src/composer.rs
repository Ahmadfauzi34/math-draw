use serde::Serialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::catalog::find_constructor;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl, parse_f32_values, sha256_json};

pub const RECURSIVE_SPEC_SCHEMA: &str = "burn-math.recursive-spec.v2";
const MAX_RECURSION_DEPTH: usize = 32;

#[derive(Debug, Clone, Serialize)]
pub struct ExpansionEvent {
    pub kind: String,
    pub target: String,
    pub prefix: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tie_weights: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExpansionReceipt {
    pub schema: String,
    pub num_slots: usize,
    pub layers_count: usize,
    pub steps_count: usize,
    pub compiled_sha256: String,
    pub events: Vec<ExpansionEvent>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub port_names: BTreeMap<String, usize>,
}

pub fn is_recursive_spec_v2(raw: &Value) -> bool {
    let Some(obj) = raw.as_object() else {
        return false;
    };
    if obj.get("schema").and_then(|v| v.as_str()) == Some(RECURSIVE_SPEC_SCHEMA) {
        return true;
    }
    obj.contains_key("subgraphs")
        || (obj.contains_key("output") && !obj.contains_key("outputSlot"))
}

struct Allocator {
    next_layer_id: usize,
    next_slot_id: usize,
    slot_names: HashMap<String, usize>,
}

impl Allocator {
    fn new() -> Self {
        Self {
            next_layer_id: 0,
            next_slot_id: 0,
            slot_names: HashMap::new(),
        }
    }

    fn bind_fixed_slot(&mut self, name: &str, slot: usize) {
        self.slot_names.insert(name.to_string(), slot);
        if slot >= self.next_slot_id {
            self.next_slot_id = slot + 1;
        }
    }

    fn resolve_slot(&mut self, sym: &Value, create_if_missing: bool) -> Result<usize, String> {
        if let Some(n) = sym.as_u64() {
            let slot = n as usize;
            if slot >= self.next_slot_id {
                self.next_slot_id = slot + 1;
            }
            return Ok(slot);
        }
        let Some(k) = sym.as_str() else {
            return Err(format!("invalid slot reference {sym}"));
        };
        if let Some(&slot) = self.slot_names.get(k) {
            return Ok(slot);
        }
        if crate::spec::is_canonical_uint_key(k) && !create_if_missing {
            let num: usize = k.parse().map_err(|_| format!("invalid slot '{k}'"))?;
            if num >= self.next_slot_id {
                self.next_slot_id = num + 1;
            }
            return Ok(num);
        }
        if !create_if_missing {
            return Err(format!(
                "unresolved_symbolic_slot: symbolic slot '{k}' is read before it is declared in ports or produced by an earlier step"
            ));
        }
        let slot = self.next_slot_id;
        self.next_slot_id += 1;
        self.slot_names.insert(k.to_string(), slot);
        Ok(slot)
    }

    fn fresh_slot(&mut self) -> usize {
        let s = self.next_slot_id;
        self.next_slot_id += 1;
        s
    }

    fn fresh_layer_id(&mut self) -> usize {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        id
    }
}

struct CompileState {
    alloc: Allocator,
    layers: Vec<LayerDecl>,
    steps: Vec<StepDecl>,
    weights: BTreeMap<String, Vec<f32>>,
    subgraphs: HashMap<String, Value>,
    events: Vec<ExpansionEvent>,
}

fn is_builtin_macro(name: &str) -> bool {
    matches!(
        name,
        "mlp" | "gated_unit" | "residual_mlp" | "transformer_ffn" | "recurrent_cell"
    )
}

fn need_usize(params: &Map<String, Value>, key: &str) -> Result<usize, String> {
    params
        .get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| format!("macro param '{key}' (positive integer) is required"))
}

fn expand_builtin_macro(
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
        _ => {
            return Err(format!("unknown built-in macro '{macro_name}'"));
        }
    }

    state.layers.extend(local_layers);
    state.steps.extend(local_steps);

    if let Some(obj) = step_weights.and_then(|v| v.as_object()) {
        for (k, v) in obj {
            if let Ok(local_idx) = k.parse::<usize>()
                && let Some(arr) = v.as_array()
            {
                let floats = parse_f32_values(arr, &format!("{prefix}:{macro_name}.weights[{k}]"))?;
                state.weights.insert((base_layer_idx + local_idx).to_string(), floats);
            }
        }
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

fn build_args_from_params(op: &str, reg_id: usize, params: &Map<String, Value>) -> Result<Vec<Value>, String> {
    let Some(meta) = find_constructor(op) else {
        return Err(format!("unknown_layer_constructor: '{op}'"));
    };
    let mut out = vec![Value::from(reg_id)];
    for p in meta.parameters.iter().skip(1) {
        if let Some(v) = params.get(p.name) {
            out.push(v.clone());
        } else if !p.optional {
            return Err(format!(
                "missing required parameter '{}' for constructor '{op}'",
                p.name
            ));
        }
    }
    Ok(out)
}

fn substitute_value(v: &Value, param_map: &Map<String, Value>) -> Value {
    if let Some(s) = v.as_str()
        && let Some(key) = s.strip_prefix('$')
        && let Some(rep) = param_map.get(key)
    {
        return rep.clone();
    }
    v.clone()
}

fn substitute_step(step_val: &Value, param_map: &Map<String, Value>, caller_weights: Option<&Value>) -> Value {
    let mut s = step_val.clone();
    if let Some(obj) = s.as_object_mut() {
        if let Some(args) = obj.get_mut("args").and_then(|a| a.as_array_mut()) {
            for item in args.iter_mut() {
                *item = substitute_value(item, param_map);
            }
        }
        if let Some(p_obj) = obj.get_mut("params").and_then(|p| p.as_object_mut()) {
            for (_, v) in p_obj.iter_mut() {
                *v = substitute_value(v, param_map);
            }
        }
        if !obj.contains_key("weights")
            && let Some(cw) = caller_weights.and_then(|w| w.as_object())
            && let Some(wk) = obj.get("weight_key").and_then(|v| v.as_str())
            && let Some(w_val) = cw.get(wk)
        {
            obj.insert("weights".to_string(), w_val.clone());
        }
    }
    s
}

fn expand_step_list(
    steps: &[Value],
    scope_env: &mut HashMap<String, usize>,
    state: &mut CompileState,
    call_stack: &[String],
    prefix: &str,
    base_dir: &Path,
    mut tied_cache: Option<&mut HashMap<String, usize>>,
) -> Result<(), String> {
    if call_stack.len() > MAX_RECURSION_DEPTH {
        return Err(format!(
            "recursion_depth_exceeded: maximum recursion depth ({MAX_RECURSION_DEPTH}) exceeded at {}",
            call_stack.join(" -> ")
        ));
    }

    for (si, st_val) in steps.iter().enumerate() {
        let Some(st) = st_val.as_object() else {
            return Err(format!("step {prefix}.s{si} must be an object"));
        };
        let step_prefix = format!("{prefix}.s{si}");

        let resolve_in = |sym: &Value, env: &HashMap<String, usize>, alloc: &mut Allocator| -> Result<usize, String> {
            if let Some(s) = sym.as_str()
                && let Some(&slot) = env.get(s)
            {
                return Ok(slot);
            }
            alloc.resolve_slot(sym, false)
        };

        let resolve_out = |sym: Option<&Value>, env: &mut HashMap<String, usize>, alloc: &mut Allocator| -> Result<usize, String> {
            let Some(sym) = sym else {
                return Ok(alloc.fresh_slot());
            };
            if let Some(n) = sym.as_u64() {
                return alloc.resolve_slot(&Value::from(n), false);
            }
            let Some(k) = sym.as_str() else {
                return Err(format!("invalid output slot ref {sym}"));
            };
            if let Some(&slot) = env.get(k) {
                return Ok(slot);
            }
            let scoped_key = if prefix == "root" {
                k.to_string()
            } else {
                format!("{prefix}/{k}")
            };
            let slot = alloc.resolve_slot(&Value::String(scoped_key), true)?;
            env.insert(k.to_string(), slot);
            Ok(slot)
        };

        // Direct v1 step pass-through
        if let (Some(kind), Some(layer), Some(slots_arr)) = (
            st.get("kind").and_then(|v| v.as_str()),
            st.get("layer").and_then(|v| v.as_u64()),
            st.get("slots").and_then(|v| v.as_array()),
        ) && (kind == "unary" || kind == "binary")
            && !slots_arr.is_empty()
        {
            let mut mapped = Vec::with_capacity(slots_arr.len());
            for (idx, s) in slots_arr.iter().enumerate() {
                if idx + 1 == slots_arr.len() {
                    mapped.push(resolve_out(Some(s), scope_env, &mut state.alloc)?);
                } else {
                    mapped.push(resolve_in(s, scope_env, &mut state.alloc)?);
                }
            }
            state.steps.push(StepDecl {
                kind: kind.to_string(),
                layer: layer as usize,
                slots: mapped,
            });
            continue;
        }

        let op = st
            .get("op")
            .or_else(|| st.get("constructor"))
            .or_else(|| st.get("kind"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("step {step_prefix} is missing 'op' or 'constructor'"))?;

        let raw_in = st.get("in").or_else(|| st.get("inputs"));
        let in_list: Vec<Value> = match raw_in {
            Some(Value::Array(arr)) => arr.clone(),
            Some(v) => vec![v.clone()],
            None => match st.get("slots").and_then(|v| v.as_array()) {
                Some(arr) if !arr.is_empty() => arr[..arr.len() - 1].to_vec(),
                _ => Vec::new(),
            },
        };
        let raw_out = st.get("out").or_else(|| st.get("output")).or_else(|| {
            st.get("slots")
                .and_then(|v| v.as_array())
                .and_then(|a| a.last())
        });

        if op == "repeat" {
            let depth = st
                .get("depth")
                .or_else(|| st.get("count"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            if depth == 0 || depth > MAX_RECURSION_DEPTH {
                return Err(format!(
                    "invalid_repeat_depth: repeat depth must be in 1..{MAX_RECURSION_DEPTH}, got {depth}"
                ));
            }
            let target_sub = st
                .get("subgraph")
                .or_else(|| st.get("macro"))
                .or_else(|| st.get("target"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("missing_repeat_target: repeat at {step_prefix} requires 'subgraph' or 'macro'"))?;
            if in_list.is_empty() {
                return Err(format!("missing_repeat_input: repeat at {step_prefix} requires >= 1 input"));
            }
            let final_out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
            let tie_weights = st.get("tie_weights").and_then(|v| v.as_bool()).unwrap_or(false);
            if tie_weights && st.get("weights_per_depth").is_some() {
                return Err(format!(
                    "tied_weights_per_depth_conflict: repeat at {step_prefix} cannot combine tie_weights=true with weights_per_depth"
                ));
            }
            let mut shared_tied_map: Option<HashMap<String, usize>> =
                if tie_weights { Some(HashMap::new()) } else { None };

            let mut cur_in_slot = resolve_in(&in_list[0], scope_env, &mut state.alloc)?;
            let mut extra_in_slots = Vec::new();
            for item in in_list.iter().skip(1) {
                extra_in_slots.push(resolve_in(item, scope_env, &mut state.alloc)?);
            }

            let target_op = if target_sub.starts_with("macro:") || is_builtin_macro(target_sub) {
                format!("macro:{}", target_sub.trim_start_matches("macro:"))
            } else {
                format!("subgraph:{}", target_sub.trim_start_matches("subgraph:"))
            };

            for d in 0..depth {
                let iter_out_slot = if d + 1 == depth {
                    final_out_slot
                } else {
                    state.alloc.fresh_slot()
                };
                let mut iter_in = vec![Value::from(cur_in_slot)];
                for &ex in &extra_in_slots {
                    iter_in.push(Value::from(ex));
                }
                let iter_weights = if tie_weights {
                    st.get("weights").cloned()
                } else if let Some(per_depth) = st.get("weights_per_depth").and_then(|v| v.as_array()) {
                    per_depth.get(d).cloned()
                } else {
                    st.get("weights").cloned()
                };

                let mut sub_obj = Map::new();
                sub_obj.insert("op".into(), Value::String(target_op.clone()));
                sub_obj.insert("params".into(), st.get("params").cloned().unwrap_or_else(|| Value::Object(Map::new())));
                if let Some(w) = iter_weights {
                    sub_obj.insert("weights".into(), w);
                }
                sub_obj.insert("in".into(), Value::Array(iter_in));
                sub_obj.insert("out".into(), Value::from(iter_out_slot));

                let mut child_stack = call_stack.to_vec();
                child_stack.push(format!("repeat({target_sub}#{d})"));
                let mut child_env = scope_env.clone();
                expand_step_list(
                    &[Value::Object(sub_obj)],
                    &mut child_env,
                    state,
                    &child_stack,
                    &format!("{step_prefix}.r{d}"),
                    base_dir,
                    shared_tied_map.as_mut(),
                )?;
                cur_in_slot = iter_out_slot;
            }

            state.events.push(ExpansionEvent {
                kind: "repeat".to_string(),
                target: target_sub.to_string(),
                prefix: step_prefix,
                depth: Some(depth),
                tie_weights: Some(tie_weights),
            });
            continue;
        }

        if op == "residual" {
            let target_sub = st
                .get("subgraph")
                .or_else(|| st.get("macro"))
                .or_else(|| st.get("target"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("invalid_residual: at {step_prefix}"))?;
            if in_list.is_empty() {
                return Err(format!("invalid_residual: {step_prefix} requires >= 1 input"));
            }
            let x_slot = resolve_in(&in_list[0], scope_env, &mut state.alloc)?;
            let mut resolved_ins = vec![Value::from(x_slot)];
            for item in in_list.iter().skip(1) {
                resolved_ins.push(Value::from(resolve_in(item, scope_env, &mut state.alloc)?));
            }
            let fx_slot = state.alloc.fresh_slot();
            let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
            let inner_op = if target_sub.starts_with("macro:") || is_builtin_macro(target_sub) {
                format!("macro:{}", target_sub.trim_start_matches("macro:"))
            } else {
                format!("subgraph:{}", target_sub.trim_start_matches("subgraph:"))
            };
            let mut inner_obj = Map::new();
            inner_obj.insert("op".into(), Value::String(inner_op));
            inner_obj.insert("params".into(), st.get("params").cloned().unwrap_or_else(|| Value::Object(Map::new())));
            if let Some(w) = st.get("weights") {
                inner_obj.insert("weights".into(), w.clone());
            }
            inner_obj.insert("in".into(), Value::Array(resolved_ins));
            inner_obj.insert("out".into(), Value::from(fx_slot));

            let mut child_stack = call_stack.to_vec();
            child_stack.push(format!("residual({target_sub})"));
            let mut child_env = scope_env.clone();
            expand_step_list(
                &[Value::Object(inner_obj)],
                &mut child_env,
                state,
                &child_stack,
                &format!("{step_prefix}.inner"),
                base_dir,
                tied_cache.as_deref_mut(),
            )?;

            let add_layer_idx = state.layers.len();
            let add_id = state.alloc.fresh_layer_id();
            state.layers.push(LayerDecl {
                constructor: "add".into(),
                args: vec![add_id.into()],
            });
            state.steps.push(StepDecl {
                kind: "binary".into(),
                layer: add_layer_idx,
                slots: vec![fx_slot, x_slot, out_slot],
            });
            state.events.push(ExpansionEvent {
                kind: "residual".to_string(),
                target: target_sub.to_string(),
                prefix: step_prefix,
                depth: None,
                tie_weights: None,
            });
            continue;
        }

        if op == "parallel_concat" {
            let branches = st
                .get("branches")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("invalid_parallel_concat at {step_prefix}"))?;
            if branches.len() < 2 {
                return Err(format!("parallel_concat at {step_prefix} requires >= 2 branches"));
            }
            let concat_dim = st.get("dim").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
            let mut branch_outs = Vec::with_capacity(branches.len());

            for (bi, b_spec) in branches.iter().enumerate() {
                let b_out = state.alloc.fresh_slot();
                branch_outs.push(b_out);
                let mut b_steps = if let Some(arr) = b_spec.as_array() {
                    arr.clone()
                } else if let Some(obj) = b_spec.as_object() {
                    let mut c = obj.clone();
                    if !c.contains_key("in") {
                        c.insert("in".into(), Value::Array(in_list.clone()));
                    }
                    c.insert("out".into(), Value::from(b_out));
                    vec![Value::Object(c)]
                } else {
                    return Err(format!("parallel_concat branch {bi} must be object or array"));
                };
                if b_spec.is_array()
                    && let Some(last) = b_steps.last_mut().and_then(|v| v.as_object_mut())
                {
                    last.insert("out".into(), Value::from(b_out));
                }
                let mut child_stack = call_stack.to_vec();
                child_stack.push(format!("branch#{bi}"));
                let mut child_env = scope_env.clone();
                expand_step_list(
                    &b_steps,
                    &mut child_env,
                    state,
                    &child_stack,
                    &format!("{step_prefix}.b{bi}"),
                    base_dir,
                    tied_cache.as_deref_mut(),
                )?;
            }

            let concat_layer_idx = state.layers.len();
            let concat_id = state.alloc.fresh_layer_id();
            state.layers.push(LayerDecl {
                constructor: "concat".into(),
                args: vec![concat_id.into(), concat_dim.into()],
            });
            let mut acc = branch_outs[0];
            for (bi, &bo) in branch_outs.iter().enumerate().skip(1) {
                let target = if bi + 1 == branch_outs.len() {
                    out_slot
                } else {
                    state.alloc.fresh_slot()
                };
                state.steps.push(StepDecl {
                    kind: "binary".into(),
                    layer: concat_layer_idx,
                    slots: vec![acc, bo, target],
                });
                acc = target;
            }
            continue;
        }

        if op == "ref" || st.contains_key("$ref") {
            let ref_path = st
                .get("$ref")
                .or_else(|| st.get("file"))
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("missing_ref_path at {step_prefix}"))?;
            let abs = base_dir.join(ref_path);
            let canon = abs.canonicalize().unwrap_or(abs);
            let token = format!("file:{}", canon.display());
            if call_stack.contains(&token) {
                return Err(format!(
                    "recursive_cycle_detected: cyclic file ref {}",
                    canon.display()
                ));
            }
            let raw_child = std::fs::read_to_string(&canon)
                .map_err(|e| format!("cannot read ref '{}': {e}", canon.display()))?;
            let child_val: Value = serde_json::from_str(&raw_child)
                .map_err(|e| format!("invalid JSON in ref '{}': {e}", canon.display()))?;
            let child_dir = canon.parent().unwrap_or(base_dir);
            let (child_v1, _) = compile_recursive_spec(&child_val, child_dir)?;

            // Inline child v1 into current state
            let mut in_slots = Vec::new();
            for item in &in_list {
                in_slots.push(resolve_in(item, scope_env, &mut state.alloc)?);
            }
            let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
            let mut child_slot_map = HashMap::new();
            for (idx, p) in child_v1.ports.iter().enumerate() {
                if let Some(&mapped) = in_slots.get(idx) {
                    child_slot_map.insert(p.slot, mapped);
                }
            }
            child_slot_map.insert(child_v1.output_slot, out_slot);

            let layer_offset = state.layers.len();
            for (li, mut l) in child_v1.layers.into_iter().enumerate() {
                let new_id = state.alloc.fresh_layer_id();
                if !l.args.is_empty() {
                    l.args[0] = Value::from(new_id);
                }
                if let Some(w) = child_v1.weights.get(&li.to_string()) {
                    state.weights.insert((layer_offset + li).to_string(), w.clone());
                }
                state.layers.push(l);
            }
            for cs in child_v1.steps {
                let mapped_slots: Vec<usize> = cs
                    .slots
                    .into_iter()
                    .map(|sl| *child_slot_map.entry(sl).or_insert_with(|| state.alloc.fresh_slot()))
                    .collect();
                state.steps.push(StepDecl {
                    kind: cs.kind,
                    layer: cs.layer + layer_offset,
                    slots: mapped_slots,
                });
            }
            continue;
        }

        let clean_name = op
            .strip_prefix("subgraph:")
            .or_else(|| op.strip_prefix("macro:"))
            .unwrap_or(op);

        if !op.starts_with("macro:") && state.subgraphs.contains_key(clean_name) {
            let cycle_token = format!("subgraph:{clean_name}");
            if call_stack.contains(&cycle_token) {
                return Err(format!(
                    "recursive_cycle_detected: {} -> {cycle_token}",
                    call_stack.join(" -> ")
                ));
            }
            let sg = state.subgraphs.get(clean_name).unwrap().clone();
            let sg_in_names: Vec<String> = sg
                .get("in")
                .or_else(|| sg.get("inputs"))
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();

            let mut in_slots = Vec::with_capacity(in_list.len());
            for item in &in_list {
                in_slots.push(resolve_in(item, scope_env, &mut state.alloc)?);
            }
            if sg_in_names.len() != in_slots.len() {
                return Err(format!(
                    "subgraph_arity_mismatch: subgraph '{clean_name}' expects {} inputs, got {}",
                    sg_in_names.len(),
                    in_slots.len()
                ));
            }
            let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;

            let mut child_env = HashMap::new();
            for (name, slot) in sg_in_names.iter().zip(in_slots.iter()) {
                child_env.insert(name.clone(), *slot);
            }
            if let Some(sg_out) = sg.get("out").or_else(|| sg.get("output")).and_then(|v| v.as_str()) {
                child_env.insert(sg_out.to_string(), out_slot);
            }

            let mut param_map = sg
                .get("defaults")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            if let Some(caller_params) = st.get("params").and_then(|v| v.as_object()) {
                for (k, v) in caller_params {
                    param_map.insert(k.clone(), v.clone());
                }
            }
            let sg_steps: Vec<Value> = sg
                .get("steps")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|cs| substitute_step(cs, &param_map, st.get("weights")))
                        .collect()
                })
                .unwrap_or_default();

            let mut child_stack = call_stack.to_vec();
            child_stack.push(cycle_token);
            expand_step_list(
                &sg_steps,
                &mut child_env,
                state,
                &child_stack,
                &format!("{step_prefix}:{clean_name}"),
                base_dir,
                tied_cache.as_deref_mut(),
            )?;
            state.events.push(ExpansionEvent {
                kind: "subgraph".to_string(),
                target: clean_name.to_string(),
                prefix: step_prefix,
                depth: None,
                tie_weights: None,
            });
            continue;
        }

        if op.starts_with("macro:") || is_builtin_macro(clean_name) {
            let mut in_slots = Vec::with_capacity(in_list.len());
            for item in &in_list {
                in_slots.push(resolve_in(item, scope_env, &mut state.alloc)?);
            }
            let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
            let params = st
                .get("params")
                .and_then(|v| v.as_object())
                .cloned()
                .unwrap_or_default();
            expand_builtin_macro(
                clean_name,
                &params,
                st.get("weights"),
                &in_slots,
                out_slot,
                state,
                &step_prefix,
            )?;
            continue;
        }

        // Primitive operation
        let mut in_slots = Vec::with_capacity(in_list.len());
        for item in &in_list {
            in_slots.push(resolve_in(item, scope_env, &mut state.alloc)?);
        }
        let out_slot = resolve_out(raw_out, scope_env, &mut state.alloc)?;
        let arity = in_slots.len();
        if arity != 1 && arity != 2 {
            return Err(format!(
                "invalid_primitive_arity: primitive op '{op}' at {step_prefix} requires 1 or 2 inputs, got {arity}"
            ));
        }

        let tie_key = tied_cache.as_ref().map(|_| {
            let sig = st
                .get("args")
                .or_else(|| st.get("params"))
                .map(|v| v.to_string())
                .unwrap_or_else(|| "[]".to_string());
            let tie_id = st
                .get("tie_id")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| si.to_string());
            format!("{op}:{sig}:{tie_id}")
        });

        let layer_idx = if let (Some(cache), Some(tk)) = (tied_cache.as_deref_mut(), tie_key.as_ref()) {
            if let Some(&existing) = cache.get(tk) {
                existing
            } else {
                let idx = push_primitive_layer(op, st, state, &step_prefix)?;
                cache.insert(tk.clone(), idx);
                idx
            }
        } else {
            push_primitive_layer(op, st, state, &step_prefix)?
        };

        let mut step_slots = in_slots;
        step_slots.push(out_slot);
        state.steps.push(StepDecl {
            kind: if arity == 2 { "binary" } else { "unary" }.to_string(),
            layer: layer_idx,
            slots: step_slots,
        });
    }
    Ok(())
}

fn push_primitive_layer(
    op: &str,
    st: &Map<String, Value>,
    state: &mut CompileState,
    step_prefix: &str,
) -> Result<usize, String> {
    let reg_id = state.alloc.fresh_layer_id();
    let args = if let Some(raw_args) = st.get("args").and_then(|v| v.as_array()) {
        if st.get("include_id").and_then(|v| v.as_bool()) == Some(true) {
            raw_args.clone()
        } else {
            let mut v = vec![Value::from(reg_id)];
            v.extend(raw_args.iter().cloned());
            v
        }
    } else {
        let params = st
            .get("params")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        build_args_from_params(op, reg_id, &params)?
    };
    let layer_idx = state.layers.len();
    state.layers.push(LayerDecl {
        constructor: op.to_string(),
        args,
    });
    if let Some(w_arr) = st.get("weights").and_then(|v| v.as_array()) {
        let floats = parse_f32_values(w_arr, &format!("{step_prefix}:{op}.weights"))?;
        state.weights.insert(layer_idx.to_string(), floats);
    }
    Ok(layer_idx)
}

pub fn compile_recursive_spec(raw: &Value, base_dir: &Path) -> Result<(SpecV1, ExpansionReceipt), String> {
    let Some(obj) = raw.as_object() else {
        return Err("recursive spec must be a JSON object".into());
    };

    let mut alloc = Allocator::new();
    let mut ports = Vec::new();
    let mut name_to_slot = HashMap::new();

    match obj.get("ports") {
        Some(Value::Array(arr)) => {
            for p in arr {
                let p_obj = p.as_object().ok_or("port must be object")?;
                let slot = p_obj
                    .get("slot")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or_else(|| alloc.fresh_slot());
                let name = p_obj
                    .get("id")
                    .or_else(|| p_obj.get("name"))
                    .and_then(|v| v.as_str())
                    .map(String::from)
                    .unwrap_or_else(|| slot.to_string());
                alloc.bind_fixed_slot(&name, slot);
                name_to_slot.insert(name, slot);
                let shape_arr = p_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("port '{slot}' missing shape"))?;
                if shape_arr.len() != 4 {
                    return Err(format!("port '{slot}' shape must be 4-D"));
                }
                let shape = [
                    shape_arr[0].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[1].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[2].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[3].as_u64().ok_or("invalid shape dim")? as usize,
                ];
                ports.push(PortDecl {
                    slot,
                    role: p_obj.get("role").and_then(|v| v.as_str()).unwrap_or("observation").to_string(),
                    shape,
                    layout: p_obj.get("layout").and_then(|v| v.as_str()).unwrap_or("feature_axis1_singleton").to_string(),
                });
            }
        }
        Some(Value::Object(map)) => {
            let mut used = HashSet::new();
            let mut auto_idx = 0usize;
            for (name, p) in map {
                let p_obj = p.as_object().ok_or_else(|| format!("port '{name}' must be object"))?;
                let slot = if let Some(s) = p_obj.get("slot").and_then(|v| v.as_u64()) {
                    s as usize
                } else {
                    while used.contains(&auto_idx) {
                        auto_idx += 1;
                    }
                    auto_idx
                };
                used.insert(slot);
                alloc.bind_fixed_slot(name, slot);
                name_to_slot.insert(name.clone(), slot);
                let shape_arr = p_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("port '{name}' missing shape"))?;
                if shape_arr.len() != 4 {
                    return Err(format!("port '{name}' shape must be 4-D"));
                }
                let shape = [
                    shape_arr[0].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[1].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[2].as_u64().ok_or("invalid shape dim")? as usize,
                    shape_arr[3].as_u64().ok_or("invalid shape dim")? as usize,
                ];
                ports.push(PortDecl {
                    slot,
                    role: p_obj.get("role").and_then(|v| v.as_str()).unwrap_or("observation").to_string(),
                    shape,
                    layout: p_obj.get("layout").and_then(|v| v.as_str()).unwrap_or("feature_axis1_singleton").to_string(),
                });
            }
            ports.sort_by_key(|p| p.slot);
        }
        _ => return Err("recursive spec requires 'ports' array or map".into()),
    }

    let mut subgraphs = HashMap::new();
    if let Some(sgs) = obj.get("subgraphs").and_then(|v| v.as_object()) {
        for (k, v) in sgs {
            subgraphs.insert(k.clone(), v.clone());
        }
    }

    let mut layers = Vec::new();
    if let Some(v1_layers) = obj.get("layers").and_then(|v| v.as_array()) {
        for l in v1_layers {
            let parsed: LayerDecl = serde_json::from_value(l.clone())
                .map_err(|e| format!("invalid layer declaration: {e}"))?;
            if let Some(id) = parsed.args.first().and_then(|v| v.as_u64()) {
                let id_u = id as usize;
                if id_u >= alloc.next_layer_id {
                    alloc.next_layer_id = id_u + 1;
                }
            }
            layers.push(parsed);
        }
    }

    let mut weights = BTreeMap::new();
    if let Some(w_obj) = obj.get("weights").and_then(|v| v.as_object()) {
        for (k, v) in w_obj {
            if let Some(arr) = v.as_array() {
                weights.insert(k.clone(), parse_f32_values(arr, &format!("weights[{k}]"))?);
            }
        }
    }

    let mut state = CompileState {
        alloc,
        layers,
        steps: Vec::new(),
        weights,
        subgraphs,
        events: Vec::new(),
    };

    let raw_steps = obj
        .get("steps")
        .and_then(|v| v.as_array())
        .ok_or("recursive spec requires 'steps' array")?;
    let mut root_env = HashMap::new();
    expand_step_list(
        raw_steps,
        &mut root_env,
        &mut state,
        &["root".to_string()],
        "root",
        base_dir,
        None,
    )?;

    let out_sym = obj
        .get("output")
        .or_else(|| obj.get("outputSlot"))
        .ok_or("recursive spec requires 'output' or 'outputSlot'")?;
    let output_slot = if let Some(s) = out_sym.as_str() {
        if let Some(&slot) = root_env.get(s) {
            slot
        } else {
            state.alloc.resolve_slot(out_sym, false)?
        }
    } else {
        state.alloc.resolve_slot(out_sym, false)?
    };

    let mut inputs = BTreeMap::new();
    if let Some(in_obj) = obj.get("inputs").and_then(|v| v.as_object()) {
        for (k, v) in in_obj {
            let arr = v
                .as_array()
                .ok_or_else(|| format!("inputs['{k}'] must be an array"))?
                .clone();
            if let Some(&slot) = name_to_slot.get(k) {
                inputs.insert(slot.to_string(), arr);
            } else {
                inputs.insert(k.clone(), arr);
            }
        }
    }

    let spec = SpecV1 {
        num_slots: state.alloc.next_slot_id.max(output_slot + 1),
        layers: state.layers,
        steps: state.steps,
        output_slot,
        ports,
        weights: state.weights,
        inputs,
    };
    let receipt = ExpansionReceipt {
        schema: RECURSIVE_SPEC_SCHEMA.to_string(),
        num_slots: spec.num_slots,
        layers_count: spec.layers.len(),
        steps_count: spec.steps.len(),
        compiled_sha256: sha256_json(&spec),
        events: state.events,
        port_names: name_to_slot.into_iter().collect(),
    };
    Ok((spec, receipt))
}
