use serde_json::{Map, Value};

use crate::catalog::find_constructor;

pub(super) fn build_args_from_params(
    op: &str,
    reg_id: usize,
    params: &Map<String, Value>,
) -> Result<Vec<Value>, String> {
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

pub(super) fn substitute_value(v: &Value, param_map: &Map<String, Value>) -> Value {
    if let Some(s) = v.as_str()
        && let Some(key) = s.strip_prefix('$')
        && let Some(rep) = param_map.get(key)
    {
        return rep.clone();
    }
    v.clone()
}

pub(super) fn substitute_step(
    step_val: &Value,
    param_map: &Map<String, Value>,
    caller_weights: Option<&Value>,
) -> Value {
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
