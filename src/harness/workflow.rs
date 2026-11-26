use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use super::workspace::{SemanticDigest, compute_semantic_digest, ensure_workspace_dir};
use crate::bundle::{export_bundle, verify_bundle};
use crate::diff_ref::diff_ref_spec;
use crate::engine::run_spec;
use crate::grad_spec::{GradSpecOptions, compile_grad_spec};
use crate::gradcheck::gradcheck_spec;
use crate::liveness::{auto_repair_spec, compact_slots_by_liveness};
use crate::quant::{BudgetMode, QuantScheme, quant_budget};
use crate::recurse::{SearchOptions, search_graph_tree_with_options, solve_gd_autodiff};
use crate::rollout::{rollout_fixpoint, rollout_scan, rollout_unroll_with_options};
use crate::shard::shard_spec;
use crate::soak::soak_spec;
use crate::spec::{LoadedSpec, load_spec_file, load_spec_str, parse_f32_values};
use crate::validate::validate_spec;
use crate::verify::TargetsFile;

fn ensure_allowed_keys(obj: &serde_json::Map<String, Value>, ctx: &str, allowed: &[&str]) -> Result<(), String> {
    for k in obj.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("workflow {ctx}: unknown field '{k}'"));
        }
    }
    Ok(())
}

fn wf_opt_str<'a>(st: &'a Value, ctx: &str, key: &str) -> Result<Option<&'a str>, String> {
    match st.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(other) => Err(format!(
            "workflow {ctx}: field '{key}' must be a string, got {other}"
        )),
    }
}

fn wf_str<'a>(st: &'a Value, ctx: &str, key: &str, default: &'a str) -> Result<&'a str, String> {
    Ok(wf_opt_str(st, ctx, key)?.unwrap_or(default))
}

fn wf_bool(st: &Value, ctx: &str, key: &str, default: bool) -> Result<bool, String> {
    match st.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Bool(b)) => Ok(*b),
        Some(other) => Err(format!(
            "workflow {ctx}: field '{key}' must be a boolean, got {other}"
        )),
    }
}

fn wf_opt_u64(st: &Value, ctx: &str, key: &str) -> Result<Option<u64>, String> {
    match st.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(n)) => n.as_u64().ok_or_else(|| {
            format!("workflow {ctx}: field '{key}' must be a non-negative integer, got {n}")
        }).map(Some),
        Some(other) => Err(format!(
            "workflow {ctx}: field '{key}' must be a non-negative integer, got {other}"
        )),
    }
}

fn wf_u64(st: &Value, ctx: &str, key: &str, default: u64) -> Result<u64, String> {
    Ok(wf_opt_u64(st, ctx, key)?.unwrap_or(default))
}

fn wf_opt_usize(st: &Value, ctx: &str, key: &str) -> Result<Option<usize>, String> {
    match wf_opt_u64(st, ctx, key)? {
        Some(v) => usize::try_from(v)
            .map(Some)
            .map_err(|_| format!("workflow {ctx}: field '{key}' overflows usize: {v}")),
        None => Ok(None),
    }
}

fn wf_usize(st: &Value, ctx: &str, key: &str, default: usize) -> Result<usize, String> {
    Ok(wf_opt_usize(st, ctx, key)?.unwrap_or(default))
}

fn wf_usize_pos(st: &Value, ctx: &str, key: &str, default: usize) -> Result<usize, String> {
    let v = wf_usize(st, ctx, key, default)?;
    if v == 0 {
        return Err(format!(
            "workflow {ctx}: field '{key}' must be >= 1, got 0"
        ));
    }
    Ok(v)
}

fn wf_f64(st: &Value, ctx: &str, key: &str, default: f64) -> Result<f64, String> {
    match st.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => {
            let v = n.as_f64().ok_or_else(|| {
                format!("workflow {ctx}: field '{key}' must be a finite number, got {n}")
            })?;
            if !v.is_finite() {
                return Err(format!(
                    "workflow {ctx}: field '{key}' must be finite, got {v}"
                ));
            }
            Ok(v)
        }
        Some(other) => Err(format!(
            "workflow {ctx}: field '{key}' must be a finite number, got {other}"
        )),
    }
}

fn wf_f64_non_neg(st: &Value, ctx: &str, key: &str, default: f64) -> Result<f64, String> {
    let v = wf_f64(st, ctx, key, default)?;
    if v < 0.0 {
        return Err(format!(
            "workflow {ctx}: field '{key}' must be >= 0, got {v}"
        ));
    }
    Ok(v)
}

fn wf_f64_pos(st: &Value, ctx: &str, key: &str, default: f64) -> Result<f64, String> {
    let v = wf_f64(st, ctx, key, default)?;
    if v <= 0.0 {
        return Err(format!(
            "workflow {ctx}: field '{key}' must be > 0, got {v}"
        ));
    }
    Ok(v)
}

pub fn execute_harness_workflow(
    workflow: &Value,
    base_dir: &Path,
    workspace_override: Option<&Path>,
) -> Result<Value, String> {
    let wf_obj = workflow
        .as_object()
        .ok_or_else(|| "workflow JSON root must be an object".to_string())?;
    ensure_allowed_keys(
        wf_obj,
        "root",
        &["schema", "name", "description", "workspace", "spec", "pipeline"],
    )?;

    let ws_from_wf = match workflow.get("workspace") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => {
            if s.trim().is_empty() {
                return Err("workflow field 'workspace' must be a non-empty string".into());
            }
            Some(PathBuf::from(s))
        }
        Some(other) => {
            return Err(format!(
                "workflow field 'workspace' must be a string path, got {other}"
            ));
        }
    };
    let ws_rel = workspace_override
        .map(PathBuf::from)
        .or(ws_from_wf)
        .unwrap_or_else(|| PathBuf::from(".burn-harness"));
    let ws_path = if ws_rel.is_absolute() {
        ws_rel
    } else {
        base_dir.join(ws_rel)
    };
    let ws = ensure_workspace_dir(&ws_path)?;

    let loaded_initial: LoadedSpec = match workflow.get("spec") {
        Some(Value::String(s_path)) => {
            let resolved = base_dir.join(s_path);
            load_spec_file(&resolved)?
        }
        Some(s_obj @ Value::Object(_)) => {
            let raw_str = serde_json::to_string(s_obj).map_err(|e| e.to_string())?;
            load_spec_str(&raw_str, base_dir)?
        }
        Some(other) => {
            return Err(format!(
                "workflow field 'spec' must be a file path string or inline spec object, got {other}"
            ));
        }
        None => {
            return Err("workflow must provide 'spec' (file path or inline spec object)".into());
        }
    };

    let mut active_spec = loaded_initial.spec;
    let active_expansion = loaded_initial.expansion;
    let mut stage_reports = Vec::new();
    let mut events = Vec::new();
    let mut proofs = Vec::new();
    let mut last_run_digest: Option<SemanticDigest> = None;

    let sync_state = |events: &[Value], proofs: &[Value]| -> Result<(), String> {
        let raw_st = std::fs::read_to_string(&ws.state_file).map_err(|e| {
            format!(
                "failed to read workspace state '{}': {e}",
                ws.state_file.display()
            )
        })?;
        let mut st: Value = serde_json::from_str(&raw_st).map_err(|e| {
            format!(
                "corrupted workspace state '{}': {e}",
                ws.state_file.display()
            )
        })?;
        let ev_arr = st
            .get_mut("tables")
            .and_then(|t| t.get_mut("events"))
            .and_then(|a| a.as_array_mut())
            .ok_or_else(|| {
                format!(
                    "workspace state '{}' missing 'tables.events' array",
                    ws.state_file.display()
                )
            })?;
        ev_arr.extend(events.iter().cloned());
        let pr_arr = st
            .get_mut("tables")
            .and_then(|t| t.get_mut("proofs"))
            .and_then(|a| a.as_array_mut())
            .ok_or_else(|| {
                format!(
                    "workspace state '{}' missing 'tables.proofs' array",
                    ws.state_file.display()
                )
            })?;
        pr_arr.extend(proofs.iter().cloned());
        let serialized = serde_json::to_string_pretty(&st)
            .map_err(|e| format!("failed to serialize workspace state JSON: {e}"))?;
        std::fs::write(&ws.state_file, format!("{serialized}\n")).map_err(|e| {
            format!(
                "failed to write workspace state '{}': {e}",
                ws.state_file.display()
            )
        })?;
        Ok(())
    };

    let pipeline = match workflow.get("pipeline") {
        Some(Value::Array(arr)) => arr.clone(),
        Some(other) => {
            return Err(format!(
                "workflow field 'pipeline' must be an array of stage objects, got {other}"
            ));
        }
        None => {
            return Err("workflow must provide 'pipeline' array of stages".into());
        }
    };

    for (idx, st) in pipeline.iter().enumerate() {
        let Some(st_obj) = st.as_object() else {
            return Err(format!(
                "workflow pipeline[{idx}] must be a JSON object, got {st}"
            ));
        };
        let stage_val = st_obj.get("stage").or_else(|| st_obj.get("op"));
        let stage_name = match stage_val {
            Some(Value::String(s)) if !s.trim().is_empty() => s.as_str(),
            Some(other) => {
                return Err(format!(
                    "workflow pipeline[{idx}] 'stage' must be a non-empty string, got {other}"
                ));
            }
            None => {
                return Err(format!(
                    "workflow pipeline[{idx}] is missing required 'stage' (or 'op') string field"
                ));
            }
        };

        let step_res: Result<Value, String> = (|| match stage_name {
            "compile" | "compile_v2" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "auto_repair", "max_weights", "compact_slots"],
                )?;
                let auto_repair = wf_bool(st, stage_name, "auto_repair", false)?;
                let max_weights = wf_usize(st, stage_name, "max_weights", 65535)?;
                let compact = wf_bool(st, stage_name, "compact_slots", false)?;
                let mut remediation = Vec::new();
                if auto_repair {
                    let ar = auto_repair_spec(&active_spec, max_weights)?;
                    active_spec = ar.spec;
                    remediation = ar.remediation_receipt;
                }
                let mut slots_saved = 0usize;
                if compact {
                    let comp = compact_slots_by_liveness(&active_spec)?;
                    active_spec = comp.spec;
                    slots_saved = comp.slots_saved;
                }
                events.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "detail": format!("layers={};steps={};slots={}", active_spec.layers.len(), active_spec.steps.len(), active_spec.num_slots),
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "layers_count": active_spec.layers.len(),
                    "steps_count": active_spec.steps.len(),
                    "num_slots": active_spec.num_slots,
                    "slots_saved": slots_saved,
                    "remediation_count": remediation.len(),
                }))
            }
            "validate" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op", "explain"])?;
                let explain = wf_bool(st, stage_name, "explain", true)?;
                let rep = validate_spec(&active_spec, explain);
                if !rep.valid {
                    return Err(format!("validate stage failed: {}", rep.code));
                }
                events.push(json!({ "stage": "validate", "index": idx, "valid": rep.valid }));
                Ok(json!({
                    "stage": "validate",
                    "ok": rep.valid,
                    "total_params": rep.explain.as_ref().map(|e| e.total_params),
                    "peak_bytes": rep.explain.as_ref().map(|e| e.liveness.peak_bytes),
                }))
            }
            "rollout_unroll" | "unroll" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "steps", "state_port", "compact_slots"],
                )?;
                let steps = wf_usize_pos(st, stage_name, "steps", 3)?;
                let state_port = wf_opt_str(st, stage_name, "state_port")?;
                let compact = wf_bool(st, stage_name, "compact_slots", false)?;
                let ur = rollout_unroll_with_options(
                    &active_spec,
                    active_expansion.as_ref(),
                    steps,
                    state_port,
                    compact,
                )?;
                active_spec = ur.unrolled_spec;
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "temporal_horizon": steps,
                    "layers_count": active_spec.layers.len(),
                    "steps_count": active_spec.steps.len(),
                    "num_slots": active_spec.num_slots,
                }))
            }
            "search_rewrites" | "recursive_rewrite" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &[
                        "stage",
                        "op",
                        "max_depth",
                        "abs_tol",
                        "rel_tol",
                        "quant_tol",
                        "quant_scheme",
                    ],
                )?;
                let max_depth = wf_usize(st, stage_name, "max_depth", 3)?;
                let abs_tol = wf_f64_non_neg(st, stage_name, "abs_tol", 1e-5)?;
                let rel_tol = wf_f64_non_neg(st, stage_name, "rel_tol", 1e-5)?;
                let quant_tol = if st_obj.contains_key("quant_tol")
                    && !st.get("quant_tol").is_some_and(|v| v.is_null())
                {
                    Some(wf_f64_non_neg(st, stage_name, "quant_tol", 0.0)?)
                } else {
                    None
                };
                let quant_scheme =
                    QuantScheme::parse(wf_str(st, stage_name, "quant_scheme", "q8-per-block")?)?;
                let opts = SearchOptions {
                    max_depth,
                    abs_tol,
                    rel_tol,
                    quant_tol,
                    quant_scheme,
                };
                let sr = search_graph_tree_with_options(&active_spec, None, &opts)?;
                let best_max_abs_diff = sr
                    .nodes
                    .iter()
                    .find(|n| n.id == sr.best_node_id)
                    .map(|n| n.max_abs_diff)
                    .unwrap_or(0.0);
                active_spec = sr.best_spec;
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": true,
                    "max_abs_diff": best_max_abs_diff,
                    "rewrite_chain": sr.transform_chain,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "explored_nodes": sr.visited_nodes,
                    "best_steps": sr.best_steps,
                    "best_weight_bytes": sr.best_weight_bytes,
                    "best_rewrite_chain": sr.transform_chain,
                }))
            }
            "quant_budget" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "tol", "scheme", "mode"],
                )?;
                let tol = wf_f64_non_neg(st, stage_name, "tol", 0.05)?;
                let scheme = QuantScheme::parse(wf_str(st, stage_name, "scheme", "q8-per-block")?)?;
                let mode = BudgetMode::parse(wf_str(st, stage_name, "mode", "bound")?)?;
                let qb = quant_budget(&active_spec, None, tol, scheme, mode)?;
                active_spec = qb.planned_spec;
                Ok(json!({
                    "stage": stage_name,
                    "ok": qb.ok,
                    "quantized_layers": qb.quantized_layers,
                    "total_saved_bytes": qb.total_saved_bytes,
                    "verified_delta": qb.verified_delta,
                }))
            }
            "fit" | "fit_hybrid" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &[
                        "stage",
                        "op",
                        "targets",
                        "generations",
                        "gens",
                        "pop",
                        "mu",
                        "sigma",
                        "lr",
                        "seed",
                        "sensitivity_guided",
                        "freeze_threshold",
                        "strategy",
                        "rng",
                    ],
                )?;
                if st_obj.contains_key("generations") && st_obj.contains_key("gens") {
                    return Err(format!(
                        "workflow {stage_name}: specify either 'generations' or 'gens', not both"
                    ));
                }
                let t_val = st
                    .get("targets")
                    .ok_or_else(|| "workflow fit stage requires 'targets'".to_string())?;
                let tf: TargetsFile = if let Some(tp) = t_val.as_str() {
                    let raw =
                        std::fs::read_to_string(base_dir.join(tp)).map_err(|e| e.to_string())?;
                    serde_json::from_str(&raw).map_err(|e| e.to_string())?
                } else {
                    serde_json::from_value(t_val.clone()).map_err(|e| e.to_string())?
                };
                let gens = match wf_opt_usize(st, stage_name, "generations")? {
                    Some(g) => g,
                    None => wf_usize(st, stage_name, "gens", 25)?,
                };
                let pop = wf_usize(st, stage_name, "pop", 12)?;
                let mu = wf_opt_usize(st, stage_name, "mu")?;
                let sigma = wf_f64(st, stage_name, "sigma", 0.2)?;
                let lr = wf_f64(st, stage_name, "lr", 0.2)?;
                let seed = wf_u64(st, stage_name, "seed", 42)?;
                let sens = wf_bool(st, stage_name, "sensitivity_guided", true)?;
                let freeze_threshold = wf_f64_non_neg(st, stage_name, "freeze_threshold", 1e-6)?;
                let strategy = match wf_opt_str(st, stage_name, "strategy")? {
                    Some(s) => crate::recurse::EsStrategyKind::parse(s)?,
                    None => crate::recurse::EsStrategyKind::OpenEs,
                };
                let rng = match wf_opt_str(st, stage_name, "rng")? {
                    Some(r) => crate::recurse::RngKind::parse(r)?,
                    None => crate::recurse::RngKind::Xorshift64,
                };
                let cfg = crate::recurse::FitEsConfig {
                    generations: gens,
                    pop,
                    mu,
                    sigma,
                    lr,
                    seed,
                    strategy,
                    rng,
                    sensitivity_guided: sens,
                    freeze_threshold,
                };
                let fr = crate::recurse::fit_es_with_config(&active_spec, &tf, &cfg)?;
                active_spec = fr.optimized_spec;
                Ok(json!({
                    "stage": stage_name,
                    "ok": fr.ok,
                    "strategy": fr.strategy,
                    "rng": fr.rng,
                    "start_loss": fr.start_loss,
                    "final_loss": fr.final_loss,
                    "flags": fr.flags,
                }))
            }
            "solve_gd" | "solve-gd" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &[
                        "stage",
                        "op",
                        "targets",
                        "steps",
                        "lr",
                        "precondition",
                        "layer",
                    ],
                )?;
                let t_val = st.get("targets").ok_or_else(|| {
                    "workflow solve_gd stage requires explicit 'targets'".to_string()
                })?;
                let tf: TargetsFile = if let Some(tp) = t_val.as_str() {
                    let raw =
                        std::fs::read_to_string(base_dir.join(tp)).map_err(|e| e.to_string())?;
                    serde_json::from_str(&raw).map_err(|e| e.to_string())?
                } else {
                    serde_json::from_value(t_val.clone()).map_err(|e| e.to_string())?
                };
                let steps = wf_usize_pos(st, stage_name, "steps", 20)?;
                let lr = wf_f64_pos(st, stage_name, "lr", 0.1)?;
                let pre = wf_bool(st, stage_name, "precondition", false)?;
                let layer = wf_opt_usize(st, stage_name, "layer")?;
                let gd = solve_gd_autodiff(&active_spec, Some(&tf), layer, steps, lr, pre)?;
                active_spec = gd.optimized_spec;
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": gd.ok,
                    "final_loss": gd.final_loss,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": gd.ok,
                    "steps_taken": gd.steps_taken,
                    "start_loss": gd.start_loss,
                    "final_loss": gd.final_loss,
                }))
            }
            "rollout_scan" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "steps", "state_port"],
                )?;
                let steps = wf_usize_pos(st, stage_name, "steps", 8)?;
                let state_port = wf_opt_str(st, stage_name, "state_port")?;
                let sc =
                    rollout_scan(&active_spec, active_expansion.as_ref(), steps, state_port, None)?;
                let dig = compute_semantic_digest(
                    &sc.final_state.values,
                    sc.state_shape,
                    Some(&ws.root),
                    1,
                    false,
                )?;
                last_run_digest = Some(dig.clone());
                Ok(json!({
                    "stage": stage_name,
                    "ok": sc.ok,
                    "steps": sc.steps,
                    "output_digest": dig,
                }))
            }
            "rollout_fixpoint" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "tol", "max_steps", "state_port"],
                )?;
                let tol = wf_f64_non_neg(st, stage_name, "tol", 1e-6)?;
                let max_steps = wf_usize_pos(st, stage_name, "max_steps", 100)?;
                let state_port = wf_opt_str(st, stage_name, "state_port")?;
                let fp = rollout_fixpoint(
                    &active_spec,
                    active_expansion.as_ref(),
                    tol,
                    max_steps,
                    state_port,
                )?;
                if !fp.converged {
                    return Err(format!(
                        "rollout_fixpoint did not converge within {max_steps} steps (final_delta={})",
                        fp.final_delta_linf
                    ));
                }
                let dig = compute_semantic_digest(
                    &fp.final_state.values,
                    fp.state_shape,
                    Some(&ws.root),
                    1,
                    false,
                )?;
                last_run_digest = Some(dig.clone());
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": true,
                    "final_delta_linf": fp.final_delta_linf,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "steps_taken": fp.steps_run,
                    "final_delta_linf": fp.final_delta_linf,
                    "fixed_point_digest": dig,
                }))
            }
            "run" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op"])?;
                let out = run_spec(&active_spec)?;
                let dig = compute_semantic_digest(
                    &out.outputs.values,
                    out.outputs.shape,
                    Some(&ws.root),
                    1,
                    false,
                )?;
                last_run_digest = Some(dig.clone());
                Ok(json!({
                    "stage": "run",
                    "ok": true,
                    "output_digest": dig,
                }))
            }
            "grad_spec" | "grad-spec" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &[
                        "stage",
                        "op",
                        "wrt",
                        "order",
                        "dout",
                        "hvp_vec",
                        "compact_slots",
                        "replace_active",
                    ],
                )?;
                let wrt = match st.get("wrt") {
                    None | Some(Value::Null) => "0".to_string(),
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n
                        .as_u64()
                        .ok_or_else(|| {
                            format!("workflow {stage_name}: field 'wrt' must be string or non-negative integer")
                        })?
                        .to_string(),
                    Some(other) => {
                        return Err(format!(
                            "workflow {stage_name}: field 'wrt' must be string or non-negative integer, got {other}"
                        ));
                    }
                };
                let order = wf_usize_pos(st, stage_name, "order", 1)?;
                let dout = match st.get("dout") {
                    None | Some(Value::Null) => None,
                    Some(Value::Array(arr)) => {
                        Some(parse_f32_values(arr, "workflow grad_spec.dout")?)
                    }
                    Some(other) => {
                        return Err(format!(
                            "workflow {stage_name}: field 'dout' must be an array of numbers, got {other}"
                        ));
                    }
                };
                let hvp_vec = match st.get("hvp_vec") {
                    None | Some(Value::Null) => None,
                    Some(Value::Array(arr)) => {
                        Some(parse_f32_values(arr, "workflow grad_spec.hvp_vec")?)
                    }
                    Some(other) => {
                        return Err(format!(
                            "workflow {stage_name}: field 'hvp_vec' must be an array of numbers, got {other}"
                        ));
                    }
                };
                let compact = wf_bool(st, stage_name, "compact_slots", false)?;
                let replace_active = wf_bool(st, stage_name, "replace_active", true)?;
                let opts = GradSpecOptions {
                    wrt,
                    order,
                    dout,
                    hvp_vec,
                    compact_slots: compact,
                };
                let gs = compile_grad_spec(&active_spec, active_expansion.as_ref(), &opts)?;
                if replace_active {
                    active_spec = gs.spec.clone();
                }
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": gs.ok,
                    "order": gs.order,
                    "wrt_slot": gs.wrt_slot,
                    "dout_slot": gs.dout_slot,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": gs.ok,
                    "order": gs.order,
                    "wrt_slot": gs.wrt_slot,
                    "dout_slot": gs.dout_slot,
                    "hvp_slot": gs.hvp_slot,
                    "steps_count": gs.steps_count,
                    "num_slots": gs.num_slots,
                }))
            }
            "soak" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op", "runs"])?;
                let runs = wf_usize_pos(st, stage_name, "runs", 10)?;
                let rep = soak_spec(&active_spec, runs)?;
                if !rep.identical {
                    return Err(format!(
                        "soak detected non-determinism at run {:?}",
                        rep.first_mismatch_index
                    ));
                }
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": rep.identical,
                    "runs": rep.runs,
                    "sha256": rep.sha256,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": rep.identical,
                    "runs": rep.runs,
                    "identical": rep.identical,
                    "sha256": rep.sha256,
                }))
            }
            "gradcheck" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "eps", "tol", "weights", "autodiff"],
                )?;
                let eps = wf_f64_pos(st, stage_name, "eps", 1e-3)?;
                let tol = wf_f64_non_neg(st, stage_name, "tol", 1e-3)?;
                let include_weights = wf_bool(st, stage_name, "weights", true)?;
                let check_autodiff = wf_bool(st, stage_name, "autodiff", false)?;
                let rep = gradcheck_spec(
                    &active_spec,
                    None,
                    eps,
                    tol,
                    include_weights,
                    check_autodiff,
                )?;
                if !rep.ok {
                    return Err("gradcheck stage failed tolerance check".into());
                }
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": rep.ok,
                    "autodiff": check_autodiff,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": rep.ok,
                    "eps": eps,
                    "tol": tol,
                    "checked_autodiff": check_autodiff,
                }))
            }
            "shard" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op", "max_weights"])?;
                let max_weights = wf_usize_pos(st, stage_name, "max_weights", 65535)?;
                let rep = shard_spec(&active_spec, max_weights)?;
                active_spec = rep.spec;
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": rep.ok,
                    "sharded": rep.sharded,
                    "reassembly_exact": rep.reassembly_exact,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": rep.ok,
                    "sharded": rep.sharded,
                    "shards_count": rep.shards.len(),
                    "reassembly_exact": rep.reassembly_exact,
                }))
            }
            "bundle_export" => {
                ensure_allowed_keys(
                    st_obj,
                    stage_name,
                    &["stage", "op", "include_exec", "attest", "out"],
                )?;
                let include_exec = wf_bool(st, stage_name, "include_exec", true)?;
                let attest_labels: Vec<String> = match st.get("attest") {
                    None | Some(Value::Null) => Vec::new(),
                    Some(Value::Array(arr)) => arr
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            v.as_str()
                                .map(|s| s.to_string())
                                .ok_or_else(|| {
                                    format!("workflow {stage_name}: attest[{i}] must be a string")
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    Some(other) => {
                        return Err(format!(
                            "workflow {stage_name}: field 'attest' must be an array of strings, got {other}"
                        ));
                    }
                };
                let bundle = export_bundle(&active_spec, include_exec, &attest_labels)?;
                let ver = verify_bundle(&bundle);
                if !ver.ok {
                    return Err("bundle_export self-verification failed".into());
                }
                if let Some(out_rel) = wf_opt_str(st, stage_name, "out")? {
                    let out_path = if Path::new(out_rel).is_absolute() {
                        PathBuf::from(out_rel)
                    } else {
                        base_dir.join(out_rel)
                    };
                    let txt = serde_json::to_string_pretty(&bundle)
                        .map_err(|e| format!("failed serializing bundle: {e}"))?;
                    std::fs::write(&out_path, format!("{txt}\n"))
                        .map_err(|e| format!("failed writing bundle '{}': {e}", out_path.display()))?;
                }
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": true,
                    "program_identity": bundle.program_identity,
                    "bundle_sha256": bundle.bundle_sha256,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "program_identity": bundle.program_identity,
                    "bundle_sha256": bundle.bundle_sha256,
                    "verified": ver.ok,
                }))
            }
            "diff_ref" | "diff-ref" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op", "tol"])?;
                let tol = wf_f64_pos(st, stage_name, "tol", 1e-5)?;
                let rep = diff_ref_spec(&active_spec, None, None, tol)?;
                if !rep.ok {
                    return Err(format!(
                        "diff_ref stage signal: max_abs_error={} > tol={tol}",
                        rep.max_abs_error
                    ));
                }
                proofs.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "ok": rep.ok,
                    "verdict": rep.verdict,
                    "max_abs_error": rep.max_abs_error,
                    "max_rel_error": rep.max_rel_error,
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": rep.ok,
                    "verdict": rep.verdict,
                    "tol": rep.tol,
                    "cases_checked": rep.cases_checked,
                    "max_abs_error": rep.max_abs_error,
                    "max_rel_error": rep.max_rel_error,
                }))
            }
            other => Err(format!("unknown workflow pipeline stage '{other}'")),
        })();

        match step_res {
            Ok(rep) => stage_reports.push(rep),
            Err(err_msg) => {
                let classified = crate::error::classify_error(&err_msg, Some(stage_name));
                stage_reports.push(json!({
                    "stage": stage_name,
                    "ok": false,
                    "code": classified.code,
                    "error": err_msg,
                    "hint": classified.hint,
                }));
                sync_state(&events, &proofs)?;
                return Ok(json!({
                    "ok": false,
                    "code": classified.code,
                    "error": err_msg,
                    "hint": classified.hint,
                    "failed_stage": stage_name,
                    "workspace": ws.root.display().to_string(),
                    "stages_completed": stage_reports.len(),
                    "stage_reports": stage_reports,
                    "final_output_digest": last_run_digest,
                }));
            }
        }
    }

    sync_state(&events, &proofs)?;
    Ok(json!({
        "ok": true,
        "workspace": ws.root.display().to_string(),
        "stages_completed": stage_reports.len(),
        "stage_reports": stage_reports,
        "final_output_digest": last_run_digest,
        "active_spec_summary": {
            "layers_count": active_spec.layers.len(),
            "steps_count": active_spec.steps.len(),
            "num_slots": active_spec.num_slots,
            "output_slot": active_spec.output_slot,
        }
    }))
}
