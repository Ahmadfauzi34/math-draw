use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use super::workspace::{SemanticDigest, compute_semantic_digest, ensure_workspace_dir};
use crate::engine::run_spec;
use crate::quant::{BudgetMode, QuantScheme, quant_budget};
use crate::recurse::{search_graph_tree, solve_gd_autodiff};
use crate::rollout::{rollout_fixpoint, rollout_scan, rollout_unroll};
use crate::spec::{LoadedSpec, load_spec_file, load_spec_str};
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
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op"])?;
                events.push(json!({
                    "stage": stage_name,
                    "index": idx,
                    "detail": format!("layers={};steps={}", active_spec.layers.len(), active_spec.steps.len()),
                }));
                Ok(json!({
                    "stage": stage_name,
                    "ok": true,
                    "layers_count": active_spec.layers.len(),
                    "steps_count": active_spec.steps.len(),
                    "num_slots": active_spec.num_slots,
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
                }))
            }
            "rollout_unroll" | "unroll" => {
                ensure_allowed_keys(st_obj, stage_name, &["stage", "op", "steps", "state_port"])?;
                let steps = wf_usize_pos(st, stage_name, "steps", 3)?;
                let state_port = wf_opt_str(st, stage_name, "state_port")?;
                let ur = rollout_unroll(&active_spec, active_expansion.as_ref(), steps, state_port)?;
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
                    &["stage", "op", "max_depth", "abs_tol", "rel_tol"],
                )?;
                let max_depth = wf_usize(st, stage_name, "max_depth", 3)?;
                let abs_tol = wf_f64_non_neg(st, stage_name, "abs_tol", 1e-5)?;
                let rel_tol = wf_f64_non_neg(st, stage_name, "rel_tol", 1e-5)?;
                let sr = search_graph_tree(&active_spec, max_depth, None, abs_tol, rel_tol)?;
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
