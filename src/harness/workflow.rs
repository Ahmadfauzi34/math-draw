use serde_json::{Value, json};
use std::path::{Path, PathBuf};

use super::workspace::{SemanticDigest, compute_semantic_digest, ensure_workspace_dir};
use crate::engine::run_spec;
use crate::quant::{BudgetMode, QuantScheme, quant_budget};
use crate::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use crate::rollout::{rollout_fixpoint, rollout_scan, rollout_unroll};
use crate::spec::{LoadedSpec, load_spec_file, load_spec_str};
use crate::validate::validate_spec;
use crate::verify::TargetsFile;

pub fn execute_harness_workflow(
    workflow: &Value,
    base_dir: &Path,
    workspace_override: Option<&Path>,
) -> Result<Value, String> {
    let ws_rel = workspace_override
        .map(PathBuf::from)
        .or_else(|| {
            workflow
                .get("workspace")
                .and_then(|v| v.as_str())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| PathBuf::from(".burn-harness"));
    let ws_path = if ws_rel.is_absolute() {
        ws_rel
    } else {
        base_dir.join(ws_rel)
    };
    let ws = ensure_workspace_dir(&ws_path)?;

    let loaded_initial: LoadedSpec =
        if let Some(s_path) = workflow.get("spec").and_then(|v| v.as_str()) {
            let resolved = base_dir.join(s_path);
            load_spec_file(&resolved)?
        } else if let Some(s_obj) = workflow.get("spec") {
            let raw_str = serde_json::to_string(s_obj).map_err(|e| e.to_string())?;
            load_spec_str(&raw_str, base_dir)?
        } else {
            return Err("workflow must provide 'spec' (file path or inline spec object)".into());
        };

    let mut active_spec = loaded_initial.spec;
    let active_expansion = loaded_initial.expansion;
    let mut stage_reports = Vec::new();
    let mut events = Vec::new();
    let mut proofs = Vec::new();
    let mut last_run_digest: Option<SemanticDigest> = None;

    let sync_state = |events: &[Value], proofs: &[Value]| {
        if let Ok(raw_st) = std::fs::read_to_string(&ws.state_file)
            && let Ok(mut st) = serde_json::from_str::<Value>(&raw_st)
        {
            if let Some(ev_arr) = st
                .get_mut("tables")
                .and_then(|t| t.get_mut("events"))
                .and_then(|a| a.as_array_mut())
            {
                ev_arr.extend(events.iter().cloned());
            }
            if let Some(pr_arr) = st
                .get_mut("tables")
                .and_then(|t| t.get_mut("proofs"))
                .and_then(|a| a.as_array_mut())
            {
                pr_arr.extend(proofs.iter().cloned());
            }
            let _ = std::fs::write(
                &ws.state_file,
                format!("{}\n", serde_json::to_string_pretty(&st).unwrap()),
            );
        }
    };

    let pipeline = workflow
        .get("pipeline")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    for (idx, st) in pipeline.iter().enumerate() {
        let stage_name = st
            .get("stage")
            .or_else(|| st.get("op"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let step_res: Result<Value, String> = (|| match stage_name {
            "compile" | "compile_v2" => {
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
                let rep = validate_spec(&active_spec, true);
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
                let steps = st.get("steps").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
                let state_port = st.get("state_port").and_then(|v| v.as_str());
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
                let max_depth = st.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
                let abs_tol = st.get("abs_tol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
                let rel_tol = st.get("rel_tol").and_then(|v| v.as_f64()).unwrap_or(1e-5);
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
                let tol = st.get("tol").and_then(|v| v.as_f64()).unwrap_or(0.05);
                let scheme = QuantScheme::parse(
                    st.get("scheme")
                        .and_then(|v| v.as_str())
                        .unwrap_or("q8-per-block"),
                )?;
                let mode = BudgetMode::parse(
                    st.get("mode").and_then(|v| v.as_str()).unwrap_or("bound"),
                )?;
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
                let gens = st.get("generations").and_then(|v| v.as_u64()).unwrap_or(25) as usize;
                let seed = st.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
                let sens = st
                    .get("sensitivity_guided")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                let fr = fit_es(&active_spec, &tf, gens, 12, 0.2, 0.2, seed, sens, 1e-6)?;
                active_spec = fr.optimized_spec;
                Ok(json!({
                    "stage": stage_name,
                    "ok": fr.ok,
                    "start_loss": fr.start_loss,
                    "final_loss": fr.final_loss,
                }))
            }
            "solve_gd" | "solve-gd" => {
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
                let steps = st.get("steps").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let lr = st.get("lr").and_then(|v| v.as_f64()).unwrap_or(0.1);
                let pre = st
                    .get("precondition")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let gd = solve_gd_autodiff(&active_spec, Some(&tf), None, steps, lr, pre)?;
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
                let steps = st.get("steps").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
                let state_port = st.get("state_port").and_then(|v| v.as_str());
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
                let tol = st.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-6);
                let max_steps =
                    st.get("max_steps").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
                let state_port = st.get("state_port").and_then(|v| v.as_str());
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
                stage_reports.push(json!({
                    "stage": stage_name,
                    "ok": false,
                    "error": err_msg,
                }));
                sync_state(&events, &proofs);
                return Ok(json!({
                    "ok": false,
                    "error": err_msg,
                    "failed_stage": stage_name,
                    "workspace": ws.root.display().to_string(),
                    "stages_completed": stage_reports.len(),
                    "stage_reports": stage_reports,
                    "final_output_digest": last_run_digest,
                }));
            }
        }
    }

    sync_state(&events, &proofs);
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
