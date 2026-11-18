use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::engine::{PreparedGraph, run_spec};
use crate::quant::{BudgetMode, QuantScheme, quant_budget};
use crate::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use crate::rollout::{rollout_fixpoint, rollout_scan, rollout_unroll};
use crate::spec::{LoadedSpec, SpecV1, load_spec_file, load_spec_str, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{EvalCase, TargetsFile, verify_specs};

pub const HARNESS_SCHEMA: &str = "burn-math.harness-session.v1";
pub const ARTIFACT_URI_PREFIX: &str = "burn://sha256:";

pub struct WorkspacePaths {
    pub root: PathBuf,
    pub artifacts_dir: PathBuf,
    pub checkpoints_dir: PathBuf,
    pub branches_dir: PathBuf,
    pub state_file: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

pub fn ensure_workspace_dir(workspace_dir: &Path) -> Result<WorkspacePaths, String> {
    let root = if workspace_dir.is_absolute() {
        workspace_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(workspace_dir)
    };
    let artifacts_dir = root.join("artifacts");
    let checkpoints_dir = root.join("checkpoints");
    let branches_dir = root.join("branches");
    std::fs::create_dir_all(&artifacts_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&checkpoints_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&branches_dir).map_err(|e| e.to_string())?;

    let state_file = root.join("workspace-state.json");
    if !state_file.exists() {
        let initial = json!({
            "schema": HARNESS_SCHEMA,
            "tables": {
                "search_frontier": {},
                "artifacts": {},
                "proofs": [],
                "events": []
            }
        });
        std::fs::write(
            &state_file,
            format!("{}\n", serde_json::to_string_pretty(&initial).unwrap()),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(WorkspacePaths {
        root,
        artifacts_dir,
        checkpoints_dir,
        branches_dir,
        state_file,
    })
}

pub fn store_artifact(
    values: &[f32],
    shape: [usize; 4],
    workspace_dir: &Path,
    kind: &str,
) -> Result<(String, String, PathBuf), String> {
    let ws = ensure_workspace_dir(workspace_dir)?;
    let vals_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let payload_val = json!({
        "kind": kind,
        "shape": shape,
        "values": vals_f64,
    });
    let payload = serde_json::to_string(&payload_val).map_err(|e| e.to_string())?;
    let digest = sha256_hex(payload.as_bytes());
    let file = ws.artifacts_dir.join(format!("{digest}.json"));
    let expected_content = format!("{payload}\n");
    let need_write = match std::fs::read_to_string(&file) {
        Ok(existing) => existing != expected_content,
        Err(_) => true,
    };
    if need_write {
        std::fs::write(&file, &expected_content).map_err(|e| e.to_string())?;
    }

    let uri = format!("{ARTIFACT_URI_PREFIX}{digest}");
    if let Ok(raw_st) = std::fs::read_to_string(&ws.state_file)
        && let Ok(mut st) = serde_json::from_str::<Value>(&raw_st)
    {
        if let Some(arts) = st
            .get_mut("tables")
            .and_then(|t| t.get_mut("artifacts"))
            .and_then(|a| a.as_object_mut())
        {
            arts.insert(
                digest.clone(),
                json!({
                    "uri": uri,
                    "kind": kind,
                    "shape": shape,
                    "numel": values.len(),
                    "bytes": payload.len(),
                }),
            );
        }
        let _ = std::fs::write(
            &ws.state_file,
            format!("{}\n", serde_json::to_string_pretty(&st).unwrap()),
        );
    }

    Ok((uri, format!("sha256:{digest}"), file))
}

pub fn resolve_artifact_uri(uri: &str, workspace_dir: &Path) -> Result<Value, String> {
    if !uri.starts_with(ARTIFACT_URI_PREFIX) {
        return Err(format!(
            "invalid artifact URI '{uri}' (expected {ARTIFACT_URI_PREFIX}<64-hex>)"
        ));
    }
    let expected_hex = &uri[ARTIFACT_URI_PREFIX.len()..];
    if expected_hex.len() != 64 || !expected_hex.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
        return Err(format!("malformed sha256 in artifact URI: {uri}"));
    }
    let ws = ensure_workspace_dir(workspace_dir)?;
    let file = ws.artifacts_dir.join(format!("{expected_hex}.json"));
    if !file.exists() {
        return Err(format!("artifact not found in workspace: {uri}"));
    }
    let raw = std::fs::read_to_string(&file).map_err(|e| e.to_string())?;
    let trimmed = raw.trim();
    let actual_hex = sha256_hex(trimmed.as_bytes());
    if actual_hex != expected_hex {
        return Err(format!(
            "artifact_integrity_failed for {uri}: expected {expected_hex}, got {actual_hex}"
        ));
    }
    serde_json::from_str(trimmed).map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticStats {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub l2_norm: f64,
    pub l_inf_norm: Option<f64>,
    pub zero_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SemanticDigest {
    pub shape: [usize; 4],
    pub numel: usize,
    pub value_sha256: String,
    pub stats: SemanticStats,
    pub preview_head: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub values: Option<Vec<f32>>,
}

pub fn compute_semantic_digest(
    values: &[f32],
    shape: [usize; 4],
    workspace_dir: Option<&Path>,
    persist_threshold: usize,
    include_full_values: bool,
) -> Result<SemanticDigest, String> {
    let mut hasher = Sha256::new();
    for &v in values {
        hasher.update(v.to_le_bytes());
    }
    let value_sha256 = format!("sha256:{:x}", hasher.finalize());

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut zero_count = 0usize;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for &v in values {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if !v.is_finite() {
            inf_count += 1;
            continue;
        }
        let vf = v as f64;
        if vf == 0.0 {
            zero_count += 1;
        }
        min = min.min(vf);
        max = max.max(vf);
        sum += vf;
        sum_sq += vf * vf;
    }

    let finite_count = values.len() - nan_count - inf_count;
    let mean = if finite_count > 0 {
        Some(sum / (finite_count as f64))
    } else {
        None
    };
    let std = mean.map(|m| (sum_sq / (finite_count as f64) - m * m).max(0.0).sqrt());
    let l2_norm = sum_sq.sqrt();
    let l_inf_norm = if finite_count > 0 {
        Some(min.abs().max(max.abs()))
    } else {
        None
    };

    let artifact_ref = if let Some(ws) = workspace_dir
        && values.len() >= persist_threshold
    {
        let (uri, _, _) = store_artifact(values, shape, ws, "tensor_f32")?;
        Some(uri)
    } else {
        None
    };

    Ok(SemanticDigest {
        shape,
        numel: values.len(),
        value_sha256,
        stats: SemanticStats {
            min: if finite_count > 0 { Some(min) } else { None },
            max: if finite_count > 0 { Some(max) } else { None },
            mean,
            std,
            l2_norm,
            l_inf_norm,
            zero_count,
            nan_count,
            inf_count,
        },
        preview_head: values.iter().take(8).copied().collect(),
        artifact_ref,
        values: if include_full_values || values.len() <= persist_threshold {
            Some(values.to_vec())
        } else {
            None
        },
    })
}

pub fn get_capability_catalog(domain: Option<&str>) -> Value {
    let domains = json!({
        "compose": {
            "summary": "Build, expand, and lower recursive v2 specs, macros, sharded layers, and weight expression trees.",
            "tools": [
                { "cmd": "math-draw recurse expand", "role": "Compile burn-math.recursive-spec.v2 (subgraphs, repeat, residual, parallel_concat, $ref, macros) to flat v1" },
                { "cmd": "math-draw shard", "role": "Split oversized linear layers along output rows with concat(dim=1) and exact reassembly verification" },
                { "cmd": "math-draw weight-lab", "role": "Manual-weighting workbench (run, gate, certify, compose, eval, selftest)" }
            ]
        },
        "verify": {
            "summary": "Validate plans, prove numerical and exact algebraic equivalence, zoom-fuzz worst cases, and localize step divergences.",
            "tools": [
                { "cmd": "math-draw validate --explain", "role": "52-constructor static validation, 4D shape inference, and dtype dataflow trace" },
                { "cmd": "math-draw verify [--fuzz] [--zoom]", "role": "Equivalence checker with domain-aware fuzzing and recursive adversarial interval narrowing (--zoom)" },
                { "cmd": "math-draw linear-equiv", "role": "Exact symbolic affine equivalence prover over linear, add, sub, and concat(dim=1) graphs" },
                { "cmd": "math-draw trace-hash [--isolate]", "role": "Per-step SHA-256 tensor fingerprinting, divergence step localization, and prefix subspec isolation" },
                { "cmd": "math-draw smoke", "role": "End-to-end deterministic smoke test suite" }
            ]
        },
        "differentiate": {
            "summary": "Compute central finite-difference Jacobians, verify analytic derivatives, and cross-check against Burn autodiff.",
            "tools": [
                { "cmd": "math-draw gradcheck [--weights] [--autodiff]", "role": "Central finite-difference input & weight Jacobians with Burn Device::flex().autodiff() verification" }
            ]
        },
        "optimize": {
            "summary": "Search rewrite trees, apply fixpoint rewrites, mutate graphs across competing branches, plan mixed-precision quantization, and optimize weights via GD or ES.",
            "tools": [
                { "cmd": "math-draw rewrite <list|apply>", "role": "Pure graph rewrites and recursive fixpoint simplification" },
                { "cmd": "math-draw recurse search", "role": "Equivalence-verified tree rewrite search" },
                { "cmd": "math-draw mutate", "role": "Transactional & multi-branch equivalence-gated graph mutations" },
                { "cmd": "math-draw quant <verify|budget>", "role": "int8/q8-per-block/fp16 weight quantization verifier and analytical Lipschitz/measured budget planner" },
                { "cmd": "math-draw recurse solve-gd [--precondition]", "role": "Burn autodiff gradient descent with backtracking and diagonal Hessian preconditioning" },
                { "cmd": "math-draw recurse fit [--sensitivity-guided]", "role": "Evolutionary Strategy parameter optimizer with sensitivity-guided layer freezing" },
                { "cmd": "math-draw profile / compare / cost-model", "role": "Whole-graph latency benchmarking and isolated per-op microbenchmark cost modeling" }
            ]
        },
        "rollout": {
            "summary": "Execute graphs one-shot, recurrently across time (scan), iteratively to convergence (fixpoint), unroll temporal horizons, or bisect trajectory divergence.",
            "tools": [
                { "cmd": "math-draw run", "role": "One-shot deterministic execution on Burn flex backend" },
                { "cmd": "math-draw rollout <scan|fixpoint|unroll|bisect>", "role": "Stateful recurrent rollout, fixed-point solver, weight-tied unroller, and tick/step bisection" }
            ]
        },
        "state": {
            "summary": "Manage persistent .burn-harness workspaces, content-addressed tensor artifacts, multi-stage workflows, and subagent branch coordination.",
            "tools": [
                { "cmd": "math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status>", "role": "Unified agent-harness co-processor interface and pipeline runner" },
                { "cmd": "math-draw find-op", "role": "Query the 52-constructor catalog and parameter schemas" }
            ]
        }
    });

    if let Some(d) = domain {
        if let Some(entry) = domains.get(d) {
            let count = entry
                .get("tools")
                .and_then(|t| t.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            json!({
                "ok": true,
                "domain": d,
                "summary": entry["summary"],
                "tool_count": count,
                "tools": entry["tools"],
            })
        } else {
            json!({
                "ok": false,
                "error": format!("unknown capability domain '{d}' (known: compose, verify, differentiate, optimize, rollout, state)")
            })
        }
    } else {
        json!({
            "ok": true,
            "domains": domains,
        })
    }
}

pub fn export_subagent_branches(spec: &SpecV1, workspace_dir: &Path) -> Result<Value, String> {
    let ws = ensure_workspace_dir(workspace_dir)?;
    let baseline_sha = sha256_json(spec);
    let baseline_path = ws.branches_dir.join("baseline.spec.json");
    std::fs::write(
        &baseline_path,
        format!("{}\n", serde_json::to_string_pretty(spec).unwrap()),
    )
    .map_err(|e| e.to_string())?;

    let default_branches = [
        (
            "sub-rewrite",
            "structural_rewrite",
            "Run math-draw recurse search on baseline.spec.json and save to sub-rewrite.candidate.spec.json",
        ),
        (
            "sub-quant",
            "quantization",
            "Run math-draw quant budget on baseline.spec.json and save to sub-quant.candidate.spec.json",
        ),
    ];

    let mut packets = Vec::new();
    for (bid, goal, instr) in default_branches {
        let cand_path = ws.branches_dir.join(format!("{bid}.candidate.spec.json"));
        let pkt = json!({
            "branch_id": bid,
            "goal": goal,
            "baseline_state_sha256": baseline_sha,
            "baseline_spec_path": baseline_path.display().to_string(),
            "candidate_output_path": cand_path.display().to_string(),
            "instructions": instr,
        });
        let pkt_path = ws.branches_dir.join(format!("{bid}.packet.json"));
        std::fs::write(
            &pkt_path,
            format!("{}\n", serde_json::to_string_pretty(&pkt).unwrap()),
        )
        .map_err(|e| e.to_string())?;
        packets.push(pkt);
    }

    Ok(json!({
        "ok": true,
        "baseline_state_sha256": baseline_sha,
        "baseline_spec_path": baseline_path.display().to_string(),
        "branch_packets": packets,
    }))
}

pub fn merge_subagent_branches(
    baseline: &SpecV1,
    candidate_files: &[String],
    tol: f64,
) -> Result<Value, String> {
    let base_graph = PreparedGraph::compile(baseline)?;
    let baseline_sha = sha256_json(baseline);

    let derive_inputs = |scale: f64, const_val: Option<f64>| -> BTreeMap<String, Vec<Value>> {
        let mut out = BTreeMap::new();
        for (k, arr) in &baseline.inputs {
            let mapped = arr
                .iter()
                .map(|v| {
                    if let Some(f) = v.as_f64() {
                        Value::from(const_val.unwrap_or(f * scale))
                    } else {
                        v.clone()
                    }
                })
                .collect();
            out.insert(k.clone(), mapped);
        }
        out
    };

    let named_candidates = [
        ("baseline", baseline.inputs.clone()),
        ("zeros", derive_inputs(0.0, Some(0.0))),
        ("ones", derive_inputs(0.0, Some(1.0))),
        ("negated", derive_inputs(-1.0, None)),
        ("scaled_x2", derive_inputs(2.0, None)),
        ("scaled_half", derive_inputs(0.5, None)),
    ];

    let mut cases = Vec::new();
    let mut case_names = Vec::new();
    for (name, inp) in named_candidates {
        if let Ok(out) = base_graph.run_once(baseline, Some(&inp))
            && out.values.iter().all(|v| v.is_finite())
        {
            cases.push(EvalCase {
                inputs: inp,
                expected: None,
            });
            case_names.push(name);
        }
    }

    let mut evaluations = Vec::new();
    let mut best_winner: Option<(usize, f64, String, String, SpecV1)> = None;

    for (i, file) in candidate_files.iter().enumerate() {
        let branch_id = format!("merge.{}", i + 1);
        let cand = match load_spec_file(Path::new(file)) {
            Ok(l) => l.spec,
            Err(e) => {
                evaluations.push(json!({
                    "file": file,
                    "branch_id": branch_id,
                    "accepted": false,
                    "reason": format!("unreadable spec: {e}"),
                }));
                continue;
            }
        };
        let ver = verify_specs(baseline, &cand, &cases, tol, tol);
        let cand_sha = sha256_json(&cand);
        let receipt = sha256_json(&(&baseline_sha, &cand_sha, ver.equivalent, ver.max_abs_diff));
        let steps_len = cand.steps.len();
        evaluations.push(json!({
            "file": file,
            "branch_id": branch_id,
            "accepted": ver.equivalent,
            "equivalent": ver.equivalent,
            "max_abs_error": ver.max_abs_diff,
            "receipt_digest": receipt,
            "steps_count": steps_len,
        }));
        if ver.equivalent {
            let take = match &best_winner {
                None => true,
                Some((best_steps, best_err, ..)) => {
                    steps_len < *best_steps
                        || (steps_len == *best_steps && ver.max_abs_diff < *best_err)
                }
            };
            if take {
                best_winner = Some((
                    steps_len,
                    ver.max_abs_diff,
                    file.clone(),
                    branch_id,
                    cand,
                ));
            }
        }
    }

    let ok = best_winner.is_some();
    let (promoted_winner, promoted_spec) = if let Some((steps_count, max_err, file, bid, sp)) = best_winner {
        let csha = sha256_json(&sp);
        let receipt = sha256_json(&(&baseline_sha, &csha, true, max_err));
        (
            Some(json!({
                "file": file,
                "branch_id": bid,
                "steps_count": steps_count,
                "max_abs_error": max_err,
                "receipt_digest": receipt,
            })),
            Some(sp),
        )
    } else {
        (None, None)
    };

    Ok(json!({
        "ok": ok,
        "baseline_state_sha256": baseline_sha,
        "equivalence_cases": case_names,
        "cases_count": cases.len(),
        "candidates_evaluated": evaluations.len(),
        "evaluations": evaluations,
        "promoted_winner": promoted_winner,
        "promoted_spec": promoted_spec,
    }))
}

pub fn execute_harness_workflow(
    workflow: &Value,
    base_dir: &Path,
    workspace_override: Option<&Path>,
) -> Result<Value, String> {
    let ws_rel = workspace_override
        .map(PathBuf::from)
        .or_else(|| workflow.get("workspace").and_then(|v| v.as_str()).map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(".burn-harness"));
    let ws_path = if ws_rel.is_absolute() {
        ws_rel
    } else {
        base_dir.join(ws_rel)
    };
    let ws = ensure_workspace_dir(&ws_path)?;

    let loaded_initial: LoadedSpec = if let Some(s_path) = workflow.get("spec").and_then(|v| v.as_str()) {
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
                    let raw = std::fs::read_to_string(base_dir.join(tp)).map_err(|e| e.to_string())?;
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
                let t_val = st
                    .get("targets")
                    .ok_or_else(|| "workflow solve_gd stage requires explicit 'targets'".to_string())?;
                let tf: TargetsFile = if let Some(tp) = t_val.as_str() {
                    let raw = std::fs::read_to_string(base_dir.join(tp)).map_err(|e| e.to_string())?;
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
                let sc = rollout_scan(&active_spec, active_expansion.as_ref(), steps, state_port, None)?;
                let dig = compute_semantic_digest(&sc.final_state.values, sc.state_shape, Some(&ws.root), 1, false)?;
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
                let max_steps = st.get("max_steps").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
                let state_port = st.get("state_port").and_then(|v| v.as_str());
                let fp = rollout_fixpoint(&active_spec, active_expansion.as_ref(), tol, max_steps, state_port)?;
                if !fp.converged {
                    return Err(format!(
                        "rollout_fixpoint did not converge within {max_steps} steps (final_delta={})",
                        fp.final_delta_linf
                    ));
                }
                let dig = compute_semantic_digest(&fp.final_state.values, fp.state_shape, Some(&ws.root), 1, false)?;
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
                let dig = compute_semantic_digest(&out.outputs.values, out.outputs.shape, Some(&ws.root), 1, false)?;
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
