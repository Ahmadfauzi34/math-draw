use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

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
        let initial_txt = serde_json::to_string_pretty(&initial)
            .map_err(|e| format!("failed serializing workspace state: {e}"))?;
        std::fs::write(&state_file, format!("{initial_txt}\n"))
            .map_err(|e| format!("failed writing workspace state '{}': {e}", state_file.display()))?;
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
    let raw_st = std::fs::read_to_string(&ws.state_file)
        .map_err(|e| format!("failed reading workspace state '{}': {e}", ws.state_file.display()))?;
    let mut st: Value = serde_json::from_str(&raw_st)
        .map_err(|e| format!("corrupted workspace state '{}': {e}", ws.state_file.display()))?;
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
    let state_txt = serde_json::to_string_pretty(&st)
        .map_err(|e| format!("failed serializing workspace state: {e}"))?;
    std::fs::write(&ws.state_file, format!("{state_txt}\n"))
        .map_err(|e| format!("failed writing workspace state '{}': {e}", ws.state_file.display()))?;

    Ok((uri, format!("sha256:{digest}"), file))
}

pub fn resolve_artifact_uri(uri: &str, workspace_dir: &Path) -> Result<Value, String> {
    if !uri.starts_with(ARTIFACT_URI_PREFIX) {
        return Err(format!(
            "invalid artifact URI '{uri}' (expected {ARTIFACT_URI_PREFIX}<64-hex>)"
        ));
    }
    let expected_hex = &uri[ARTIFACT_URI_PREFIX.len()..];
    if expected_hex.len() != 64
        || !expected_hex
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
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
            "summary": "Build, expand, auto-repair, compact, and lower recursive v2 specs, macros (incl. attention_head), sharded layers, 3D geometry pipelines/scene trees/IK, and weight expression trees.",
            "tools": [
                { "cmd": "math-draw recurse expand [--auto-repair] [--compact-slots]", "role": "Compile burn-math.recursive-spec.v2 (subgraphs, repeat, residual, parallel_concat, $ref, 6 built-in macros incl. attention_head) to flat v1 with optional auto-repair and linear-scan liveness slot compaction" },
                { "cmd": "math-draw shard", "role": "Split oversized linear layers along output rows with concat(dim=1) and exact reassembly verification" },
                { "cmd": "math-draw geom <build|scene-tree|solve-ik|project|batch|selftest>", "role": "3D geometry domain pack: 4x4 transform chains (incl. repeat), hierarchical scene trees, 3D IK solver, single & batched (BATCH_MAX=63) Burn projection with near-plane clipping" },
                { "cmd": "math-draw weight-lab", "role": "Manual-weighting workbench (run, gate, certify, compose, eval, selftest)" }
            ]
        },
        "verify": {
            "summary": "Validate plans with liveness analysis, differential-test Burn against pure-Rust f64 oracle, prove numerical/algebraic equivalence, soak-test determinism, certify PRNG streams, zoom-fuzz worst cases, localize step divergences, and verify portable program bundles.",
            "tools": [
                { "cmd": "math-draw validate --explain", "role": "95-constructor static validation, 4D shape inference, dtype dataflow trace, and peak-byte slot liveness analysis" },
                { "cmd": "math-draw diff-ref [--cases <cases.json>] [--ref <ref.json>] [--tol 1e-5]", "role": "Differential test Burn Device::flex() f32 outputs against an independent pure-Rust f64 interpreter (all 87 f32 constructors) or external reference vectors" },
                { "cmd": "math-draw verify [--fuzz] [--zoom]", "role": "Equivalence checker with domain-aware fuzzing and recursive adversarial interval narrowing (--zoom)" },
                { "cmd": "math-draw linear-equiv", "role": "Exact symbolic affine equivalence prover over linear, add, sub, and concat(dim=1) graphs" },
                { "cmd": "math-draw soak [--runs 100]", "role": "Long-run bit-identical determinism proof hashing raw canonical f32 output bytes" },
                { "cmd": "math-draw prng-check [algo] [seed] [count] | verify <manifest.json>", "role": "Certify and re-verify deterministic PRNG streams (xorshift64, mulberry32, pcg32, philox4x32)" },
                { "cmd": "math-draw trace-hash [--isolate]", "role": "Per-step SHA-256 tensor fingerprinting, divergence step localization, and prefix subspec isolation" },
                { "cmd": "math-draw bundle <binding|export|verify>", "role": "Canonical unique-first-use parameter binding, self-contained program bundle export, cryptographic tamper detection, and live replay verification" },
                { "cmd": "math-draw smoke", "role": "End-to-end deterministic smoke test suite" }
            ]
        },
        "differentiate": {
            "summary": "Compile symbolic in-graph 1st-order VJP and 2nd-order HVP specs, compute central finite-difference Jacobians, verify analytic derivatives, and cross-check against Burn autodiff.",
            "tools": [
                { "cmd": "math-draw recurse grad-spec [--wrt <slot|name>] [--order 1|2] [--compact-slots]", "role": "Compile a runnable 1st-order VJP or 2nd-order Hessian-Vector Product SpecV1 graph across 23 differentiable constructors" },
                { "cmd": "math-draw gradcheck [--weights] [--autodiff]", "role": "Central finite-difference input & weight Jacobians with Burn Device::flex().autodiff() verification" }
            ]
        },
        "optimize": {
            "summary": "Search rewrite+quantization trees, apply fixpoint rewrites, mutate graphs across competing branches, plan mixed-precision quantization, and optimize weights via GD or ES.",
            "tools": [
                { "cmd": "math-draw rewrite <list|apply>", "role": "Pure graph rewrites and recursive fixpoint simplification" },
                { "cmd": "math-draw recurse search [--quant-tol T]", "role": "Equivalence-verified tree rewrite and quantization branch search" },
                { "cmd": "math-draw mutate", "role": "Transactional & multi-branch equivalence-gated graph mutations" },
                { "cmd": "math-draw quant <verify|budget>", "role": "int8/q8-per-block/fp16 weight quantization verifier and analytical Lipschitz/measured budget planner" },
                { "cmd": "math-draw recurse solve-gd [--precondition]", "role": "Burn autodiff gradient descent with backtracking and diagonal Hessian preconditioning" },
                { "cmd": "math-draw recurse fit [--strategy openes|mu-lambda] [--rng xorshift64|mulberry32|pcg32|philox4x32] [--sensitivity-guided]", "role": "Evolutionary Strategy optimizer supporting OpenES & (mu, lambda), 4 deterministic PRNGs, population diagnostics, and sensitivity-guided freezing" },
                { "cmd": "math-draw profile / compare / cost-model", "role": "Whole-graph latency benchmarking and isolated per-op microbenchmark cost modeling" }
            ]
        },
        "rollout": {
            "summary": "Execute graphs one-shot, run spec-free direct math (arith/int/kernel), step recurrently across time (scan), solve fixed points, unroll temporal horizons, or bisect trajectory divergence.",
            "tools": [
                { "cmd": "math-draw run", "role": "One-shot deterministic execution on Burn flex backend" },
                { "cmd": "math-draw arith / int / kernel [--batch <ops.json>]", "role": "Spec-free deterministic f32 arithmetic/linalg/stats, exact checked i64 arithmetic, and Lane-E elementwise kernels with $prev/$<idx> batch register chaining" },
                { "cmd": "math-draw rollout <scan|fixpoint|unroll|bisect> [--compact-slots]", "role": "Stateful recurrent rollout, fixed-point solver, weight-tied unroller (with optional liveness compaction), and tick/step bisection" }
            ]
        },
        "state": {
            "summary": "Manage persistent .burn-harness workspaces, content-addressed tensor artifacts, portable program bundles, multi-stage workflows, and subagent branch coordination.",
            "tools": [
                { "cmd": "math-draw harness <capabilities|digest|resolve-artifact|run-workflow|branch-export|branch-merge|workspace-status>", "role": "Unified agent-harness co-processor interface and pipeline runner" },
                { "cmd": "math-draw bundle <binding|export|verify>", "role": "Self-contained portable program bundle export and provenance verification" },
                { "cmd": "math-draw find-op", "role": "Query the 95-constructor catalog and parameter schemas" }
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
