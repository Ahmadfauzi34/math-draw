use serde_json::json;
use std::path::Path;

use super::{SmokeCheck, run_check};
use crate::cost_model::run_cost_model;
use crate::gradcheck::gradcheck_spec;
use crate::harness::{compute_semantic_digest, execute_harness_workflow, get_capability_catalog};
use crate::linear_equiv::equiv_linear;
use crate::mutate::{MutationRequest, mutate_spec};
use crate::quant::{BudgetMode, QuantScheme, quant_budget, quant_verify};
use crate::shard::shard_spec;
use crate::spec::load_spec_str;
use crate::verify::{FuzzConfig, zoom_fuzz_specs};
use crate::weight_lab::run_weight_lab_selftest;

pub(crate) fn collect_advanced_checks() -> Vec<SmokeCheck> {
    let mut checks = Vec::new();

    // 16. gradcheck numeric Jacobian + Burn autodiff cross-check
    checks.push(run_check("gradcheck_jacobian_and_autodiff", || {
        let raw = r#"{"inputs":{"0":[1.0,2.0]},"layers":[{"args":[0,2,2,true],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}],"weights":{"0":[1.0,2.0,3.0,4.0,0.5,-0.5]}}"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let analytic = json!({
            "0": [[1.0, 3.0], [2.0, 4.0]]
        });
        let rep = gradcheck_spec(&spec, Some(&analytic), 1e-3, 1e-2, true, true)?;
        if !rep.ok || !rep.passed || !rep.autodiff_check.as_ref().map(|a| a.ok).unwrap_or(false) {
            return Err(format!("gradcheck failed: max_abs={}", rep.max_abs_error));
        }
        Ok(format!(
            "max_abs={:.2e} autodiff_weights={}",
            rep.max_abs_error,
            rep.autodiff_check.unwrap().checked_weights
        ))
    }));

    // 17. quant verify (int8 + fp16) & quant budget (bound + measured)
    checks.push(run_check("quant_verify_and_budget", || {
        let raw = r#"{
          "numSlots": 4,
          "layers": [
            { "constructor": "linear", "args": [0, 4, 4, true] },
            { "constructor": "relu", "args": [1] },
            { "constructor": "linear", "args": [2, 4, 2, true] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] },
            { "kind": "unary", "layer": 2, "slots": [2, 3] }
          ],
          "outputSlot": 3,
          "ports": [{ "slot": 0, "shape": [1, 4, 1, 1] }],
          "weights": {
            "0": [0.5,-0.25,0.1,0.3, -0.4,0.6,0.2,-0.1, 0.2,0.1,-0.5,0.4, 0.3,-0.2,0.1,0.5, 0.1,0.0,-0.1,0.2],
            "2": [0.4,-0.2, 0.3,0.5, -0.1,0.2, 0.5,-0.3, 0.1,-0.1]
          },
          "inputs": { "0": [1.0, -0.5, 0.25, 2.0] }
        }"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let qv = quant_verify(&spec, QuantScheme::Fp16, None, 1e-2, 1e-2)?;
        if !qv.ok {
            return Err(format!("fp16 quant_verify failed: diff={}", qv.verification.max_abs_diff));
        }
        let qb_bound = quant_budget(&spec, None, 0.5, QuantScheme::Q8PerBlock, BudgetMode::Bound)?;
        let qb_meas = quant_budget(&spec, None, 0.5, QuantScheme::Int8, BudgetMode::Measured)?;
        if !qb_bound.ok || !qb_meas.ok || qb_meas.quantized_layers.is_empty() {
            return Err("quant_budget failed to find valid plan".into());
        }
        Ok(format!(
            "fp16_diff={:.5} saved_measured={}B",
            qv.verification.max_abs_diff, qb_meas.total_saved_bytes
        ))
    }));

    // 18. mutate multi-branch equivalence gate
    checks.push(run_check("mutate_branch_equivalence_gate", || {
        let raw = r#"{
          "numSlots": 4,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "relu", "args": [1] },
            { "constructor": "relu", "args": [2] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] },
            { "kind": "unary", "layer": 2, "slots": [2, 3] }
          ],
          "outputSlot": 3,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [1, 0, 0, 1] },
          "inputs": { "0": [1.5, -2.0] }
        }"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let req: MutationRequest = serde_json::from_value(json!({
            "commit": true,
            "branches": [
                { "branch_id": "bad", "edits": [{ "op": "setWeights", "layer_index": 0, "values": [9.0, 0.0, 0.0, 1.0] }] },
                { "branch_id": "good", "edits": [{ "op": "removeStep", "index": 2 }, { "op": "setOutput", "slot": 2 }] }
            ]
        }))
        .map_err(|e| e.to_string())?;
        let rep = mutate_spec(&spec, &req, None, 1e-6, 1e-6)?;
        if !rep.ok || !rep.committed || rep.winning_branch.as_deref() != Some("good") {
            return Err(format!("expected winning_branch=good, got {:?}", rep.winning_branch));
        }
        Ok(format!("winner=good diff_bad={:.2}", rep.branches[0].max_abs_error))
    }));

    // 19. shard linear layer output-row splitting + exact reassembly
    checks.push(run_check("shard_linear_exact_reassembly", || {
        let raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "linear", "args": [0, 3, 5, true] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 3, 1, 1] }],
          "weights": { "0": [1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15, 0.1,0.2,0.3,0.4,0.5] },
          "inputs": { "0": [1.0, -1.0, 0.5] }
        }"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let rep = shard_spec(&spec, 8)?;
        if !rep.ok
            || !rep.sharded
            || !rep.reassembly_exact
            || !rep.verified
            || rep.shards[0].num_shards != 3
        {
            return Err(format!(
                "shard failed: exact={} verified={} shards={}",
                rep.reassembly_exact,
                rep.verified,
                rep.shards.len()
            ));
        }
        Ok(format!(
            "num_shards={} max_abs={:.2e}",
            rep.shards[0].num_shards, rep.max_abs_error
        ))
    }));

    // 20. weight-lab selftest
    checks.push(run_check("weight_lab_workbench_and_selftest", || {
        let res = run_weight_lab_selftest()?;
        let ok = res.get("ok").and_then(|v| v.as_bool()).unwrap_or(false);
        let passed = res.get("passed").and_then(|v| v.as_u64()).unwrap_or(0);
        let total = res.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
        if !ok || passed != total {
            return Err(format!("weight_lab selftest failed: {passed}/{total}"));
        }
        Ok(format!("selftests={passed}/{total}"))
    }));

    // 21. linear-equiv exact symbolic affine equivalence
    checks.push(run_check("linear_equiv_symbolic_affine", || {
        let raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "linear", "args": [0, 3, 4, true] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 3, 1, 1] }],
          "weights": { "0": [1,2,3,4, 5,6,7,8, 9,10,11,12, 0.5,-0.5,1.0,-1.0] },
          "inputs": { "0": [1.0, 2.0, 3.0] }
        }"#;
        let base = load_spec_str(raw, Path::new("."))?.spec;
        let sharded = shard_spec(&base, 8)?.spec;
        let eq = equiv_linear(&base, &sharded, 0.0);
        if !eq.ok || eq.verdict != "ACCEPT" {
            return Err(format!("linear_equiv rejected sharded spec: {}", eq.reason));
        }
        Ok(format!(
            "verdict={} fp={}",
            eq.verdict,
            eq.effective_fingerprint.unwrap_or_default()
        ))
    }));

    // 22. cost-model per-op microbenchmark & zoom-fuzz adversarial interval narrowing
    checks.push(run_check("cost_model_and_zoom_fuzz", || {
        let raw = r#"{"inputs":{"0":[1.0,-2.0]},"layers":[{"args":[0,2,2,false],"constructor":"linear"},{"args":[1],"constructor":"relu"}],"numSlots":3,"outputSlot":2,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]},{"kind":"unary","layer":1,"slots":[1,2]}],"weights":{"0":[1.0,0.0,0.0,1.0]}}"#;
        let base = load_spec_str(raw, Path::new("."))?.spec;
        let cm = run_cost_model(&base, 4)?;
        if !cm.ok || cm.per_op.len() != 2 {
            return Err("cost_model failed".into());
        }
        let mut cand = base.clone();
        cand.weights.insert("0".into(), vec![1.2, 0.0, 0.0, 1.0]);
        let zf = zoom_fuzz_specs(
            &base,
            &cand,
            &FuzzConfig {
                count: 6,
                seed: 42,
                lo: -4.0,
                hi: 4.0,
            },
            3,
        )?;
        if !zf.ok || zf.zoomed_max_abs_error < zf.initial_max_abs_error {
            return Err("zoom_fuzz failed to maintain or amplify worst-case error".into());
        }
        Ok(format!(
            "ops={} zoom_err={:.3}->{:.3}",
            cm.per_op.len(),
            zf.initial_max_abs_error,
            zf.zoomed_max_abs_error
        ))
    }));

    // 23. harness capabilities, semantic digest & multi-stage workflow
    checks.push(run_check("harness_digest_and_workflow", || {
        let cap = get_capability_catalog(Some("optimize"));
        if !cap.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err("capability catalog lookup failed".into());
        }
        let dig = compute_semantic_digest(&[1.0, -2.0, 0.0, 3.0], [1, 4, 1, 1], None, 16, false)?;
        if dig.stats.zero_count != 1 || dig.numel != 4 {
            return Err("semantic digest stats mismatch".into());
        }
        let tmp_ws = std::env::temp_dir().join(format!("math_draw_smoke_ws_{}", std::process::id()));
        let wf = json!({
            "spec": {
                "numSlots": 4,
                "layers": [
                    { "constructor": "linear", "args": [0, 2, 2, false] },
                    { "constructor": "relu", "args": [1] },
                    { "constructor": "relu", "args": [2] }
                ],
                "steps": [
                    { "kind": "unary", "layer": 0, "slots": [0, 1] },
                    { "kind": "unary", "layer": 1, "slots": [1, 2] },
                    { "kind": "unary", "layer": 2, "slots": [2, 3] }
                ],
                "outputSlot": 3,
                "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
                "weights": { "0": [1.0, 0.0, 0.0, 1.0] },
                "inputs": { "0": [2.0, -1.0] }
            },
            "pipeline": [
                { "stage": "validate" },
                { "stage": "search_rewrites", "max_depth": 2 },
                { "stage": "run" }
            ]
        });
        let wf_res = execute_harness_workflow(&wf, Path::new("."), Some(&tmp_ws));
        let _ = std::fs::remove_dir_all(&tmp_ws);
        let wf_val = wf_res?;
        if !wf_val.get("ok").and_then(|v| v.as_bool()).unwrap_or(false) {
            return Err(format!("harness workflow failed: {wf_val}"));
        }
        let final_steps = wf_val["active_spec_summary"]["steps_count"]
            .as_u64()
            .unwrap_or(99);
        Ok(format!("workflow_stages=3 optimized_steps={final_steps}"))
    }));

    // 24. Ported 30 pure-math ops end-to-end pipeline
    checks.push(run_check("ported_math_ops_end_to_end", || {
        let raw = r#"{
          "numSlots": 11,
          "layers": [
            { "constructor": "indicesLike", "args": [0, 1] },
            { "constructor": "abs", "args": [1] },
            { "constructor": "clamp", "args": [2, 0.25, 3.5] },
            { "constructor": "add", "args": [3] },
            { "constructor": "normalize", "args": [4] },
            { "constructor": "entropy", "args": [5] },
            { "constructor": "expandLike", "args": [6] },
            { "constructor": "sumAxis", "args": [7, 1] },
            { "constructor": "variancePopulation", "args": [8] },
            { "constructor": "lessEqual01", "args": [9] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [0, 2] },
            { "kind": "unary", "layer": 2, "slots": [2, 3] },
            { "kind": "binary", "layer": 3, "slots": [1, 3, 4] },
            { "kind": "unary", "layer": 4, "slots": [4, 5] },
            { "kind": "unary", "layer": 5, "slots": [5, 6] },
            { "kind": "binary", "layer": 6, "slots": [6, 4, 7] },
            { "kind": "unary", "layer": 7, "slots": [7, 8] },
            { "kind": "unary", "layer": 8, "slots": [4, 9] },
            { "kind": "binary", "layer": 9, "slots": [8, 9, 10] }
          ],
          "outputSlot": 10,
          "ports": [{ "slot": 0, "shape": [1, 3, 1, 1] }],
          "inputs": { "0": [-1.0, 2.0, -4.0] }
        }"#;
        let spec: crate::spec::SpecV1 =
            serde_json::from_str(raw).map_err(|e| format!("parse: {e}"))?;
        let res = crate::engine::run_spec(&spec)?;
        if res.outputs.shape != [1, 1, 1, 1] || res.outputs.values.len() != 1 {
            return Err(format!("unexpected shape {:?}", res.outputs.shape));
        }
        // indices=[0,1,2], abs=[-1,2,-4]->[1,2,4], clamp(0.25,3.5)->[1,2,3.5], sum=[1,3,5.5]
        // normalize -> entropy ~ 0.903 -> expandLike [1,3,1,1] -> sumAxis ~ 2.710 <= variance ~ 3.389 -> 1.0
        if res.outputs.values[0] != 1.0 {
            return Err(format!("expected 1.0, got {:?}", res.outputs.values));
        }
        Ok(format!("all_math_families predicate={}", res.outputs.values[0]))
    }));

    // 25. Parity 13 math ops + ternary fma end-to-end (compiled from v2)
    checks.push(run_check("parity_13_math_ops_and_ternary_fma", || {
        let raw = r#"{
          "schema": "burn-math.recursive-spec.v2",
          "ports": {
            "x0": { "slot": 0, "shape": [1, 2, 1, 1] },
            "x2": { "slot": 1, "shape": [1, 2, 1, 1] }
          },
          "inputs": {
            "x0": [0.0, 1.0],
            "x2": [2.5, 3.5]
          },
          "steps": [
            { "op": "sin", "in": ["x0"], "out": "s_sin" },
            { "op": "cos", "in": ["x0"], "out": "s_cos" },
            { "op": "tan", "in": ["x0"], "out": "s_tan" },
            { "op": "floor", "in": ["x2"], "out": "s_flr" },
            { "op": "ceil", "in": ["x2"], "out": "s_ceil" },
            { "op": "round", "in": ["x2"], "out": "s_rnd" },
            { "op": "atan2", "in": ["s_sin", "s_cos"], "out": "s_at2" },
            { "op": "hypot", "in": ["s_flr", "s_rnd"], "out": "s_hyp" },
            { "op": "powf", "in": ["s_flr", "s_ceil"], "out": "s_pow" },
            { "op": "fma", "in": ["s_at2", "s_hyp", "s_pow"], "out": "s_fma" },
            { "op": "varianceSample", "in": ["s_fma"], "out": "s_vs" },
            { "op": "stdSample", "in": ["s_fma"], "out": "s_ss" },
            { "op": "allFinite", "in": ["s_tan"], "out": "t_fin" },
            { "op": "fma", "in": ["s_vs", "t_fin", "s_ss"], "out": "final_out" }
          ],
          "output": "final_out"
        }"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        let ternary_count = loaded.spec.steps.iter().filter(|s| s.kind == "ternary").count();
        if ternary_count != 2 {
            return Err(format!("expected 2 ternary steps, got {ternary_count}"));
        }
        let res = crate::engine::run_spec(&loaded.spec)?;
        let expected = 3042.0f32 + 3042.0f32.sqrt();
        if res.outputs.shape != [1, 1, 1, 1]
            || res.outputs.values.len() != 1
            || (res.outputs.values[0] - expected).abs() > 1e-2
        {
            return Err(format!(
                "expected [{expected}], got {:?} (shape {:?})",
                res.outputs.values, res.outputs.shape
            ));
        }
        Ok(format!(
            "13_ops_ternary_fma out={:.4} ternary_steps={ternary_count}",
            res.outputs.values[0]
        ))
    }));

    checks
}
