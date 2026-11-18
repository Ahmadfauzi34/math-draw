use serde::Serialize;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

use crate::cost_model::run_cost_model;
use crate::engine::{eval_spec_with_inputs, run_spec};
use crate::gradcheck::gradcheck_spec;
use crate::harness::{compute_semantic_digest, execute_harness_workflow, get_capability_catalog};
use crate::linear_equiv::equiv_linear;
use crate::mutate::{MutationRequest, mutate_spec};
use crate::profile::{compare_specs_perf, profile_spec};
use crate::quant::{BudgetMode, QuantScheme, quant_budget, quant_verify};
use crate::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use crate::rollout::{rollout_bisect, rollout_fixpoint, rollout_scan, rollout_unroll};
use crate::shard::shard_spec;
use crate::spec::{load_spec_str, SpecV1};
use crate::trace::{DEFAULT_MAX_TENSOR_BYTES, compare_trace_hashes};
use crate::validate::validate_spec;
use crate::verify::{FuzzConfig, TargetsFile, generate_fuzz_cases, verify_specs, zoom_fuzz_specs};
use crate::weight_lab::run_weight_lab_selftest;

#[derive(Debug, Clone, Serialize)]
pub struct SmokeCheck {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmokeReport {
    pub ok: bool,
    pub passed: usize,
    pub total: usize,
    pub checks: Vec<SmokeCheck>,
}

fn run_check<F: FnOnce() -> Result<String, String>>(name: &'static str, f: F) -> SmokeCheck {
    let t0 = Instant::now();
    match f() {
        Ok(detail) => SmokeCheck {
            name,
            ok: true,
            detail,
            elapsed_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        },
        Err(detail) => SmokeCheck {
            name,
            ok: false,
            detail,
            elapsed_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        },
    }
}

fn approx_eq_slice(actual: &[f32], expected: &[f32], tol: f32) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("len {} != {}", actual.len(), expected.len()));
    }
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        if (a - b).abs() > tol {
            return Err(format!("mismatch at [{i}]: got {a}, expected {b}"));
        }
    }
    Ok(())
}

pub fn run_smoke_suite() -> SmokeReport {
    let mut checks = Vec::new();

    // 1. linear-relu exact parity ([4.5, 10.5])
    checks.push(run_check("linear_relu_exact", || {
        let raw = r#"{"inputs":{"0":[1,-2],"1":[3,10.5]},"layers":[{"args":[0,2,2,false],"constructor":"linear"},{"args":[1],"constructor":"relu"},{"args":[2],"constructor":"add"}],"numSlots":5,"outputSlot":4,"ports":[{"layout":"feature_axis1_singleton","role":"observation","shape":[1,2,1,1],"slot":0},{"layout":"feature_axis1_singleton","role":"observation","shape":[1,2,1,1],"slot":1}],"steps":[{"kind":"unary","layer":0,"slots":[0,2]},{"kind":"unary","layer":1,"slots":[2,3]},{"kind":"binary","layer":2,"slots":[3,1,4]}],"weights":{"0":[1,0.5,-0.25,2]}}"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        let res = run_spec(&loaded.spec)?;
        approx_eq_slice(&res.outputs.values, &[4.5, 10.5], 1e-5)?;
        Ok(format!("outputs={:?} sha={}", res.outputs.values, res.trace_sha256))
    }));

    // 2. linear-relu-bias parity ([5.0, 10.5])
    checks.push(run_check("linear_relu_bias_exact", || {
        let raw = r#"{"inputs":{"0":[1,-2],"1":[3,10.5]},"layers":[{"args":[0,2,2,true],"constructor":"linear"},{"args":[1],"constructor":"relu"},{"args":[2],"constructor":"add"}],"numSlots":5,"outputSlot":4,"ports":[{"layout":"feature_axis1_singleton","role":"observation","shape":[1,2,1,1],"slot":0},{"layout":"feature_axis1_singleton","role":"observation","shape":[1,2,1,1],"slot":1}],"steps":[{"kind":"unary","layer":0,"slots":[0,2]},{"kind":"unary","layer":1,"slots":[2,3]},{"kind":"binary","layer":2,"slots":[3,1,4]}],"weights":{"0":[1,0.5,-0.25,2,0.5,-1]}}"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        let res = run_spec(&loaded.spec)?;
        approx_eq_slice(&res.outputs.values, &[5.0, 10.5], 1e-5)?;
        Ok(format!("outputs={:?}", res.outputs.values))
    }));

    // 3. v2 recursive repeat + tie_weights -> exact [1.5, 3.0]
    checks.push(run_check("recursive_v2_repeat_tie_weights", || {
        let raw = r#"{
          "schema": "burn-math.recursive-spec.v2",
          "ports": {
            "state": { "slot": 0, "role": "state", "shape": [1, 2, 1, 1] },
            "bias_ctx": { "slot": 1, "role": "context", "shape": [1, 2, 1, 1] }
          },
          "subgraphs": {
            "contract_step": {
              "in": ["s", "b"],
              "out": "s_next",
              "steps": [
                { "op": "linear", "args": [2, 2, false], "weights": [0.5, 0.0, 0.0, 0.5], "in": ["s"], "out": "scaled" },
                { "op": "add", "in": ["scaled", "b"], "out": "s_next" }
              ]
            }
          },
          "steps": [
            { "op": "repeat", "subgraph": "contract_step", "depth": 2, "tie_weights": true, "in": ["state", "bias_ctx"], "out": "final_state" }
          ],
          "output": "final_state",
          "inputs": { "state": [0.0, 0.0], "bias_ctx": [1.0, 2.0] }
        }"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        if !loaded.was_v2 || loaded.spec.layers.len() != 2 {
            return Err("expected 2 tied layers from repeat depth=2".into());
        }
        let res = run_spec(&loaded.spec)?;
        approx_eq_slice(&res.outputs.values, &[1.5, 3.0], 1e-5)?;
        Ok(format!("outputs={:?} layers={} steps={}", res.outputs.values, loaded.spec.layers.len(), loaded.spec.steps.len()))
    }));

    // 4. Builtin macro mlp + residual
    checks.push(run_check("recursive_v2_macro_residual", || {
        let raw = r#"{
          "schema": "burn-math.recursive-spec.v2",
          "ports": [{ "id": "x", "slot": 0, "shape": [1, 2, 1, 1] }],
          "steps": [
            {
              "op": "residual",
              "macro": "mlp",
              "params": { "in_dim": 2, "hidden_dim": 2, "out_dim": 2, "bias": false },
              "weights": { "0": [1, 0, 0, 1], "2": [1, 0, 0, 1] },
              "in": ["x"],
              "out": "y"
            }
          ],
          "output": "y",
          "inputs": { "x": [2.0, -3.0] }
        }"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        let res = run_spec(&loaded.spec)?;
        // relu([2, -3]) + [2, -3] = [4, -3]
        approx_eq_slice(&res.outputs.values, &[4.0, -3.0], 1e-5)?;
        Ok(format!("outputs={:?}", res.outputs.values))
    }));

    // 5. validate ok + fail-closed invalid + explain
    checks.push(run_check("validate_explain_and_fail_closed", || {
        let bad = json!({
            "numSlots": 2,
            "layers": [{ "constructor": "unknownOp", "args": [0] }],
            "steps": [],
            "outputSlot": 1,
            "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }]
        });
        let s_bad: SpecV1 = serde_json::from_value(bad).map_err(|e| e.to_string())?;
        let rep_bad = validate_spec(&s_bad, false);
        if rep_bad.valid || rep_bad.code != "unknown_layer_constructor" {
            return Err(format!("expected unknown_layer_constructor, got {:?}", rep_bad.code));
        }
        let good = r#"{"inputs":{"0":[1,-2]},"layers":[{"args":[0,2,2,false],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}]}"#;
        let s_good = load_spec_str(good, Path::new("."))?.spec;
        let rep_good = validate_spec(&s_good, true);
        if !rep_good.valid || rep_good.explain.is_none() {
            return Err("expected valid + explain".into());
        }
        Ok(format!("bad_code={} explain_params={}", rep_bad.code, rep_good.explain.unwrap().total_params))
    }));

    // 6. verify + fuzz (equivalent vs mutated)
    checks.push(run_check("verify_and_fuzz", || {
        let base_raw = r#"{"inputs":{"0":[1,-2]},"layers":[{"args":[0,2,2,false],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}],"weights":{"0":[1,0,0,1]}}"#;
        let mut_raw = r#"{"inputs":{"0":[1,-2]},"layers":[{"args":[0,2,2,false],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}],"weights":{"0":[2,0,0,1]}}"#;
        let base = load_spec_str(base_raw, Path::new("."))?.spec;
        let mutated = load_spec_str(mut_raw, Path::new("."))?.spec;
        let cases = generate_fuzz_cases(&base, &FuzzConfig { count: 4, ..Default::default() });
        let eq = verify_specs(&base, &base, &cases, 1e-5, 1e-5);
        let neq = verify_specs(&base, &mutated, &cases, 1e-5, 1e-5);
        if !eq.equivalent || neq.equivalent {
            return Err("verify failed accept/reject distinction".into());
        }
        Ok(format!("self={} mutated={} max_diff={:.4}", eq.verdict, neq.verdict, neq.max_abs_diff))
    }));

    // 7. recurse search (linear chain fusion + verification)
    checks.push(run_check("recurse_search_fusion", || {
        let raw = r#"{
          "numSlots": 3,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, true] },
            { "constructor": "linear", "args": [1, 2, 1, true] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] }
          ],
          "outputSlot": 2,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [1, 2, 3, 4, 0.5, -0.5], "1": [2, -1, 1.0] },
          "inputs": { "0": [1.5, -0.5] }
        }"#;
        let base = load_spec_str(raw, Path::new("."))?.spec;
        let sres = search_graph_tree(&base, 3, None, 1e-5, 1e-5)?;
        if sres.best_steps >= sres.baseline_steps {
            return Err(format!("expected step reduction, got {} -> {}", sres.baseline_steps, sres.best_steps));
        }
        Ok(format!("steps {} -> {}, chain={:?}", sres.baseline_steps, sres.best_steps, sres.transform_chain))
    }));

    // 8. recurse solve-gd (Burn 0.22.1 autodiff lowers loss)
    checks.push(run_check("recurse_solve_gd_autodiff", || {
        let raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "linear", "args": [0, 2, 1, true] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [0.0, 0.0, 0.0] },
          "inputs": { "0": [1.0, 2.0] }
        }"#;
        let base = load_spec_str(raw, Path::new("."))?.spec;
        let targets: TargetsFile = serde_json::from_str(
            r#"{"cases":[{"inputs":{"0":[1.0,2.0]},"expected":[5.0]}]}"#,
        ).map_err(|e| e.to_string())?;
        let res = solve_gd_autodiff(&base, Some(&targets), Some(0), 25, 0.1, false)?;
        if !(res.final_loss < 1e-3 && res.final_loss < res.start_loss) {
            return Err(format!("loss did not converge: {} -> {}", res.start_loss, res.final_loss));
        }
        Ok(format!("loss {:.4} -> {:.6} steps={}", res.start_loss, res.final_loss, res.steps_taken))
    }));

    // 9. recurse fit (gradient-free ES lowers loss)
    checks.push(run_check("recurse_fit_es", || {
        let raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "linear", "args": [0, 2, 1, true] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [0.0, 0.0, 0.0] },
          "inputs": { "0": [1.0, 2.0] }
        }"#;
        let base = load_spec_str(raw, Path::new("."))?.spec;
        let targets: TargetsFile = serde_json::from_str(
            r#"{"cases":[{"inputs":{"0":[1.0,2.0]},"expected":[5.0]}]}"#,
        ).map_err(|e| e.to_string())?;
        let res = fit_es(&base, &targets, 25, 16, 0.2, 0.2, 42, false, 1e-6)?;
        if !(res.final_loss < res.start_loss) {
            return Err(format!("ES loss did not improve: {} -> {}", res.start_loss, res.final_loss));
        }
        Ok(format!("loss {:.4} -> {:.6}", res.start_loss, res.final_loss))
    }));

    // 10. mixed i64 + f32 pipeline + batch isolation check
    checks.push(run_check("mixed_int_i64_and_batch_isolation", || {
        let int_raw = r#"{
          "numSlots": 4,
          "layers": [
            { "constructor": "intAdd", "args": [0] },
            { "constructor": "intToFloat", "args": [1] }
          ],
          "steps": [
            { "kind": "binary", "layer": 0, "slots": [0, 1, 2] },
            { "kind": "unary", "layer": 1, "slots": [2, 3] }
          ],
          "outputSlot": 3,
          "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
          ],
          "inputs": { "0": [10, -4], "1": [5, 7] }
        }"#;
        let int_spec = load_spec_str(int_raw, Path::new("."))?.spec;
        let r_int = run_spec(&int_spec)?;
        approx_eq_slice(&r_int.outputs.values, &[15.0, 3.0], 1e-6)?;

        // Batch isolation: Batch [A; B] output[0] == run(A) and output[1] == run(B)
        let b2_raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "layerNorm", "args": [0, 2, 1e-5] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [2, 1, 1, 2] }],
          "inputs": { "0": [1.0, 3.0, 10.0, 20.0] }
        }"#;
        let b2_spec = load_spec_str(b2_raw, Path::new("."))?.spec;
        let r_b2 = eval_spec_with_inputs(&b2_spec, None)?;
        let b1_raw = r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "layerNorm", "args": [0, 2, 1e-5] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 1, 1, 2] }],
          "inputs": { "0": [10.0, 20.0] }
        }"#;
        let b1_spec = load_spec_str(b1_raw, Path::new("."))?.spec;
        let r_b1 = eval_spec_with_inputs(&b1_spec, None)?;
        approx_eq_slice(&r_b2.values[2..4], &r_b1.values, 1e-5)?;
        Ok(format!("int_out={:?} batch_isolated_ok", r_int.outputs.values))
    }));

    // 11. trace-hash step divergence localization + prefix isolate
    checks.push(run_check("trace_hash_divergence_isolate", || {
        let base_raw = r#"{
          "numSlots": 4,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "relu", "args": [1] },
            { "constructor": "linear", "args": [2, 2, 2, false] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] },
            { "kind": "unary", "layer": 2, "slots": [2, 3] }
          ],
          "outputSlot": 3,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [1, 0, 0, 1], "2": [1, 0, 0, 1] },
          "inputs": { "0": [2.0, 3.0] }
        }"#;
        let mut_raw = r#"{
          "numSlots": 4,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "relu", "args": [1] },
            { "constructor": "linear", "args": [2, 2, 2, false] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] },
            { "kind": "unary", "layer": 2, "slots": [2, 3] }
          ],
          "outputSlot": 3,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [1, 0, 0, 1], "2": [2, 0, 0, 1] },
          "inputs": { "0": [2.0, 3.0] }
        }"#;
        let base = load_spec_str(base_raw, Path::new("."))?.spec;
        let mutated = load_spec_str(mut_raw, Path::new("."))?.spec;
        let cmp = compare_trace_hashes(&base, &mutated, DEFAULT_MAX_TENSOR_BYTES, true)?;
        if cmp.first_divergent_step != Some(2) || cmp.left_prefix_subspec.is_none() {
            return Err(format!("expected divergence at step 2 with isolate subspec, got {:?}", cmp.first_divergent_step));
        }
        Ok(format!("first_divergent_step=2 reason={}", cmp.divergent_step_detail.unwrap().reason))
    }));

    // 12. rollout fixpoint contraction + unroll parity
    checks.push(run_check("rollout_fixpoint_and_unroll", || {
        let raw = r#"{
          "schema": "burn-math.recursive-spec.v2",
          "ports": {
            "state": { "slot": 0, "role": "state", "shape": [1, 2, 1, 1] },
            "bias_ctx": { "slot": 1, "role": "context", "shape": [1, 2, 1, 1] }
          },
          "subgraphs": {
            "contract_step": {
              "in": ["s", "b"],
              "out": "s_next",
              "steps": [
                { "op": "linear", "args": [2, 2, false], "weights": [0.5, 0.0, 0.0, 0.5], "in": ["s"], "out": "scaled" },
                { "op": "add", "in": ["scaled", "b"], "out": "s_next" }
              ]
            }
          },
          "steps": [
            { "op": "repeat", "subgraph": "contract_step", "depth": 1, "tie_weights": true, "in": ["state", "bias_ctx"], "out": "final_state" }
          ],
          "output": "final_state",
          "inputs": { "state": [0.0, 0.0], "bias_ctx": [1.0, 2.0] }
        }"#;
        let loaded = load_spec_str(raw, Path::new("."))?;
        let scan = rollout_scan(&loaded.spec, loaded.expansion.as_ref(), 4, Some("state"), None)?;
        let unroll = rollout_unroll(&loaded.spec, loaded.expansion.as_ref(), 4, Some("state"))?;
        let unrolled_out = eval_spec_with_inputs(&unroll.unrolled_spec, None)?;
        approx_eq_slice(&scan.final_state.values, &unrolled_out.values, 1e-5)?;
        let fp = rollout_fixpoint(&loaded.spec, loaded.expansion.as_ref(), 1e-5, 50, Some("state"))?;
        if !fp.converged {
            return Err(format!("expected fixpoint convergence, final_delta={}", fp.final_delta_linf));
        }
        // s_{t+1} = 0.5 s_t + [1, 2] converges to [2.0, 4.0]
        approx_eq_slice(&fp.final_state.values, &[2.0, 4.0], 1e-4)?;
        Ok(format!("converged in {} steps final={:?}", fp.steps_run, fp.final_state.values))
    }));

    // 13. rollout bisect (tick + step localization)
    checks.push(run_check("rollout_bisect_localization", || {
        let base_raw = r#"{
          "numSlots": 3,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "add", "args": [1] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 2] },
            { "kind": "binary", "layer": 1, "slots": [2, 1, 2] }
          ],
          "outputSlot": 2,
          "ports": [
            { "slot": 0, "role": "state", "shape": [1, 2, 1, 1] },
            { "slot": 1, "role": "context", "shape": [1, 2, 1, 1] }
          ],
          "weights": { "0": [0.5, 0.0, 0.0, 0.5] },
          "inputs": { "0": [0.0, 0.0], "1": [1.0, 2.0] }
        }"#;
        let cand_raw = r#"{
          "numSlots": 3,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "add", "args": [1] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 2] },
            { "kind": "binary", "layer": 1, "slots": [2, 1, 2] }
          ],
          "outputSlot": 2,
          "ports": [
            { "slot": 0, "role": "state", "shape": [1, 2, 1, 1] },
            { "slot": 1, "role": "context", "shape": [1, 2, 1, 1] }
          ],
          "weights": { "0": [0.25, 0.0, 0.0, 0.5] },
          "inputs": { "0": [0.0, 0.0], "1": [1.0, 2.0] }
        }"#;
        let base = load_spec_str(base_raw, Path::new("."))?.spec;
        let cand = load_spec_str(cand_raw, Path::new("."))?.spec;
        // At tick 0 state=[0,0], linear([0,0])=[0,0] matches; at tick 1 state=[1,2], linear diverges at step 0!
        let bis = rollout_bisect(&base, &cand, None, 5, Some("state"), None, 1e-5)?;
        if bis.equivalent || bis.divergent_tick != Some(1) || bis.first_divergent_step != Some(0) {
            return Err(format!("expected tick=1 step=0 divergence, got tick={:?} step={:?}", bis.divergent_tick, bis.first_divergent_step));
        }
        Ok("diverged cleanly at tick=1 step=0".into())
    }));

    // 14. profile & compare
    checks.push(run_check("profile_and_compare", || {
        let raw = r#"{"inputs":{"0":[1,-2]},"layers":[{"args":[0,2,2,false],"constructor":"linear"},{"args":[1],"constructor":"relu"}],"numSlots":3,"outputSlot":2,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]},{"kind":"unary","layer":1,"slots":[1,2]}],"weights":{"0":[1,0,0,1]}}"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let prof = profile_spec(&spec, 2, 5)?;
        let cmp = compare_specs_perf(&spec, &spec, 2, 5)?;
        if !cmp.outputs_match || prof.steps_count != 2 {
            return Err("profile/compare mismatch".into());
        }
        Ok(format!("median_ms={:.3} speedup={:.2}x", prof.median_ms, cmp.speedup))
    }));

    // 15. preconditioned solve-gd + sensitivity-guided fit
    checks.push(run_check("recurse_preconditioned_and_sensitivity", || {
        let raw = r#"{
          "numSlots": 3,
          "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "linear", "args": [1, 2, 1, true] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] }
          ],
          "outputSlot": 2,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [1.0, 0.0, 0.0, 1.0], "1": [0.0, 0.0, 0.0] },
          "inputs": { "0": [1.0, 2.0] }
        }"#;
        let spec = load_spec_str(raw, Path::new("."))?.spec;
        let tf: TargetsFile = serde_json::from_str(
            r#"{"cases":[{"inputs":{"0":[1.0,2.0]},"expected":[4.0]}]}"#,
        ).map_err(|e| e.to_string())?;
        let gd = solve_gd_autodiff(&spec, Some(&tf), None, 15, 0.1, true)?;
        if !gd.preconditioned || gd.final_loss >= gd.start_loss {
            return Err("preconditioned GD failed to reduce loss".into());
        }
        let fit = fit_es(&spec, &tf, 15, 12, 0.2, 0.2, 42, true, 1e-3)?;
        if !fit.sensitivity_guided || !fit.frozen_layers.contains(&0) || fit.final_loss >= fit.start_loss {
            return Err(format!("sensitivity-guided fit unexpected result: frozen={:?} loss {}->{}", fit.frozen_layers, fit.start_loss, fit.final_loss));
        }
        Ok(format!("gd_trust={:?} fit_frozen={:?}", gd.hessian_trust, fit.frozen_layers))
    }));

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
        Ok(format!("max_abs={:.2e} autodiff_weights={}", rep.max_abs_error, rep.autodiff_check.unwrap().checked_weights))
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
        Ok(format!("fp16_diff={:.5} saved_measured={}B", qv.verification.max_abs_diff, qb_meas.total_saved_bytes))
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
        })).map_err(|e| e.to_string())?;
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
        if !rep.ok || !rep.sharded || !rep.reassembly_exact || !rep.verified || rep.shards[0].num_shards != 3 {
            return Err(format!("shard failed: exact={} verified={} shards={}", rep.reassembly_exact, rep.verified, rep.shards.len()));
        }
        Ok(format!("num_shards={} max_abs={:.2e}", rep.shards[0].num_shards, rep.max_abs_error))
    }));

    // 20. weight-lab selftest (engine probe, spectral radius, gate, norm_bound/orthonormal/measured compose, expr tree)
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

    // 21. linear-equiv exact symbolic affine equivalence (sharded linear + concat chain == original)
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
        Ok(format!("verdict={} fp={}", eq.verdict, eq.effective_fingerprint.unwrap_or_default()))
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
        let zf = zoom_fuzz_specs(&base, &cand, &FuzzConfig { count: 6, seed: 42, lo: -4.0, hi: 4.0 }, 3)?;
        if !zf.ok || zf.zoomed_max_abs_error < zf.initial_max_abs_error {
            return Err("zoom_fuzz failed to maintain or amplify worst-case error".into());
        }
        Ok(format!("ops={} zoom_err={:.3}->{:.3}", cm.per_op.len(), zf.initial_max_abs_error, zf.zoomed_max_abs_error))
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
        let final_steps = wf_val["active_spec_summary"]["steps_count"].as_u64().unwrap_or(99);
        Ok(format!("workflow_stages=3 optimized_steps={final_steps}"))
    }));

    let passed = checks.iter().filter(|c| c.ok).count();
    let total = checks.len();
    SmokeReport {
        ok: passed == total,
        passed,
        total,
        checks,
    }
}
