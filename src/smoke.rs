use serde::Serialize;
use serde_json::json;
use std::path::Path;
use std::time::Instant;

use crate::engine::{eval_spec_with_inputs, run_spec};
use crate::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use crate::spec::{load_spec_str, SpecV1};
use crate::validate::validate_spec;
use crate::verify::{FuzzConfig, TargetsFile, generate_fuzz_cases, verify_specs};

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
        let res = solve_gd_autodiff(&base, Some(&targets), Some(0), 25, 0.1)?;
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
        let res = fit_es(&base, &targets, 25, 16, 0.2, 0.2, 42)?;
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

    let passed = checks.iter().filter(|c| c.ok).count();
    let total = checks.len();
    SmokeReport {
        ok: passed == total,
        passed,
        total,
        checks,
    }
}
