use math_draw::catalog::CATALOG;
use math_draw::engine::run_spec;
use math_draw::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
use math_draw::rewrite::{apply_rewrite, apply_rewrites_recursive};
use math_draw::smoke::run_smoke_suite;
use math_draw::spec::{LayerDecl, PortDecl, SpecV1, StepDecl, load_spec_str};
use math_draw::validate::validate_spec;
use math_draw::verify::{FuzzConfig, TargetsFile, generate_fuzz_cases, verify_specs};
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

#[test]
fn catalog_has_all_52_constructors() {
    assert_eq!(CATALOG.len(), 52);
}

#[test]
fn smoke_suite_passes_all_checks() {
    let report = run_smoke_suite();
    for c in &report.checks {
        assert!(c.ok, "smoke check '{}' failed: {}", c.name, c.detail);
    }
    assert!(report.ok);
}

#[test]
fn all_52_constructors_execute_cleanly() {
    let single_unary = |ctor: &str, args: Vec<serde_json::Value>, shape: [usize; 4], input: Vec<serde_json::Value>, weights: Option<Vec<f32>>| {
        let mut w_map = BTreeMap::new();
        if let Some(w) = weights {
            w_map.insert("0".to_string(), w);
        }
        let mut in_map = BTreeMap::new();
        in_map.insert("0".to_string(), input);
        let spec = SpecV1 {
            num_slots: 2,
            layers: vec![LayerDecl {
                constructor: ctor.to_string(),
                args,
            }],
            steps: vec![StepDecl {
                kind: "unary".into(),
                layer: 0,
                slots: vec![0, 1],
            }],
            output_slot: 1,
            ports: vec![PortDecl {
                slot: 0,
                role: "observation".into(),
                shape,
                layout: "any_rank4".into(),
            }],
            weights: w_map,
            inputs: in_map,
        };
        run_spec(&spec).unwrap_or_else(|e| panic!("constructor {ctor} failed: {e}"))
    };

    // Unary float activations & transforms
    for ctor in ["relu", "gelu", "sigmoid", "tanh", "hardSwish", "mish", "sqrt"] {
        single_unary(ctor, vec![0.into()], [1, 2, 2, 2], vec![1.0.into(); 8], None);
    }
    single_unary("leakyRelu", vec![0.into(), 0.2.into()], [1, 2, 1, 1], vec![(-2.0).into(), 3.0.into()], None);
    single_unary("prelu", vec![0.into(), 2.into(), 0.25.into()], [1, 2, 1, 1], vec![(-2.0).into(), 3.0.into()], None);
    single_unary("swiGlu", vec![0.into(), 2.into(), 2.into(), false.into()], [1, 1, 1, 2], vec![1.0.into(), 2.0.into()], None);
    single_unary("hardSigmoid", vec![0.into(), 0.2.into(), 0.5.into()], [1, 2, 1, 1], vec![1.0.into(), (-1.0).into()], None);
    single_unary("softplus", vec![0.into(), 1.0.into()], [1, 2, 1, 1], vec![0.0.into(), 1.0.into()], None);
    single_unary("softmax", vec![0.into(), 1.into()], [1, 2, 1, 1], vec![1.0.into(), 2.0.into()], None);
    single_unary("logSoftmax", vec![0.into(), 1.into()], [1, 2, 1, 1], vec![1.0.into(), 2.0.into()], None);
    single_unary("glu", vec![0.into(), 1.into()], [1, 4, 1, 1], vec![1.0.into(), 2.0.into(), 0.0.into(), 0.0.into()], None);
    single_unary("transpose", vec![0.into(), 2.into(), 3.into()], [1, 1, 2, 3], vec![1.0.into(); 6], None);

    // Linear & Norms & Embedding
    single_unary("linear", vec![0.into(), 2.into(), 3.into(), true.into()], [1, 2, 1, 1], vec![1.0.into(), 2.0.into()], None);
    single_unary("batchNorm", vec![0.into(), 2.into(), 1e-5.into()], [2, 2, 1, 1], vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()], None);
    single_unary("groupNorm", vec![0.into(), 2.into(), 4.into()], [1, 4, 2, 2], vec![1.0.into(); 16], None);
    single_unary("instanceNorm", vec![0.into(), 2.into()], [1, 2, 2, 2], vec![1.0.into(); 8], None);
    single_unary("layerNorm", vec![0.into(), 2.into()], [1, 1, 2, 2], vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()], None);
    single_unary("rmsNorm", vec![0.into(), 2.into()], [1, 1, 1, 2], vec![3.0.into(), 4.0.into()], None);
    single_unary("embedding", vec![0.into(), 5.into(), 3.into()], [1, 2, 1, 1], vec![0.into(), 4.into()], None);

    // Convs & Pools
    single_unary("conv1d", vec![0.into(), 2.into(), 3.into(), 3.into(), 1.into(), 1.into()], [1, 2, 4, 1], vec![1.0.into(); 8], None);
    single_unary("conv2d", vec![0.into(), 2.into(), 3.into(), 3.into(), 3.into(), 1.into(), 1.into(), 1.into(), 1.into()], [1, 2, 4, 4], vec![1.0.into(); 32], None);
    single_unary("convTranspose2d", vec![0.into(), 2.into(), 3.into(), 2.into(), 2.into()], [1, 2, 3, 3], vec![1.0.into(); 18], None);
    single_unary("maxPool1d", vec![0.into(), 2.into()], [1, 2, 4, 1], vec![1.0.into(); 8], None);
    single_unary("avgPool1d", vec![0.into(), 2.into()], [1, 2, 4, 1], vec![1.0.into(); 8], None);
    single_unary("maxPool2d", vec![0.into(), 2.into(), 2.into()], [1, 2, 4, 4], vec![1.0.into(); 32], None);
    single_unary("avgPool2d", vec![0.into(), 2.into(), 2.into()], [1, 2, 4, 4], vec![1.0.into(); 32], None);
    single_unary("adaptiveAvgPool2d", vec![0.into(), 1.into(), 1.into()], [1, 2, 4, 4], vec![1.0.into(); 32], None);

    // Custom layers
    let fn_res = single_unary("featureNorm", vec![0.into()], [1, 2, 1, 1], vec![3.0.into(), 4.0.into()], None);
    assert!((fn_res.outputs.values[0] - 0.6).abs() < 1e-5);
    assert!((fn_res.outputs.values[1] - 0.8).abs() < 1e-5);
    for ctor in ["shiftUp", "shiftDown", "shiftLeft", "shiftRight"] {
        single_unary(ctor, vec![0.into(), 1.into()], [1, 1, 3, 3], vec![1.0.into(); 9], None);
    }
    single_unary("ghost", vec![0.into(), 2.into(), 4.into(), 1.into(), 1.into(), 2.into()], [1, 2, 2, 2], vec![1.0.into(); 8], None);
    single_unary("seBlock", vec![0.into(), 4.into(), 2.into()], [1, 4, 2, 2], vec![1.0.into(); 16], None);

    // Binary float ops
    for ctor in ["add", "sub", "mul", "matmul", "div", "concat"] {
        let args = if ctor == "concat" { vec![0.into(), 1.into()] } else { vec![0.into()] };
        let spec = SpecV1 {
            num_slots: 3,
            layers: vec![LayerDecl { constructor: ctor.into(), args }],
            steps: vec![StepDecl { kind: "binary".into(), layer: 0, slots: vec![0, 1, 2] }],
            output_slot: 2,
            ports: vec![
                PortDecl { slot: 0, role: "observation".into(), shape: [1, 1, 2, 2], layout: "any_rank4".into() },
                PortDecl { slot: 1, role: "observation".into(), shape: [1, 1, 2, 2], layout: "any_rank4".into() },
            ],
            weights: BTreeMap::new(),
            inputs: BTreeMap::from([
                ("0".to_string(), vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()]),
                ("1".to_string(), vec![2.0.into(), 2.0.into(), 2.0.into(), 2.0.into()]),
            ]),
        };
        run_spec(&spec).unwrap_or_else(|e| panic!("binary {ctor} failed: {e}"));
    }

    // Int ops: intAdd, intSub, intMul, intMatmul, intDiv, intRem, intToFloat, intFromFloat
    for ctor in ["intAdd", "intSub", "intMul", "intMatmul", "intDiv", "intRem"] {
        let spec = SpecV1 {
            num_slots: 4,
            layers: vec![
                LayerDecl { constructor: ctor.into(), args: vec![0.into()] },
                LayerDecl { constructor: "intToFloat".into(), args: vec![1.into()] },
            ],
            steps: vec![
                StepDecl { kind: "binary".into(), layer: 0, slots: vec![0, 1, 2] },
                StepDecl { kind: "unary".into(), layer: 1, slots: vec![2, 3] },
            ],
            output_slot: 3,
            ports: vec![
                PortDecl { slot: 0, role: "observation".into(), shape: [1, 1, 2, 2], layout: "any_rank4".into() },
                PortDecl { slot: 1, role: "observation".into(), shape: [1, 1, 2, 2], layout: "any_rank4".into() },
            ],
            weights: BTreeMap::new(),
            inputs: BTreeMap::from([
                ("0".to_string(), vec![10.into(), 20.into(), 30.into(), 40.into()]),
                ("1".to_string(), vec![3.into(), 4.into(), 5.into(), 6.into()]),
            ]),
        };
        run_spec(&spec).unwrap_or_else(|e| panic!("int binary {ctor} failed: {e}"));
    }

    let roundtrip = SpecV1 {
        num_slots: 3,
        layers: vec![
            LayerDecl { constructor: "intFromFloat".into(), args: vec![0.into()] },
            LayerDecl { constructor: "intToFloat".into(), args: vec![1.into()] },
        ],
        steps: vec![
            StepDecl { kind: "unary".into(), layer: 0, slots: vec![0, 1] },
            StepDecl { kind: "unary".into(), layer: 1, slots: vec![1, 2] },
        ],
        output_slot: 2,
        ports: vec![PortDecl { slot: 0, role: "observation".into(), shape: [1, 2, 1, 1], layout: "any_rank4".into() }],
        weights: BTreeMap::new(),
        inputs: BTreeMap::from([("0".to_string(), vec![7.9.into(), (-3.9).into()])]),
    };
    let rt_res = run_spec(&roundtrip).unwrap();
    assert_eq!(rt_res.outputs.values, vec![7.0, -3.0]);
}

#[test]
fn div_by_zero_and_int_overflow_fail_closed() {
    let div0 = json!({
        "numSlots": 3,
        "layers": [{ "constructor": "div", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [1.0], "1": [0.0] }
    });
    let s: SpecV1 = serde_json::from_value(div0).unwrap();
    assert!(run_spec(&s).unwrap_err().contains("zero denominator"));

    let iov = json!({
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
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [i64::MAX], "1": [1] }
    });
    let s2: SpecV1 = serde_json::from_value(iov).unwrap();
    assert!(run_spec(&s2).unwrap_err().contains("overflow"));
}

#[test]
fn rewrites_and_recurse_preserve_semantics() {
    let raw = r#"{
      "numSlots": 5,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 2, false] },
        { "constructor": "linear", "args": [1, 2, 2, false] },
        { "constructor": "add", "args": [2] },
        { "constructor": "relu", "args": [3] },
        { "constructor": "relu", "args": [4] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "unary", "layer": 1, "slots": [0, 2] },
        { "kind": "binary", "layer": 2, "slots": [1, 2, 3] },
        { "kind": "unary", "layer": 3, "slots": [3, 4] },
        { "kind": "unary", "layer": 4, "slots": [4, 4] }
      ],
      "outputSlot": 4,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [1, 0, 0, 1], "1": [0.5, 0.5, 0, 0.5] },
      "inputs": { "0": [2.0, -4.0] }
    }"#;
    let base = load_spec_str(raw, Path::new(".")).unwrap().spec;
    let fused = apply_rewrite("linear-branch-fusion", &base).unwrap();
    assert_eq!(fused.steps.len(), 3);
    let cases = generate_fuzz_cases(&base, &FuzzConfig { count: 6, ..Default::default() });
    assert!(verify_specs(&base, &fused, &cases, 1e-5, 1e-5).equivalent);

    let (fix_spec, applied) = apply_rewrites_recursive(&base, 8);
    assert!(!applied.is_empty());
    assert!(validate_spec(&fix_spec, false).valid);

    let sres = search_graph_tree(&base, 3, Some(&cases), 1e-5, 1e-5).unwrap();
    assert!(sres.best_steps <= base.steps.len());

    // Multi-layer solve-gd & fit
    let ml_raw = r#"{
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
      "weights": { "0": [0.2, -0.1, 0.3, 0.4, 0.0, 0.0], "1": [0.5, -0.5, 0.0] },
      "inputs": { "0": [1.0, 2.0] }
    }"#;
    let ml = load_spec_str(ml_raw, Path::new(".")).unwrap().spec;
    let tf: TargetsFile = serde_json::from_str(r#"{"cases":[{"inputs":{"0":[1.0,2.0]},"expected":[2.5]}]}"#).unwrap();
    let gd = solve_gd_autodiff(&ml, Some(&tf), None, 20, 0.05).unwrap();
    assert!(gd.final_loss < gd.start_loss);
    let es = fit_es(&ml, &tf, 15, 12, 0.2, 0.2, 7).unwrap();
    assert!(es.final_loss < es.start_loss);
}
