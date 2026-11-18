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
fn catalog_has_all_95_constructors() {
    assert_eq!(CATALOG.len(), 95);
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
fn all_95_constructors_execute_cleanly() {
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
    let single_binary = |ctor: &str, args: Vec<serde_json::Value>, s0: [usize; 4], i0: Vec<serde_json::Value>, s1: [usize; 4], i1: Vec<serde_json::Value>| {
        let spec = SpecV1 {
            num_slots: 3,
            layers: vec![LayerDecl { constructor: ctor.into(), args }],
            steps: vec![StepDecl { kind: "binary".into(), layer: 0, slots: vec![0, 1, 2] }],
            output_slot: 2,
            ports: vec![
                PortDecl { slot: 0, role: "observation".into(), shape: s0, layout: "any_rank4".into() },
                PortDecl { slot: 1, role: "observation".into(), shape: s1, layout: "any_rank4".into() },
            ],
            weights: BTreeMap::new(),
            inputs: BTreeMap::from([("0".to_string(), i0), ("1".to_string(), i1)]),
        };
        run_spec(&spec).unwrap_or_else(|e| panic!("binary constructor {ctor} failed: {e}"))
    };
    let single_ternary = |ctor: &str,
                          args: Vec<serde_json::Value>,
                          s0: [usize; 4],
                          i0: Vec<serde_json::Value>,
                          s1: [usize; 4],
                          i1: Vec<serde_json::Value>,
                          s2: [usize; 4],
                          i2: Vec<serde_json::Value>| {
        let spec = SpecV1 {
            num_slots: 4,
            layers: vec![LayerDecl { constructor: ctor.into(), args }],
            steps: vec![StepDecl { kind: "ternary".into(), layer: 0, slots: vec![0, 1, 2, 3] }],
            output_slot: 3,
            ports: vec![
                PortDecl { slot: 0, role: "observation".into(), shape: s0, layout: "any_rank4".into() },
                PortDecl { slot: 1, role: "observation".into(), shape: s1, layout: "any_rank4".into() },
                PortDecl { slot: 2, role: "observation".into(), shape: s2, layout: "any_rank4".into() },
            ],
            weights: BTreeMap::new(),
            inputs: BTreeMap::from([
                ("0".to_string(), i0),
                ("1".to_string(), i1),
                ("2".to_string(), i2),
            ]),
        };
        run_spec(&spec).unwrap_or_else(|e| panic!("ternary constructor {ctor} failed: {e}"))
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
        single_binary(
            ctor,
            args,
            [1, 1, 2, 2],
            vec![1.0.into(), 2.0.into(), 3.0.into(), 4.0.into()],
            [1, 1, 2, 2],
            vec![2.0.into(), 2.0.into(), 2.0.into(), 2.0.into()],
        );
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

    // Ported pure math constructors (30)
    // 1. Numeric (4)
    assert_eq!(single_unary("abs", vec![0.into()], [1, 3, 1, 1], vec![(-2.0).into(), 0.0.into(), 3.0.into()], None).outputs.values, vec![2.0, 0.0, 3.0]);
    let exp_out = single_unary("exp", vec![0.into()], [1, 2, 1, 1], vec![0.0.into(), 1.0.into()], None);
    assert!((exp_out.outputs.values[0] - 1.0).abs() < 1e-6 && (exp_out.outputs.values[1] - std::f32::consts::E).abs() < 1e-5);
    let log_out = single_unary("log", vec![0.into()], [1, 2, 1, 1], vec![1.0.into(), (std::f32::consts::E as f64).into()], None);
    assert!(log_out.outputs.values[0].abs() < 1e-6 && (log_out.outputs.values[1] - 1.0).abs() < 1e-5);
    assert_eq!(single_unary("clamp", vec![0.into(), 0.0.into(), 1.0.into()], [1, 3, 1, 1], vec![(-2.0).into(), 0.5.into(), 3.0.into()], None).outputs.values, vec![0.0, 0.5, 1.0]);

    // 2. Tensor transforms (6)
    let reshaped = single_unary("reshape", vec![0.into(), json!([1, 1, 3, 2])], [1, 2, 1, 3], (1..=6).map(|v| (v as f64).into()).collect(), None);
    assert_eq!(reshaped.outputs.shape, [1, 1, 3, 2]);
    let permuted = single_unary("permute", vec![0.into(), json!([0, 2, 3, 1])], [1, 2, 1, 3], (1..=6).map(|v| (v as f64).into()).collect(), None);
    assert_eq!(permuted.outputs.shape, [1, 1, 3, 2]);
    assert_eq!(permuted.outputs.values, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let sliced = single_unary("slice", vec![0.into(), json!([0, 0, 0, 1]), json!([1, 2, 2, 3])], [1, 2, 2, 3], (1..=12).map(|v| (v as f64).into()).collect(), None);
    assert_eq!(sliced.outputs.shape, [1, 2, 2, 2]);
    assert_eq!(sliced.outputs.values, vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0, 11.0, 12.0]);
    let selected = single_unary("selectAxis", vec![0.into(), 1.into(), json!([1, 0])], [1, 2, 1, 3], (1..=6).map(|v| (v as f64).into()).collect(), None);
    assert_eq!(selected.outputs.values, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    let filled = single_unary("fillLike", vec![0.into(), 7.5.into()], [1, 2, 2, 1], vec![0.0.into(); 4], None);
    assert_eq!(filled.outputs.values, vec![7.5; 4]);
    let expanded = single_binary("expandLike", vec![0.into()], [2, 1, 1, 1], vec![10.0.into(), 20.0.into()], [2, 3, 1, 1], vec![0.0.into(); 6]);
    assert_eq!(expanded.outputs.shape, [2, 3, 1, 1]);
    assert_eq!(expanded.outputs.values, vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);

    // 3. Linear algebra (4)
    let dot_out = single_binary("dot", vec![0.into()], [2, 3, 1, 1], vec![1.0.into(), 2.0.into(), 3.0.into(), 3.0.into(), 4.0.into(), 0.0.into()], [2, 3, 1, 1], vec![3.0.into(), 1.0.into(), 2.0.into(), 0.0.into(), 4.0.into(), 3.0.into()]);
    assert_eq!(dot_out.outputs.values, vec![11.0, 16.0]);
    let norm_out = single_unary("l2Norm", vec![0.into()], [1, 2, 1, 1], vec![3.0.into(), 4.0.into()], None);
    assert!((norm_out.outputs.values[0] - 5.0).abs() < 1e-5);
    let dist_out = single_binary("l2Distance", vec![0.into()], [1, 2, 1, 1], vec![1.0.into(), 2.0.into()], [1, 2, 1, 1], vec![4.0.into(), 6.0.into()]);
    assert!((dist_out.outputs.values[0] - 5.0).abs() < 1e-5);
    let cos_out = single_binary("cosineSimilarity", vec![0.into()], [1, 2, 1, 1], vec![3.0.into(), 4.0.into()], [1, 2, 1, 1], vec![4.0.into(), 3.0.into()]);
    assert!((cos_out.outputs.values[0] - 0.96).abs() < 1e-5);

    // 4. Statistics (6) including wide-spread f64 two-pass variance/std
    let stat_in = vec![2.0.into(), 4.0.into(), 6.0.into(), 8.0.into()];
    assert_eq!(single_unary("sum", vec![0.into()], [1, 4, 1, 1], stat_in.clone(), None).outputs.values, vec![20.0]);
    assert_eq!(single_unary("mean", vec![0.into()], [1, 4, 1, 1], stat_in.clone(), None).outputs.values, vec![5.0]);
    assert_eq!(single_unary("variancePopulation", vec![0.into()], [1, 4, 1, 1], stat_in.clone(), None).outputs.values, vec![5.0]);
    assert!((single_unary("stdPopulation", vec![0.into()], [1, 4, 1, 1], stat_in.clone(), None).outputs.values[0] - 5.0f32.sqrt()).abs() < 1e-5);
    assert_eq!(single_unary("min", vec![0.into()], [1, 4, 1, 1], stat_in.clone(), None).outputs.values, vec![2.0]);
    assert_eq!(single_unary("max", vec![0.into()], [1, 4, 1, 1], stat_in, None).outputs.values, vec![8.0]);
    let wide_in = vec![1e19.into(), (-1e19).into(), 1e19.into(), (-1e19).into()];
    let wide_var = single_unary("variancePopulation", vec![0.into()], [1, 4, 1, 1], wide_in.clone(), None);
    let wide_std = single_unary("stdPopulation", vec![0.into()], [1, 4, 1, 1], wide_in, None);
    assert!((wide_var.outputs.values[0] - 1e38).abs() < 1e32);
    assert!((wide_std.outputs.values[0] - 1e19).abs() < 1e13);

    // 5. Axis reduction keepdim (4)
    let ax_in = vec![1.0.into(), 3.0.into(), 2.0.into(), 6.0.into()];
    assert_eq!(single_unary("sumAxis", vec![0.into(), 3.into()], [1, 1, 2, 2], ax_in.clone(), None).outputs.values, vec![4.0, 8.0]);
    assert_eq!(single_unary("meanAxis", vec![0.into(), 3.into()], [1, 1, 2, 2], ax_in.clone(), None).outputs.values, vec![2.0, 4.0]);
    assert_eq!(single_unary("minAxis", vec![0.into(), 3.into()], [1, 1, 2, 2], ax_in.clone(), None).outputs.values, vec![1.0, 2.0]);
    assert_eq!(single_unary("maxAxis", vec![0.into(), 3.into()], [1, 1, 2, 2], ax_in, None).outputs.values, vec![3.0, 6.0]);

    // 6. Probability (4)
    let norm_p = single_unary("normalize", vec![0.into()], [1, 2, 1, 1], vec![1.0.into(), 3.0.into()], None);
    assert_eq!(norm_p.outputs.values, vec![0.25, 0.75]);
    let ent = single_unary("entropy", vec![0.into()], [1, 2, 1, 1], vec![0.5.into(), 0.5.into()], None);
    assert!((ent.outputs.values[0] - std::f32::consts::LN_2).abs() < 1e-5);
    let ce = single_binary("crossEntropy", vec![0.into()], [1, 2, 1, 1], vec![0.5.into(), 0.5.into()], [1, 2, 1, 1], vec![0.75.into(), 0.25.into()]);
    let kl = single_binary("klDivergence", vec![0.into()], [1, 2, 1, 1], vec![0.5.into(), 0.5.into()], [1, 2, 1, 1], vec![0.75.into(), 0.25.into()]);
    let exp_ce = -0.5 * 0.75_f32.ln() - 0.5 * 0.25_f32.ln();
    assert!((ce.outputs.values[0] - exp_ce).abs() < 1e-5);
    assert!((kl.outputs.values[0] - (exp_ce - std::f32::consts::LN_2)).abs() < 1e-5);

    // 7. Comparison & Index source (2)
    let le = single_binary("lessEqual01", vec![0.into()], [1, 4, 1, 1], vec![1.0.into(), 2.0.into(), 3.0.into(), 0.0.into()], [1, 4, 1, 1], vec![2.0.into(), 2.0.into(), 1.0.into(), (-1.0).into()]);
    assert_eq!(le.outputs.values, vec![1.0, 1.0, 0.0, 0.0]);
    let idx = single_unary("indicesLike", vec![0.into(), 1.into()], [2, 3, 1, 1], vec![99.0.into(); 6], None);
    assert_eq!(idx.outputs.values, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);

    // 8. Parity 13 math constructors (11 numeric + 2 sample stats)
    let sin_out = single_unary("sin", vec![0.into()], [1, 2, 1, 1], vec![0.0.into(), (std::f64::consts::FRAC_PI_2).into()], None);
    assert!(sin_out.outputs.values[0].abs() < 1e-6 && (sin_out.outputs.values[1] - 1.0).abs() < 1e-5);
    let cos_out = single_unary("cos", vec![0.into()], [1, 2, 1, 1], vec![0.0.into(), std::f64::consts::PI.into()], None);
    assert!((cos_out.outputs.values[0] - 1.0).abs() < 1e-6 && (cos_out.outputs.values[1] + 1.0).abs() < 1e-5);
    let tan_out = single_unary("tan", vec![0.into()], [1, 2, 1, 1], vec![0.0.into(), std::f64::consts::FRAC_PI_4.into()], None);
    assert!(tan_out.outputs.values[0].abs() < 1e-6 && (tan_out.outputs.values[1] - 1.0).abs() < 1e-5);

    assert_eq!(single_unary("floor", vec![0.into()], [1, 3, 1, 1], vec![2.9.into(), (-2.1).into(), 3.0.into()], None).outputs.values, vec![2.0, -3.0, 3.0]);
    assert_eq!(single_unary("ceil", vec![0.into()], [1, 3, 1, 1], vec![2.1.into(), (-2.9).into(), 3.0.into()], None).outputs.values, vec![3.0, -2.0, 3.0]);
    let rnd_out = single_unary("round", vec![0.into()], [1, 3, 1, 1], vec![2.5.into(), 3.5.into(), (-2.5).into()], None);
    assert_eq!(rnd_out.outputs.values, vec![2.0, 4.0, -2.0]);

    let at2_out = single_binary("atan2", vec![0.into()], [1, 2, 1, 1], vec![1.0.into(), (-1.0).into()], [1, 2, 1, 1], vec![1.0.into(), 0.0.into()]);
    assert!((at2_out.outputs.values[0] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
    assert!((at2_out.outputs.values[1] + std::f32::consts::FRAC_PI_2).abs() < 1e-5);

    let pow_out = single_binary("powf", vec![0.into()], [1, 2, 1, 1], vec![2.0.into(), 9.0.into()], [1, 2, 1, 1], vec![3.0.into(), 0.5.into()]);
    assert!((pow_out.outputs.values[0] - 8.0).abs() < 1e-5 && (pow_out.outputs.values[1] - 3.0).abs() < 1e-5);

    let hyp_out = single_binary("hypot", vec![0.into()], [1, 2, 1, 1], vec![3.0.into(), 1e38.into()], [1, 2, 1, 1], vec![4.0.into(), 1e38.into()]);
    assert!((hyp_out.outputs.values[0] - 5.0).abs() < 1e-5);
    assert!((hyp_out.outputs.values[1] - 1e38 * std::f32::consts::SQRT_2).abs() < 1e32);

    let fma_out = single_ternary(
        "fma",
        vec![0.into()],
        [1, 3, 1, 1],
        vec![2.0.into(), (-3.0).into(), 0.5.into()],
        [1, 3, 1, 1],
        vec![4.0.into(), 5.0.into(), 6.0.into()],
        [1, 3, 1, 1],
        vec![1.5.into(), 2.0.into(), (-1.0).into()],
    );
    assert_eq!(fma_out.outputs.values, vec![9.5, -13.0, 2.0]);

    let fin_out = single_unary("allFinite", vec![0.into()], [1, 3, 1, 1], vec![1.0.into(), (-2.0).into(), 0.0.into()], None);
    assert_eq!(fin_out.outputs.shape, [1, 1, 1, 1]);
    assert_eq!(fin_out.outputs.values, vec![1.0]);

    let sample_in = vec![2.0.into(), 4.0.into(), 6.0.into(), 8.0.into()];
    let vs = single_unary("varianceSample", vec![0.into()], [1, 4, 1, 1], sample_in.clone(), None);
    let ss = single_unary("stdSample", vec![0.into()], [1, 4, 1, 1], sample_in, None);
    assert!((vs.outputs.values[0] - (20.0 / 3.0)).abs() < 1e-5);
    assert!((ss.outputs.values[0] - (20.0f32 / 3.0).sqrt()).abs() < 1e-5);
    let wide_sample_in = vec![1e19.into(), (-1e19).into(), 1e19.into(), (-1e19).into()];
    let wide_vs = single_unary("varianceSample", vec![0.into()], [1, 4, 1, 1], wide_sample_in.clone(), None);
    let wide_ss = single_unary("stdSample", vec![0.into()], [1, 4, 1, 1], wide_sample_in, None);
    assert!((wide_vs.outputs.values[0] - (4e38_f64 / 3.0) as f32).abs() < 1e32);
    assert!((wide_ss.outputs.values[0] - (4e38_f64 / 3.0).sqrt() as f32).abs() < 1e13);
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
    let gd = solve_gd_autodiff(&ml, Some(&tf), None, 20, 0.05, false).unwrap();
    assert!(gd.final_loss < gd.start_loss);
    let es = fit_es(&ml, &tf, 15, 12, 0.2, 0.2, 7, false, 1e-6).unwrap();
    assert!(es.final_loss < es.start_loss);
}

#[test]
fn batch1_trace_rollout_profile_and_preconditioned() {
    use math_draw::profile::{compare_specs_perf, profile_spec};
    use math_draw::rollout::{rollout_fixpoint, rollout_scan, rollout_unroll};
    use math_draw::trace::{compare_trace_hashes, trace_hash_spec};

    // 1. trace-hash budget skipping check
    let raw = r#"{"inputs":{"0":[1.0,-2.0]},"layers":[{"args":[0,2,2,false],"constructor":"linear"},{"args":[1],"constructor":"relu"}],"numSlots":3,"outputSlot":2,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]},{"kind":"unary","layer":1,"slots":[1,2]}],"weights":{"0":[1,0,0,1]}}"#;
    let spec = load_spec_str(raw, Path::new(".")).unwrap().spec;
    let rep_skip = trace_hash_spec(&spec, 4).unwrap();
    assert_eq!(rep_skip.skipped_steps, 2);
    let rep_full = trace_hash_spec(&spec, 1024).unwrap();
    assert_eq!(rep_full.captured_steps, 2);

    let cmp_eq = compare_trace_hashes(&spec, &spec, 1024, true).unwrap();
    assert!(cmp_eq.hash_chain_equal);
    assert!(cmp_eq.first_divergent_step.is_none());

    // 2. rollout with non-converging / shape mismatch fails closed cleanly
    let bad_rollout = rollout_fixpoint(&spec, None, 1e-6, 1, None);
    // [1, -2] -> relu -> [1, 0], delta = 2.0 > 1e-6 so converged == false
    assert!(!bad_rollout.unwrap().converged);

    // 3. scan + unroll agreement on step inputs
    let r2 = r#"{
      "numSlots": 3,
      "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }, { "constructor": "add", "args": [1] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 2] }, { "kind": "binary", "layer": 1, "slots": [2, 1, 2] }],
      "outputSlot": 2,
      "ports": [{ "slot": 0, "role": "state", "shape": [1, 2, 1, 1] }, { "slot": 1, "role": "context", "shape": [1, 2, 1, 1] }],
      "weights": { "0": [0.5, 0.0, 0.0, 0.5] },
      "inputs": { "0": [0.0, 0.0], "1": [1.0, 2.0] }
    }"#;
    let rs = load_spec_str(r2, Path::new(".")).unwrap().spec;
    let sc = rollout_scan(&rs, None, 3, Some("state"), None).unwrap();
    let ur = rollout_unroll(&rs, None, 3, Some("state")).unwrap();
    let ur_out = run_spec(&ur.unrolled_spec).unwrap();
    assert_eq!(sc.final_state_sha256, ur_out.trace_sha256);

    // 4. profile & compare
    let p = profile_spec(&spec, 2, 5).unwrap();
    assert_eq!(p.steps_count, 2);
    let cp = compare_specs_perf(&spec, &spec, 2, 5).unwrap();
    assert!(cp.outputs_match);

    // 5. preconditioned GD on ill-conditioned weighted linear
    let tf: TargetsFile = serde_json::from_str(r#"{"cases":[{"inputs":{"0":[1.0,-2.0]},"expected":[2.0,-4.0]}]}"#).unwrap();
    let gd_pre = solve_gd_autodiff(&spec, Some(&tf), Some(0), 20, 0.1, true).unwrap();
    assert!(gd_pre.preconditioned);
    assert!(gd_pre.final_loss < gd_pre.start_loss);
}

#[test]
fn batch2_gradcheck_quant_mutate_and_shard() {
    use math_draw::gradcheck::gradcheck_spec;
    use math_draw::mutate::{MutationRequest, mutate_spec};
    use math_draw::quant::{BudgetMode, QuantScheme, f32_to_f16_roundtrip, quant_budget, quant_verify};
    use math_draw::shard::shard_spec;

    // 1. IEEE-754 f16 roundtrip sanity
    assert_eq!(f32_to_f16_roundtrip(0.0), 0.0);
    assert_eq!(f32_to_f16_roundtrip(1.0), 1.0);
    assert_eq!(f32_to_f16_roundtrip(-2.5), -2.5);
    assert!((f32_to_f16_roundtrip(0.3333333) - 0.33325195).abs() < 1e-4);

    // 2. gradcheck detects mismatch cleanly (fail-closed)
    let raw = r#"{"inputs":{"0":[1.0,2.0]},"layers":[{"args":[0,2,1,false],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,2,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}],"weights":{"0":[3.0,-4.0]}}"#;
    let spec = load_spec_str(raw, Path::new(".")).unwrap().spec;
    let bad_an = json!({ "0": [[9.0, -4.0]] });
    let gc_bad = gradcheck_spec(&spec, Some(&bad_an), 1e-3, 1e-3, false, false).unwrap();
    assert!(!gc_bad.ok);
    let gc_ok = gradcheck_spec(&spec, None, 1e-3, 1e-2, true, true).unwrap();
    assert!(gc_ok.ok);
    assert!(gc_ok.autodiff_check.unwrap().ok);

    // 3. quant budget bound mode refuses unbounded `div` loudly
    let div_spec_raw = r#"{
      "numSlots": 4,
      "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }, { "constructor": "div", "args": [1] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 2] }, { "kind": "binary", "layer": 1, "slots": [2, 1, 3] }],
      "outputSlot": 3,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }, { "slot": 1, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [1, 0, 0, 1] },
      "inputs": { "0": [1.0, 2.0], "1": [2.0, 2.0] }
    }"#;
    let div_spec = load_spec_str(div_spec_raw, Path::new(".")).unwrap().spec;
    assert!(quant_budget(&div_spec, None, 0.1, QuantScheme::Int8, BudgetMode::Bound).is_err());
    // While measured mode works cleanly on bounded inputs
    assert!(quant_budget(&div_spec, None, 0.1, QuantScheme::Int8, BudgetMode::Measured).unwrap().ok);
    assert!(quant_verify(&div_spec, QuantScheme::Q8PerBlock, None, 1e-2, 1e-2).unwrap().ok);

    // 4. mutate refuses commit when no branch passes equivalence gate
    let req_fail: MutationRequest = serde_json::from_value(json!({
        "commit": true,
        "edits": [{ "op": "setWeights", "layer_index": 0, "values": [10.0, 10.0] }]
    })).unwrap();
    let mr = mutate_spec(&spec, &req_fail, None, 1e-5, 1e-5).unwrap();
    assert!(!mr.ok && !mr.committed && !mr.equivalent);

    // 5. shard fail-closed when in_dim exceeds max_weights
    let wide_raw = r#"{"inputs":{"0":[1.0,2.0,3.0,4.0]},"layers":[{"args":[0,4,2,true],"constructor":"linear"}],"numSlots":2,"outputSlot":1,"ports":[{"shape":[1,4,1,1],"slot":0}],"steps":[{"kind":"unary","layer":0,"slots":[0,1]}],"weights":{"0":[1,1,1,1,1,1,1,1,0,0]}}"#;
    let wide = load_spec_str(wide_raw, Path::new(".")).unwrap().spec;
    assert!(shard_spec(&wide, 3).unwrap_err().contains("unshardable"));
}

#[test]
fn batch3_weight_lab_linear_equiv_cost_model_and_harness() {
    use math_draw::cost_model::run_cost_model;
    use math_draw::harness::{
        export_subagent_branches, get_capability_catalog, merge_subagent_branches,
        resolve_artifact_uri, store_artifact,
    };
    use math_draw::linear_equiv::equiv_linear;
    use math_draw::verify::{FuzzConfig, zoom_fuzz_specs};
    use math_draw::weight_lab::{
        WeightProbe, WeightRequires, WeightSet, certify_weight_set, run_weight_lab_selftest,
        spectral_radius,
    };

    // 1. weight-lab selftest + anti-diagonal oscillatory spectral radius
    let st = run_weight_lab_selftest().unwrap();
    assert_eq!(st["ok"], json!(true));
    // Anti-diagonal [[0, 5], [0.05, 0]] has true eigenvalues +-0.5 (not single-step spike 5.0)
    let (rho_osc, _) = spectral_radius(&[vec![0.0, 5.0], vec![0.05, 0.0]], 200).unwrap();
    assert!((rho_osc - 0.5).abs() < 1e-6);

    // 2. weight-lab certify refuses when probe error exceeds probe_tol
    let bad_probe_ws = WeightSet {
        name: Some("bad_probe".into()),
        kind: "map".into(),
        w: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        b: vec![0.0, 0.0],
        probes: vec![WeightProbe {
            input: vec![1.0, 2.0],
            want: Some(vec![99.0, 99.0]),
        }],
        probe_tol: Some(1e-4),
        requires: WeightRequires::default(),
    };
    let cert_bad = certify_weight_set(&bad_probe_ws, "bad", None).unwrap();
    assert_eq!(cert_bad["ok"], json!(false));
    assert_eq!(cert_bad["verdict"], json!("REFUSED"));

    // 3. linear-equiv proves linear-branch-fusion & linear-chain-fusion exactness, and rejects non-linear
    let branch_raw = r#"{
      "numSlots": 4,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 2, false] },
        { "constructor": "linear", "args": [1, 2, 2, false] },
        { "constructor": "add", "args": [2] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "unary", "layer": 1, "slots": [0, 2] },
        { "kind": "binary", "layer": 2, "slots": [1, 2, 3] }
      ],
      "outputSlot": 3,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [1, 2, 3, 4], "1": [0.5, -0.5, 1.0, 2.0] },
      "inputs": { "0": [1.0, -1.0] }
    }"#;
    let b_spec = load_spec_str(branch_raw, Path::new(".")).unwrap().spec;
    let b_fused = apply_rewrite("linear-branch-fusion", &b_spec).unwrap();
    let eq = equiv_linear(&b_spec, &b_fused, 1e-7);
    assert!(eq.ok && eq.verdict == "ACCEPT");

    // 4. cost-model + zoom-fuzz
    let cm = run_cost_model(&b_spec, 3).unwrap();
    assert!(cm.ok && cm.per_op.len() == 3);
    let mut pert = b_fused.clone();
    let active_layer = pert.steps[0].layer.to_string();
    pert.weights.get_mut(&active_layer).unwrap()[0] += 0.25;
    let zf = zoom_fuzz_specs(&b_spec, &pert, &FuzzConfig { count: 6, seed: 7, lo: -3.0, hi: 3.0 }, 3).unwrap();
    assert!(zf.ok && zf.zoomed_max_abs_error > 0.0 && zf.zoomed_max_abs_error >= zf.initial_max_abs_error);

    // 5. harness artifact store integrity check + branch export/merge
    assert_eq!(get_capability_catalog(Some("unknown_domain"))["ok"], json!(false));
    let tmp_ws = std::env::temp_dir().join(format!("math_draw_test_ws_{}", std::process::id()));
    let (uri, _, art_file) = store_artifact(&[1.5, -2.5], [1, 2, 1, 1], &tmp_ws, "tensor_f32").unwrap();
    let resolved = resolve_artifact_uri(&uri, &tmp_ws).unwrap();
    assert_eq!(resolved["shape"], json!([1, 2, 1, 1]));
    // Tamper with stored artifact and verify fail-closed integrity check
    std::fs::write(&art_file, "{\"tampered\":true}\n").unwrap();
    assert!(
        resolve_artifact_uri(&uri, &tmp_ws)
            .unwrap_err()
            .contains("artifact_integrity_failed")
    );

    let exp = export_subagent_branches(&b_spec, &tmp_ws).unwrap();
    assert_eq!(exp["ok"], json!(true));
    let f_good = tmp_ws.join("good_cand.json");
    let f_bad = tmp_ws.join("bad_cand.json");
    std::fs::write(&f_good, serde_json::to_string(&b_fused).unwrap()).unwrap();
    std::fs::write(&f_bad, serde_json::to_string(&pert).unwrap()).unwrap();
    let merged = merge_subagent_branches(
        &b_spec,
        &[
            f_bad.display().to_string(),
            f_good.display().to_string(),
        ],
        1e-4,
    )
    .unwrap();
    let _ = std::fs::remove_dir_all(&tmp_ws);
    assert_eq!(merged["ok"], json!(true));
    assert_eq!(merged["promoted_winner"]["branch_id"], json!("merge.2"));
    assert_eq!(merged["promoted_winner"]["steps_count"], json!(1));
}

#[test]
fn ported_math_ops_domain_and_shape_fail_closed() {
    use math_draw::quant::{BudgetMode, QuantScheme, quant_budget};

    let expect_run_err = |raw: serde_json::Value, needle: &str| {
        let s: SpecV1 = serde_json::from_value(raw).unwrap();
        let err = run_spec(&s).unwrap_err();
        assert!(err.contains(needle), "expected '{needle}' in '{err}'");
    };
    let expect_invalid = |raw: serde_json::Value| {
        let s: SpecV1 = serde_json::from_value(raw.clone()).unwrap();
        assert!(!validate_spec(&s, false).valid, "expected invalid spec: {raw}");
    };

    // Numeric domain failures
    expect_run_err(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "log", "args": [0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "inputs": { "0": [1.0, 0.0] }
    }), "value must be > 0");
    expect_run_err(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "exp", "args": [0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
        "inputs": { "0": [100.0] }
    }), "exp output: non-finite");
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "clamp", "args": [0, 2.0, 1.0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
        "inputs": { "0": [0.5] }
    }));

    // Tensor transform static rejections
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "reshape", "args": [0, [1, 2, 2, 2]] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 3] }],
        "inputs": { "0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }
    }));
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "permute", "args": [0, [0, 1, 1, 3]] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 3] }],
        "inputs": { "0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }
    }));
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "slice", "args": [0, [0, 0, 0, 2], [1, 2, 2, 2]] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 2, 3] }]
    }));
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "selectAxis", "args": [0, 1, [2]] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 3] }]
    }));
    expect_invalid(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "expandLike", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [2, 2, 1, 1] },
            { "slot": 1, "shape": [2, 3, 1, 1] }
        ]
    }));

    // Linalg & Reduction & Index & Comparison invalid args/shapes
    expect_invalid(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "cosineSimilarity", "args": [0, 0.0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
        ]
    }));
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "sumAxis", "args": [0, 4] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }]
    }));
    expect_invalid(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "lessEqual01", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 1, 2, 1] }
        ]
    }));
    expect_invalid(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "indicesLike", "args": [0, 1] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 16777218, 1, 1] }]
    }));

    // Probability domain rejections
    expect_run_err(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "normalize", "args": [0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "inputs": { "0": [-1.0, 2.0] }
    }), "must be >= 0");
    expect_run_err(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "entropy", "args": [0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "inputs": { "0": [0.2, 0.2] }
    }), "must sum to 1");
    expect_run_err(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "klDivergence", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
        ],
        "inputs": { "0": [1.0, 0.0], "1": [0.0, 1.0] }
    }), "zero support");

    // Quant budget bound mode: succeeds with bounded `clamp` downstream, refuses `exp`
    let q_clamp: SpecV1 = serde_json::from_value(json!({
        "numSlots": 3,
        "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "clamp", "args": [1, -2.0, 2.0] }
        ],
        "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] }
        ],
        "outputSlot": 2,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "weights": { "0": [0.5, -0.5, 0.25, 0.75] },
        "inputs": { "0": [1.0, -1.0] }
    })).unwrap();
    let qb_ok = quant_budget(&q_clamp, None, 0.1, QuantScheme::Q8PerBlock, BudgetMode::Bound).unwrap();
    assert!(qb_ok.ok);

    let q_exp: SpecV1 = serde_json::from_value(json!({
        "numSlots": 3,
        "layers": [
            { "constructor": "linear", "args": [0, 2, 2, false] },
            { "constructor": "exp", "args": [1] }
        ],
        "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] }
        ],
        "outputSlot": 2,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "weights": { "0": [0.5, -0.5, 0.25, 0.75] },
        "inputs": { "0": [1.0, -1.0] }
    })).unwrap();
    assert!(quant_budget(&q_exp, None, 0.1, QuantScheme::Q8PerBlock, BudgetMode::Bound).is_err());
}

#[test]
fn parity_13_math_ops_fail_closed_and_ternary_pipelines() {
    use burn::tensor::{Device, Tensor, TensorData};
    use math_draw::cost_model::run_cost_model;
    use math_draw::layers::{RuntimeLayer, SlotValue, exec_layer};
    use math_draw::mutate::{MutationRequest, mutate_spec};
    use math_draw::quant::{BudgetMode, QuantScheme, quant_budget};
    use math_draw::trace::trace_hash_spec;

    let expect_run_err = |raw: serde_json::Value, needle: &str| {
        let s: SpecV1 = serde_json::from_value(raw).unwrap();
        let err = run_spec(&s).unwrap_err();
        assert!(err.contains(needle), "expected '{needle}' in '{err}'");
    };
    let expect_invalid = |raw: serde_json::Value| {
        let s: SpecV1 = serde_json::from_value(raw.clone()).unwrap();
        assert!(!validate_spec(&s, false).valid, "expected invalid spec: {raw}");
    };

    // 1. powf fail-closed domains & overflows
    expect_run_err(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "powf", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [-8.0], "1": [0.33333334] }
    }), "non-finite");
    expect_run_err(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "powf", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [0.0], "1": [-1.0] }
    }), "non-finite");
    expect_run_err(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "powf", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [1e38], "1": [2.0] }
    }), "non-finite");

    // 2. hypot overflow above f32::MAX fails closed (while (1e38, 1e38) passes via scaled sqrt)
    expect_run_err(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "hypot", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 1, 1, 1] },
            { "slot": 1, "shape": [1, 1, 1, 1] }
        ],
        "inputs": { "0": [3e38], "1": [3e38] }
    }), "non-finite");

    // 3. sample variance/std fail closed when features < 2
    for ctor in ["varianceSample", "stdSample"] {
        expect_invalid(json!({
            "numSlots": 2,
            "layers": [{ "constructor": ctor, "args": [0] }],
            "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
            "outputSlot": 1,
            "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
            "inputs": { "0": [42.0] }
        }));
    }

    // 4. fma structural validation (shape mismatch, wrong kind, wrong slot arity)
    expect_invalid(json!({
        "numSlots": 4,
        "layers": [{ "constructor": "fma", "args": [0] }],
        "steps": [{ "kind": "ternary", "layer": 0, "slots": [0, 1, 2, 3] }],
        "outputSlot": 3,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 3, 1, 1] },
            { "slot": 2, "shape": [1, 2, 1, 1] }
        ]
    }));
    expect_invalid(json!({
        "numSlots": 4,
        "layers": [{ "constructor": "fma", "args": [0] }],
        "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2, 3] }],
        "outputSlot": 3,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] },
            { "slot": 2, "shape": [1, 2, 1, 1] }
        ]
    }));
    expect_invalid(json!({
        "numSlots": 3,
        "layers": [{ "constructor": "fma", "args": [0] }],
        "steps": [{ "kind": "ternary", "layer": 0, "slots": [0, 1, 2] }],
        "outputSlot": 2,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
        ]
    }));

    // 5. allFinite direct tensor check with non-finite entry returns 0.0 (no error)
    let dev = Device::flex();
    let bad_t = Tensor::<4>::from_data(TensorData::new(vec![1.0f32, f32::NAN], [1, 2, 1, 1]), &dev);
    let out_fin = exec_layer(&RuntimeLayer::AllFinite, &[SlotValue::F32(bad_t)], &dev).unwrap();
    let fin_vals = out_fin.as_f32("test").unwrap().into_data().try_to_vec::<f32>().unwrap();
    assert_eq!(fin_vals, vec![0.0]);

    // 6. v2 composer + cost-model + trace-hash + quant_budget + mutate with ternary fma
    let v2_fma = r#"{
        "schema": "burn-math.recursive-spec.v2",
        "ports": {
            "a": { "slot": 0, "shape": [1, 2, 1, 1] },
            "b": { "slot": 1, "shape": [1, 2, 1, 1] },
            "c": { "slot": 2, "shape": [1, 2, 1, 1] }
        },
        "inputs": {
            "a": [1.0, -2.0],
            "b": [3.0, 4.0],
            "c": [0.5, 1.5]
        },
        "steps": [
            { "op": "linear", "args": [2, 2, false], "weights": [1.0, 0.0, 0.0, 1.0], "in": ["a"], "out": "la" },
            { "op": "fma", "in": ["la", "b", "c"], "out": "y" }
        ],
        "output": "y"
    }"#;
    let compiled = load_spec_str(v2_fma, Path::new(".")).unwrap().spec;
    assert_eq!(compiled.steps[1].kind, "ternary");
    let res = run_spec(&compiled).unwrap();
    assert_eq!(res.outputs.values, vec![3.5, -6.5]);

    let cm = run_cost_model(&compiled, 3).unwrap();
    assert!(cm.ok && cm.per_op.len() == 2 && cm.per_op.iter().any(|r| r.layer == "fma" && r.kind == "ternary"));

    let th = trace_hash_spec(&compiled, 1024).unwrap();
    assert!(th.ok && th.captured_steps == 2);

    let qb_bound = quant_budget(&compiled, None, 0.5, QuantScheme::Q8PerBlock, BudgetMode::Bound).unwrap();
    let qb_meas = quant_budget(&compiled, None, 0.5, QuantScheme::Int8, BudgetMode::Measured).unwrap();
    assert!(qb_bound.ok && qb_meas.ok);

    // Mutate: fuse mul(a,b) + add(t,c) into a single equivalent ternary fma step
    let base_raw = r#"{
        "numSlots": 5,
        "layers": [
            { "constructor": "mul", "args": [0] },
            { "constructor": "add", "args": [1] }
        ],
        "steps": [
            { "kind": "binary", "layer": 0, "slots": [0, 1, 3] },
            { "kind": "binary", "layer": 1, "slots": [3, 2, 4] }
        ],
        "outputSlot": 4,
        "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] },
            { "slot": 2, "shape": [1, 2, 1, 1] }
        ],
        "inputs": { "0": [1.5, -2.0], "1": [2.0, 3.0], "2": [0.25, -1.5] }
    }"#;
    let base = load_spec_str(base_raw, Path::new(".")).unwrap().spec;
    let mreq: MutationRequest = serde_json::from_value(json!({
        "commit": true,
        "edits": [
            { "op": "removeStep", "index": 1 },
            { "op": "replaceStep", "index": 0, "layer": { "constructor": "fma", "args": [2] }, "slots": [0, 1, 2, 4] }
        ]
    })).unwrap();
    let mrep = mutate_spec(&base, &mreq, None, 1e-6, 1e-6).unwrap();
    assert!(mrep.ok && mrep.committed, "mutate failed: {:?}", mrep.branches);
    let mutated = mrep.resulting_spec.as_ref().unwrap();
    assert_eq!(mutated.steps.len(), 1);
    assert_eq!(mutated.steps[0].kind, "ternary");
}
