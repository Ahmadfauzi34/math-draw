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

#[test]
fn es_rng_vectors_mu_lambda_and_strict_validation() {
    use math_draw::recurse::{
        AnyRng, EsStrategyKind, FitEsConfig, Mulberry32, Pcg32, Philox4x32, RngKind, diversity,
        fit_es_with_config,
    };

    // 1. Published PCG32 pcg-c-basic test vector (seed=42, seq=54)
    let mut pcg = Pcg32::new(42, 54);
    assert_eq!(
        [
            pcg.next_u32(),
            pcg.next_u32(),
            pcg.next_u32(),
            pcg.next_u32(),
            pcg.next_u32()
        ],
        [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b]
    );

    // 2. Published Philox4x32-10 Random123 test vectors
    let mut ph0 = Philox4x32::with_key([0, 0]);
    assert_eq!(
        [ph0.next_u32(), ph0.next_u32(), ph0.next_u32(), ph0.next_u32()],
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    );
    let mut ph_ff = Philox4x32::with_key_counter(
        [0xFFFF_FFFF, 0xFFFF_FFFF],
        [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF],
    );
    assert_eq!(
        [ph_ff.next_u32(), ph_ff.next_u32(), ph_ff.next_u32(), ph_ff.next_u32()],
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
    );
    let mut ph_12 = Philox4x32::with_key_counter([1, 2], [3, 4, 5, 6]);
    assert_eq!(
        [ph_12.next_u32(), ph_12.next_u32(), ph_12.next_u32(), ph_12.next_u32()],
        [0x8850da9c, 0xc159ff4c, 0x75752dca, 0x7d607cf9]
    );

    // 3. Philox seek equivalence & fail-closed seek on non-Philox PRNGs
    let mut seq = Philox4x32::new(7);
    let seq_out: Vec<u32> = (0..12).map(|_| seq.next_u32()).collect();
    let mut jumped = Philox4x32::new(7);
    jumped.seek(2);
    let jumped_out: Vec<u32> = (0..4).map(|_| jumped.next_u32()).collect();
    assert_eq!(jumped_out, seq_out[8..12]);

    let mut pcg_any = AnyRng::new(RngKind::Pcg32, 7);
    assert!(pcg_any.philox_seek(0).is_err());
    let mut ph_any = AnyRng::new(RngKind::Philox4x32, 7);
    assert!(ph_any.philox_seek(2).is_ok());

    // 4. Mulberry32 oracle vector (matching ai_agent_builder::es::rng::Rng)
    let mut mb = Mulberry32::new(7);
    assert_eq!(mb.next_u32(), 0x02ff_152c);

    // 5. Ragged diversity returns NaN rather than panicking
    assert!(diversity(&[vec![1.0, 2.0], vec![1.0]]).is_nan());
    assert_eq!(diversity(&[vec![1.0, 2.0]]), 0.0);

    // 6. Convergence + determinism across MuLambda & all PRNGs + strict validation
    let spec: SpecV1 = serde_json::from_value(json!({
        "numSlots": 2,
        "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "weights": { "0": [0.1, 0.0, 0.0, 0.1] },
        "inputs": { "0": [1.0, 1.0] }
    }))
    .unwrap();
    let tf: TargetsFile = serde_json::from_value(json!({
        "cases": [
            { "inputs": { "0": [1.0, 0.0] }, "expected": [1.5, 0.0] },
            { "inputs": { "0": [0.0, 1.0] }, "expected": [0.0, -1.5] }
        ]
    }))
    .unwrap();

    for rng in [
        RngKind::Xorshift64,
        RngKind::Mulberry32,
        RngKind::Pcg32,
        RngKind::Philox4x32,
    ] {
        let cfg = FitEsConfig {
            generations: 18,
            pop: 10,
            mu: Some(3),
            sigma: 0.25,
            lr: 0.0,
            seed: 123,
            strategy: EsStrategyKind::MuLambda,
            rng,
            sensitivity_guided: false,
            freeze_threshold: 1e-6,
        };
        let r1 = fit_es_with_config(&spec, &tf, &cfg).unwrap();
        let r2 = fit_es_with_config(&spec, &tf, &cfg).unwrap();
        assert!(r1.ok && r1.final_loss < r1.start_loss);
        assert_eq!(r1.best_weights, r2.best_weights);
        assert_eq!(r1.trajectory.len(), 18);
        assert!(r1.trajectory.iter().all(|t| t.diversity.is_finite() && t.worst_loss >= t.best_loss));
    }

    // Fail-closed hyperparameter checks
    let odd_openes = FitEsConfig {
        pop: 7,
        strategy: EsStrategyKind::OpenEs,
        ..FitEsConfig::default()
    };
    assert!(fit_es_with_config(&spec, &tf, &odd_openes).is_err());

    let bad_mu = FitEsConfig {
        pop: 6,
        mu: Some(7),
        strategy: EsStrategyKind::MuLambda,
        ..FitEsConfig::default()
    };
    assert!(fit_es_with_config(&spec, &tf, &bad_mu).is_err());

    let bad_sigma = FitEsConfig {
        sigma: 0.0,
        ..FitEsConfig::default()
    };
    assert!(fit_es_with_config(&spec, &tf, &bad_sigma).is_err());

    let bad_lr = FitEsConfig {
        lr: f64::NAN,
        strategy: EsStrategyKind::OpenEs,
        ..FitEsConfig::default()
    };
    assert!(fit_es_with_config(&spec, &tf, &bad_lr).is_err());

    assert!(EsStrategyKind::parse("bogus").is_err());
    assert!(RngKind::parse("bogus").is_err());
}

#[test]
fn program_bundle_binding_replay_and_tamper_detection() {
    use math_draw::bundle::{
        apply_bound_flat_weights, compute_parameter_binding, compute_program_identity,
        export_bundle, read_bound_flat_weights, verify_bundle,
    };

    // Weight-tied graph: layer 0 (linear 2->2, 6 params) used twice + layer 1 (relu, stateless) + layer 2 (linear 2->1, 3 params)
    let spec: SpecV1 = serde_json::from_value(json!({
        "numSlots": 5,
        "layers": [
            { "constructor": "linear", "args": [11, 2, 2, true] },
            { "constructor": "relu", "args": [12] },
            { "constructor": "linear", "args": [13, 2, 1, true] }
        ],
        "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 2] },
            { "kind": "unary", "layer": 0, "slots": [2, 3] },
            { "kind": "unary", "layer": 2, "slots": [3, 4] }
        ],
        "outputSlot": 4,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "weights": {
            "0": [1.0, 0.5, -0.5, 1.0, 0.1, -0.1],
            "2": [2.0, -1.0, 0.25]
        },
        "inputs": { "0": [1.5, -0.5] }
    }))
    .unwrap();

    // 1. Unique-first-use binding deduplicates repeated layer 0
    let binding = compute_parameter_binding(&spec).unwrap();
    assert_eq!(binding.ordering, "unique_first_use_graph_plan");
    assert_eq!(binding.owners.len(), 2);
    assert_eq!(binding.owners[0].layer_index, 0);
    assert_eq!(binding.owners[0].offset, 0);
    assert_eq!(binding.owners[0].len, 6);
    assert_eq!(binding.owners[1].layer_index, 2);
    assert_eq!(binding.owners[1].offset, 6);
    assert_eq!(binding.owners[1].len, 3);
    assert_eq!(binding.total_params, 9);
    assert_eq!(binding.trainable_params, 9);
    assert_eq!(binding.stateless_layers_count, 1);

    // 2. Round-trip read/apply flat weights preserves structural program_identity
    let flat = read_bound_flat_weights(&spec, &binding).unwrap();
    assert_eq!(flat.len(), 9);
    let shifted: Vec<f32> = flat.iter().map(|&v| v + 0.25).collect();
    let updated = apply_bound_flat_weights(&spec, &binding, &shifted).unwrap();
    assert_eq!(compute_program_identity(&updated), binding.program_identity);
    assert_eq!(read_bound_flat_weights(&updated, &binding).unwrap(), shifted);

    // Non-finite or wrong-length flat vector fails closed
    let mut bad_nan = shifted.clone();
    bad_nan[3] = f32::NAN;
    assert!(apply_bound_flat_weights(&spec, &binding, &bad_nan).is_err());
    assert!(apply_bound_flat_weights(&spec, &binding, &shifted[..8]).is_err());

    // 3. Export & verify clean bundle (with custom user attestation)
    let bundle = export_bundle(&spec, true, &["ci_release_gate".to_string()]).unwrap();
    assert_eq!(bundle.attestations.len(), 3);
    let rep = verify_bundle(&bundle);
    assert!(rep.ok, "clean bundle verification failed: {:?}", rep.checks);
    assert!(rep.replayed_execution);
    assert_eq!(rep.checked_attestations, 3);

    // 4. Tamper detection: weight tamper triggers bundle_integrity_failed + parameter_binding_mismatch + execution_receipt_mismatch
    let mut tamper_w = bundle.clone();
    tamper_w.spec.weights.insert("2".into(), vec![9.0, -1.0, 0.25]);
    let rep_w = verify_bundle(&tamper_w);
    assert!(!rep_w.ok);
    assert_eq!(rep_w.code.as_deref(), Some("bundle_integrity_failed"));
    assert!(rep_w.checks.iter().any(|c| c.name == "parameter_binding" && !c.passed));
    assert!(rep_w.checks.iter().any(|c| c.name == "execution_replay" && !c.passed));

    // 5. Tamper detection: output receipt tamper
    let mut tamper_out = bundle.clone();
    if let Some(ref mut exec) = tamper_out.reference_execution {
        exec.output_values[0] += 1.0;
    }
    let rep_out = verify_bundle(&tamper_out);
    assert!(!rep_out.ok);
    assert!(rep_out.checks.iter().any(|c| c.name == "execution_replay" && !c.passed));

    // 6. Tamper detection: attestation digest tamper
    let mut tamper_att = bundle.clone();
    tamper_att.attestations[0].digest = "sha256:0000".to_string();
    let rep_att = verify_bundle(&tamper_att);
    assert!(!rep_att.ok);
    assert!(rep_att.checks.iter().any(|c| c.name == "attestation_provenance" && !c.passed));

    // 7. Tamper detection: wrong schema
    let mut tamper_schema = bundle.clone();
    tamper_schema.schema = "burn-math.program-bundle.v999".to_string();
    let rep_schema = verify_bundle(&tamper_schema);
    assert!(!rep_schema.ok);
    assert_eq!(rep_schema.code.as_deref(), Some("invalid_bundle_schema"));
}

#[test]
fn error_handling_phase1_cli_and_library_fail_closed() {
    use math_draw::harness::{
        ensure_workspace_dir, execute_harness_workflow, store_artifact,
    };
    use math_draw::recurse::{FitEsConfig, fit_es_with_config, solve_gd_autodiff};
    use math_draw::verify::TargetsFile;
    use math_draw::weight_lab::{WeightRequires, WeightSet, certify_weight_set};
    use std::process::Command;

    let tmp_dir = std::env::temp_dir().join(format!(
        "math_draw_phase1_err_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let spec_raw = r#"{
      "numSlots": 2,
      "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
      "outputSlot": 1,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [1.0, 0.0, 0.0, 1.0] },
      "inputs": { "0": [1.0, 2.0] }
    }"#;
    let spec_path = tmp_dir.join("spec.json");
    std::fs::write(&spec_path, spec_raw).unwrap();
    let spec = load_spec_str(spec_raw, Path::new(".")).unwrap().spec;

    // --- P1 Library guarantees ---
    // 1. Missing 'expected' in TargetsFile returns Err instead of panicking
    let missing_expected: TargetsFile =
        serde_json::from_str(r#"{"cases":[{"inputs":{"0":[1.0,2.0]}}]}"#).unwrap();
    let gd_err = solve_gd_autodiff(&spec, Some(&missing_expected), Some(0), 2, 0.1, false)
        .unwrap_err();
    assert!(
        gd_err.contains("expected"),
        "expected missing-expected error in solve_gd, got: {gd_err}"
    );
    let es_err =
        fit_es_with_config(&spec, &missing_expected, &FitEsConfig::default()).unwrap_err();
    assert!(
        es_err.contains("expected"),
        "expected missing-expected error in fit_es, got: {es_err}"
    );

    // 2. Corrupted workspace-state.json fails closed in store_artifact & execute_harness_workflow
    let bad_ws_dir = tmp_dir.join("corrupted_ws");
    let ws = ensure_workspace_dir(&bad_ws_dir).unwrap();
    std::fs::write(&ws.state_file, "{not-valid-json").unwrap();
    let art_err = store_artifact(&[1.0, 2.0], [1, 2, 1, 1], &bad_ws_dir, "tensor_f32")
        .unwrap_err();
    assert!(
        art_err.contains("corrupted workspace state"),
        "expected corrupted workspace state error, got: {art_err}"
    );
    let wf = json!({
        "spec": spec_path.display().to_string(),
        "pipeline": [{ "stage": "validate" }]
    });
    let wf_err = execute_harness_workflow(&wf, &tmp_dir, Some(&bad_ws_dir)).unwrap_err();
    assert!(
        wf_err.contains("corrupted workspace state"),
        "expected corrupted workspace state error from workflow, got: {wf_err}"
    );

    // 3. Corrupted existing repertoire file fails closed in certify_weight_set (no silent overwrite)
    let bad_rep = tmp_dir.join("bad_repertoire.json");
    std::fs::write(&bad_rep, "{broken-repertoire").unwrap();
    let ws_ok = WeightSet {
        name: Some("id_set".into()),
        kind: "map".into(),
        w: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        b: vec![0.0, 0.0],
        probes: vec![],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    let cert_err = certify_weight_set(&ws_ok, "id_set", Some(&bad_rep)).unwrap_err();
    assert!(
        cert_err.contains("failed to parse repertoire"),
        "expected repertoire parse error, got: {cert_err}"
    );

    // --- P0 CLI guarantees ---
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let run_cli = |args: &[&str]| -> (i32, serde_json::Value) {
        let out = Command::new(bin).args(args).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
            panic!("CLI stdout must be valid JSON for args {args:?}: {e}\nstdout={stdout}")
        });
        (code, parsed)
    };
    let sp = spec_path.to_str().unwrap();

    // Unknown flag rejected fail-closed
    for args in [
        vec!["run", sp, "--typo"],
        vec!["validate", sp, "--bogus"],
        vec!["profile", sp, "--unknown-flag"],
        vec!["smoke", "--extra"],
        vec!["find-op", "-z"],
    ] {
        let (code, val) = run_cli(&args);
        assert_eq!(code, 2, "expected exit 2 for {args:?}");
        assert_eq!(val["ok"], json!(false));
        assert!(
            val["error"].as_str().unwrap_or("").contains("unknown flag"),
            "expected 'unknown flag' for {args:?}, got {val}"
        );
    }

    // Missing value on valued flag rejected
    for args in [
        vec!["profile", sp, "--iters"],
        vec!["verify", sp, sp, "--abs-tol", "--zoom"],
        vec!["bundle", "export", sp, "--attest"],
    ] {
        let (code, val) = run_cli(&args);
        assert_eq!(code, 2, "expected exit 2 for {args:?}");
        assert_eq!(val["ok"], json!(false));
        assert!(
            val["error"].as_str().unwrap_or("").contains("requires a value"),
            "expected 'requires a value' for {args:?}, got {val}"
        );
    }

    // Duplicate single-valued flag rejected
    let (code, val) = run_cli(&["profile", sp, "--iters", "3", "--iters", "5"]);
    assert_eq!(code, 2);
    assert!(val["error"].as_str().unwrap_or("").contains("duplicate flag"));

    // Invalid / non-finite / out-of-domain numeric flag values rejected
    for (args, needle) in [
        (vec!["profile", sp, "--iters", "abc"], "invalid --iters"),
        (vec!["linear-equiv", sp, sp, "--tol", "NaN"], "must be finite"),
        (vec!["linear-equiv", sp, sp, "--tol", "inf"], "must be finite"),
        (vec!["linear-equiv", sp, sp, "--tol", "-0.1"], "must be >= 0"),
        (vec!["gradcheck", sp, "--eps", "0"], "must be > 0"),
        (vec!["verify", sp, sp, "--fuzz", "abc"], "invalid --fuzz"),
        (vec!["verify", sp, sp, "--fuzz", "4", "--lo", "5.0", "--hi", "-5.0"], "invalid fuzz range"),
    ] {
        let (code, val) = run_cli(&args);
        assert_eq!(code, 2, "expected exit 2 for {args:?}");
        assert_eq!(val["ok"], json!(false));
        assert!(
            val["error"].as_str().unwrap_or("").contains(needle),
            "expected '{needle}' for {args:?}, got {val}"
        );
    }

    // Unexpected extra positional rejected
    let (code, val) = run_cli(&["validate", sp, sp]);
    assert_eq!(code, 2);
    assert!(val["error"].as_str().unwrap_or("").contains("unexpected positional"));

    // Corrupted --repertoire fails closed in weight-lab eval
    let expr_path = tmp_dir.join("expr.json");
    std::fs::write(&expr_path, r#"{"expr":{"ref":"id_set"}}"#).unwrap();
    let (code, val) = run_cli(&[
        "weight-lab",
        "eval",
        expr_path.to_str().unwrap(),
        "--repertoire",
        bad_rep.to_str().unwrap(),
    ]);
    assert_eq!(code, 2);
    assert!(val["error"].as_str().unwrap_or("").contains("invalid repertoire JSON"));

    // Corrupted workspace-state.json fails closed in harness workspace-status
    let (code, val) = run_cli(&[
        "harness",
        "workspace-status",
        "--workspace",
        bad_ws_dir.to_str().unwrap(),
    ]);
    assert_eq!(code, 2);
    assert!(val["error"].as_str().unwrap_or("").contains("corrupted workspace state"));

    // Valid special CLI forms still succeed cleanly with exit 0
    for args in [
        vec!["verify", sp, sp, "--fuzz"],
        vec!["verify", "--fuzz", sp, sp],
        vec!["verify", sp, sp, "--fuzz", "4", "--lo", "-3.5", "--hi", "3.5"],
        vec!["bundle", "export", sp, "--attest", "ci", "--attest", "prod"],
    ] {
        let (code, val) = run_cli(&args);
        assert_eq!(code, 0, "expected exit 0 for valid args {args:?}, got {val}");
        assert_eq!(val["ok"], json!(true));
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn error_handling_phase2_typed_codes_and_envelopes() {
    use math_draw::error::{ErrorCode, ErrorEnvelope, MathDrawError, classify_error};
    use math_draw::harness::{ensure_workspace_dir, execute_harness_workflow};
    use std::process::Command;

    // 1. Direct MathDrawError / ErrorEnvelope JSON structure & trait impls
    let err = MathDrawError::new(
        ErrorCode::CliInvalidRange,
        "invalid --tol value '-0.1': must be >= 0",
    )
    .with_stage("cli_parse")
    .with_hint("Use a non-negative finite tolerance.");
    assert_eq!(err.to_string(), "invalid --tol value '-0.1': must be >= 0");
    let as_string: String = err.clone().into();
    assert_eq!(as_string, err.message);
    let env: ErrorEnvelope = err.to_envelope();
    let env_val = serde_json::to_value(&env).unwrap();
    assert_eq!(env_val["ok"], json!(false));
    assert_eq!(env_val["code"], json!("cli_invalid_range"));
    assert_eq!(env_val["stage"], json!("cli_parse"));
    assert_eq!(
        env_val["hint"],
        json!("Use a non-negative finite tolerance.")
    );
    assert_eq!(
        env_val["error"],
        json!("invalid --tol value '-0.1': must be >= 0")
    );

    // 2. Deterministic classify_error mappings across domain errors
    let cases: &[(&str, Option<&str>, ErrorCode)] = &[
        (
            "div step 1: division by zero",
            Some("run"),
            ErrorCode::DivisionByZero,
        ),
        (
            "intMul: integer overflow",
            Some("run"),
            ErrorCode::IntOverflow,
        ),
        (
            "artifact_integrity_failed: digest mismatch",
            Some("harness.resolve"),
            ErrorCode::ArtifactIntegrityFailed,
        ),
        (
            "corrupted workspace state in workspace-state.json",
            Some("harness.workspace_status"),
            ErrorCode::WorkspaceCorrupted,
        ),
        (
            "layer 0 is unshardable under max_weights=3",
            Some("shard"),
            ErrorCode::ShardFailed,
        ),
        (
            "case 0 missing 'expected' target vector",
            Some("recurse.solve_gd"),
            ErrorCode::MissingExpectedTarget,
        ),
        (
            "fit: targets.cases must be non-empty",
            Some("recurse.fit"),
            ErrorCode::MissingExpectedTarget,
        ),
        (
            "fit: openes requires an even pop >= 2, got 7",
            Some("recurse.fit"),
            ErrorCode::EsConfigInvalid,
        ),
        (
            "fit: sigma must be finite and > 0, got 0",
            Some("fit"),
            ErrorCode::EsConfigInvalid,
        ),
        (
            "unknown ES strategy 'bogus'",
            Some("recurse.fit"),
            ErrorCode::EsConfigInvalid,
        ),
        (
            "unknown rng 'bogus'",
            Some("fit"),
            ErrorCode::EsConfigInvalid,
        ),
        (
            "quant bound mode does not support unbounded op exp",
            Some("quant"),
            ErrorCode::QuantBoundUnsupported,
        ),
        (
            "unknown workflow pipeline stage 'bogus'",
            Some("bogus"),
            ErrorCode::WorkflowFailed,
        ),
        (
            "invalid repertoire JSON: expected object",
            Some("weight_lab.eval"),
            ErrorCode::RepertoireInvalid,
        ),
        (
            "bundle signature failed",
            Some("bundle"),
            ErrorCode::BundleInvalid,
        ),
        (
            "unknown rewrite rule 'nope'",
            Some("rewrite"),
            ErrorCode::RewriteFailed,
        ),
        (
            "tensor shape mismatch at slot 2",
            Some("run"),
            ErrorCode::ShapeMismatch,
        ),
    ];
    for (msg, stage, expected_code) in cases {
        let classified = classify_error(msg, *stage);
        assert_eq!(
            classified.code, *expected_code,
            "classify_error({msg:?}, {stage:?}) returned {:?}",
            classified.code
        );
        assert_eq!(classified.stage.as_deref(), *stage);
        assert!(
            classified.hint.as_deref().is_some_and(|h| !h.is_empty()),
            "expected non-empty hint for {:?}",
            expected_code
        );
    }

    // 3. Workflow stage failure emits typed `code` and `hint` at both stage_reports and top-level
    let tmp_dir = std::env::temp_dir().join(format!(
        "math_draw_phase2_err_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let spec_raw = r#"{
      "numSlots": 2,
      "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
      "outputSlot": 1,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [1.0, 0.0, 0.0, 1.0] },
      "inputs": { "0": [1.0, 2.0] }
    }"#;
    let spec_path = tmp_dir.join("spec.json");
    std::fs::write(&spec_path, spec_raw).unwrap();

    let targets_path = tmp_dir.join("targets.json");
    std::fs::write(
        &targets_path,
        r#"{"cases":[{"inputs":{"0":[1.0,0.0]},"expected":[2.0,0.0]}]}"#,
    )
    .unwrap();

    let ws_dir = tmp_dir.join("ws");
    let wf_unknown = json!({
        "spec": spec_path.display().to_string(),
        "pipeline": [{ "stage": "bogus_stage" }]
    });
    let wf_rep = execute_harness_workflow(&wf_unknown, &tmp_dir, Some(&ws_dir)).unwrap();
    assert_eq!(wf_rep["ok"], json!(false));
    assert_eq!(wf_rep["code"], json!("workflow_failed"));
    assert_eq!(wf_rep["failed_stage"], json!("bogus_stage"));
    assert!(wf_rep["hint"].as_str().is_some_and(|h| !h.is_empty()));
    assert_eq!(wf_rep["stage_reports"][0]["code"], json!("workflow_failed"));
    assert!(
        wf_rep["stage_reports"][0]["hint"]
            .as_str()
            .is_some_and(|h| !h.is_empty())
    );

    let wf_bad_fit = json!({
        "spec": spec_path.display().to_string(),
        "pipeline": [{
            "stage": "fit",
            "targets": targets_path.display().to_string(),
            "strategy": "openes",
            "pop": 7,
            "gens": 2
        }]
    });
    let wf_fit_rep = execute_harness_workflow(&wf_bad_fit, &tmp_dir, Some(&ws_dir)).unwrap();
    assert_eq!(wf_fit_rep["ok"], json!(false));
    assert_eq!(wf_fit_rep["code"], json!("es_config_invalid"));
    assert_eq!(wf_fit_rep["failed_stage"], json!("fit"));
    assert_eq!(
        wf_fit_rep["stage_reports"][0]["code"],
        json!("es_config_invalid")
    );
    assert!(
        wf_fit_rep["stage_reports"][0]["hint"]
            .as_str()
            .is_some_and(|h| !h.is_empty())
    );

    // 4. Live CLI binary tests asserting structured JSON ErrorEnvelope fields (ok, code, stage, error, hint)
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let run_cli = |args: &[&str]| -> (i32, serde_json::Value) {
        let out = Command::new(bin).args(args).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
            panic!("CLI stdout must be valid JSON for args {args:?}: {e}\nstdout={stdout}")
        });
        (code, parsed)
    };

    let bad_rep = tmp_dir.join("bad_rep.json");
    std::fs::write(&bad_rep, "{not-json").unwrap();
    let expr_path = tmp_dir.join("expr.json");
    std::fs::write(&expr_path, r#"{"expr":{"ref":"id_set"}}"#).unwrap();

    let bad_ws_dir = tmp_dir.join("bad_ws");
    let bad_ws = ensure_workspace_dir(&bad_ws_dir).unwrap();
    std::fs::write(&bad_ws.state_file, "{broken-ws").unwrap();

    let div0_path = tmp_dir.join("div0.json");
    std::fs::write(
        &div0_path,
        r#"{
          "numSlots": 3,
          "layers": [{ "constructor": "div", "args": [0] }],
          "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
          "outputSlot": 2,
          "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
          ],
          "inputs": { "0": [1.0, 2.0], "1": [1.0, 0.0] }
        }"#,
    )
    .unwrap();

    let sp = spec_path.to_str().unwrap();
    let missing = tmp_dir.join("does_not_exist.json");
    let missing_s = missing.to_str().unwrap();
    let div0_s = div0_path.to_str().unwrap();
    let expr_s = expr_path.to_str().unwrap();
    let bad_rep_s = bad_rep.to_str().unwrap();
    let bad_ws_s = bad_ws_dir.to_str().unwrap();

    let cli_checks: Vec<(Vec<&str>, &str, &str)> = vec![
        (vec!["nope-cmd"], "cli_unknown_command", "cli_dispatch"),
        (vec!["run", sp, "--typo"], "cli_unknown_flag", "cli_parse"),
        (vec!["profile", sp, "--iters"], "cli_missing_value", "cli_parse"),
        (
            vec!["profile", sp, "--iters", "2", "--iters", "3"],
            "cli_duplicate_flag",
            "cli_parse",
        ),
        (
            vec!["profile", sp, "--iters", "abc"],
            "cli_invalid_number",
            "cli_parse",
        ),
        (
            vec!["linear-equiv", sp, sp, "--tol", "NaN"],
            "cli_invalid_number",
            "cli_parse",
        ),
        (
            vec!["linear-equiv", sp, sp, "--tol", "-0.1"],
            "cli_invalid_range",
            "cli_parse",
        ),
        (
            vec!["gradcheck", sp, "--eps", "0"],
            "cli_invalid_range",
            "cli_parse",
        ),
        (
            vec!["verify", sp, sp, "--fuzz", "4", "--lo", "5", "--hi", "-5"],
            "cli_invalid_range",
            "cli_parse",
        ),
        (
            vec!["validate", sp, sp],
            "cli_unexpected_positional",
            "cli_parse",
        ),
        (vec!["run", missing_s], "io_read_failed", "load_spec"),
        (
            vec!["weight-lab", "eval", expr_s, "--repertoire", bad_rep_s],
            "repertoire_invalid",
            "weight_lab.eval",
        ),
        (
            vec!["harness", "workspace-status", "--workspace", bad_ws_s],
            "workspace_corrupted",
            "harness.workspace_status",
        ),
        (
            vec!["rewrite", "bogus-sub"],
            "cli_unknown_command",
            "rewrite",
        ),
        (vec!["run", div0_s], "division_by_zero", "run"),
    ];

    for (args, expected_code, expected_stage) in cli_checks {
        let (exit_code, val) = run_cli(&args);
        assert_eq!(exit_code, 2, "expected exit 2 for {args:?}, got {val}");
        assert_eq!(
            val["ok"],
            json!(false),
            "expected ok=false for {args:?}, got {val}"
        );
        assert_eq!(
            val["code"],
            json!(expected_code),
            "expected code={expected_code} for {args:?}, got {val}"
        );
        assert_eq!(
            val["stage"],
            json!(expected_stage),
            "expected stage={expected_stage} for {args:?}, got {val}"
        );
        assert!(
            val["error"].as_str().is_some_and(|e| !e.is_empty()),
            "expected non-empty error string for {args:?}, got {val}"
        );
        assert!(
            val["hint"].as_str().is_some_and(|h| !h.is_empty()),
            "expected non-empty hint string for {args:?}, got {val}"
        );
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn error_handling_phase3_workflow_schema_and_domain_guards() {
    use math_draw::harness::{execute_harness_workflow, merge_subagent_branches};
    use math_draw::linear_equiv::equiv_linear;
    use math_draw::mutate::{MutationRequest, mutate_spec};
    use math_draw::quant::{QuantScheme, quant_verify};
    use math_draw::recurse::{fit_es, search_graph_tree, solve_gd_autodiff};
    use math_draw::rollout::rollout_bisect;
    use math_draw::verify::{FuzzConfig, default_cases_from_spec, verify_specs, zoom_fuzz_specs};
    use math_draw::weight_lab::{WeightRequires, WeightSet, gate_sets, spectral_radius};

    let tmp_dir = std::env::temp_dir().join(format!(
        "math_draw_phase3_err_test_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let spec_raw = r#"{
      "numSlots": 2,
      "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
      "outputSlot": 1,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1], "role": "state" }],
      "weights": { "0": [0.5, 0.0, 0.0, 0.5] },
      "inputs": { "0": [1.0, 2.0] }
    }"#;
    let spec_path = tmp_dir.join("spec.json");
    std::fs::write(&spec_path, spec_raw).unwrap();
    let spec = load_spec_str(spec_raw, Path::new(".")).unwrap().spec;

    let targets_path = tmp_dir.join("targets.json");
    let tf_raw = r#"{"cases":[{"inputs":{"0":[1.0,0.0]},"expected":[0.5,0.0]}]}"#;
    std::fs::write(&targets_path, tf_raw).unwrap();
    let tf: TargetsFile = serde_json::from_str(tf_raw).unwrap();

    let ws_dir = tmp_dir.join("ws");

    // 1. Top-level workflow structural & field-type errors fail closed with Err
    for (bad_wf, needle) in [
        (json!(["not-an-object"]), "workflow JSON root must be an object"),
        (
            json!({ "spec": spec_path.display().to_string(), "pipeline": [], "typo_key": 1 }),
            "unknown field 'typo_key'",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "workspace": 123, "pipeline": [] }),
            "field 'workspace' must be a string",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "workspace": "  ", "pipeline": [] }),
            "field 'workspace' must be a non-empty string",
        ),
        (
            json!({ "spec": 42, "pipeline": [] }),
            "field 'spec' must be a file path string or inline spec object",
        ),
        (
            json!({ "spec": spec_path.display().to_string() }),
            "workflow must provide 'pipeline' array",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "pipeline": "not-an-array" }),
            "field 'pipeline' must be an array",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "pipeline": [123] }),
            "pipeline[0] must be a JSON object",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "pipeline": [{}] }),
            "pipeline[0] is missing required 'stage'",
        ),
        (
            json!({ "spec": spec_path.display().to_string(), "pipeline": [{ "stage": 99 }] }),
            "pipeline[0] 'stage' must be a non-empty string",
        ),
    ] {
        let err = execute_harness_workflow(&bad_wf, &tmp_dir, Some(&ws_dir)).unwrap_err();
        assert!(
            err.contains(needle),
            "expected '{needle}' for workflow {bad_wf}, got: {err}"
        );
    }

    // 2. Per-stage unknown keys, wrong types, and out-of-domain numerics fail closed with code="workflow_failed"
    for (stage_obj, expected_stage, needle) in [
        (
            json!({ "stage": "validate", "explain": "yes" }),
            "validate",
            "field 'explain' must be a boolean",
        ),
        (
            json!({ "stage": "validate", "bogus_opt": true }),
            "validate",
            "unknown field 'bogus_opt'",
        ),
        (
            json!({ "stage": "rollout_unroll", "steps": "three" }),
            "rollout_unroll",
            "field 'steps' must be a non-negative integer",
        ),
        (
            json!({ "stage": "rollout_unroll", "steps": 0 }),
            "rollout_unroll",
            "field 'steps' must be >= 1",
        ),
        (
            json!({ "stage": "search_rewrites", "abs_tol": -0.01 }),
            "search_rewrites",
            "field 'abs_tol' must be >= 0",
        ),
        (
            json!({ "stage": "quant_budget", "tol": -1.0 }),
            "quant_budget",
            "field 'tol' must be >= 0",
        ),
        (
            json!({
                "stage": "fit",
                "targets": targets_path.display().to_string(),
                "generations": 5,
                "gens": 5
            }),
            "fit",
            "specify either 'generations' or 'gens', not both",
        ),
        (
            json!({
                "stage": "fit",
                "targets": targets_path.display().to_string(),
                "pop": "twelve"
            }),
            "fit",
            "field 'pop' must be a non-negative integer",
        ),
        (
            json!({
                "stage": "solve_gd",
                "targets": targets_path.display().to_string(),
                "lr": 0.0
            }),
            "solve_gd",
            "field 'lr' must be > 0",
        ),
        (
            json!({
                "stage": "solve_gd",
                "targets": targets_path.display().to_string(),
                "steps": 0
            }),
            "solve_gd",
            "field 'steps' must be >= 1",
        ),
        (
            json!({ "stage": "rollout_scan", "steps": -2 }),
            "rollout_scan",
            "field 'steps' must be a non-negative integer",
        ),
        (
            json!({ "stage": "rollout_fixpoint", "tol": -1e-5 }),
            "rollout_fixpoint",
            "field 'tol' must be >= 0",
        ),
        (
            json!({ "stage": "rollout_fixpoint", "max_steps": 0 }),
            "rollout_fixpoint",
            "field 'max_steps' must be >= 1",
        ),
        (
            json!({ "stage": "run", "extra": 1 }),
            "run",
            "unknown field 'extra'",
        ),
    ] {
        let wf = json!({
            "spec": spec_path.display().to_string(),
            "pipeline": [stage_obj]
        });
        let rep = execute_harness_workflow(&wf, &tmp_dir, Some(&ws_dir)).unwrap();
        assert_eq!(rep["ok"], json!(false), "expected ok=false for {wf}");
        assert_eq!(
            rep["failed_stage"],
            json!(expected_stage),
            "unexpected failed_stage for {wf}: {rep}"
        );
        assert_eq!(
            rep["code"],
            json!("workflow_failed"),
            "expected code=workflow_failed for {wf}: {rep}"
        );
        let err_str = rep["error"].as_str().unwrap_or("");
        assert!(
            err_str.contains(needle),
            "expected '{needle}' in '{err_str}' for {wf}"
        );
    }

    // Valid workflow with 'gens' alias in 'fit' and 'layer' in 'solve_gd' succeeds cleanly
    let wf_ok = json!({
        "name": "phase3_valid_pipeline",
        "spec": spec_path.display().to_string(),
        "pipeline": [
            { "stage": "validate", "explain": true },
            { "stage": "fit", "targets": targets_path.display().to_string(), "gens": 2, "pop": 4 },
            { "stage": "solve_gd", "targets": targets_path.display().to_string(), "steps": 2, "lr": 0.05, "layer": 0 },
            { "stage": "run" }
        ]
    });
    let rep_ok = execute_harness_workflow(&wf_ok, &tmp_dir, Some(&ws_dir)).unwrap();
    assert_eq!(rep_ok["ok"], json!(true), "valid workflow failed: {rep_ok}");
    assert_eq!(rep_ok["stages_completed"], json!(4));

    // 3. Uniform domain numeric boundary guards across public Rust library APIs
    assert!(solve_gd_autodiff(&spec, Some(&tf), None, 0, 0.1, false).is_err());
    assert!(solve_gd_autodiff(&spec, Some(&tf), None, 5, 0.0, false).is_err());
    assert!(solve_gd_autodiff(&spec, Some(&tf), None, 5, -0.1, false).is_err());
    assert!(solve_gd_autodiff(&spec, Some(&tf), None, 5, f64::NAN, false).is_err());

    assert!(search_graph_tree(&spec, 2, None, -1e-5, 1e-5).is_err());
    assert!(search_graph_tree(&spec, 2, None, 1e-5, f64::NAN).is_err());

    // Legacy fit_es no longer silently clamps pop=0 or odd pop=3
    assert!(fit_es(&spec, &tf, 5, 0, 0.1, 0.1, 42, false, 1e-6).is_err());
    assert!(fit_es(&spec, &tf, 5, 3, 0.1, 0.1, 42, false, 1e-6).is_err());

    let cases = default_cases_from_spec(&spec);
    let vr_neg = verify_specs(&spec, &spec, &cases, -1e-5, 1e-5);
    assert!(!vr_neg.ok && !vr_neg.equivalent && vr_neg.verdict == "REJECT");
    let vr_nan = verify_specs(&spec, &spec, &cases, f64::NAN, 1e-5);
    assert!(!vr_nan.ok && !vr_nan.equivalent && vr_nan.verdict == "REJECT");

    assert!(
        zoom_fuzz_specs(
            &spec,
            &spec,
            &FuzzConfig {
                count: 0,
                seed: 1,
                lo: -1.0,
                hi: 1.0
            },
            2
        )
        .is_err()
    );
    assert!(
        zoom_fuzz_specs(
            &spec,
            &spec,
            &FuzzConfig {
                count: 4,
                seed: 1,
                lo: 2.0,
                hi: -2.0
            },
            2
        )
        .is_err()
    );
    assert!(
        zoom_fuzz_specs(
            &spec,
            &spec,
            &FuzzConfig {
                count: 4,
                seed: 1,
                lo: f64::NAN,
                hi: 1.0
            },
            2
        )
        .is_err()
    );

    assert!(quant_verify(&spec, QuantScheme::Int8, None, -1e-3, 1e-3).is_err());
    assert!(quant_verify(&spec, QuantScheme::Int8, None, 1e-3, f64::NAN).is_err());

    let le_neg = equiv_linear(&spec, &spec, -1e-6);
    assert!(!le_neg.ok && le_neg.verdict == "REJECT");
    let le_nan = equiv_linear(&spec, &spec, f64::NAN);
    assert!(!le_nan.ok && le_nan.verdict == "REJECT");

    assert!(rollout_bisect(&spec, &spec, None, 4, None, None, -1e-5).is_err());
    assert!(rollout_bisect(&spec, &spec, None, 4, None, None, f64::NAN).is_err());

    let mreq: MutationRequest = serde_json::from_value(json!({
        "commit": false,
        "edits": [{ "op": "setWeights", "layer_index": 0, "values": [0.5, 0.0, 0.0, 0.5] }]
    }))
    .unwrap();
    assert!(mutate_spec(&spec, &mreq, None, -1e-5, 1e-5).is_err());
    assert!(mutate_spec(&spec, &mreq, None, 1e-5, f64::NAN).is_err());

    assert!(merge_subagent_branches(&spec, &[], -1e-4).is_err());
    assert!(merge_subagent_branches(&spec, &[], f64::NAN).is_err());

    assert!(spectral_radius(&[vec![1.0, 0.0], vec![0.0, 1.0]], 0).is_err());
    let ws_set = WeightSet {
        name: Some("s".into()),
        kind: "map".into(),
        w: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        b: vec![0.0, 0.0],
        probes: vec![],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    assert!(gate_sets(&ws_set, &ws_set, -0.1).is_err());
    assert!(gate_sets(&ws_set, &ws_set, f64::NAN).is_err());

    let _ = std::fs::remove_dir_all(&tmp_dir);
}


