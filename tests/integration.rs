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

#[test]
fn liveness_compaction_auto_repair_and_attention_head_macros() {
    use math_draw::liveness::{auto_repair_spec, compact_slots_by_liveness, ssa_normalize_spec};
    use math_draw::rollout::rollout_unroll_with_options;

    // 1. Explain liveness + SSA normalization + linear-scan compaction on a deep chain
    let chain_raw = r#"{
      "numSlots": 7,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 2, true] },
        { "constructor": "relu", "args": [1] },
        { "constructor": "sigmoid", "args": [2] },
        { "constructor": "tanh", "args": [3] },
        { "constructor": "fma", "args": [4] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "unary", "layer": 1, "slots": [1, 2] },
        { "kind": "unary", "layer": 2, "slots": [2, 3] },
        { "kind": "unary", "layer": 3, "slots": [3, 4] },
        { "kind": "ternary", "layer": 4, "slots": [4, 2, 0, 5] }
      ],
      "outputSlot": 5,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [0.8, -0.3, 0.4, 0.7, 0.1, -0.2] },
      "inputs": { "0": [1.5, -0.5] }
    }"#;
    let chain = load_spec_str(chain_raw, Path::new(".")).unwrap().spec;
    let val = validate_spec(&chain, true);
    assert!(val.valid);
    let explain = val.explain.as_ref().unwrap();
    assert!(!explain.liveness.peak_is_lower_bound);
    assert!(explain.liveness.peak_bytes >= 24);
    assert_eq!(explain.liveness.live_before.len(), chain.steps.len() + 1);

    let comp = compact_slots_by_liveness(&chain).unwrap();
    assert!(comp.slots_saved >= 2, "expected >=2 slots saved, got {}", comp.slots_saved);
    let base_out = run_spec(&chain).unwrap().outputs.values;
    let comp_out = run_spec(&comp.spec).unwrap().outputs.values;
    assert_eq!(base_out, comp_out);

    // 2. SSA normalize a spec that overwrites slots in-place
    let ssa_rep = ssa_normalize_spec(&comp.spec);
    assert!(validate_spec(&ssa_rep.ssa_spec, false).valid);
    assert_eq!(run_spec(&ssa_rep.ssa_spec).unwrap().outputs.values, base_out);

    // 3. Rollout unroll with --compact-slots reduces slot count while preserving bit-identity
    let rec_raw = r#"{
      "numSlots": 3,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 2, true] },
        { "constructor": "tanh", "args": [1] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "unary", "layer": 1, "slots": [1, 2] }
      ],
      "outputSlot": 2,
      "ports": [{ "slot": 0, "role": "state", "shape": [1, 2, 1, 1] }],
      "weights": { "0": [0.5, 0.0, 0.0, 0.5, 0.1, -0.1] },
      "inputs": { "0": [1.0, -1.0] }
    }"#;
    let rec = load_spec_str(rec_raw, Path::new(".")).unwrap().spec;
    let unrolled_plain = rollout_unroll_with_options(&rec, None, 6, None, false).unwrap();
    let unrolled_comp = rollout_unroll_with_options(&rec, None, 6, None, true).unwrap();
    assert!(unrolled_comp.unrolled_num_slots < unrolled_plain.unrolled_num_slots);
    assert_eq!(
        run_spec(&unrolled_plain.unrolled_spec).unwrap().outputs.values,
        run_spec(&unrolled_comp.unrolled_spec).unwrap().outputs.values
    );

    // 4. Auto-repair: mixed f32/i64 boundary cast + terminal intToFloat + unused port + linear sharding
    let repair_raw = r#"{
      "numSlots": 5,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 4, true] },
        { "constructor": "intAdd", "args": [1] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "binary", "layer": 1, "slots": [1, 2, 3] }
      ],
      "outputSlot": 3,
      "ports": [
        { "slot": 0, "shape": [1, 2, 1, 1] },
        { "slot": 2, "shape": [1, 4, 1, 1] },
        { "slot": 4, "shape": [1, 2, 1, 1] }
      ],
      "weights": { "0": [1,0,0,0, 0,1,0,0, 0,0,0,0] },
      "inputs": { "0": [3.0, 4.0], "2": [10, 20, 30, 40], "4": [1.0, 1.0] }
    }"#;
    let repair_in = load_spec_str(repair_raw, Path::new(".")).unwrap().spec;
    let rep = auto_repair_spec(&repair_in, 6).unwrap();
    assert!(rep.repaired);
    let kinds: Vec<&str> = rep
        .remediation_receipt
        .iter()
        .filter_map(|v| v["repair"].as_str())
        .collect();
    assert!(kinds.contains(&"insert_dtype_boundary_cast"), "receipts={kinds:?}");
    assert!(kinds.contains(&"insert_terminal_intToFloat"), "receipts={kinds:?}");
    assert!(kinds.contains(&"drain_unused_port"), "receipts={kinds:?}");
    assert!(kinds.contains(&"recursive_shard_linear"), "receipts={kinds:?}");
    assert!(validate_spec(&rep.spec, false).valid);
    let out = run_spec(&rep.spec).unwrap().outputs.values;
    assert_eq!(out, vec![13.0, 24.0, 30.0, 40.0]);

    // 5. Built-in macro `attention_head` in both `seq_axis1` (cross-attention Sq=2, Skv=3) and `heads_axis1`
    let seq_attn_raw = r#"{
      "schema": "burn-math.recursive-spec.v2",
      "ports": {
        "q": { "slot": 0, "shape": [1, 2, 2, 1] },
        "k": { "slot": 1, "shape": [1, 3, 2, 1] },
        "v": { "slot": 2, "shape": [1, 3, 2, 1] }
      },
      "steps": [
        { "op": "attention_head", "params": { "d_model": 2, "layout": "seq_axis1" }, "in": ["q", "k", "v"], "out": "y" }
      ],
      "output": "y",
      "inputs": {
        "q": [1.0, 0.0, 0.0, 1.0],
        "k": [1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
        "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      }
    }"#;
    let seq_attn = load_spec_str(seq_attn_raw, Path::new(".")).unwrap().spec;
    let seq_out = run_spec(&seq_attn).unwrap();
    assert_eq!(seq_out.outputs.shape, [1, 2, 2, 1]);
    assert!(seq_out.outputs.values.iter().all(|v| v.is_finite()));

    let bad_layout = r#"{
      "schema": "burn-math.recursive-spec.v2",
      "ports": {
        "q": { "slot": 0, "shape": [1, 1, 2, 2] },
        "k": { "slot": 1, "shape": [1, 1, 2, 2] },
        "v": { "slot": 2, "shape": [1, 1, 2, 2] }
      },
      "steps": [
        { "op": "attention_head", "params": { "d_model": 2, "layout": "invalid_layout" }, "in": ["q", "k", "v"], "out": "y" }
      ],
      "output": "y"
    }"#;
    assert!(load_spec_str(bad_layout, Path::new(".")).is_err());
}

#[test]
fn symbolic_grad_spec_vjp_hvp_and_finite_diff_parity() {
    use math_draw::grad_spec::{GradSpecOptions, compile_grad_spec};
    use math_draw::gradcheck::gradcheck_spec;

    // 1. Nonlinear multi-op forward graph: linear -> sigmoid + exp -> sin + fma -> div -> sqrt
    let nl_raw = r#"{
      "schema": "burn-math.recursive-spec.v2",
      "ports": {
        "x": { "slot": 0, "shape": [1, 2, 1, 1] },
        "c": { "slot": 1, "shape": [1, 2, 1, 1] }
      },
      "steps": [
        { "op": "linear", "params": { "in_dim": 2, "out_dim": 2, "bias": true }, "in": "x", "out": "h1", "weights": [0.4, -0.2, 0.3, 0.5, 0.1, 0.2] },
        { "op": "sigmoid", "in": "h1", "out": "s1" },
        { "op": "tanh", "in": "h1", "out": "t1" },
        { "op": "fma", "in": ["s1", "t1", "c"], "out": "f1" },
        { "op": "exp", "in": "s1", "out": "e1" },
        { "op": "add", "in": ["f1", "e1"], "out": "y" }
      ],
      "output": "y",
      "inputs": { "x": [0.7, -0.4], "c": [1.5, 2.0] }
    }"#;
    let loaded = load_spec_str(nl_raw, Path::new(".")).unwrap();
    let vjp = compile_grad_spec(
        &loaded.spec,
        loaded.expansion.as_ref(),
        &GradSpecOptions {
            wrt: "x".into(),
            order: 1,
            dout: Some(vec![1.0, -0.5]),
            hvp_vec: None,
            compact_slots: true,
        },
    )
    .unwrap();
    let vjp_vals = run_spec(&vjp.spec).unwrap().outputs.values;
    assert!(
        vjp.forward_output_slot < vjp.num_slots,
        "compacted forward_output_slot {} must be < num_slots {}",
        vjp.forward_output_slot,
        vjp.num_slots
    );
    assert_eq!(
        Some(vjp.forward_output_slot),
        vjp.spec.steps[loaded.spec.steps.len() - 1].output_slot(),
        "compacted forward_output_slot must match the remapped final forward step output slot"
    );

    // Cross-check against central finite-difference Jacobian J^T * dout
    let gc = gradcheck_spec(&loaded.spec, None, 1e-3, 1e-2, false, false).unwrap();
    let jac = &gc.input_jacobians["0"].values;
    let dout = [1.0f64, -0.5f64];
    for j in 0..2 {
        let expected: f64 = (0..2).map(|i| jac[i][j] * dout[i]).sum();
        assert!(
            ((vjp_vals[j] as f64) - expected).abs() < 2e-3,
            "VJP[{j}]={} vs fd={expected}",
            vjp_vals[j]
        );
    }

    // 2. 2nd-order HVP on exp(x): f(x) = exp(x), f'(x) = exp(x), f''(x) = exp(x)
    let exp_raw = r#"{
      "numSlots": 2,
      "layers": [{ "constructor": "exp", "args": [0] }],
      "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
      "outputSlot": 1,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "inputs": { "0": [0.5, -0.3] }
    }"#;
    let exp_spec = load_spec_str(exp_raw, Path::new(".")).unwrap().spec;
    let hvp = compile_grad_spec(
        &exp_spec,
        None,
        &GradSpecOptions {
            wrt: "0".into(),
            order: 2,
            dout: Some(vec![1.0, 1.0]),
            hvp_vec: Some(vec![2.0, -3.0]),
            compact_slots: false,
        },
    )
    .unwrap();
    assert_eq!(hvp.kind, "symbolic_hvp_spec");
    assert!(hvp.hvp_slot.is_some());
    let hvp_vals = run_spec(&hvp.spec).unwrap().outputs.values;
    let exp0 = 0.5_f32.exp() * 2.0;
    let exp1 = (-0.3_f32).exp() * (-3.0);
    assert!((hvp_vals[0] - exp0).abs() < 1e-4, "{} vs {exp0}", hvp_vals[0]);
    assert!((hvp_vals[1] - exp1).abs() < 1e-4, "{} vs {exp1}", hvp_vals[1]);

    // 3. Test gelu, mish, and swiGlu VJPs against central finite differences
    for (name, raw) in [
        (
            "gelu",
            r#"{
              "numSlots": 2,
              "layers": [{ "constructor": "gelu", "args": [0] }],
              "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
              "outputSlot": 1,
              "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
              "inputs": { "0": [0.8, -0.6] }
            }"#,
        ),
        (
            "mish",
            r#"{
              "numSlots": 2,
              "layers": [{ "constructor": "mish", "args": [0] }],
              "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
              "outputSlot": 1,
              "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
              "inputs": { "0": [0.8, -0.6] }
            }"#,
        ),
        (
            "swiGlu",
            r#"{
              "numSlots": 2,
              "layers": [{ "constructor": "swiGlu", "args": [0, 2, 2, true] }],
              "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
              "outputSlot": 1,
              "ports": [{ "slot": 0, "shape": [1, 1, 1, 2], "layout": "feature_last" }],
              "weights": { "0": [0.5, -0.2, 0.3, 0.4, 0.1, -0.1, 0.2, 0.6, -0.3, 0.5, 0.05, 0.15] },
              "inputs": { "0": [0.8, -0.6] }
            }"#,
        ),
    ] {
        let sp = load_spec_str(raw, Path::new(".")).unwrap().spec;
        let gs = compile_grad_spec(&sp, None, &GradSpecOptions::default()).unwrap();
        let gvals = run_spec(&gs.spec).unwrap().outputs.values;
        let gc_op = gradcheck_spec(&sp, None, 1e-3, 1e-2, false, false).unwrap();
        let jmat = &gc_op.input_jacobians["0"].values;
        for col in 0..2 {
            let expected: f64 = (0..2).map(|row| jmat[row][col]).sum();
            assert!(
                ((gvals[col] as f64) - expected).abs() < 3e-3,
                "{name} VJP[{col}]={} vs fd={expected}",
                gvals[col]
            );
        }
    }

    // 3b. GELU VJP (order=1) & HVP (order=2) at exact origin x = 0.0 (must not divide by zero)
    {
        let gelu_zero = load_spec_str(
            r#"{
              "numSlots": 2,
              "layers": [{ "constructor": "gelu", "args": [0] }],
              "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
              "outputSlot": 1,
              "ports": [{ "slot": 0, "shape": [1, 4, 1, 1] }],
              "inputs": { "0": [0.0, 1e-7, 0.8, -0.6] }
            }"#,
            Path::new("."),
        )
        .unwrap()
        .spec;
        let vjp_zero = compile_grad_spec(
            &gelu_zero,
            None,
            &GradSpecOptions {
                wrt: "0".into(),
                order: 1,
                dout: Some(vec![1.0, 1.0, 1.0, 1.0]),
                hvp_vec: None,
                compact_slots: true,
            },
        )
        .unwrap();
        let vjp_z_vals = run_spec(&vjp_zero.spec).unwrap().outputs.values;
        assert!(
            (vjp_z_vals[0] - 0.5).abs() < 1e-6,
            "gelu'(0) expected 0.5, got {}",
            vjp_z_vals[0]
        );
        assert!(
            (vjp_z_vals[1] - 0.5).abs() < 1e-5,
            "gelu'(1e-7) expected ~0.5, got {}",
            vjp_z_vals[1]
        );

        let hvp_zero = compile_grad_spec(
            &gelu_zero,
            None,
            &GradSpecOptions {
                wrt: "0".into(),
                order: 2,
                dout: Some(vec![1.0, 1.0, 1.0, 1.0]),
                hvp_vec: Some(vec![1.0, 1.0, 1.0, 1.0]),
                compact_slots: false,
            },
        )
        .unwrap();
        let hvp_z_vals = run_spec(&hvp_zero.spec).unwrap().outputs.values;
        let expected_d2_0 = (2.0 / std::f64::consts::PI).sqrt() as f32;
        assert!(
            (hvp_z_vals[0] - expected_d2_0).abs() < 1e-4,
            "gelu''(0) expected {expected_d2_0}, got {}",
            hvp_z_vals[0]
        );
    }

    // 4. Fail-closed errors
    assert!(
        compile_grad_spec(
            &exp_spec,
            None,
            &GradSpecOptions {
                order: 3,
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        compile_grad_spec(
            &exp_spec,
            None,
            &GradSpecOptions {
                order: 1,
                hvp_vec: Some(vec![1.0, 1.0]),
                ..Default::default()
            }
        )
        .is_err()
    );
    assert!(
        compile_grad_spec(
            &exp_spec,
            None,
            &GradSpecOptions {
                dout: Some(vec![1.0]),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn soak_prng_manifest_quant_search_and_new_workflow_stages() {
    use math_draw::harness::execute_harness_workflow;
    use math_draw::quant::QuantScheme;
    use math_draw::recurse::{RngKind, SearchOptions, search_graph_tree_with_options};
    use math_draw::soak::{certify_prng_stream, soak_spec, verify_prng_manifest};

    let spec_raw = r#"{
      "numSlots": 4,
      "layers": [
        { "constructor": "linear", "args": [0, 2, 4, true] },
        { "constructor": "sigmoid", "args": [1] }
      ],
      "steps": [
        { "kind": "unary", "layer": 0, "slots": [0, 1] },
        { "kind": "unary", "layer": 1, "slots": [1, 2] }
      ],
      "outputSlot": 2,
      "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
      "weights": { "0": [0.5, -0.2, 0.3, 0.4, 0.1, 0.6, -0.4, 0.2, 0.05, -0.05, 0.1, -0.1] },
      "inputs": { "0": [1.0, -0.5] }
    }"#;
    let spec = load_spec_str(spec_raw, Path::new(".")).unwrap().spec;

    // 1. Soak determinism & domain guard
    let sk = soak_spec(&spec, 12).unwrap();
    assert!(sk.identical && sk.runs == 12 && sk.first_mismatch_index.is_none());
    assert!(soak_spec(&spec, 1).is_err());

    // 2. PRNG stream certification across all 4 engines + tamper check
    for kind in [
        RngKind::Xorshift64,
        RngKind::Mulberry32,
        RngKind::Pcg32,
        RngKind::Philox4x32,
    ] {
        let m = certify_prng_stream(kind, 123, 50).unwrap();
        assert_eq!(m.preview.len(), 5);
        assert!(verify_prng_manifest(&m.algorithm, m.seed, m.count, &m.sha256).unwrap().verified);
        assert!(!verify_prng_manifest(&m.algorithm, m.seed, m.count + 1, &m.sha256).unwrap().verified);
    }
    assert!(certify_prng_stream(RngKind::Mulberry32, 1, 0).is_err());

    // 3. Search with --quant-tol reduces weight bytes when within tolerance & rejects negative quant_tol
    let sr = search_graph_tree_with_options(
        &spec,
        None,
        &SearchOptions {
            max_depth: 2,
            abs_tol: 1e-5,
            rel_tol: 1e-5,
            quant_tol: Some(0.1),
            quant_scheme: QuantScheme::Int8,
        },
    )
    .unwrap();
    assert!(sr.best_weight_bytes < sr.baseline_weight_bytes);
    assert!(sr.transform_chain.iter().any(|t| t == "quant:int8"));
    assert!(
        search_graph_tree_with_options(
            &spec,
            None,
            &SearchOptions {
                quant_tol: Some(-0.01),
                ..Default::default()
            }
        )
        .is_err()
    );

    // 4. End-to-end harness workflow exercising new v0.8.0 stages
    let tmp_dir = std::env::temp_dir().join(format!("math_draw_v08_wf_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let spec_path = tmp_dir.join("spec.json");
    std::fs::write(&spec_path, spec_raw).unwrap();
    let bundle_path = tmp_dir.join("out.bundle.json");

    let wf = json!({
        "spec": spec_path.display().to_string(),
        "pipeline": [
            { "stage": "compile", "compact_slots": true },
            { "stage": "validate", "explain": true },
            { "stage": "shard", "max_weights": 6 },
            { "stage": "gradcheck", "eps": 1e-3, "tol": 1e-2, "weights": true, "autodiff": true },
            { "stage": "soak", "runs": 5 },
            { "stage": "bundle_export", "attest": ["v08_wf"], "out": bundle_path.display().to_string() },
            { "stage": "grad_spec", "wrt": 0, "order": 1, "compact_slots": true },
            { "stage": "run" }
        ]
    });
    let ws = tmp_dir.join("ws");
    let res = execute_harness_workflow(&wf, &tmp_dir, Some(&ws)).unwrap();
    assert_eq!(res["ok"], json!(true), "workflow failed: {res}");
    assert_eq!(res["stages_completed"], json!(8));
    assert!(bundle_path.exists());

    // Strict unknown field rejected on new stages
    let bad_wf = json!({
        "spec": spec_path.display().to_string(),
        "pipeline": [{ "stage": "soak", "runs": 4, "bogus": true }]
    });
    let bad_res = execute_harness_workflow(&bad_wf, &tmp_dir, Some(&ws)).unwrap();
    assert_eq!(bad_res["ok"], json!(false));
    assert_eq!(bad_res["code"], json!("workflow_failed"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn direct_math_cli_and_library_arith_int_kernel_end_to_end() {
    use math_draw::direct_math::{
        DirectBatchKind, eval_arith_op, eval_direct_batch, eval_int_op, eval_kernel_op,
    };
    use std::process::Command;

    // 1. Arith ops (broadcast, stats, clamp, matmul, axis reductions, fail-closed div0)
    let add = eval_arith_op("add", "1,2,3", Some("10")).unwrap();
    assert_eq!(add["broadcast"], json!("b"));
    assert_eq!(add["values"], json!([11.0, 12.0, 13.0]));
    let cl = eval_arith_op("clamp", "-5,0.5,9", Some("0,1")).unwrap();
    assert_eq!(cl["values"], json!([0.0, 0.5, 1.0]));
    let ax = eval_arith_op("sum_axis", "1,2,3;4,5,6", Some("0")).unwrap();
    assert_eq!(ax["values"], json!([5.0, 7.0, 9.0]));
    assert!(eval_arith_op("div", "1,2", Some("1,0")).is_err());
    assert!(eval_arith_op("clamp", "1,2", Some("5,1")).is_err());

    // 2. Exact i64 ops, overflow guard, --wrap, to-f32 exactness guard, from-f32
    let im = eval_int_op("matmul", "1,2;3,4", Some("5,6;7,8"), false, false).unwrap();
    assert_eq!(im["values"], json!(["19", "22", "43", "50"]));
    assert!(eval_int_op("add", "9223372036854775807", Some("1"), false, false).is_err());
    let wrapped = eval_int_op("add", "9223372036854775807", Some("1"), true, false).unwrap();
    assert_eq!(wrapped["values"], json!(["-9223372036854775808"]));
    assert!(eval_int_op("div", "10", Some("0"), true, false).is_err());
    assert!(eval_int_op("to-f32", "16777217", None, false, false).is_err());
    assert!(eval_int_op("to-f32", "16777217", None, false, true).is_ok());
    let fr = eval_int_op("from-f32", "3.9,-4.7,0", None, false, false).unwrap();
    assert_eq!(fr["values"], json!(["3", "-4", "0"]));

    // 3. Lane-E kernel ops: round half-to-even, atan2(y,x), fma, strict no-broadcast
    let rd = eval_kernel_op("round", "2.5,3.5", None, None).unwrap();
    assert_eq!(rd["values"], json!([2.0, 4.0]));
    let at = eval_kernel_op("atan2", "1,0", Some("0,1"), None).unwrap();
    let at0 = at["values"][0].as_f64().unwrap();
    let at1 = at["values"][1].as_f64().unwrap();
    assert!((at0 - std::f64::consts::FRAC_PI_2).abs() < 1e-5);
    assert!(at1.abs() < 1e-6);
    assert!(eval_kernel_op("hypot", "3,4", Some("1"), None).is_err());

    // 4. Batch chaining for int and kernel
    let int_batch = eval_direct_batch(
        DirectBatchKind::Int,
        &json!([
            { "op": "add", "a": "10,20", "b": "5" },
            { "op": "mul", "a": "$prev", "b": "2" }
        ]),
    )
    .unwrap();
    assert_eq!(int_batch["results"][1]["values"], json!(["30", "50"]));

    // 5. Live CLI binary verification for new v0.8.0 commands
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let run_cli = |args: &[&str]| -> (i32, serde_json::Value) {
        let out = Command::new(bin).args(args).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let parsed: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("non-JSON stdout for {args:?}: {e}\n{stdout}"));
        (code, parsed)
    };

    let (c, v) = run_cli(&["arith", "add", "1,-2", "3,4"]);
    assert_eq!(c, 0);
    assert_eq!(v["values"], json!([4.0, 2.0]));

    let (c, v) = run_cli(&["int", "add", "16777217", "1"]);
    assert_eq!(c, 0);
    assert_eq!(v["values"], json!(["16777218"]));

    let (c, v) = run_cli(&["kernel", "fma", "2,3", "4,5", "1,-1"]);
    assert_eq!(c, 0);
    assert_eq!(v["values"], json!([9.0, 14.0]));

    let tmp_dir = std::env::temp_dir().join(format!("math_draw_v08_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let manifest_p = tmp_dir.join("prng.json");
    let mp = manifest_p.to_str().unwrap();

    let (c, v) = run_cli(&["prng-check", "philox4x32", "7", "32", "--out", mp]);
    assert_eq!(c, 0);
    assert_eq!(v["ok"], json!(true));
    let (c, v) = run_cli(&["prng-check", "verify", mp]);
    assert_eq!(c, 0);
    assert_eq!(v["verified"], json!(true));

    let (c, v) = run_cli(&["soak", "examples/linear-relu.json", "--runs", "5"]);
    assert_eq!(c, 0);
    assert_eq!(v["identical"], json!(true));

    let (c, v) = run_cli(&[
        "recurse",
        "expand",
        "examples/recursive-gated-residual.v2.json",
        "--auto-repair",
        "--compact-slots",
    ]);
    assert_eq!(c, 0);
    assert_eq!(v["ok"], json!(true));
    assert!(v["slot_compaction"]["slots_saved"].as_u64().unwrap() > 0);

    let (c, v) = run_cli(&[
        "rollout",
        "unroll",
        "examples/recursive-gated-residual.v2.json",
        "--steps",
        "4",
        "--compact-slots",
    ]);
    assert_eq!(c, 0);
    assert_eq!(v["ok"], json!(true));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn diff_ref_all_f32_families_and_external_ref_parity() {
    use math_draw::diff_ref::{SUPPORTED_REF_OPS, diff_ref_spec, run_reference_f64};
    use std::process::Command;

    // 1. Verify exact 87-op alignment with non-integer constructors in CATALOG
    assert_eq!(SUPPORTED_REF_OPS.len(), 87);
    for meta in CATALOG {
        if !math_draw::catalog::is_int_constructor(meta.name) {
            assert!(
                SUPPORTED_REF_OPS.contains(&meta.name),
                "missing f32 constructor '{}' in SUPPORTED_REF_OPS",
                meta.name
            );
        }
    }

    let check_unary = |ctor: &str,
                       args: Vec<serde_json::Value>,
                       shape: [usize; 4],
                       input: Vec<serde_json::Value>,
                       weights: Option<Vec<f32>>,
                       tol: f64| {
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
        let rep = diff_ref_spec(&spec, None, None, tol)
            .unwrap_or_else(|e| panic!("diff_ref_spec({ctor}) error: {e}"));
        assert!(
            rep.ok,
            "diff_ref_spec({ctor}) SIGNAL: abs={:.3e} rel={:.3e} tol={tol}",
            rep.max_abs_error, rep.max_rel_error
        );
    };

    let check_binary = |ctor: &str,
                        args: Vec<serde_json::Value>,
                        s0: [usize; 4],
                        i0: Vec<serde_json::Value>,
                        s1: [usize; 4],
                        i1: Vec<serde_json::Value>,
                        tol: f64| {
        let mut in_map = BTreeMap::new();
        in_map.insert("0".to_string(), i0);
        in_map.insert("1".to_string(), i1);
        let spec = SpecV1 {
            num_slots: 3,
            layers: vec![LayerDecl {
                constructor: ctor.to_string(),
                args,
            }],
            steps: vec![StepDecl {
                kind: "binary".into(),
                layer: 0,
                slots: vec![0, 1, 2],
            }],
            output_slot: 2,
            ports: vec![
                PortDecl {
                    slot: 0,
                    role: "observation".into(),
                    shape: s0,
                    layout: "any_rank4".into(),
                },
                PortDecl {
                    slot: 1,
                    role: "observation".into(),
                    shape: s1,
                    layout: "any_rank4".into(),
                },
            ],
            weights: BTreeMap::new(),
            inputs: in_map,
        };
        let rep = diff_ref_spec(&spec, None, None, tol)
            .unwrap_or_else(|e| panic!("diff_ref_spec({ctor}) error: {e}"));
        assert!(
            rep.ok,
            "diff_ref_spec({ctor}) SIGNAL: abs={:.3e} rel={:.3e} tol={tol}",
            rep.max_abs_error, rep.max_rel_error
        );
    };

    let f4 = vec![json!(0.4), json!(-0.7), json!(1.2), json!(0.25)];
    let pos4 = vec![json!(0.4), json!(0.7), json!(1.2), json!(2.5)];
    let prob_p = vec![json!(0.2), json!(0.3), json!(0.5)];
    let prob_q = vec![json!(0.25), json!(0.25), json!(0.5)];
    let img2x2: Vec<serde_json::Value> = (1..=8).map(|i| json!(i as f64 * 0.25 - 0.5)).collect();

    // Activations (15)
    for op in ["relu", "gelu", "sigmoid", "tanh", "hardSwish", "mish"] {
        check_unary(op, vec![json!(0)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    }
    check_unary("leakyRelu", vec![json!(0), json!(0.1)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("prelu", vec![json!(0), json!(4), json!(0.25)], [1, 4, 1, 1], f4.clone(), Some(vec![0.1, 0.2, 0.3, 0.4]), 1e-5);
    check_unary("swiGlu", vec![json!(0), json!(2), json!(2), json!(true)], [1, 1, 1, 2], vec![json!(0.8), json!(-0.6)], Some(vec![0.5, -0.2, 0.3, 0.4, 0.1, -0.1, 0.2, 0.6, -0.3, 0.5, 0.05, 0.15]), 1e-5);
    check_unary("hardSigmoid", vec![json!(0), json!(0.2), json!(0.5)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("softplus", vec![json!(0), json!(1.5)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("softmax", vec![json!(0), json!(1)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("logSoftmax", vec![json!(0), json!(1)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("glu", vec![json!(0), json!(1)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_unary("sqrt", vec![json!(0)], [1, 4, 1, 1], pos4.clone(), None, 1e-5);

    // Linear (1)
    check_unary("linear", vec![json!(0), json!(2), json!(3), json!(true)], [1, 2, 1, 1], vec![json!(1.5), json!(-0.5)], Some(vec![0.5, -0.2, 0.3, 0.4, 0.1, 0.6, -0.1, 0.2, 0.05]), 1e-5);

    // Norms (5)
    check_unary("batchNorm", vec![json!(0), json!(2), json!(1e-5)], [2, 2, 2, 1], img2x2.clone(), Some(vec![1.1, 0.9, 0.1, -0.2]), 1e-4);
    check_unary("groupNorm", vec![json!(0), json!(2), json!(2), json!(1e-5)], [1, 2, 2, 2], img2x2.clone(), Some(vec![1.2, 0.8, 0.05, -0.05]), 1e-4);
    check_unary("instanceNorm", vec![json!(0), json!(2), json!(1e-5)], [1, 2, 2, 2], img2x2.clone(), Some(vec![1.1, 0.7, 0.2, -0.1]), 1e-4);
    check_unary("layerNorm", vec![json!(0), json!(4), json!(1e-5)], [1, 1, 1, 4], f4.clone(), Some(vec![1.0, 1.2, 0.8, 0.9, 0.1, -0.1, 0.05, 0.0]), 1e-4);
    check_unary("rmsNorm", vec![json!(0), json!(4), json!(1e-6)], [1, 1, 1, 4], f4.clone(), Some(vec![1.1, 0.9, 1.2, 0.8]), 1e-5);

    // Convs & Embedding (4)
    check_unary("conv1d", vec![json!(0), json!(2), json!(2), json!(2), json!(1), json!(0)], [1, 2, 4, 1], img2x2.clone(), Some(vec![0.2, -0.1, 0.3, 0.4, -0.2, 0.5, 0.1, -0.3, 0.05, -0.05]), 1e-5);
    check_unary("conv2d", vec![json!(0), json!(2), json!(2), json!(2), json!(2), json!(1), json!(1), json!(0), json!(0)], [1, 2, 2, 2], img2x2.clone(), Some((0..18).map(|i| (i as f32) * 0.07 - 0.5).collect()), 1e-5);
    check_unary("convTranspose2d", vec![json!(0), json!(2), json!(2), json!(2), json!(2), json!(1), json!(1), json!(0), json!(0)], [1, 2, 2, 2], img2x2.clone(), Some((0..18).map(|i| (i as f32) * 0.06 - 0.4).collect()), 1e-5);
    check_unary("embedding", vec![json!(0), json!(4), json!(3)], [1, 2, 1, 1], vec![json!(1), json!(3)], Some((0..12).map(|i| i as f32 * 0.25).collect()), 1e-5);

    // Pools (5)
    check_unary("maxPool1d", vec![json!(0), json!(2), json!(1), json!(0)], [1, 2, 4, 1], img2x2.clone(), None, 1e-5);
    check_unary("avgPool1d", vec![json!(0), json!(2), json!(1), json!(0)], [1, 2, 4, 1], img2x2.clone(), None, 1e-5);
    check_unary("maxPool2d", vec![json!(0), json!(2), json!(2), json!(1), json!(1), json!(0), json!(0)], [1, 2, 2, 2], img2x2.clone(), None, 1e-5);
    check_unary("avgPool2d", vec![json!(0), json!(2), json!(2), json!(1), json!(1), json!(0), json!(0)], [1, 2, 2, 2], img2x2.clone(), None, 1e-5);
    check_unary("adaptiveAvgPool2d", vec![json!(0), json!(1), json!(1)], [1, 2, 2, 2], img2x2.clone(), None, 1e-5);

    // Custom (7)
    check_unary("featureNorm", vec![json!(0), json!(1e-6)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    for sh in ["shiftUp", "shiftDown", "shiftLeft", "shiftRight"] {
        check_unary(sh, vec![json!(0), json!(1)], [1, 2, 2, 2], img2x2.clone(), None, 1e-6);
    }
    check_unary("ghost", vec![json!(0), json!(2), json!(4), json!(1), json!(1), json!(2), json!(1), json!(1), json!(0), json!(0)], [1, 2, 2, 2], img2x2.clone(), Some((0..8).map(|i| (i as f32) * 0.1 - 0.3).collect()), 1e-5);
    check_unary("seBlock", vec![json!(0), json!(2), json!(1)], [1, 2, 2, 2], img2x2.clone(), Some(vec![0.4, -0.2, 0.3, 0.5, 0.1, -0.1, 0.2, 0.6, -0.3, 0.5, 0.05, 0.1]), 1e-5);

    // Binary f32 (6) & Transform (1)
    for op in ["add", "sub", "mul", "div"] {
        check_binary(op, vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);
    }
    check_binary("matmul", vec![json!(0)], [1, 1, 2, 2], f4.clone(), [1, 1, 2, 2], pos4.clone(), 1e-5);
    check_binary("concat", vec![json!(0), json!(1)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-6);
    check_unary("transpose", vec![json!(0), json!(1), json!(2)], [1, 2, 2, 2], img2x2.clone(), None, 1e-6);

    // Numeric primitives (15)
    for op in ["abs", "exp", "sin", "cos", "tan", "floor", "ceil", "round", "allFinite"] {
        check_unary(op, vec![json!(0)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    }
    check_unary("log", vec![json!(0)], [1, 4, 1, 1], pos4.clone(), None, 1e-5);
    check_unary("clamp", vec![json!(0), json!(-0.2), json!(0.9)], [1, 4, 1, 1], f4.clone(), None, 1e-6);
    check_binary("atan2", vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);
    check_binary("powf", vec![json!(0)], [1, 4, 1, 1], pos4.clone(), [1, 4, 1, 1], f4.clone(), 1e-5);
    check_binary("hypot", vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);

    // Ternary fma
    {
        let mut in_map = BTreeMap::new();
        in_map.insert("0".to_string(), f4.clone());
        in_map.insert("1".to_string(), pos4.clone());
        in_map.insert("2".to_string(), vec![json!(0.1), json!(0.2), json!(0.3), json!(0.4)]);
        let fma_spec = SpecV1 {
            num_slots: 4,
            layers: vec![LayerDecl { constructor: "fma".into(), args: vec![json!(0)] }],
            steps: vec![StepDecl { kind: "ternary".into(), layer: 0, slots: vec![0, 1, 2, 3] }],
            output_slot: 3,
            ports: vec![
                PortDecl { slot: 0, role: "observation".into(), shape: [1, 4, 1, 1], layout: "any_rank4".into() },
                PortDecl { slot: 1, role: "observation".into(), shape: [1, 4, 1, 1], layout: "any_rank4".into() },
                PortDecl { slot: 2, role: "observation".into(), shape: [1, 4, 1, 1], layout: "any_rank4".into() },
            ],
            weights: BTreeMap::new(),
            inputs: in_map,
        };
        let rep = diff_ref_spec(&fma_spec, None, None, 1e-5).unwrap();
        assert!(rep.ok);
    }

    // Transforms (6)
    check_unary("reshape", vec![json!(0), json!([1, 2, 2, 1])], [1, 4, 1, 1], f4.clone(), None, 1e-6);
    check_unary("permute", vec![json!(0), json!([0, 2, 3, 1])], [1, 2, 2, 2], img2x2.clone(), None, 1e-6);
    check_unary("slice", vec![json!(0), json!([0, 0, 0, 0]), json!([1, 2, 1, 2])], [1, 2, 2, 2], img2x2.clone(), None, 1e-6);
    check_unary("selectAxis", vec![json!(0), json!(1), json!([3, 0, 2])], [1, 4, 1, 1], f4.clone(), None, 1e-6);
    check_unary("fillLike", vec![json!(0), json!(3.5)], [1, 4, 1, 1], f4.clone(), None, 1e-6);
    check_binary("expandLike", vec![json!(0)], [1, 1, 1, 1], vec![json!(2.25)], [1, 4, 1, 1], f4.clone(), 1e-6);

    // Linalg (4)
    check_binary("dot", vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);
    check_unary("l2Norm", vec![json!(0)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    check_binary("l2Distance", vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);
    check_binary("cosineSimilarity", vec![json!(0), json!(1e-8)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-5);

    // Statistics (8) & Reductions (4)
    for op in [
        "sum",
        "mean",
        "variancePopulation",
        "stdPopulation",
        "varianceSample",
        "stdSample",
        "min",
        "max",
    ] {
        check_unary(op, vec![json!(0)], [1, 4, 1, 1], f4.clone(), None, 1e-5);
    }
    for op in ["sumAxis", "meanAxis", "minAxis", "maxAxis"] {
        check_unary(op, vec![json!(0), json!(2)], [1, 2, 2, 2], img2x2.clone(), None, 1e-5);
    }

    // Probability (4) & Comparison/Index (2)
    check_unary("normalize", vec![json!(0)], [1, 3, 1, 1], prob_p.clone(), None, 1e-5);
    check_unary("entropy", vec![json!(0)], [1, 3, 1, 1], prob_p.clone(), None, 1e-5);
    check_binary("crossEntropy", vec![json!(0)], [1, 3, 1, 1], prob_p.clone(), [1, 3, 1, 1], prob_q.clone(), 1e-5);
    check_binary("klDivergence", vec![json!(0)], [1, 3, 1, 1], prob_p.clone(), [1, 3, 1, 1], prob_q.clone(), 1e-5);
    check_binary("lessEqual01", vec![json!(0)], [1, 4, 1, 1], f4.clone(), [1, 4, 1, 1], pos4.clone(), 1e-6);
    check_unary("indicesLike", vec![json!(0), json!(2)], [1, 2, 2, 2], img2x2.clone(), None, 1e-6);

    // 2. Fail-closed checks: bad tol, int op without --ref, ref length mismatch
    let lin_spec = math_draw::spec::load_spec_file(Path::new("examples/linear-relu.json"))
        .unwrap()
        .spec;
    assert!(diff_ref_spec(&lin_spec, None, None, 0.0).is_err());
    assert!(diff_ref_spec(&lin_spec, None, None, -1e-4).is_err());
    assert!(diff_ref_spec(&lin_spec, None, None, f64::NAN).is_err());
    assert!(diff_ref_spec(&lin_spec, None, Some(&[vec![1.0]]), 1e-4).is_err());
    assert!(diff_ref_spec(&lin_spec, None, Some(&[vec![f64::NAN, 0.0]]), 1e-4).is_err());

    let int_spec = load_spec_str(
        r#"{
          "numSlots": 3,
          "layers": [{ "constructor": "intAdd", "args": [0] }],
          "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
          "outputSlot": 2,
          "ports": [
            { "slot": 0, "shape": [1, 2, 1, 1] },
            { "slot": 1, "shape": [1, 2, 1, 1] }
          ],
          "inputs": { "0": [1, 2], "1": [3, 4] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    assert!(run_reference_f64(&int_spec, None).is_err());

    // 3. Live CLI `math-draw diff-ref` (oracle PASS + external --ref PASS & SIGNAL)
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let run_cli = |args: &[&str]| -> (i32, serde_json::Value) {
        let out = Command::new(bin).args(args).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let val: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        (code, val)
    };

    let (c, v) = run_cli(&["diff-ref", "examples/linear-relu.json", "--tol", "1e-5"]);
    assert_eq!(c, 0, "diff-ref CLI failed: {v}");
    assert_eq!(v["ok"], json!(true));
    assert_eq!(v["verdict"], json!("PASS"));
    assert_eq!(v["reference_source"], json!("pure_rust_f64"));

    let tmp_dir = std::env::temp_dir().join(format!("math_draw_diffref_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let ref_good = tmp_dir.join("ref_good.json");
    let ref_bad = tmp_dir.join("ref_bad.json");
    let ref_t = run_reference_f64(&lin_spec, None).unwrap();
    std::fs::write(&ref_good, serde_json::to_string(&ref_t.values).unwrap()).unwrap();
    let mut bad_vals = ref_t.values.clone();
    bad_vals[0] += 5.0;
    std::fs::write(&ref_bad, serde_json::to_string(&bad_vals).unwrap()).unwrap();

    let (c, v) = run_cli(&["diff-ref", "examples/linear-relu.json", "--ref", ref_good.to_str().unwrap()]);
    assert_eq!(c, 0);
    assert_eq!(v["verdict"], json!("PASS"));
    assert_eq!(v["reference_source"], json!("external_ref"));

    let (c, v) = run_cli(&["diff-ref", "examples/linear-relu.json", "--ref", ref_bad.to_str().unwrap()]);
    assert_eq!(c, 2);
    assert_eq!(v["ok"], json!(false));
    assert_eq!(v["verdict"], json!("SIGNAL"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn geom3_builders_scene_tree_ik_projection_batch_and_cli() {
    use math_draw::geom3::{
        BATCH_MAX, build_chain_spec, compile_scene_tree, project_model, run_geom_selftest,
        solve_ik,
    };
    use math_draw::harness::execute_harness_workflow;
    use std::process::Command;

    // 1. Selftest 17/17
    let st = run_geom_selftest().unwrap();
    assert_eq!(st["ok"], json!(true), "geom selftest failed: {st}");
    assert_eq!(st["passed"], json!(17));
    assert_eq!(st["total"], json!(17));

    // 2. build_chain_spec composed vs uncomposed equivalence on Burn engine + repeat op
    let chain_doc = json!({
        "input": [1.0, 2.0, 3.0],
        "chain": [
            { "op": "scale", "by": [2.0, 1.0, 0.5] },
            { "op": "euler", "roll": 15.0, "pitch": 25.0, "yaw": 35.0, "order": "ZYX" },
            { "op": "repeat", "depth": 2, "chain": [{ "op": "rotate_z", "degrees": 45.0 }] },
            { "op": "translate", "by": [4.0, -2.0, 1.5] }
        ]
    });
    let uncomposed = build_chain_spec(&chain_doc, false).unwrap();
    let composed = build_chain_spec(&chain_doc, true).unwrap();
    assert_eq!(uncomposed.steps, 4);
    assert_eq!(composed.steps, 1);
    let out_u = run_spec(&uncomposed.spec).unwrap().outputs.values;
    let out_c = run_spec(&composed.spec).unwrap().outputs.values;
    for k in 0..3 {
        assert!((out_u[k] - out_c[k]).abs() < 1e-4, "k={k}: {} vs {}", out_u[k], out_c[k]);
        assert!(
            ((out_c[k] as f64) - composed.reference_point_f64[k]).abs() < 1e-4,
            "k={k}: engine={} vs ref={}",
            out_c[k],
            composed.reference_point_f64[k]
        );
    }

    // 3. compile_scene_tree cycle/duplicate/unknown-parent fail-closed
    assert!(
        compile_scene_tree(&json!({
            "nodes": [
                { "id": "a", "parent": "b" },
                { "id": "b", "parent": "a" }
            ]
        }))
        .is_err()
    );
    assert!(
        compile_scene_tree(&json!({
            "nodes": [
                { "id": "a" },
                { "id": "a" }
            ]
        }))
        .is_err()
    );
    assert!(
        compile_scene_tree(&json!({
            "nodes": [{ "id": "a", "parent": "missing" }]
        }))
        .is_err()
    );

    // 4. solve_ik with joint limits + engine parity
    let ik_doc = json!({
        "joints": [
            { "id": "j0", "axis": "z", "length": 1.0, "initial_deg": 5.0, "min_deg": -90.0, "max_deg": 90.0 },
            { "id": "j1", "axis": "z", "length": 1.0, "initial_deg": 5.0, "min_deg": -90.0, "max_deg": 90.0 },
            { "id": "j2", "axis": "y", "length": 0.5, "initial_deg": 0.0, "min_deg": -90.0, "max_deg": 90.0 }
        ],
        "target": [1.5, 1.0, 0.3],
        "tol": 1e-3,
        "max_iters": 200
    });
    let ik = solve_ik(&ik_doc).unwrap();
    assert!(ik.ok && ik.converged, "IK did not converge: err={}", ik.final_error);
    assert!(ik.engine_max_abs_diff < 1e-4);
    for &ang in &ik.solved_angles_deg {
        assert!((-90.0..=90.0).contains(&ang));
    }

    // 5. Multi-chunk batched projection (> BATCH_MAX = 63) matches single projection + triangles clipping
    let n_verts = BATCH_MAX + 7; // 70 vertices -> 2 chunks (63 + 7)
    let mut verts = Vec::with_capacity(n_verts);
    for i in 0..n_verts {
        let a = (i as f64) * 0.1;
        verts.push([a.cos(), a.sin(), -5.0 - (i % 5) as f64]);
    }
    verts[0] = [0.0, 0.0, 5.0]; // behind camera
    let model_doc = json!({
        "vertices": verts,
        "triangles": [
            [1, 2, 3],   // fully inside
            [0, 1, 2]    // spans camera (v0 has w <= 0)
        ],
        "camera": {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "aspect": 1.5,
            "near": 0.1,
            "far": 100.0
        },
        "viewport": { "w": 640.0, "h": 480.0 }
    });
    let p_single = project_model(&model_doc, false).unwrap();
    let p_batch = project_model(&model_doc, true).unwrap();
    assert_eq!(p_batch.chunks, 2);
    assert_eq!(p_single.behind_camera, 1);
    assert_eq!(p_batch.behind_camera, 1);
    assert_eq!(p_batch.triangles_spans_camera, Some(1));
    assert_eq!(p_batch.triangles.as_ref().unwrap()[0].clipped_count, 1);
    assert!(p_batch.triangles.as_ref().unwrap()[1].spans_camera);
    for (a, b) in p_single.results.iter().zip(p_batch.results.iter()) {
        assert_eq!(a.behind, b.behind);
        for k in 0..4 {
            assert!((a.clip[k] - b.clip[k]).abs() < 1e-5);
        }
    }

    // 5b. Overflow-inducing huge viewport dimensions fail closed instead of returning Inf screen coords
    let overflow_vp_doc = json!({
        "vertices": [[10.0, 10.0, -2.0]],
        "camera": {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "aspect": 1.0,
            "near": 0.1,
            "far": 100.0
        },
        "viewport": { "w": 1e308, "h": 1e308 }
    });
    assert!(
        project_model(&overflow_vp_doc, false).is_err(),
        "expected huge viewport causing Inf screen coords to fail closed"
    );

    // 5c. Malformed explicit optional fields fail closed instead of silently substituting defaults
    let bad_aspect_doc = json!({
        "vertices": [[0.0, 0.0, -5.0]],
        "camera": {
            "eye": [0.0, 0.0, 0.0],
            "target": [0.0, 0.0, -1.0],
            "up": [0.0, 1.0, 0.0],
            "fov_deg": 60.0,
            "aspect": "16:9",
            "near": 0.1,
            "far": 100.0
        },
        "viewport": { "w": 800.0, "h": 600.0 }
    });
    assert!(project_model(&bad_aspect_doc, false).is_err());
    assert!(
        build_chain_spec(
            &json!({
                "input": [1.0, 0.0, 0.0],
                "chain": [{ "op": "euler", "roll": 10.0, "order": 123 }]
            }),
            false
        )
        .is_err()
    );
    assert!(
        solve_ik(&json!({
            "joints": [{ "id": "j0", "axis": 42, "length": 1.0 }],
            "target": [1.0, 0.0, 0.0]
        }))
        .is_err()
    );

    // 6. Workflow `diff_ref` stage + live CLI `math-draw geom` subcommands
    let tmp_dir = std::env::temp_dir().join(format!("math_draw_geom_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let wf = json!({
        "spec": "examples/linear-relu.json",
        "pipeline": [
            { "stage": "validate" },
            { "stage": "diff_ref", "tol": 1e-5 },
            { "stage": "run" }
        ]
    });
    let ws = tmp_dir.join("ws");
    let wf_rep = execute_harness_workflow(&wf, Path::new("."), Some(&ws)).unwrap();
    assert_eq!(wf_rep["ok"], json!(true), "diff_ref workflow failed: {wf_rep}");
    assert_eq!(wf_rep["stage_reports"][1]["verdict"], json!("PASS"));

    let bin = env!("CARGO_BIN_EXE_math-draw");
    let run_cli = |args: &[&str]| -> (i32, serde_json::Value) {
        let out = Command::new(bin).args(args).output().unwrap();
        let code = out.status.code().unwrap_or(-1);
        let val: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        (code, val)
    };

    let (c, v) = run_cli(&["geom", "selftest"]);
    assert_eq!(c, 0);
    assert_eq!(v["ok"], json!(true));
    assert_eq!(v["passed"], json!(17));

    let chain_p = tmp_dir.join("chain.json");
    let built_p = tmp_dir.join("built.spec.json");
    let model_p = tmp_dir.join("model.json");
    let ik_p = tmp_dir.join("ik.json");
    std::fs::write(&chain_p, serde_json::to_string(&chain_doc).unwrap()).unwrap();
    std::fs::write(&model_p, serde_json::to_string(&model_doc).unwrap()).unwrap();
    std::fs::write(&ik_p, serde_json::to_string(&ik_doc).unwrap()).unwrap();

    let (c, v) = run_cli(&[
        "geom",
        "build",
        chain_p.to_str().unwrap(),
        "--compose",
        "--out",
        built_p.to_str().unwrap(),
    ]);
    assert_eq!(c, 0);
    assert_eq!(v["ok"], json!(true));
    assert_eq!(v["composed"], json!(true));

    let (c, v) = run_cli(&["diff-ref", built_p.to_str().unwrap(), "--tol", "1e-5"]);
    assert_eq!(c, 0);
    assert_eq!(v["verdict"], json!("PASS"));

    let (c, v) = run_cli(&["geom", "solve-ik", ik_p.to_str().unwrap()]);
    assert_eq!(c, 0);
    assert_eq!(v["converged"], json!(true));

    let (c, v) = run_cli(&["geom", "batch", model_p.to_str().unwrap()]);
    assert_eq!(c, 0);
    assert_eq!(v["chunks"], json!(2));

    let (c, v) = run_cli(&["geom", "unknown-sub"]);
    assert_eq!(c, 2);
    assert_eq!(v["code"], json!("cli_unknown_command"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn muse_audit_edge_case_regressions() {
    use math_draw::diff_ref::{diff_ref_spec, run_reference_f64};
    use math_draw::direct_math::{DirectBatchKind, eval_direct_batch};
    use math_draw::geom3::{
        build_chain_spec, compile_scene_tree, dehomogenize, normal_mat3, perspective4,
    };
    use math_draw::grad_spec::{GradSpecOptions, compile_grad_spec};
    use math_draw::harness::execute_harness_workflow;
    use math_draw::liveness::auto_repair_spec;
    use std::process::Command;

    // 1. H1 + M-r2: short weights in linear / swiGlu return Err (never panic) in both grad_spec and diff_ref
    let short_lin = load_spec_str(
        r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "linear", "args": [0, 2, 2, false] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "weights": { "0": [0.5, 0.5] },
          "inputs": { "0": [1.0, 2.0] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    assert!(compile_grad_spec(&short_lin, None, &GradSpecOptions::default()).is_err());
    assert!(run_reference_f64(&short_lin, None).is_err());

    let short_swiglu = load_spec_str(
        r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "swiGlu", "args": [0, 2, 2, true] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 1, 1, 2], "layout": "feature_last" }],
          "weights": { "0": [0.5] },
          "inputs": { "0": [0.8, -0.6] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    assert!(compile_grad_spec(&short_swiglu, None, &GradSpecOptions::default()).is_err());
    assert!(run_reference_f64(&short_swiglu, None).is_err());

    // 2. H2: order=2 HVP on overwritten intermediate slot differentiates the exact same SSA-final variable
    let overwrite_spec = load_spec_str(
        r#"{
          "numSlots": 3,
          "layers": [
            { "constructor": "linear", "args": [0, 1, 1, false] },
            { "constructor": "sigmoid", "args": [1] },
            { "constructor": "exp", "args": [2] }
          ],
          "steps": [
            { "kind": "unary", "layer": 0, "slots": [0, 1] },
            { "kind": "unary", "layer": 1, "slots": [1, 1] },
            { "kind": "unary", "layer": 2, "slots": [1, 2] }
          ],
          "outputSlot": 2,
          "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
          "weights": { "0": [1.0] },
          "inputs": { "0": [0.5] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    let s_val = 1.0f32 / (1.0f32 + (-0.5f32).exp());
    let expected_exp_s = s_val.exp(); // d/ds exp(s) == d^2/ds^2 exp(s) == exp(s) ~= 1.8635
    let vjp_mid = compile_grad_spec(
        &overwrite_spec,
        None,
        &GradSpecOptions {
            wrt: "1".into(),
            order: 1,
            dout: Some(vec![1.0]),
            hvp_vec: None,
            compact_slots: false,
        },
    )
    .unwrap();
    let hvp_mid = compile_grad_spec(
        &overwrite_spec,
        None,
        &GradSpecOptions {
            wrt: "1".into(),
            order: 2,
            dout: Some(vec![1.0]),
            hvp_vec: Some(vec![1.0]),
            compact_slots: false,
        },
    )
    .unwrap();
    let vjp_out = run_spec(&vjp_mid.spec).unwrap().outputs.values[0];
    let hvp_out = run_spec(&hvp_mid.spec).unwrap().outputs.values[0];
    assert!((vjp_out - expected_exp_s).abs() < 1e-4, "vjp={vjp_out} vs {expected_exp_s}");
    assert!((hvp_out - expected_exp_s).abs() < 1e-4, "hvp={hvp_out} vs {expected_exp_s}");

    // 3. M-l3: undefined wrt slot fails closed with unresolved_wrt_slot
    let undef_slot_spec = load_spec_str(
        r#"{
          "numSlots": 4,
          "layers": [{ "constructor": "exp", "args": [0] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "inputs": { "0": [0.5, -0.5] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    let err_undef = compile_grad_spec(
        &undef_slot_spec,
        None,
        &GradSpecOptions {
            wrt: "3".into(),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err_undef.contains("unresolved_wrt_slot"), "got: {err_undef}");

    // 4. M-l5: auto_repair_spec fails closed if spec remains invalid after repair
    let bad_spec = load_spec_str(
        r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "nonexistentOp", "args": [0] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "inputs": { "0": [1.0, 2.0] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    assert!(auto_repair_spec(&bad_spec, 65535).is_err());

    // 5. H6 + M-r1: cosineSimilarity default eps=1e-12 matches engine; mixed-magnitude per-element pass rule
    let cos_default = load_spec_str(
        r#"{
          "numSlots": 3,
          "layers": [{ "constructor": "cosineSimilarity", "args": [0] }],
          "steps": [{ "kind": "binary", "layer": 0, "slots": [0, 1, 2] }],
          "outputSlot": 2,
          "ports": [
            { "slot": 0, "shape": [1, 4, 1, 1] },
            { "slot": 1, "shape": [1, 4, 1, 1] }
          ],
          "inputs": {
            "0": [1e-5, 1e-5, 1e-5, 1e-5],
            "1": [1e-5, 1e-5, 1e-5, 1e-5]
          }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    let cos_rep = diff_ref_spec(&cos_default, None, None, 1e-5).unwrap();
    assert!(cos_rep.ok && cos_rep.verdict == "PASS", "false SIGNAL on default cosineSimilarity: {cos_rep:?}");

    // Mixed magnitude [1e-8, 1e6]: abs diff at idx 1 is 0.5 (rel=5e-7 <= 1e-5), rel diff at idx 0 is 0.1 (abs=1e-9 <= 1e-5)
    let id_spec = load_spec_str(
        r#"{
          "numSlots": 2,
          "layers": [{ "constructor": "relu", "args": [0] }],
          "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
          "outputSlot": 1,
          "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
          "inputs": { "0": [1e-8, 1e6] }
        }"#,
        Path::new("."),
    )
    .unwrap()
    .spec;
    let mixed_rep = diff_ref_spec(&id_spec, None, Some(&[vec![1.1e-8, 1_000_000.5]]), 1e-5).unwrap();
    assert!(mixed_rep.ok && mixed_rep.cases[0].first_diff_index.is_none());

    // 6. H4 + M-g1 + M-g2 + M-g4 + M-g5: geom3 overflow & deep-tree guards
    assert!(
        build_chain_spec(
            &json!({
                "input": [1.0, 1.0, 1.0],
                "chain": [
                    { "op": "scale", "by": [1e308, 1e308, 1e308] },
                    { "op": "scale", "by": [10.0, 10.0, 10.0] }
                ]
            }),
            true
        )
        .is_err()
    );
    assert!(
        build_chain_spec(
            &json!({
                "input": [1.0, 1.0, 1.0],
                "chain": [{ "op": "repeat", "depth": 1024, "chain": [{ "op": "scale", "by": [2.0, 2.0, 2.0] }] }]
            }),
            true
        )
        .is_err()
    );
    let near_sing = [
        1e-320, 0.0, 0.0, 0.0,
        0.0, 1e154, 0.0, 0.0,
        0.0, 0.0, 1e154, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    assert!(normal_mat3(&near_sing).is_err());
    assert!(perspective4(1e-320, 1.0, 0.1, 100.0).is_err());
    assert!(perspective4(60.0, 1.0, 8e307, 1.7e308).is_err());
    assert!(dehomogenize([1e308, 0.0, 0.0, 1e-10]).is_err());

    let deep_nodes: Vec<serde_json::Value> = (0..520)
        .map(|i| {
            json!({
                "id": format!("n{i}"),
                "parent": if i == 0 { serde_json::Value::Null } else { json!(format!("n{}", i - 1)) },
                "point": [0.0, 0.0, 0.0]
            })
        })
        .collect();
    assert!(compile_scene_tree(&json!({ "nodes": deep_nodes })).is_err());

    // 7. H5 + M-d1 + M-d3: direct_math batch strict field allowlist, 2D->1D chaining, CLI --batch + --wrap rejection
    assert!(
        eval_direct_batch(
            DirectBatchKind::Arith,
            &json!([{ "op": "add", "a": "1,2", "axis": 5 }])
        )
        .is_err()
    );
    assert!(
        eval_direct_batch(
            DirectBatchKind::Arith,
            &json!([{ "op": "clamp", "a": "-5,0.5,9", "b": "0,1", "min": 0, "max": 1 }])
        )
        .is_err()
    );
    let mat_chain = eval_direct_batch(
        DirectBatchKind::Arith,
        &json!([
            { "op": "matmul", "a": "1,2;3,4", "b": "5,6;7,8" },
            { "op": "add", "a": "$prev", "b": [1, 1, 1, 1] }
        ]),
    )
    .unwrap();
    assert_eq!(mat_chain["results"][1]["values"], json!([20.0, 23.0, 44.0, 51.0]));

    // 8. H3 + M-w1 + M-w2 + M-w4 + M-w6: harness workflow strictness & error codes
    let tmp_dir = std::env::temp_dir().join(format!("math_draw_muse_audit_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let ws = tmp_dir.join("ws");

    // Empty pipeline rejected (M-w2)
    assert!(
        execute_harness_workflow(
            &json!({ "spec": "examples/linear-relu.json", "pipeline": [] }),
            Path::new("."),
            Some(&ws),
        )
        .is_err()
    );
    // Explicit null param rejected (M-w4) & compile.max_weights=0 rejected (M-w6)
    let null_rep = execute_harness_workflow(
        &json!({
            "spec": "examples/linear-relu.json",
            "pipeline": [{ "stage": "soak", "runs": null }]
        }),
        Path::new("."),
        Some(&ws),
    )
    .unwrap();
    assert_eq!(null_rep["ok"], json!(false));
    assert_ne!(null_rep["code"], json!("internal_error"));

    let mw0_rep = execute_harness_workflow(
        &json!({
            "spec": "examples/linear-relu.json",
            "pipeline": [{ "stage": "compile", "auto_repair": true, "max_weights": 0 }]
        }),
        Path::new("."),
        Some(&ws),
    )
    .unwrap();
    assert_eq!(mw0_rep["ok"], json!(false));
    assert_ne!(mw0_rep["code"], json!("internal_error"));

    // Stage error codes are never internal_error (M-w1)
    for stage_obj in [
        json!({ "stage": "quant_budget", "scheme": "bogus" }),
        json!({ "stage": "grad_spec", "order": 3 }),
        json!({ "stage": "gradcheck", "tol": 0.0 }),
        json!({ "stage": "fit", "targets": "does-not-exist-targets.json" }),
    ] {
        let rep = execute_harness_workflow(
            &json!({ "spec": "examples/linear-relu.json", "pipeline": [stage_obj] }),
            Path::new("."),
            Some(&ws),
        )
        .unwrap();
        assert_eq!(rep["ok"], json!(false));
        assert_ne!(rep["code"], json!("internal_error"), "stage returned internal_error: {rep}");
        assert!(rep["hint"].as_str().is_some_and(|h| !h.is_empty()), "missing hint: {rep}");
    }

    // M-d3 CLI test: `int --batch <file> --wrap` rejected with exit 2
    let batch_p = tmp_dir.join("int_batch.json");
    std::fs::write(&batch_p, r#"[{"op":"add","a":"1","b":"2"}]"#).unwrap();
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let out = Command::new(bin)
        .args(["int", "--batch", batch_p.to_str().unwrap(), "--wrap"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["code"], json!("cli_usage"));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn adversarial_verify_zoom_tol_and_bundle_unknown_fields() {
    use math_draw::bundle::{ProgramBundle, export_bundle, verify_bundle_json};
    use math_draw::spec::SpecV1;
    use math_draw::verify::{FuzzConfig, zoom_fuzz_specs_with_tol};
    use serde_json::json;
    use std::process::Command;

    let tmp_dir = std::env::temp_dir().join(format!(
        "math_draw_adv_hardening_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp_dir);
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let base_val = json!({
        "numSlots": 2,
        "layers": [{ "constructor": "linear", "args": [0, 2, 2, true] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 2, 1, 1] }],
        "weights": { "0": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] },
        "inputs": { "0": [1.0, 2.0] }
    });
    let mut pert_val = base_val.clone();
    pert_val["weights"]["0"] = json!([1.05, 0.0, 0.0, 1.0, 0.0, 0.0]);

    let base_spec: SpecV1 = serde_json::from_value(base_val.clone()).unwrap();
    let pert_spec: SpecV1 = serde_json::from_value(pert_val.clone()).unwrap();

    let base_path = tmp_dir.join("base.json");
    let pert_path = tmp_dir.join("pert.json");
    std::fs::write(&base_path, serde_json::to_string_pretty(&base_val).unwrap()).unwrap();
    std::fs::write(&pert_path, serde_json::to_string_pretty(&pert_val).unwrap()).unwrap();

    // 1. Library API: zoom_fuzz_specs_with_tol rejects invalid tolerances and gates violations
    let cfg = FuzzConfig {
        count: 10,
        seed: 42,
        lo: -5.0,
        hi: 5.0,
    };
    assert!(zoom_fuzz_specs_with_tol(&base_spec, &pert_spec, &cfg, 3, Some(-1e-4), None).is_err());
    assert!(zoom_fuzz_specs_with_tol(&base_spec, &pert_spec, &cfg, 3, None, Some(f64::NAN)).is_err());

    let rep_rej =
        zoom_fuzz_specs_with_tol(&base_spec, &pert_spec, &cfg, 4, Some(1e-4), None).unwrap();
    assert!(!rep_rej.ok);
    assert_eq!(rep_rej.verdict, Some("REJECT"));
    assert!(rep_rej.zoomed_max_abs_error > 1e-4);

    let rep_acc =
        zoom_fuzz_specs_with_tol(&base_spec, &base_spec, &cfg, 4, Some(1e-4), None).unwrap();
    assert!(rep_acc.ok);
    assert_eq!(rep_acc.verdict, Some("ACCEPT"));
    assert_eq!(rep_acc.zoomed_max_abs_error, 0.0);

    // 2. CLI verify --zoom with --abs-tol / --rel-tol fails closed with exit 2 on violation
    let bin = env!("CARGO_BIN_EXE_math-draw");
    let out_rej = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            pert_path.to_str().unwrap(),
            "--fuzz",
            "10",
            "--zoom",
            "--abs-tol",
            "1e-4",
        ])
        .output()
        .unwrap();
    assert_eq!(out_rej.status.code(), Some(2));
    let v_rej: serde_json::Value = serde_json::from_slice(&out_rej.stdout).unwrap();
    assert_eq!(v_rej["ok"], json!(false));
    assert_eq!(v_rej["verdict"], json!("REJECT"));
    assert!(v_rej["zoomed_max_abs_error"].as_f64().unwrap() > 1e-4);

    let out_rel_rej = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            pert_path.to_str().unwrap(),
            "--fuzz",
            "10",
            "--zoom",
            "--rel-tol",
            "1e-4",
        ])
        .output()
        .unwrap();
    assert_eq!(out_rel_rej.status.code(), Some(2));
    let v_rel_rej: serde_json::Value = serde_json::from_slice(&out_rel_rej.stdout).unwrap();
    assert_eq!(v_rel_rej["ok"], json!(false));
    assert_eq!(v_rel_rej["verdict"], json!("REJECT"));

    let out_acc = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            base_path.to_str().unwrap(),
            "--fuzz",
            "10",
            "--zoom",
            "--abs-tol",
            "1e-4",
        ])
        .output()
        .unwrap();
    assert_eq!(out_acc.status.code(), Some(0));
    let v_acc: serde_json::Value = serde_json::from_slice(&out_acc.stdout).unwrap();
    assert_eq!(v_acc["ok"], json!(true));
    assert_eq!(v_acc["verdict"], json!("ACCEPT"));

    // Exploratory mode without tolerance flags remains backward-compatible (ok: true, exit 0)
    let out_exp = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            pert_path.to_str().unwrap(),
            "--fuzz",
            "10",
            "--zoom",
        ])
        .output()
        .unwrap();
    assert_eq!(out_exp.status.code(), Some(0));
    let v_exp: serde_json::Value = serde_json::from_slice(&out_exp.stdout).unwrap();
    assert_eq!(v_exp["ok"], json!(true));
    assert!(v_exp.get("verdict").is_none());

    // --zoom combined with --cases fails closed with cli_usage
    let cases_path = tmp_dir.join("cases.json");
    std::fs::write(&cases_path, r#"{"cases":[{"inputs":{"0":[1.0,2.0]}}]}"#).unwrap();
    let out_cases = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            pert_path.to_str().unwrap(),
            "--zoom",
            "--cases",
            cases_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(out_cases.status.code(), Some(2));
    let v_cases: serde_json::Value = serde_json::from_slice(&out_cases.stdout).unwrap();
    assert_eq!(v_cases["code"], json!("cli_usage"));

    // 3. Bundle verify rejects injected unknown JSON keys anywhere in ProgramBundle or nested SpecV1
    let bundle = export_bundle(&base_spec, true, &[]).unwrap();
    let clean_raw = serde_json::to_string_pretty(&bundle).unwrap();
    let (_, clean_rep) = verify_bundle_json(&clean_raw).unwrap();
    assert!(clean_rep.ok);

    // Inject unknown key inside spec.layers[0].weights
    let mut injected_layer: serde_json::Value = serde_json::to_value(&bundle).unwrap();
    injected_layer["spec"]["layers"][0]["weights"] = json!([99.0, 99.0]);
    let injected_layer_raw = serde_json::to_string_pretty(&injected_layer).unwrap();

    // Direct serde deserialization of ProgramBundle also fails closed (deny_unknown_fields)
    assert!(serde_json::from_str::<ProgramBundle>(&injected_layer_raw).is_err());

    let (_, rep_layer) = verify_bundle_json(&injected_layer_raw).unwrap();
    assert!(!rep_layer.ok);
    assert_eq!(rep_layer.code.as_deref(), Some("bundle_integrity_failed"));
    assert!(
        rep_layer
            .checks
            .iter()
            .any(|c| c.name == "bundle_integrity" && !c.passed && c.detail.contains("spec.layers[0].weights")),
        "expected spec.layers[0].weights in detail: {:?}",
        rep_layer.checks
    );

    let tampered_bundle_path = tmp_dir.join("tampered_layer_key.bundle.json");
    std::fs::write(&tampered_bundle_path, &injected_layer_raw).unwrap();
    let out_bundle = Command::new(bin)
        .args(["bundle", "verify", tampered_bundle_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(out_bundle.status.code(), Some(2));
    let v_bundle: serde_json::Value = serde_json::from_slice(&out_bundle.stdout).unwrap();
    assert_eq!(v_bundle["ok"], json!(false));
    assert_eq!(v_bundle["code"], json!("bundle_integrity_failed"));

    // Inject unknown key at root level and inside spec.steps[0]
    let mut injected_root: serde_json::Value = serde_json::to_value(&bundle).unwrap();
    injected_root["smuggled_meta"] = json!("evil");
    injected_root["spec"]["steps"][0]["note"] = json!("hidden");
    let injected_root_raw = serde_json::to_string_pretty(&injected_root).unwrap();
    assert!(serde_json::from_str::<ProgramBundle>(&injected_root_raw).is_err());
    let (_, rep_root) = verify_bundle_json(&injected_root_raw).unwrap();
    assert!(!rep_root.ok);
    assert_eq!(rep_root.code.as_deref(), Some("bundle_integrity_failed"));

    // 4. All remaining unknown-field strip locations + null-valued unknown fields (M-t1)
    type MutateBundle = fn(&mut serde_json::Value);
    let remaining_cases: [(&str, MutateBundle); 8] = [
        ("spec.extra_spec", |v| { v["spec"]["extra_spec"] = json!(123); }),
        ("spec.ports[0].extra_port", |v| { v["spec"]["ports"][0]["extra_port"] = json!("x"); }),
        ("parameter_binding.extra_pb", |v| { v["parameter_binding"]["extra_pb"] = json!(1); }),
        ("parameter_binding.owners[0].extra_owner", |v| { v["parameter_binding"]["owners"][0]["extra_owner"] = json!(true); }),
        ("reference_execution.extra_exec", |v| { v["reference_execution"]["extra_exec"] = json!(0); }),
        ("attestations[0].extra_att", |v| { v["attestations"][0]["extra_att"] = json!("y"); }),
        ("null_root", |v| { v["null_root"] = serde_json::Value::Null; }),
        ("spec.null_spec", |v| { v["spec"]["null_spec"] = serde_json::Value::Null; }),
    ];
    for (expected_path, mutate_fn) in remaining_cases {
        let mut tampered: serde_json::Value = serde_json::to_value(&bundle).unwrap();
        mutate_fn(&mut tampered);
        let raw = serde_json::to_string_pretty(&tampered).unwrap();
        assert!(
            serde_json::from_str::<ProgramBundle>(&raw).is_err(),
            "serde failed to reject {expected_path}"
        );
        let (_, rep) = verify_bundle_json(&raw).unwrap();
        assert!(!rep.ok, "verify_bundle_json passed for {expected_path}");
        assert_eq!(rep.code.as_deref(), Some("bundle_integrity_failed"));
        assert!(
            rep.checks
                .iter()
                .any(|c| c.name == "bundle_integrity" && !c.passed && c.detail.contains(expected_path)),
            "expected {expected_path} in bundle_integrity detail: {:?}",
            rep.checks
        );
    }

    // verify_bundle_json Err path -> CLI bundle_invalid exit 2
    for (name, bad_payload) in [
        ("malformed.bundle.json", "{not valid json"),
        ("missing_fields.bundle.json", r#"{"schema":"burn-math.program-bundle.v1"}"#),
    ] {
        let p = tmp_dir.join(name);
        std::fs::write(&p, bad_payload).unwrap();
        let out_bad = Command::new(bin)
            .args(["bundle", "verify", p.to_str().unwrap()])
            .output()
            .unwrap();
        assert_eq!(out_bad.status.code(), Some(2));
        let v_bad: serde_json::Value = serde_json::from_slice(&out_bad.stdout).unwrap();
        assert_eq!(v_bad["ok"], json!(false));
        assert_eq!(v_bad["code"], json!("bundle_invalid"));
    }

    // 5. Tolerance boundaries: --abs-tol 0, combined --abs-tol + --rel-tol OR semantics,
    // --fuzz 0, --zoom-depth 0 / > MAX_ZOOM_DEPTH, and --zoom --cases before load_spec (M-t1/L-z1/L-z2)
    let out_zero_acc = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            base_path.to_str().unwrap(),
            "--fuzz",
            "6",
            "--zoom",
            "--abs-tol",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(out_zero_acc.status.code(), Some(0));
    let v_zero_acc: serde_json::Value = serde_json::from_slice(&out_zero_acc.stdout).unwrap();
    assert_eq!(v_zero_acc["verdict"], json!("ACCEPT"));

    let out_zero_rej = Command::new(bin)
        .args([
            "verify",
            base_path.to_str().unwrap(),
            pert_path.to_str().unwrap(),
            "--fuzz",
            "6",
            "--zoom",
            "--abs-tol",
            "0",
        ])
        .output()
        .unwrap();
    assert_eq!(out_zero_rej.status.code(), Some(2));
    let v_zero_rej: serde_json::Value = serde_json::from_slice(&out_zero_rej.stdout).unwrap();
    assert_eq!(v_zero_rej["verdict"], json!("REJECT"));

    // Combined OR semantics: large magnitude where abs > 1e-4 but rel <= 1e-4
    let mut big_base = base_val.clone();
    big_base["weights"]["0"] = json!([1000.0, 0.0, 0.0, 1000.0, 0.0, 0.0]);
    let mut big_pert = base_val.clone();
    big_pert["weights"]["0"] = json!([1000.01, 0.0, 0.0, 1000.0, 0.0, 0.0]);
    let big_base_p = tmp_dir.join("big_base.json");
    let big_pert_p = tmp_dir.join("big_pert.json");
    std::fs::write(&big_base_p, serde_json::to_string_pretty(&big_base).unwrap()).unwrap();
    std::fs::write(&big_pert_p, serde_json::to_string_pretty(&big_pert).unwrap()).unwrap();

    let out_or_acc = Command::new(bin)
        .args([
            "verify",
            big_base_p.to_str().unwrap(),
            big_pert_p.to_str().unwrap(),
            "--fuzz",
            "6",
            "--zoom",
            "--abs-tol",
            "1e-4",
            "--rel-tol",
            "1e-4",
        ])
        .output()
        .unwrap();
    assert_eq!(out_or_acc.status.code(), Some(0));
    let v_or_acc: serde_json::Value = serde_json::from_slice(&out_or_acc.stdout).unwrap();
    assert_eq!(v_or_acc["verdict"], json!("ACCEPT"));
    assert!(v_or_acc["zoomed_max_abs_error"].as_f64().unwrap() > 1e-4);

    let out_or_rej = Command::new(bin)
        .args([
            "verify",
            big_base_p.to_str().unwrap(),
            big_pert_p.to_str().unwrap(),
            "--fuzz",
            "6",
            "--zoom",
            "--abs-tol",
            "1e-4",
            "--rel-tol",
            "1e-7",
        ])
        .output()
        .unwrap();
    assert_eq!(out_or_rej.status.code(), Some(2));
    let v_or_rej: serde_json::Value = serde_json::from_slice(&out_or_rej.stdout).unwrap();
    assert_eq!(v_or_rej["verdict"], json!("REJECT"));

    // --fuzz 0 rejected, --zoom-depth 0 accepted, --zoom-depth 65 rejected
    let out_fuzz0 = Command::new(bin)
        .args(["verify", base_path.to_str().unwrap(), pert_path.to_str().unwrap(), "--zoom", "--fuzz", "0"])
        .output()
        .unwrap();
    assert_eq!(out_fuzz0.status.code(), Some(2));

    let out_depth0 = Command::new(bin)
        .args(["verify", base_path.to_str().unwrap(), base_path.to_str().unwrap(), "--zoom", "--zoom-depth", "0", "--abs-tol", "1e-4"])
        .output()
        .unwrap();
    assert_eq!(out_depth0.status.code(), Some(0));

    let out_depth65 = Command::new(bin)
        .args(["verify", base_path.to_str().unwrap(), base_path.to_str().unwrap(), "--zoom", "--zoom-depth", "65"])
        .output()
        .unwrap();
    assert_eq!(out_depth65.status.code(), Some(2));

    // --zoom --cases returns cli_usage BEFORE load_spec even when spec files do not exist (L-z1)
    let out_conflict_missing = Command::new(bin)
        .args(["verify", "no_such_a.json", "no_such_b.json", "--zoom", "--cases", "no_such_cases.json"])
        .output()
        .unwrap();
    assert_eq!(out_conflict_missing.status.code(), Some(2));
    let v_conflict_missing: serde_json::Value = serde_json::from_slice(&out_conflict_missing.stdout).unwrap();
    assert_eq!(v_conflict_missing["code"], json!("cli_usage"));

    // 6. Codex P2 / @muse M-z1: refined zoom trial runtime error fails closed instead of being silently skipped
    // base = log(x), cand = 0*x; initial samples in (0, 1] are finite, zoom chases x -> 0+ and crosses <= 0
    let log_base_val = json!({
        "numSlots": 2,
        "layers": [{ "constructor": "log", "args": [0] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
        "weights": {},
        "inputs": { "0": [0.05] }
    });
    let zero_cand_val = json!({
        "numSlots": 2,
        "layers": [{ "constructor": "linear", "args": [0, 1, 1, false] }],
        "steps": [{ "kind": "unary", "layer": 0, "slots": [0, 1] }],
        "outputSlot": 1,
        "ports": [{ "slot": 0, "shape": [1, 1, 1, 1] }],
        "weights": { "0": [0.0] },
        "inputs": { "0": [0.05] }
    });
    let log_base_spec: SpecV1 = serde_json::from_value(log_base_val.clone()).unwrap();
    let zero_cand_spec: SpecV1 = serde_json::from_value(zero_cand_val.clone()).unwrap();
    let log_base_p = tmp_dir.join("log_base.json");
    let zero_cand_p = tmp_dir.join("zero_cand.json");
    std::fs::write(&log_base_p, serde_json::to_string_pretty(&log_base_val).unwrap()).unwrap();
    std::fs::write(&zero_cand_p, serde_json::to_string_pretty(&zero_cand_val).unwrap()).unwrap();

    // Prove initial sweep (depth 0) is clean (0 errors) while refined zoom (depth 5) crosses <= 0
    let err_cfg = FuzzConfig {
        count: 4,
        seed: FuzzConfig::default().seed,
        lo: -0.02,
        hi: 2.0,
    };
    let rep_d0 =
        zoom_fuzz_specs_with_tol(&log_base_spec, &zero_cand_spec, &err_cfg, 0, Some(1e9), None)
            .unwrap();
    assert!(rep_d0.ok && rep_d0.error_trials == 0, "expected initial sweep to be clean: {rep_d0:?}");

    // Even with huge abs_tol = 1e9, a domain error on a refined zoom trial must REJECT
    let rep_err =
        zoom_fuzz_specs_with_tol(&log_base_spec, &zero_cand_spec, &err_cfg, 5, Some(1e9), None)
            .unwrap();
    assert!(!rep_err.ok, "zoom gate falsely accepted despite refined runtime error trial");
    assert_eq!(rep_err.verdict, Some("REJECT"));
    assert!(rep_err.error_trials >= 1);
    assert!(rep_err.first_error.is_some());

    // Exploratory mode also propagates the refined runtime error rather than silently skipping it
    assert!(
        zoom_fuzz_specs_with_tol(&log_base_spec, &zero_cand_spec, &err_cfg, 5, None, None).is_err()
    );

    // CLI verify --zoom --abs-tol 1e9 fails closed with exit 2 and verdict REJECT
    let out_err_cli = Command::new(bin)
        .args([
            "verify",
            log_base_p.to_str().unwrap(),
            zero_cand_p.to_str().unwrap(),
            "--zoom",
            "--fuzz",
            "4",
            "--lo",
            "-0.02",
            "--hi",
            "2.0",
            "--zoom-depth",
            "5",
            "--abs-tol",
            "1e9",
        ])
        .output()
        .unwrap();
    assert_eq!(out_err_cli.status.code(), Some(2));
    let v_err_cli: serde_json::Value = serde_json::from_slice(&out_err_cli.stdout).unwrap();
    assert_eq!(v_err_cli["ok"], json!(false));
    assert_eq!(v_err_cli["verdict"], json!("REJECT"));
    assert!(v_err_cli["error_trials"].as_u64().unwrap() >= 1);

    let _ = std::fs::remove_dir_all(&tmp_dir);
}





