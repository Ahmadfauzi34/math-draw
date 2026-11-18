use serde_json::{Value, json};
use std::collections::BTreeMap;

use super::certify::{check_properties, compose_properties, evaluate_weight_expr, gate_sets};
use super::linalg::engine_probe;
use super::{WeightProbe, WeightRequires, WeightSet};

pub fn run_weight_lab_selftest() -> Result<Value, String> {
    let mut results = Vec::new();
    let mut record = |name: &str, ok: bool, detail: Value| {
        results.push(json!({ "name": name, "ok": ok, "detail": detail }));
    };

    // 1. Elastic preset execution
    let w_el = vec![vec![1.2, -0.4], vec![0.5, 0.8]];
    let b_el = vec![20.0, -10.0];
    let eng = &engine_probe(&w_el, &b_el, &[vec![80.0, 80.0]])?[0];
    let ok1 = (eng[0] - 84.0).abs() < 1e-5 && (eng[1] - 94.0).abs() < 1e-5;
    record("engine elastic [80,80] = [84,94]", ok1, json!({ "engine": eng }));

    // 2. Stable vs unstable spectral radius
    let inv_ws = WeightSet {
        name: Some("inv".into()),
        kind: "dynamics".into(),
        w: vec![vec![-0.8, 0.0], vec![0.0, -0.8]],
        b: vec![0.0, 0.0],
        probes: vec![],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    let p_inv = check_properties(&inv_ws)?;
    record(
        "inverted preset: spectral radius 0.8 < 1",
        p_inv.len() == 1 && p_inv[0].ok,
        json!({ "rho": p_inv[0].value }),
    );

    let unst_ws = WeightSet {
        name: Some("unst".into()),
        kind: "dynamics".into(),
        w: vec![vec![2.0, 0.0], vec![0.0, 0.5]],
        b: vec![0.0, 0.0],
        probes: vec![],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    let p_un = check_properties(&unst_ws)?;
    record(
        "unstable W (rho=2) is flagged",
        p_un.len() == 1 && !p_un[0].ok,
        json!({ "rho": p_un[0].value }),
    );

    // 3 & 4. Gate ACCEPT & REJECT
    let cur = WeightSet {
        name: Some("elastic".into()),
        kind: "map".into(),
        w: w_el.clone(),
        b: b_el.clone(),
        probes: vec![WeightProbe {
            input: vec![80.0, 80.0],
            want: Some(vec![84.0, 94.0]),
        }],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    let g_acc = gate_sets(&cur, &cur, 1.0)?;
    record(
        "gate identical set -> ACCEPT",
        g_acc.ok,
        json!({ "max_jump": g_acc.max_jump }),
    );

    let mut pert = cur.clone();
    pert.w[0][0] += 5.0;
    let g_rej = gate_sets(&cur, &pert, 1.0)?;
    record(
        "gate perturbed set -> REJECT",
        !g_rej.ok,
        json!({ "max_jump": g_rej.max_jump }),
    );

    // 5. Norm-bound composition
    let dyn_set = |s: f64, b: Vec<f64>| WeightSet {
        name: Some(format!("d{s}")),
        kind: "dynamics".into(),
        w: vec![vec![s, 0.0], vec![0.0, s]],
        b,
        probes: vec![],
        probe_tol: None,
        requires: WeightRequires::default(),
    };
    let c_norm = compose_properties(&dyn_set(0.5, vec![1.0, 2.0]), &dyn_set(0.6, vec![3.0, 4.0]))?;
    record(
        "compose 0.5I+0.6I -> norm_bound COMPOSED",
        c_norm.ok && c_norm.method == "norm_bound",
        c_norm.certificate,
    );

    // 6. Orthonormal composition Rz(30) + Rz(15) = Rz(45)
    let rz = |deg: f64, nm: &str| {
        let rad = deg.to_radians();
        let (s, c) = rad.sin_cos();
        WeightSet {
            name: Some(nm.into()),
            kind: "calibration".into(),
            w: vec![vec![c, -s], vec![s, c]],
            b: vec![0.0, 0.0],
            probes: vec![],
            probe_tol: None,
            requires: WeightRequires {
                det_positive: Some(true),
                ..Default::default()
            },
        }
    };
    let c_rot = compose_properties(&rz(30.0, "r30"), &rz(15.0, "r15"))?;
    record(
        "compose Rz30+Rz15 -> orthonormal COMPOSED",
        c_rot.ok && c_rot.method == "orthonormal",
        c_rot.certificate,
    );

    // 7. Bound refuses (-0.8I)^2 (Frobenius norm 1.28 >= 1), measured accepts (rho = 0.64 < 1)
    let c_inv = compose_properties(&inv_ws, &inv_ws)?;
    record(
        "compose inverted^2 -> bound refuses, measured ACCEPTS",
        c_inv.ok && c_inv.method == "measured",
        c_inv.certificate,
    );

    // 8. Rectangular 3x2 after 2x3 + recursive weight expression tree
    let mut rep = BTreeMap::new();
    rep.insert(
        "up_3x2".into(),
        WeightSet {
            name: Some("up_3x2".into()),
            kind: "map".into(),
            w: vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
            b: vec![1.0, 2.0, 3.0],
            probes: vec![],
            probe_tol: None,
            requires: WeightRequires::default(),
        },
    );
    rep.insert(
        "down_2x3".into(),
        WeightSet {
            name: Some("down_2x3".into()),
            kind: "map".into(),
            w: vec![vec![1.0, 1.0, 0.0], vec![0.0, 1.0, 1.0]],
            b: vec![-1.0, -2.0],
            probes: vec![],
            probe_tol: None,
            requires: WeightRequires::default(),
        },
    );
    let expr = json!({
        "op": "repeat",
        "steps": 2,
        "expr": {
            "op": "compose",
            "first": { "op": "ref", "name": "up_3x2" },
            "second": { "op": "ref", "name": "down_2x3" }
        }
    });
    let ev = evaluate_weight_expr(&expr, &rep)?;
    record(
        "rectangular 3x2 & recursive weight expression tree",
        ev.ok && ev.w.len() == 2 && ev.w[0].len() == 2,
        json!({ "W": ev.w, "b": ev.b }),
    );

    let all_ok = results
        .iter()
        .all(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false));
    Ok(json!({
        "ok": all_ok,
        "passed": results.iter().filter(|r| r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false)).count(),
        "total": results.len(),
        "tests": results,
    }))
}
