use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

use super::linalg::{
    blend_bias, blend_weights, det, engine_probe, frobenius_norm, mat_mul, mat_vec_mul,
    orthonormality_error, spectral_radius, weight_fingerprint,
};
use super::{WeightRequires, WeightSet};

#[derive(Debug, Clone, Serialize)]
pub struct PropertyCheck {
    pub name: &'static str,
    pub ok: bool,
    pub value: f64,
    pub bound: f64,
}

pub fn check_properties(set: &WeightSet) -> Result<Vec<PropertyCheck>, String> {
    set.validate_dims()?;
    let mut out = Vec::new();
    match set.kind.as_str() {
        "dynamics" => {
            let bound = set.requires.spectral_radius_lt.unwrap_or(1.0);
            let (rho, _) = spectral_radius(&set.w, 500)?;
            out.push(PropertyCheck {
                name: "spectral_radius_lt",
                ok: rho < bound,
                value: rho,
                bound,
            });
        }
        "calibration" => {
            let tol = set.requires.orthonormal_tol.unwrap_or(1e-6);
            let e = orthonormality_error(&set.w);
            out.push(PropertyCheck {
                name: "orthonormal",
                ok: e <= tol,
                value: e,
                bound: tol,
            });
            if set.requires.det_positive.unwrap_or(false) {
                let d = det(&set.w).unwrap_or(f64::NAN);
                out.push(PropertyCheck {
                    name: "det_positive",
                    ok: d > 0.0,
                    value: d,
                    bound: 0.0,
                });
            }
        }
        _ => {}
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    pub ok: bool,
    pub verdict: &'static str,
    pub properties_ok: bool,
    pub property_failures: Vec<PropertyCheck>,
    pub max_jump: f64,
    pub jump_tol: f64,
    pub jump_ok: bool,
    pub probes: usize,
}

pub fn gate_sets(cur: &WeightSet, cand: &WeightSet, jump_tol: f64) -> Result<GateReport, String> {
    let (_, in_cur) = cur.validate_dims()?;
    let (_, in_cand) = cand.validate_dims()?;
    if in_cur != in_cand || cur.w.len() != cand.w.len() {
        return Err(
            "gate_sets: current and candidate weight sets must have matching dimensions".into(),
        );
    }
    let props = check_properties(cand)?;
    let prop_fails: Vec<PropertyCheck> = props.into_iter().filter(|p| !p.ok).collect();

    let mut seen = BTreeMap::new();
    for p in cur.probes.iter().chain(cand.probes.iter()) {
        let key = format!("{:?}", p.input);
        seen.entry(key).or_insert_with(|| p.input.clone());
    }
    let inputs: Vec<Vec<f64>> = if seen.is_empty() {
        vec![vec![0.0; in_cand]]
    } else {
        seen.into_values().collect()
    };

    let out_cur = engine_probe(&cur.w, &cur.b, &inputs)?;
    let out_cand = engine_probe(&cand.w, &cand.b, &inputs)?;
    let mut max_jump = 0.0f64;
    for (oc, on) in out_cur.iter().zip(out_cand.iter()) {
        for (&a, &b) in oc.iter().zip(on.iter()) {
            max_jump = max_jump.max((b - a).abs());
        }
    }
    let jump_ok = max_jump <= jump_tol;
    let properties_ok = prop_fails.is_empty();
    let accepted = properties_ok && jump_ok;

    Ok(GateReport {
        ok: accepted,
        verdict: if accepted { "ACCEPT" } else { "REJECT" },
        properties_ok,
        property_failures: prop_fails,
        max_jump,
        jump_tol,
        jump_ok,
        probes: inputs.len(),
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct ComposeReport {
    pub ok: bool,
    pub name: String,
    pub kind: String,
    #[serde(rename = "W")]
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub verdict: &'static str,
    pub method: &'static str,
    pub certificate: Value,
    pub note: &'static str,
}

pub fn compose_properties(a: &WeightSet, b: &WeightSet) -> Result<ComposeReport, String> {
    let (out_a, in_a) = a.validate_dims()?;
    let (out_b, in_b) = b.validate_dims()?;
    if out_a != in_b {
        return Err(format!(
            "compose_properties: inner dim mismatch (A out_dim {out_a} != B in_dim {in_b})"
        ));
    }
    let w = mat_mul(&b.w, &a.w)?;
    let w_ba = mat_vec_mul(&b.w, &a.b)?;
    let bias: Vec<f64> = w_ba
        .into_iter()
        .zip(b.b.iter())
        .map(|(x, &bi)| x + bi)
        .collect();
    let is_square = out_a == in_a && out_b == in_b && w.len() == w[0].len();
    let name = format!(
        "{}_then_{}",
        a.name.as_deref().unwrap_or("A"),
        b.name.as_deref().unwrap_or("B")
    );

    if is_square && a.kind == "calibration" && b.kind == "calibration" {
        let tol = a
            .requires
            .orthonormal_tol
            .unwrap_or(1e-6)
            .min(b.requires.orthonormal_tol.unwrap_or(1e-6));
        let ea = orthonormality_error(&a.w);
        let eb = orthonormality_error(&b.w);
        let pre_ok = ea <= tol && eb <= tol;
        if !pre_ok {
            return Ok(ComposeReport {
                ok: false,
                name,
                kind: "map".into(),
                w,
                b: bias,
                verdict: "REJECTED",
                method: "orthonormal",
                certificate: json!({
                    "checks": [{ "name": "factors_orthonormal", "ok": false, "err_A": ea, "err_B": eb, "tol": tol }]
                }),
                note: "factors not orthonormal within tol; no composition claim",
            });
        }
        let ec = orthonormality_error(&w);
        let prod_ok = ec <= tol;
        let det_pos_req =
            a.requires.det_positive.unwrap_or(false) && b.requires.det_positive.unwrap_or(false);
        let dc = if det_pos_req { Some(det(&w)?) } else { None };
        let det_ok = dc.map(|d| d > 0.0).unwrap_or(true);
        let all_ok = prod_ok && det_ok;
        return Ok(ComposeReport {
            ok: all_ok,
            name,
            kind: "calibration".into(),
            w,
            b: bias,
            verdict: if all_ok { "COMPOSED" } else { "REJECTED" },
            method: "orthonormal",
            certificate: json!({
                "rule": "(WB WA)(WB WA)^T = I; det(WB WA) = det(WB)det(WA)",
                "det_product": dc,
                "checks": [
                    { "name": "factors_orthonormal", "ok": true, "err_A": ea, "err_B": eb, "tol": tol },
                    { "name": "product_orthonormal", "ok": prod_ok, "err_product": ec, "tol": tol },
                ]
            }),
            note: "orthonormality proven by the product identity",
        });
    }

    if is_square && a.kind == "dynamics" && b.kind == "dynamics" {
        let na = frobenius_norm(&a.w);
        let nb = frobenius_norm(&b.w);
        let bound = na * nb;
        let rho_bound = a
            .requires
            .spectral_radius_lt
            .unwrap_or(1.0)
            .min(b.requires.spectral_radius_lt.unwrap_or(1.0));
        let refused = bound >= rho_bound;
        if !refused {
            let nc = frobenius_norm(&w);
            let (rho, _) = spectral_radius(&w, 500)?;
            let cons_ok = rho <= nc + 1e-9 && nc <= bound + 1e-9;
            return Ok(ComposeReport {
                ok: cons_ok,
                name,
                kind: "dynamics".into(),
                w,
                b: bias,
                verdict: if cons_ok { "COMPOSED" } else { "REJECTED" },
                method: "norm_bound",
                certificate: json!({
                    "rule": "||WB WA||_F <= ||WA||_F ||WB||_F < 1 => strict contraction",
                    "checks": [
                        { "name": "norm_product_bound", "ok": true, "norm_A": na, "norm_B": nb, "bound": bound, "rho_bound": rho_bound, "refused": false },
                        { "name": "consistency_rho_le_frob_le_bound", "ok": cons_ok, "rho_product": rho, "frob_product": nc, "bound": bound }
                    ]
                }),
                note: "contraction certified by the Frobenius norm bound",
            });
        }
        let (rho, _) = spectral_radius(&w, 500)?;
        let meas_ok = rho < rho_bound;
        return Ok(ComposeReport {
            ok: meas_ok,
            name,
            kind: if meas_ok { "dynamics" } else { "map" }.into(),
            w,
            b: bias,
            verdict: if meas_ok { "COMPOSED" } else { "REJECTED" },
            method: "measured",
            certificate: json!({
                "rule": "norm bound refused (>= 1); product measured directly",
                "checks": [
                    { "name": "norm_product_bound", "ok": false, "norm_A": na, "norm_B": nb, "bound": bound, "rho_bound": rho_bound, "refused": true },
                    { "name": "measured_rho", "ok": meas_ok, "rho_product": rho, "rho_bound": rho_bound }
                ]
            }),
            note: "bound refusal is not a rejection — the product was measured instead",
        });
    }

    Ok(ComposeReport {
        ok: true,
        name,
        kind: "map".into(),
        w,
        b: bias,
        verdict: "COMPOSED",
        method: "measured",
        certificate: json!({
            "rule": "none — kinds do not compose semantically; affine fusion only",
            "checks": []
        }),
        note: "affine fusion is exact (W_C = W_B W_A, b_C = W_B b_A + b_B)",
    })
}

pub fn evaluate_weight_expr(
    expr: &Value,
    repertoire: &BTreeMap<String, WeightSet>,
) -> Result<ComposeReport, String> {
    let obj = expr
        .as_object()
        .ok_or_else(|| "evaluate_weight_expr: expr must be a JSON object".to_string())?;

    if obj.contains_key("W") && obj.contains_key("b") && !obj.contains_key("op") {
        let ws: WeightSet = serde_json::from_value(expr.clone()).map_err(|e| e.to_string())?;
        ws.validate_dims()?;
        return Ok(ComposeReport {
            ok: true,
            name: ws.name.unwrap_or_else(|| "literal".into()),
            kind: ws.kind,
            w: ws.w,
            b: ws.b,
            verdict: "LITERAL",
            method: "literal",
            certificate: json!({}),
            note: "literal weight set",
        });
    }

    let op = obj
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "evaluate_weight_expr: missing 'op'".to_string())?;

    match op {
        "ref" => {
            let name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| "evaluate_weight_expr: ref requires 'name'".to_string())?;
            let ws = repertoire
                .get(name)
                .ok_or_else(|| format!("evaluate_weight_expr: unknown weight set '{name}'"))?;
            ws.validate_dims()?;
            Ok(ComposeReport {
                ok: true,
                name: ws.name.clone().unwrap_or_else(|| name.to_string()),
                kind: ws.kind.clone(),
                w: ws.w.clone(),
                b: ws.b.clone(),
                verdict: "CERTIFIED_REF",
                method: "ref",
                certificate: json!({ "ref": name }),
                note: "resolved from repertoire",
            })
        }
        "literal" => {
            let ws: WeightSet = serde_json::from_value(expr.clone()).map_err(|e| e.to_string())?;
            ws.validate_dims()?;
            Ok(ComposeReport {
                ok: true,
                name: ws.name.unwrap_or_else(|| "literal".into()),
                kind: ws.kind,
                w: ws.w,
                b: ws.b,
                verdict: "LITERAL",
                method: "literal",
                certificate: json!({}),
                note: "literal weight set",
            })
        }
        "blend" => {
            let a_val = obj
                .get("a")
                .or_else(|| obj.get("first"))
                .ok_or_else(|| "blend requires 'a' (or 'first')".to_string())?;
            let b_val = obj
                .get("b")
                .or_else(|| obj.get("second"))
                .ok_or_else(|| "blend requires 'b' (or 'second')".to_string())?;
            let alpha = obj.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.5);
            let ra = evaluate_weight_expr(a_val, repertoire)?;
            let rb = evaluate_weight_expr(b_val, repertoire)?;
            let w = blend_weights(&ra.w, &rb.w, alpha)?;
            let b = blend_bias(&ra.b, &rb.b, alpha)?;
            let kind = if ra.kind == rb.kind {
                ra.kind.clone()
            } else {
                "map".to_string()
            };
            let ws = WeightSet {
                name: Some(format!("blend({},{})", ra.name, rb.name)),
                kind: kind.clone(),
                w: w.clone(),
                b: b.clone(),
                probes: vec![],
                probe_tol: None,
                requires: WeightRequires::default(),
            };
            let props = if kind != "map" {
                check_properties(&ws)?
            } else {
                vec![]
            };
            let ok = props.iter().all(|p| p.ok);
            Ok(ComposeReport {
                ok,
                name: ws.name.unwrap(),
                kind: if ok { kind } else { "map".into() },
                w,
                b,
                verdict: if ok { "BLENDED" } else { "REJECTED" },
                method: "blend",
                certificate: json!({ "alpha": alpha, "properties": props }),
                note: "linear interpolation of weight sets",
            })
        }
        "repeat" => {
            let depth = obj
                .get("depth")
                .or_else(|| obj.get("steps"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            if !(1..=64).contains(&depth) {
                return Err(format!(
                    "evaluate_weight_expr: repeat depth must be 1..=64, got {depth}"
                ));
            }
            let inner = obj
                .get("expr")
                .ok_or_else(|| "repeat requires 'expr'".to_string())?;
            let base = evaluate_weight_expr(inner, repertoire)?;
            let base_ws = WeightSet {
                name: Some(base.name.clone()),
                kind: base.kind.clone(),
                w: base.w.clone(),
                b: base.b.clone(),
                probes: vec![],
                probe_tol: None,
                requires: WeightRequires::default(),
            };
            let mut acc = base;
            for _ in 1..depth {
                let acc_ws = WeightSet {
                    name: Some(acc.name.clone()),
                    kind: acc.kind.clone(),
                    w: acc.w.clone(),
                    b: acc.b.clone(),
                    probes: vec![],
                    probe_tol: None,
                    requires: WeightRequires::default(),
                };
                acc = compose_properties(&acc_ws, &base_ws)?;
                if acc.verdict == "REJECTED" {
                    break;
                }
            }
            acc.name = format!("repeat({},{depth})", base_ws.name.unwrap_or_default());
            Ok(acc)
        }
        "compose" => {
            let list: Vec<Value> = if let Some(arr) =
                obj.get("steps").or_else(|| obj.get("chain")).and_then(|v| v.as_array())
            {
                arr.clone()
            } else if let (Some(f), Some(s)) = (obj.get("first"), obj.get("second")) {
                vec![f.clone(), s.clone()]
            } else {
                return Err("compose requires 'steps' array or 'first'/'second'".into());
            };
            if list.is_empty() {
                return Err("compose steps must be non-empty".into());
            }
            let mut acc = evaluate_weight_expr(&list[0], repertoire)?;
            for item in list.iter().skip(1) {
                let next = evaluate_weight_expr(item, repertoire)?;
                let acc_ws = WeightSet {
                    name: Some(acc.name.clone()),
                    kind: acc.kind.clone(),
                    w: acc.w.clone(),
                    b: acc.b.clone(),
                    probes: vec![],
                    probe_tol: None,
                    requires: WeightRequires::default(),
                };
                let next_ws = WeightSet {
                    name: Some(next.name.clone()),
                    kind: next.kind.clone(),
                    w: next.w.clone(),
                    b: next.b.clone(),
                    probes: vec![],
                    probe_tol: None,
                    requires: WeightRequires::default(),
                };
                acc = compose_properties(&acc_ws, &next_ws)?;
                if acc.verdict == "REJECTED" {
                    break;
                }
            }
            Ok(acc)
        }
        other => Err(format!("evaluate_weight_expr: unknown op '{other}'")),
    }
}

pub fn certify_weight_set(
    set: &WeightSet,
    default_name: &str,
    repertoire_path: Option<&Path>,
) -> Result<Value, String> {
    let props = check_properties(set)?;
    if props.iter().any(|p| !p.ok) {
        return Ok(json!({
            "ok": false,
            "verdict": "REFUSED",
            "reason": "property checks failed",
            "properties": props,
        }));
    }
    let probe_tol = set.probe_tol.unwrap_or(1e-4);
    let mut probe_max_err = 0.0f64;
    let mut probe_results = Vec::new();
    if !set.probes.is_empty() {
        let inputs: Vec<Vec<f64>> = set.probes.iter().map(|p| p.input.clone()).collect();
        let outs = engine_probe(&set.w, &set.b, &inputs)?;
        for (p, got) in set.probes.iter().zip(outs.iter()) {
            let js_ref: Vec<f64> = mat_vec_mul(&set.w, &p.input)?
                .into_iter()
                .zip(set.b.iter())
                .map(|(x, &bi)| x + bi)
                .collect();
            let target = p.want.as_ref().unwrap_or(&js_ref);
            let err = got
                .iter()
                .zip(target.iter())
                .map(|(&g, &w)| (g - w).abs())
                .fold(0.0f64, f64::max);
            probe_max_err = probe_max_err.max(err);
            probe_results.push(json!({
                "in": p.input,
                "got": got,
                "want": target,
                "max_abs_err": err,
            }));
        }
        if !probe_max_err.is_finite() || probe_max_err > probe_tol {
            return Ok(json!({
                "ok": false,
                "verdict": "REFUSED",
                "reason": format!("probe verification failed: probe_max_abs_err {probe_max_err} > probe_tol {probe_tol}"),
                "probe_max_abs_err": probe_max_err,
                "probe_tol": probe_tol,
                "probe_results": probe_results,
                "properties": props,
            }));
        }
    }

    let name = set.name.clone().unwrap_or_else(|| default_name.to_string());
    let fp = weight_fingerprint(&set.w, &set.b);
    let entry = json!({
        "name": name,
        "kind": set.kind,
        "W": set.w,
        "b": set.b,
        "fingerprint": fp,
        "properties": props,
        "probe_max_abs_err": probe_max_err,
        "probes": set.probes.len(),
        "engine": "f32 math-draw (burn-flex 0.22.1)",
    });

    if let Some(rp) = repertoire_path {
        let mut rep_val: Value = if rp.exists() {
            let raw = std::fs::read_to_string(rp).map_err(|e| e.to_string())?;
            serde_json::from_str(&raw).unwrap_or_else(|_| json!({ "version": 1, "sets": [] }))
        } else {
            json!({ "version": 1, "sets": [] })
        };
        let sets_arr = rep_val
            .get_mut("sets")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| "repertoire JSON missing 'sets' array".to_string())?;
        sets_arr.retain(|e| e.get("name").and_then(|n| n.as_str()) != Some(&name));
        sets_arr.push(entry.clone());
        let count = sets_arr.len();
        if let Some(parent) = rp.parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(
            rp,
            format!("{}\n", serde_json::to_string_pretty(&rep_val).unwrap()),
        )
        .map_err(|e| e.to_string())?;
        return Ok(json!({
            "ok": true,
            "verdict": "CERTIFIED",
            "entry": entry,
            "repertoire": rp.display().to_string(),
            "sets": count,
        }));
    }

    Ok(json!({
        "ok": true,
        "verdict": "CERTIFIED",
        "entry": entry,
    }))
}
