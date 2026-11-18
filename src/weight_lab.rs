use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use crate::engine::PreparedGraph;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WeightRequires {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_radius_lt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orthonormal_tol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub det_positive: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeightProbe {
    #[serde(rename = "in", alias = "input")]
    pub input: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub want: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeightSet {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_weight_kind")]
    pub kind: String,
    #[serde(rename = "W")]
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    #[serde(default)]
    pub probes: Vec<WeightProbe>,
    #[serde(default)]
    pub probe_tol: Option<f64>,
    #[serde(default)]
    pub requires: WeightRequires,
}

fn default_weight_kind() -> String {
    "map".to_string()
}

impl WeightSet {
    pub fn validate_dims(&self) -> Result<(usize, usize), String> {
        let out_dim = self.w.len();
        if out_dim == 0 {
            return Err("W must be a non-empty 2D matrix".into());
        }
        let in_dim = self.w[0].len();
        if in_dim == 0 || !self.w.iter().all(|r| r.len() == in_dim) {
            return Err("all rows of W must have equal non-zero length".into());
        }
        if self.b.len() != out_dim {
            return Err(format!(
                "b length ({}) must match W row count out_dim ({out_dim})",
                self.b.len()
            ));
        }
        Ok((out_dim, in_dim))
    }
}

pub fn mat_vec_mul(w: &[Vec<f64>], v: &[f64]) -> Result<Vec<f64>, String> {
    if w.is_empty() || w[0].len() != v.len() {
        return Err(format!(
            "mat_vec_mul: W cols ({}) != v len ({})",
            w.first().map(|r| r.len()).unwrap_or(0),
            v.len()
        ));
    }
    Ok(w.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(&wij, &vj)| wij * vj).sum())
        .collect())
}

pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
        return Err("mat_mul: inner dimensions mismatch".into());
    }
    let n = a.len();
    let k = b.len();
    let m = b[0].len();
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for t in 0..k {
            let ait = a[i][t];
            if ait == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += ait * b[t][j];
            }
        }
    }
    Ok(out)
}

/// Convert row-major math matrix W (out_dim x in_dim) to Burn linear weight layout [in_dim, out_dim].
pub fn to_column_major(w: &[Vec<f64>]) -> Result<Vec<f32>, String> {
    let out_dim = w.len();
    if out_dim == 0 || w[0].is_empty() {
        return Err("to_column_major: W must be a non-empty 2D matrix".into());
    }
    let in_dim = w[0].len();
    if !w.iter().all(|r| r.len() == in_dim) {
        return Err("to_column_major: all rows of W must have equal length".into());
    }
    let mut out = Vec::with_capacity(in_dim * out_dim);
    for j in 0..in_dim {
        for row in w.iter().take(out_dim) {
            out.push(row[j] as f32);
        }
    }
    Ok(out)
}

pub fn affine_spec(w: &[Vec<f64>], b: &[f64], v: &[f64]) -> Result<SpecV1, String> {
    let out_dim = w.len();
    let in_dim = w.first().map(|r| r.len()).unwrap_or(0);
    if out_dim == 0 || in_dim == 0 || !w.iter().all(|r| r.len() == in_dim) {
        return Err("affine_spec: invalid W dimensions".into());
    }
    if b.len() != out_dim || v.len() != in_dim {
        return Err(format!(
            "affine_spec: b len ({}) != out_dim ({out_dim}) or v len ({}) != in_dim ({in_dim})",
            b.len(),
            v.len()
        ));
    }
    let mut weights = BTreeMap::new();
    weights.insert("0".to_string(), to_column_major(w)?);
    let mut inputs = BTreeMap::new();
    inputs.insert(
        "0".to_string(),
        v.iter().map(|&x| Value::from(x)).collect(),
    );
    inputs.insert(
        "1".to_string(),
        b.iter().map(|&x| Value::from(x)).collect(),
    );

    Ok(SpecV1 {
        num_slots: 4,
        layers: vec![
            LayerDecl {
                constructor: "linear".into(),
                args: vec![
                    Value::from(100),
                    Value::from(in_dim),
                    Value::from(out_dim),
                    Value::from(false),
                ],
            },
            LayerDecl {
                constructor: "add".into(),
                args: vec![Value::from(200)],
            },
        ],
        steps: vec![
            StepDecl {
                kind: "unary".into(),
                layer: 0,
                slots: vec![0, 2],
            },
            StepDecl {
                kind: "binary".into(),
                layer: 1,
                slots: vec![2, 1, 3],
            },
        ],
        output_slot: 3,
        ports: vec![
            PortDecl {
                slot: 0,
                role: "observation".into(),
                shape: [1, in_dim, 1, 1],
                layout: "feature_axis1_singleton".into(),
            },
            PortDecl {
                slot: 1,
                role: "observation".into(),
                shape: [1, out_dim, 1, 1],
                layout: "feature_axis1_singleton".into(),
            },
        ],
        weights,
        inputs,
    })
}

pub fn engine_probe(w: &[Vec<f64>], b: &[f64], inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    if inputs.is_empty() {
        return Ok(vec![]);
    }
    let out_dim = w.len();
    let spec = affine_spec(w, b, &inputs[0])?;
    let graph = PreparedGraph::compile(&spec)?;
    let b_vals: Vec<Value> = b.iter().map(|&x| Value::from(x)).collect();
    let mut outs = Vec::with_capacity(inputs.len());
    for v in inputs {
        let mut ov = BTreeMap::new();
        ov.insert(
            "0".to_string(),
            v.iter().map(|&x| Value::from(x)).collect(),
        );
        ov.insert("1".to_string(), b_vals.clone());
        let run = graph.run_once(&spec, Some(&ov))?;
        outs.push(
            run.values
                .iter()
                .take(out_dim)
                .map(|&x| x as f64)
                .collect(),
        );
    }
    Ok(outs)
}

/// Dominant eigenvalue magnitude via power iteration + 2-step geometric norm sqrt(||W^2 v||_2).
pub fn spectral_radius(w: &[Vec<f64>], iters: usize) -> Result<(f64, usize), String> {
    let n = w.len();
    if n == 0 || !w.iter().all(|r| r.len() == n) {
        return Err("spectral_radius: W must be a non-empty square matrix".into());
    }
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    let inv_sqrt = 1.0 / (n as f64).sqrt();
    seeds.push(vec![inv_sqrt; n]);
    for k in 0..n.min(4) {
        let mut e = vec![0.0; n];
        e[k] = 1.0;
        seeds.push(e);
    }
    // Deterministic pseudo-random unit vector
    let mut state = 0x5eed_1234u32;
    let mut rv = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x6D2B79F5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        let u = ((t ^ (t >> 14)) as f64) / 4294967296.0;
        rv.push(u * 2.0 - 1.0);
    }
    let r_norm = rv.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    seeds.push(rv.into_iter().map(|x| x / r_norm).collect());

    let mut best_rho = 0.0f64;
    let mut best_iters = 0usize;
    for init_v in seeds {
        let mut v = init_v;
        let mut rho = 0.0f64;
        let mut used_iters = iters;
        for it in 0..iters {
            let wv = mat_vec_mul(w, &v)?;
            let norm = wv.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !(norm > 0.0) {
                rho = 0.0;
                used_iters = it;
                break;
            }
            v = wv.into_iter().map(|x| x / norm).collect();
            rho = norm;
        }
        if rho > 0.0 {
            let w1 = mat_vec_mul(w, &v)?;
            let w2 = mat_vec_mul(w, &w1)?;
            let norm2 = w2.iter().map(|x| x * x).sum::<f64>().sqrt();
            rho = if norm2 > 0.0 { norm2.sqrt() } else { 0.0 };
        }
        if rho > best_rho {
            best_rho = rho;
            best_iters = used_iters;
        }
    }
    Ok((best_rho, best_iters))
}

pub fn orthonormality_error(w: &[Vec<f64>]) -> f64 {
    let n = w.len();
    if n == 0 || !w.iter().all(|r| r.len() == n) {
        return f64::INFINITY;
    }
    let mut err = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = w[i].iter().zip(w[j].iter()).map(|(&a, &b)| a * b).sum();
            let target = if i == j { 1.0 } else { 0.0 };
            err = err.max((dot - target).abs());
        }
    }
    err
}

pub fn frobenius_norm(w: &[Vec<f64>]) -> f64 {
    w.iter()
        .flat_map(|r| r.iter())
        .map(|&x| x * x)
        .sum::<f64>()
        .sqrt()
}

pub fn det(w: &[Vec<f64>]) -> Result<f64, String> {
    let n = w.len();
    if n == 0 || !w.iter().all(|r| r.len() == n) {
        return Err("det: W must be square".into());
    }
    if n == 2 {
        return Ok(w[0][0] * w[1][1] - w[0][1] * w[1][0]);
    }
    let mut a = w.to_vec();
    let mut d = 1.0f64;
    for i in 0..n {
        let mut p = i;
        for k in (i + 1)..n {
            if a[k][i].abs() > a[p][i].abs() {
                p = k;
            }
        }
        if a[p][i].abs() < 1e-300 {
            return Ok(0.0);
        }
        if p != i {
            a.swap(p, i);
            d = -d;
        }
        d *= a[i][i];
        let pivot = a[i][i];
        for k in (i + 1)..n {
            let f = a[k][i] / pivot;
            for j in i..n {
                a[k][j] -= f * a[i][j];
            }
        }
    }
    Ok(d)
}

pub fn blend_weights(a: &[Vec<f64>], b: &[Vec<f64>], alpha: f64) -> Result<Vec<Vec<f64>>, String> {
    if a.is_empty() || a.len() != b.len() || a[0].len() != b[0].len() {
        return Err("blend_weights: dimension mismatch".into());
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| {
            ra.iter()
                .zip(rb.iter())
                .map(|(&wa, &wb)| (1.0 - alpha) * wa + alpha * wb)
                .collect()
        })
        .collect())
}

pub fn blend_bias(a: &[f64], b: &[f64], alpha: f64) -> Result<Vec<f64>, String> {
    if a.len() != b.len() {
        return Err("blend_bias: dimension mismatch".into());
    }
    Ok(a.iter()
        .zip(b.iter())
        .map(|(&xa, &xb)| (1.0 - alpha) * xa + alpha * xb)
        .collect())
}

pub fn weight_fingerprint(w: &[Vec<f64>], b: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for row in w {
        for &v in row {
            hasher.update((v as f32).to_le_bytes());
        }
    }
    for &v in b {
        hasher.update((v as f32).to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

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
        return Err("gate_sets: current and candidate weight sets must have matching dimensions".into());
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
    let bias: Vec<f64> = w_ba.into_iter().zip(b.b.iter()).map(|(x, &bi)| x + bi).collect();
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
                return Err(format!("evaluate_weight_expr: repeat depth must be 1..=64, got {depth}"));
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
            let list: Vec<Value> = if let Some(arr) = obj.get("steps").or_else(|| obj.get("chain")).and_then(|v| v.as_array()) {
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
    record("gate identical set -> ACCEPT", g_acc.ok, json!({ "max_jump": g_acc.max_jump }));

    let mut pert = cur.clone();
    pert.w[0][0] += 5.0;
    let g_rej = gate_sets(&cur, &pert, 1.0)?;
    record("gate perturbed set -> REJECT", !g_rej.ok, json!({ "max_jump": g_rej.max_jump }));

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
