use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use crate::engine::PreparedGraph;
use crate::spec::{LayerDecl, PortDecl, SpecV1, StepDecl};

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
