use serde_json::Value;

use super::{RefTensor, coords4, flat4, numel};

fn arg_usize(args: &[Value], idx: usize) -> Result<usize, String> {
    args.get(idx)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| format!("ref: missing usize arg[{idx}]"))
}

fn arg_opt_usize(args: &[Value], idx: usize) -> Option<usize> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_u64().map(|n| n as usize),
    }
}

fn arg_bool(args: &[Value], idx: usize) -> Result<bool, String> {
    args.get(idx)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| format!("ref: missing bool arg[{idx}]"))
}

fn arg_f64(args: &[Value], idx: usize) -> Result<f64, String> {
    args.get(idx)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("ref: missing f64 arg[{idx}]"))
}

fn arg_opt_f64(args: &[Value], idx: usize) -> Option<f64> {
    match args.get(idx) {
        Some(Value::Null) | None => None,
        Some(v) => v.as_f64(),
    }
}

fn linear_f64(x: &[f64], w: &[f64], in_dim: usize, out_dim: usize, bias: bool) -> Vec<f64> {
    let batch = x.len() / in_dim;
    let mut y = vec![0.0f64; batch * out_dim];
    let b_off = in_dim * out_dim;
    for b in 0..batch {
        for i in 0..out_dim {
            let mut acc = if bias { w[b_off + i] } else { 0.0 };
            for j in 0..in_dim {
                acc += w[j * out_dim + i] * x[b * in_dim + j];
            }
            y[b * out_dim + i] = acc;
        }
    }
    y
}

fn erf_approx(x: f64) -> f64 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    let e = 1.0 - poly * (-ax * ax).exp();
    if x < 0.0 { -e } else { e }
}

fn softmax_axis(x: &[f64], shape: [usize; 4], dim: usize, log_mode: bool) -> Vec<f64> {
    let strides = [
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1,
    ];
    let mut y = vec![0.0f64; x.len()];
    let mut idx = [0usize; 4];

    #[allow(clippy::too_many_arguments)]
    fn walk(
        d: usize,
        dim: usize,
        shape: [usize; 4],
        strides: [usize; 4],
        idx: &mut [usize; 4],
        x: &[f64],
        y: &mut [f64],
        log_mode: bool,
    ) {
        if d == 4 {
            let mut m = f64::NEG_INFINITY;
            for k in 0..shape[dim] {
                idx[dim] = k;
                let o = idx[0] * strides[0]
                    + idx[1] * strides[1]
                    + idx[2] * strides[2]
                    + idx[3] * strides[3];
                if x[o] > m {
                    m = x[o];
                }
            }
            let mut s = 0.0f64;
            for k in 0..shape[dim] {
                idx[dim] = k;
                let o = idx[0] * strides[0]
                    + idx[1] * strides[1]
                    + idx[2] * strides[2]
                    + idx[3] * strides[3];
                s += (x[o] - m).exp();
            }
            let log_s = s.ln();
            for k in 0..shape[dim] {
                idx[dim] = k;
                let o = idx[0] * strides[0]
                    + idx[1] * strides[1]
                    + idx[2] * strides[2]
                    + idx[3] * strides[3];
                y[o] = if log_mode {
                    x[o] - m - log_s
                } else {
                    (x[o] - m).exp() / s
                };
            }
            return;
        }
        if d == dim {
            walk(d + 1, dim, shape, strides, idx, x, y, log_mode);
            return;
        }
        for k in 0..shape[d] {
            idx[d] = k;
            walk(d + 1, dim, shape, strides, idx, x, y, log_mode);
        }
    }

    walk(0, dim, shape, strides, &mut idx, x, &mut y, log_mode);
    y
}

fn shift_f64(values: &[f64], shape: [usize; 4], dir: &str, s: usize) -> Vec<f64> {
    let [d0, d1, d2, d3] = shape;
    let mut out = vec![0.0f64; values.len()];
    let s_i = s as isize;
    for a in 0..d0 {
        for b in 0..d1 {
            for c in 0..d2 {
                for d in 0..d3 {
                    let (mut sc, mut sd) = (c as isize, d as isize);
                    match dir {
                        "shiftUp" => sc += s_i,
                        "shiftDown" => sc -= s_i,
                        "shiftLeft" => sd += s_i,
                        _ => sd -= s_i,
                    }
                    if sc >= 0 && (sc as usize) < d2 && sd >= 0 && (sd as usize) < d3 {
                        out[((a * d1 + b) * d2 + c) * d3 + d] =
                            values[((a * d1 + b) * d2 + sc as usize) * d3 + sd as usize];
                    }
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv2d_f64(
    x: &[f64],
    shape: [usize; 4],
    in_ch: usize,
    out_ch: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    w: &[f64],
) -> RefTensor {
    let [n, c_in, h, ww] = shape;
    let oh = (h + 2 * ph - kh) / sh + 1;
    let ow = (ww + 2 * pw - kw) / sw + 1;
    let b_off = out_ch * in_ch * kh * kw;
    let mut out = vec![0.0f64; n * out_ch * oh * ow];
    for b in 0..n {
        for o in 0..out_ch {
            for y in 0..oh {
                for x_o in 0..ow {
                    let mut acc = w[b_off + o];
                    for i in 0..in_ch {
                        for khh in 0..kh {
                            let hh = (y * sh + khh) as isize - ph as isize;
                            if hh < 0 || (hh as usize) >= h {
                                continue;
                            }
                            for kww in 0..kw {
                                let xx = (x_o * sw + kww) as isize - pw as isize;
                                if xx < 0 || (xx as usize) >= ww {
                                    continue;
                                }
                                let w_idx = ((o * in_ch + i) * kh + khh) * kw + kww;
                                let x_idx =
                                    ((b * c_in + i) * h + hh as usize) * ww + xx as usize;
                                acc += w[w_idx] * x[x_idx];
                            }
                        }
                    }
                    out[((b * out_ch + o) * oh + y) * ow + x_o] = acc;
                }
            }
        }
    }
    RefTensor {
        values: out,
        shape: [n, out_ch, oh, ow],
    }
}

pub(super) fn try_eval_nn_op(
    ctor: &str,
    ins: &[&RefTensor],
    args: &[Value],
    weights_f32: Option<&[f32]>,
) -> Result<Option<RefTensor>, String> {
    let w_vec: Option<Vec<f64>> =
        weights_f32.map(|sl| sl.iter().map(|&v| v as f64).collect());
    let w = w_vec.as_deref();

    let out = match ctor {
        "linear" => {
            let in_dim = arg_usize(args, 1)?;
            let out_dim = arg_usize(args, 2)?;
            let bias = arg_bool(args, 3)?;
            let exp = in_dim * out_dim + if bias { out_dim } else { 0 };
            let owned;
            let ws = match w {
                Some(v) => v,
                None => {
                    owned = vec![0.0f64; exp];
                    &owned
                }
            };
            let sh = ins[0].shape;
            RefTensor {
                values: linear_f64(&ins[0].values, ws, in_dim, out_dim, bias),
                shape: [sh[0], out_dim, sh[2], sh[3]],
            }
        }
        "relu" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.max(0.0)).collect(),
            shape: ins[0].shape,
        },
        "leakyRelu" => {
            let alpha = arg_f64(args, 1)?;
            RefTensor {
                values: ins[0]
                    .values
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { alpha * v })
                    .collect(),
                shape: ins[0].shape,
            }
        }
        "prelu" => {
            let np = arg_usize(args, 1)?;
            let alpha_arg = arg_f64(args, 2)?;
            let [_, c, h, ww] = ins[0].shape;
            let hw = h * ww;
            let vals: Vec<f64> = if let Some(ws) = w {
                if ws.len() != np {
                    return Err(format!("ref prelu: weights length {} != {np}", ws.len()));
                }
                ins[0]
                    .values
                    .iter()
                    .enumerate()
                    .map(|(idx, &v)| {
                        let a = if np == 1 { ws[0] } else { ws[(idx / hw) % c] };
                        if v > 0.0 { v } else { a * v }
                    })
                    .collect()
            } else {
                ins[0]
                    .values
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { alpha_arg * v })
                    .collect()
            };
            RefTensor {
                values: vals,
                shape: ins[0].shape,
            }
        }
        "sigmoid" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .map(|&v| 1.0 / (1.0 + (-v).exp()))
                .collect(),
            shape: ins[0].shape,
        },
        "tanh" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.tanh()).collect(),
            shape: ins[0].shape,
        },
        "gelu" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .map(|&v| 0.5 * v * (1.0 + erf_approx(v / std::f64::consts::SQRT_2)))
                .collect(),
            shape: ins[0].shape,
        },
        "softplus" => {
            let beta = arg_f64(args, 1)?;
            RefTensor {
                values: ins[0]
                    .values
                    .iter()
                    .map(|&v| {
                        let bv = beta * v;
                        if bv > 20.0 {
                            v
                        } else {
                            (1.0 + bv.exp()).ln() / beta
                        }
                    })
                    .collect(),
                shape: ins[0].shape,
            }
        }
        "hardSigmoid" => {
            let alpha = arg_f64(args, 1)?;
            let beta = arg_f64(args, 2)?;
            RefTensor {
                values: ins[0]
                    .values
                    .iter()
                    .map(|&v| (alpha * v + beta).clamp(0.0, 1.0))
                    .collect(),
                shape: ins[0].shape,
            }
        }
        "hardSwish" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .map(|&v| v * (v / 6.0 + 0.5).clamp(0.0, 1.0))
                .collect(),
            shape: ins[0].shape,
        },
        "mish" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .map(|&v| {
                    let sp = if v > 20.0 { v } else { v.exp().ln_1p() };
                    v * sp.tanh()
                })
                .collect(),
            shape: ins[0].shape,
        },
        "glu" => {
            let dim = arg_usize(args, 1)?;
            let in_sh = ins[0].shape;
            let mut out_sh = in_sh;
            out_sh[dim] = in_sh[dim] / 2;
            let half = out_sh[dim];
            let mut out = vec![0.0f64; numel(out_sh)];
            for (f, dst) in out.iter_mut().enumerate() {
                let ca = coords4(out_sh, f);
                let mut cb = ca;
                cb[dim] += half;
                let a = ins[0].values[flat4(in_sh, ca)];
                let b = ins[0].values[flat4(in_sh, cb)];
                *dst = a / (1.0 + (-b).exp());
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "sqrt" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.sqrt()).collect(),
            shape: ins[0].shape,
        },
        "softmax" => {
            let dim = arg_usize(args, 1)?;
            RefTensor {
                values: softmax_axis(&ins[0].values, ins[0].shape, dim, false),
                shape: ins[0].shape,
            }
        }
        "logSoftmax" => {
            let dim = arg_usize(args, 1)?;
            RefTensor {
                values: softmax_axis(&ins[0].values, ins[0].shape, dim, true),
                shape: ins[0].shape,
            }
        }
        "add" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
            shape: ins[0].shape,
        },
        "sub" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&a, &b)| a - b)
                .collect(),
            shape: ins[0].shape,
        },
        "mul" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&a, &b)| a * b)
                .collect(),
            shape: ins[0].shape,
        },
        "div" => {
            let mut out = Vec::with_capacity(ins[0].values.len());
            for (i, (&a, &b)) in ins[0].values.iter().zip(ins[1].values.iter()).enumerate() {
                if b == 0.0 {
                    return Err(format!("ref div: division by zero at index {i}"));
                }
                out.push(a / b);
            }
            RefTensor {
                values: out,
                shape: ins[0].shape,
            }
        }
        "transpose" => {
            let d0 = arg_usize(args, 1)?;
            let d1 = arg_usize(args, 2)?;
            let in_sh = ins[0].shape;
            let mut out_sh = in_sh;
            out_sh.swap(d0, d1);
            let mut out = vec![0.0f64; ins[0].values.len()];
            for (f, dst) in out.iter_mut().enumerate() {
                let mut c = coords4(out_sh, f);
                c.swap(d0, d1);
                *dst = ins[0].values[flat4(in_sh, c)];
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "matmul" => {
            let [b0, b1, m, k] = ins[0].shape;
            let n = ins[1].shape[3];
            let mut out = vec![0.0f64; b0 * b1 * m * n];
            for a in 0..b0 {
                for b in 0..b1 {
                    for i in 0..m {
                        for j in 0..n {
                            let mut s = 0.0f64;
                            for kk in 0..k {
                                s += ins[0].values[((a * b1 + b) * m + i) * k + kk]
                                    * ins[1].values[((a * b1 + b) * k + kk) * n + j];
                            }
                            out[((a * b1 + b) * m + i) * n + j] = s;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [b0, b1, m, n],
            }
        }
        "concat" => {
            let dim = arg_usize(args, 1)?;
            let sh0 = ins[0].shape;
            let sh1 = ins[1].shape;
            let mut out_sh = sh0;
            out_sh[dim] = sh0[dim] + sh1[dim];
            let mut out = vec![0.0f64; numel(out_sh)];
            for (f, dst) in out.iter_mut().enumerate() {
                let c = coords4(out_sh, f);
                if c[dim] < sh0[dim] {
                    *dst = ins[0].values[flat4(sh0, c)];
                } else {
                    let mut c2 = c;
                    c2[dim] -= sh0[dim];
                    *dst = ins[1].values[flat4(sh1, c2)];
                }
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "shiftUp" | "shiftDown" | "shiftLeft" | "shiftRight" => {
            let s = arg_usize(args, 1)?;
            RefTensor {
                values: shift_f64(&ins[0].values, ins[0].shape, ctor, s),
                shape: ins[0].shape,
            }
        }
        "featureNorm" => {
            let eps = arg_opt_f64(args, 1).unwrap_or(1e-12);
            let [n, c, h, ww] = ins[0].shape;
            let mut y = vec![0.0f64; ins[0].values.len()];
            for b in 0..n {
                for hh in 0..h {
                    for xw in 0..ww {
                        let mut sq = 0.0f64;
                        for cc in 0..c {
                            let v = ins[0].values[((b * c + cc) * h + hh) * ww + xw];
                            sq += v * v;
                        }
                        let r = sq.sqrt().max(eps);
                        for cc in 0..c {
                            let idx = ((b * c + cc) * h + hh) * ww + xw;
                            y[idx] = ins[0].values[idx] / r;
                        }
                    }
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "rmsNorm" => {
            let f = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-6);
            let gamma: Vec<f64> = match w {
                Some(ws) => ws[..f].to_vec(),
                None => vec![1.0; f],
            };
            let outer = ins[0].values.len() / f;
            let mut y = vec![0.0f64; ins[0].values.len()];
            for o in 0..outer {
                let base = o * f;
                let mut sq = 0.0f64;
                for i in 0..f {
                    let v = ins[0].values[base + i];
                    sq += v * v;
                }
                let rms = (sq / f as f64 + eps).sqrt();
                for i in 0..f {
                    y[base + i] = (ins[0].values[base + i] / rms) * gamma[i];
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "layerNorm" => {
            let f = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let gamma: Vec<f64> = match w {
                Some(ws) => ws[..f].to_vec(),
                None => vec![1.0; f],
            };
            let beta: Vec<f64> = match w {
                Some(ws) if ws.len() >= 2 * f => ws[f..2 * f].to_vec(),
                _ => vec![0.0; f],
            };
            let outer = ins[0].values.len() / f;
            let mut y = vec![0.0f64; ins[0].values.len()];
            for o in 0..outer {
                let base = o * f;
                let mu: f64 = ins[0].values[base..base + f].iter().sum::<f64>() / f as f64;
                let va: f64 = ins[0].values[base..base + f]
                    .iter()
                    .map(|&v| (v - mu) * (v - mu))
                    .sum::<f64>()
                    / f as f64;
                let sigma = (va + eps).sqrt();
                for i in 0..f {
                    y[base + i] = ((ins[0].values[base + i] - mu) / sigma) * gamma[i] + beta[i];
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "batchNorm" => {
            let nf = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let gamma: Vec<f64> = match w {
                Some(ws) => ws[..nf].to_vec(),
                None => vec![1.0; nf],
            };
            let beta: Vec<f64> = match w {
                Some(ws) if ws.len() >= 2 * nf => ws[nf..2 * nf].to_vec(),
                _ => vec![0.0; nf],
            };
            let [n, c, h, ww] = ins[0].shape;
            let hw = h * ww;
            let inv = 1.0 / (1.0 + eps).sqrt();
            let mut y = vec![0.0f64; ins[0].values.len()];
            for b in 0..n {
                for cc in 0..c {
                    for f in 0..hw {
                        let idx = (b * c + cc) * hw + f;
                        y[idx] = gamma[cc] * ins[0].values[idx] * inv + beta[cc];
                    }
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "groupNorm" => {
            let ng = arg_usize(args, 1)?;
            let nc = arg_usize(args, 2)?;
            let eps = arg_opt_f64(args, 3).unwrap_or(1e-5);
            let gamma: Vec<f64> = match w {
                Some(ws) => ws[..nc].to_vec(),
                None => vec![1.0; nc],
            };
            let beta: Vec<f64> = match w {
                Some(ws) if ws.len() >= 2 * nc => ws[nc..2 * nc].to_vec(),
                _ => vec![0.0; nc],
            };
            let [n, c, h, ww] = ins[0].shape;
            let hw = h * ww;
            let cpg = c / ng;
            let mut y = vec![0.0f64; ins[0].values.len()];
            for b in 0..n {
                for g in 0..ng {
                    let count = cpg * hw;
                    let mut sum = 0.0f64;
                    for cc in (g * cpg)..((g + 1) * cpg) {
                        for f in 0..hw {
                            sum += ins[0].values[(b * c + cc) * hw + f];
                        }
                    }
                    let mean = sum / count as f64;
                    let mut va = 0.0f64;
                    for cc in (g * cpg)..((g + 1) * cpg) {
                        for f in 0..hw {
                            let d = ins[0].values[(b * c + cc) * hw + f] - mean;
                            va += d * d;
                        }
                    }
                    let inv = 1.0 / (va / count as f64 + eps).sqrt();
                    for cc in (g * cpg)..((g + 1) * cpg) {
                        for f in 0..hw {
                            let idx = (b * c + cc) * hw + f;
                            y[idx] = gamma[cc] * (ins[0].values[idx] - mean) * inv + beta[cc];
                        }
                    }
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "instanceNorm" => {
            let nc = arg_usize(args, 1)?;
            let eps = arg_opt_f64(args, 2).unwrap_or(1e-5);
            let gamma: Vec<f64> = match w {
                Some(ws) => ws[..nc].to_vec(),
                None => vec![1.0; nc],
            };
            let beta: Vec<f64> = match w {
                Some(ws) if ws.len() >= 2 * nc => ws[nc..2 * nc].to_vec(),
                _ => vec![0.0; nc],
            };
            let [n, c, h, ww] = ins[0].shape;
            let hw = h * ww;
            let mut y = vec![0.0f64; ins[0].values.len()];
            for b in 0..n {
                for cc in 0..c {
                    let base = (b * c + cc) * hw;
                    let mean: f64 = ins[0].values[base..base + hw].iter().sum::<f64>() / hw as f64;
                    let va: f64 = ins[0].values[base..base + hw]
                        .iter()
                        .map(|&v| (v - mean) * (v - mean))
                        .sum::<f64>()
                        / hw as f64;
                    let inv = 1.0 / (va + eps).sqrt();
                    for f in 0..hw {
                        y[base + f] =
                            gamma[cc] * (ins[0].values[base + f] - mean) * inv + beta[cc];
                    }
                }
            }
            RefTensor {
                values: y,
                shape: ins[0].shape,
            }
        }
        "embedding" => {
            let vocab = arg_usize(args, 1)?;
            let d_model = arg_usize(args, 2)?;
            let owned;
            let ws = match w {
                Some(v) => v,
                None => {
                    owned = vec![0.0f64; vocab * d_model];
                    &owned
                }
            };
            let [b, s, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b * s * d_model];
            for (i, &tok_f) in ins[0].values.iter().enumerate() {
                if !tok_f.is_finite()
                    || tok_f < 0.0
                    || tok_f.fract() != 0.0
                    || (tok_f as usize) >= vocab
                {
                    return Err(format!("ref embedding: token {tok_f} out of bounds [0, {vocab})"));
                }
                let tok = tok_f as usize;
                out[i * d_model..(i + 1) * d_model]
                    .copy_from_slice(&ws[tok * d_model..(tok + 1) * d_model]);
            }
            RefTensor {
                values: out,
                shape: [b, s, d_model, 1],
            }
        }
        "conv1d" => {
            let in_ch = arg_usize(args, 1)?;
            let out_ch = arg_usize(args, 2)?;
            let k = arg_usize(args, 3)?;
            let s = arg_opt_usize(args, 4).unwrap_or(1);
            let p = arg_opt_usize(args, 5).unwrap_or(0);
            let exp = out_ch * in_ch * k + out_ch;
            let owned;
            let ws = match w {
                Some(v) => v,
                None => {
                    owned = vec![0.0f64; exp];
                    &owned
                }
            };
            let [n, c_in, len, _] = ins[0].shape;
            let o_len = (len + 2 * p - k) / s + 1;
            let b_off = out_ch * in_ch * k;
            let mut out = vec![0.0f64; n * out_ch * o_len];
            for b in 0..n {
                for o in 0..out_ch {
                    for l in 0..o_len {
                        let mut acc = ws[b_off + o];
                        for i in 0..in_ch {
                            for kk in 0..k {
                                let ll = (l * s + kk) as isize - p as isize;
                                if ll >= 0 && (ll as usize) < len {
                                    acc += ws[(o * in_ch + i) * k + kk]
                                        * ins[0].values[(b * c_in + i) * len + ll as usize];
                                }
                            }
                        }
                        out[(b * out_ch + o) * o_len + l] = acc;
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, out_ch, o_len, 1],
            }
        }
        "conv2d" => {
            let in_ch = arg_usize(args, 1)?;
            let out_ch = arg_usize(args, 2)?;
            let kh = arg_usize(args, 3)?;
            let kw = arg_usize(args, 4)?;
            let sh = arg_opt_usize(args, 5).unwrap_or(1);
            let sw = arg_opt_usize(args, 6).unwrap_or(1);
            let ph = arg_opt_usize(args, 7).unwrap_or(0);
            let pw = arg_opt_usize(args, 8).unwrap_or(0);
            let exp = out_ch * in_ch * kh * kw + out_ch;
            let owned;
            let ws = match w {
                Some(v) => v,
                None => {
                    owned = vec![0.0f64; exp];
                    &owned
                }
            };
            conv2d_f64(&ins[0].values, ins[0].shape, in_ch, out_ch, kh, kw, sh, sw, ph, pw, ws)
        }
        "convTranspose2d" => {
            let in_ch = arg_usize(args, 1)?;
            let out_ch = arg_usize(args, 2)?;
            let kh = arg_usize(args, 3)?;
            let kw = arg_usize(args, 4)?;
            let sh = arg_opt_usize(args, 5).unwrap_or(1);
            let sw = arg_opt_usize(args, 6).unwrap_or(1);
            let ph = arg_opt_usize(args, 7).unwrap_or(0);
            let pw = arg_opt_usize(args, 8).unwrap_or(0);
            let exp = in_ch * out_ch * kh * kw + out_ch;
            let owned;
            let ws = match w {
                Some(v) => v,
                None => {
                    owned = vec![0.0f64; exp];
                    &owned
                }
            };
            let [n, c_in, h, ww] = ins[0].shape;
            let oh = (h - 1) * sh + kh - 2 * ph;
            let ow = (ww - 1) * sw + kw - 2 * pw;
            let b_off = in_ch * out_ch * kh * kw;
            let mut out = vec![0.0f64; n * out_ch * oh * ow];
            for b in 0..n {
                for o in 0..out_ch {
                    for y in 0..oh {
                        for x_o in 0..ow {
                            out[((b * out_ch + o) * oh + y) * ow + x_o] = ws[b_off + o];
                        }
                    }
                }
            }
            for b in 0..n {
                for i in 0..in_ch {
                    for hh in 0..h {
                        for xx in 0..ww {
                            let xv = ins[0].values[((b * c_in + i) * h + hh) * ww + xx];
                            for khh in 0..kh {
                                let y = (hh * sh + khh) as isize - ph as isize;
                                if y < 0 || (y as usize) >= oh {
                                    continue;
                                }
                                for kww in 0..kw {
                                    let x_o = (xx * sw + kww) as isize - pw as isize;
                                    if x_o < 0 || (x_o as usize) >= ow {
                                        continue;
                                    }
                                    for o in 0..out_ch {
                                        let w_idx = ((i * out_ch + o) * kh + khh) * kw + kww;
                                        out[((b * out_ch + o) * oh + y as usize) * ow
                                            + x_o as usize] += xv * ws[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, out_ch, oh, ow],
            }
        }
        "avgPool1d" => {
            let k = arg_usize(args, 1)?;
            let s = arg_opt_usize(args, 2).unwrap_or(k);
            let p = arg_opt_usize(args, 3).unwrap_or(0);
            let [n, c, l, ww] = ins[0].shape;
            let ol = (l + 2 * p - k) / s + 1;
            let mut out = vec![0.0f64; n * c * ol * ww];
            for b in 0..n {
                for cc in 0..c {
                    for xw in 0..ww {
                        for o in 0..ol {
                            let mut sum = 0.0f64;
                            let l0 = (o * s) as isize - p as isize;
                            for i in 0..k {
                                let ll = l0 + i as isize;
                                if ll >= 0 && (ll as usize) < l {
                                    sum += ins[0].values
                                        [((b * c + cc) * l + ll as usize) * ww + xw];
                                }
                            }
                            out[((b * c + cc) * ol + o) * ww + xw] = sum / k as f64;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, c, ol, ww],
            }
        }
        "maxPool1d" => {
            let k = arg_usize(args, 1)?;
            let s = arg_opt_usize(args, 2).unwrap_or(k);
            let p = arg_opt_usize(args, 3).unwrap_or(0);
            let [n, c, l, ww] = ins[0].shape;
            let ol = (l + 2 * p - k) / s + 1;
            let mut out = vec![0.0f64; n * c * ol * ww];
            for b in 0..n {
                for cc in 0..c {
                    for xw in 0..ww {
                        for o in 0..ol {
                            let mut mx = f64::NEG_INFINITY;
                            let l0 = (o * s) as isize - p as isize;
                            for i in 0..k {
                                let ll = l0 + i as isize;
                                if ll >= 0 && (ll as usize) < l {
                                    let v = ins[0].values
                                        [((b * c + cc) * l + ll as usize) * ww + xw];
                                    if v > mx {
                                        mx = v;
                                    }
                                }
                            }
                            out[((b * c + cc) * ol + o) * ww + xw] = mx;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, c, ol, ww],
            }
        }
        "avgPool2d" => {
            let kh = arg_usize(args, 1)?;
            let kw = arg_usize(args, 2)?;
            let sh = arg_opt_usize(args, 3).unwrap_or(kh);
            let sw = arg_opt_usize(args, 4).unwrap_or(kw);
            let ph = arg_opt_usize(args, 5).unwrap_or(0);
            let pw = arg_opt_usize(args, 6).unwrap_or(0);
            let [n, c, h, ww] = ins[0].shape;
            let oh = (h + 2 * ph - kh) / sh + 1;
            let ow = (ww + 2 * pw - kw) / sw + 1;
            let div = (kh * kw) as f64;
            let mut out = vec![0.0f64; n * c * oh * ow];
            for b in 0..n {
                for cc in 0..c {
                    for y in 0..oh {
                        for x_o in 0..ow {
                            let mut sum = 0.0f64;
                            let h0 = (y * sh) as isize - ph as isize;
                            let w0 = (x_o * sw) as isize - pw as isize;
                            for i in 0..kh {
                                for j in 0..kw {
                                    let hh = h0 + i as isize;
                                    let xx = w0 + j as isize;
                                    if hh >= 0
                                        && (hh as usize) < h
                                        && xx >= 0
                                        && (xx as usize) < ww
                                    {
                                        sum += ins[0].values
                                            [((b * c + cc) * h + hh as usize) * ww + xx as usize];
                                    }
                                }
                            }
                            out[((b * c + cc) * oh + y) * ow + x_o] = sum / div;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, c, oh, ow],
            }
        }
        "maxPool2d" => {
            let kh = arg_usize(args, 1)?;
            let kw = arg_usize(args, 2)?;
            let sh = arg_opt_usize(args, 3).unwrap_or(kh);
            let sw = arg_opt_usize(args, 4).unwrap_or(kw);
            let ph = arg_opt_usize(args, 5).unwrap_or(0);
            let pw = arg_opt_usize(args, 6).unwrap_or(0);
            let [n, c, h, ww] = ins[0].shape;
            let oh = (h + 2 * ph - kh) / sh + 1;
            let ow = (ww + 2 * pw - kw) / sw + 1;
            let mut out = vec![0.0f64; n * c * oh * ow];
            for b in 0..n {
                for cc in 0..c {
                    for y in 0..oh {
                        for x_o in 0..ow {
                            let mut mx = f64::NEG_INFINITY;
                            let h0 = (y * sh) as isize - ph as isize;
                            let w0 = (x_o * sw) as isize - pw as isize;
                            for i in 0..kh {
                                for j in 0..kw {
                                    let hh = h0 + i as isize;
                                    let xx = w0 + j as isize;
                                    if hh >= 0
                                        && (hh as usize) < h
                                        && xx >= 0
                                        && (xx as usize) < ww
                                    {
                                        let v = ins[0].values
                                            [((b * c + cc) * h + hh as usize) * ww + xx as usize];
                                        if v > mx {
                                            mx = v;
                                        }
                                    }
                                }
                            }
                            out[((b * c + cc) * oh + y) * ow + x_o] = mx;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, c, oh, ow],
            }
        }
        "adaptiveAvgPool2d" => {
            let oh = arg_usize(args, 1)?;
            let ow = arg_usize(args, 2)?;
            let [n, c, h, ww] = ins[0].shape;
            let mut out = vec![0.0f64; n * c * oh * ow];
            for b in 0..n {
                for cc in 0..c {
                    for y in 0..oh {
                        let h0 = (y * h) / oh;
                        let h1 = ((y + 1) * h).div_ceil(oh);
                        for x_o in 0..ow {
                            let w0 = (x_o * ww) / ow;
                            let w1 = ((x_o + 1) * ww).div_ceil(ow);
                            let mut s = 0.0f64;
                            let mut cnt = 0usize;
                            for hh in h0..h1 {
                                for xx in w0..w1 {
                                    s += ins[0].values[((b * c + cc) * h + hh) * ww + xx];
                                    cnt += 1;
                                }
                            }
                            out[((b * c + cc) * oh + y) * ow + x_o] = s / cnt as f64;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, c, oh, ow],
            }
        }
        "ghost" => {
            let in_ch = arg_usize(args, 1)?;
            let out_ch = arg_usize(args, 2)?;
            let kh = arg_usize(args, 3)?;
            let kw = arg_usize(args, 4)?;
            let ratio = arg_usize(args, 5)?;
            let sh = arg_opt_usize(args, 6).unwrap_or(1);
            let sw = arg_opt_usize(args, 7).unwrap_or(1);
            let ph = arg_opt_usize(args, 8).unwrap_or(0);
            let pw = arg_opt_usize(args, 9).unwrap_or(0);
            let pch = out_ch / ratio;
            let out_c = 2 * pch;
            let [n, _, h, ww] = ins[0].shape;
            let oh = (h + 2 * ph - kh) / sh + 1;
            let ow = (ww + 2 * pw - kw) / sw + 1;
            let Some(ws) = w else {
                return Ok(Some(RefTensor {
                    values: vec![0.0f64; n * out_c * oh * ow],
                    shape: [n, out_c, oh, ow],
                }));
            };
            let p_len = pch * (in_ch * kh * kw + 1);
            let primary = conv2d_f64(
                &ins[0].values,
                ins[0].shape,
                in_ch,
                pch,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                &ws[..p_len],
            );
            let cheap_w = &ws[p_len..];
            let mut out = vec![0.0f64; n * out_c * oh * ow];
            for b in 0..n {
                for c in 0..pch {
                    for y in 0..oh {
                        for x_o in 0..ow {
                            let pv = primary.values[((b * pch + c) * oh + y) * ow + x_o];
                            out[((b * out_c + c) * oh + y) * ow + x_o] = pv;
                            out[((b * out_c + pch + c) * oh + y) * ow + x_o] = cheap_w[c] * pv;
                        }
                    }
                }
            }
            RefTensor {
                values: out,
                shape: [n, out_c, oh, ow],
            }
        }
        "seBlock" => {
            let ch = arg_usize(args, 1)?;
            let red = arg_usize(args, 2)?;
            let [n, c, h, ww] = ins[0].shape;
            let Some(ws) = w else {
                return Ok(Some(RefTensor {
                    values: ins[0].values.iter().map(|&v| v * 0.5).collect(),
                    shape: ins[0].shape,
                }));
            };
            let r = ch / red;
            let fc1_len = ch * r + r;
            let fc1_w = &ws[..fc1_len];
            let fc2_w = &ws[fc1_len..];
            let hw = h * ww;
            let mut out = vec![0.0f64; ins[0].values.len()];
            for b in 0..n {
                let mut sq = vec![0.0f64; ch];
                for (ci, sq_i) in sq.iter_mut().enumerate() {
                    let base = (b * c + ci) * hw;
                    *sq_i = ins[0].values[base..base + hw].iter().sum::<f64>() / hw as f64;
                }
                let hidden: Vec<f64> = linear_f64(&sq, fc1_w, ch, r, true)
                    .into_iter()
                    .map(|v| v.max(0.0))
                    .collect();
                let gate: Vec<f64> = linear_f64(&hidden, fc2_w, r, ch, true)
                    .into_iter()
                    .map(|v| 1.0 / (1.0 + (-v).exp()))
                    .collect();
                for (ci, &g) in gate.iter().enumerate() {
                    let base = (b * c + ci) * hw;
                    for k in 0..hw {
                        out[base + k] = ins[0].values[base + k] * g;
                    }
                }
            }
            RefTensor {
                values: out,
                shape: ins[0].shape,
            }
        }
        "swiGlu" => {
            let d_in = arg_usize(args, 1)?;
            let d_out = arg_usize(args, 2)?;
            let bias = arg_bool(args, 3)?;
            let mut out_sh = ins[0].shape;
            out_sh[3] = d_out;
            let Some(ws) = w else {
                return Ok(Some(RefTensor {
                    values: vec![0.0f64; numel(out_sh)],
                    shape: out_sh,
                }));
            };
            let inner_len = d_in * d_out + if bias { d_out } else { 0 };
            let inner = linear_f64(&ins[0].values, &ws[..inner_len], d_in, d_out, bias);
            let outer = linear_f64(&ins[0].values, &ws[inner_len..], d_in, d_out, bias);
            let values: Vec<f64> = inner
                .iter()
                .zip(outer.iter())
                .map(|(&a, &o)| (a / (1.0 + (-a).exp())) * o)
                .collect();
            RefTensor {
                values,
                shape: out_sh,
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}
