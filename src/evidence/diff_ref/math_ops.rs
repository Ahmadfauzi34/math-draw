use serde_json::Value;

use super::{RefTensor, coords4, flat4, numel};

fn arg_usize(args: &[Value], idx: usize) -> Result<usize, String> {
    args.get(idx)
        .and_then(|v| v.as_u64())
        .map(|n| n as usize)
        .ok_or_else(|| format!("ref: missing usize arg[{idx}]"))
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

fn arg_usize_vec(args: &[Value], idx: usize) -> Result<Vec<usize>, String> {
    let arr = args
        .get(idx)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("ref: missing array arg[{idx}]"))?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(
            v.as_u64()
                .ok_or_else(|| format!("ref: non-integer in arg[{idx}]"))? as usize,
        );
    }
    Ok(out)
}

fn axis_reduce(x: &[f64], shape: [usize; 4], axis: usize, mode: &str) -> RefTensor {
    let mut out_sh = shape;
    out_sh[axis] = 1;
    let out_len = numel(out_sh);
    let mut out = vec![0.0f64; out_len];
    let axis_len = shape[axis];
    for (f, dst) in out.iter_mut().enumerate() {
        let mut c = coords4(out_sh, f);
        match mode {
            "sum" | "mean" => {
                let mut acc = 0.0f64;
                for k in 0..axis_len {
                    c[axis] = k;
                    acc += x[flat4(shape, c)];
                }
                *dst = if mode == "mean" {
                    acc / axis_len as f64
                } else {
                    acc
                };
            }
            "min" => {
                let mut best = f64::INFINITY;
                for k in 0..axis_len {
                    c[axis] = k;
                    best = best.min(x[flat4(shape, c)]);
                }
                *dst = best;
            }
            _ => {
                let mut best = f64::NEG_INFINITY;
                for k in 0..axis_len {
                    c[axis] = k;
                    best = best.max(x[flat4(shape, c)]);
                }
                *dst = best;
            }
        }
    }
    RefTensor {
        values: out,
        shape: out_sh,
    }
}

pub(super) fn try_eval_math_op(
    ctor: &str,
    ins: &[&RefTensor],
    args: &[Value],
) -> Result<Option<RefTensor>, String> {
    let out = match ctor {
        "abs" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.abs()).collect(),
            shape: ins[0].shape,
        },
        "exp" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.exp()).collect(),
            shape: ins[0].shape,
        },
        "log" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.ln()).collect(),
            shape: ins[0].shape,
        },
        "clamp" => {
            let mn = arg_f64(args, 1)?;
            let mx = arg_f64(args, 2)?;
            RefTensor {
                values: ins[0].values.iter().map(|&v| v.clamp(mn, mx)).collect(),
                shape: ins[0].shape,
            }
        }
        "sin" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.sin()).collect(),
            shape: ins[0].shape,
        },
        "cos" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.cos()).collect(),
            shape: ins[0].shape,
        },
        "tan" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.tan()).collect(),
            shape: ins[0].shape,
        },
        "floor" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.floor()).collect(),
            shape: ins[0].shape,
        },
        "ceil" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.ceil()).collect(),
            shape: ins[0].shape,
        },
        "round" => RefTensor {
            values: ins[0].values.iter().map(|&v| v.round_ties_even()).collect(),
            shape: ins[0].shape,
        },
        "allFinite" => {
            let all_fin = ins[0].values.iter().all(|v| v.is_finite());
            RefTensor {
                values: vec![if all_fin { 1.0 } else { 0.0 }],
                shape: [1, 1, 1, 1],
            }
        }
        "atan2" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&y, &x)| y.atan2(x))
                .collect(),
            shape: ins[0].shape,
        },
        "powf" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&b, &e)| b.powf(e))
                .collect(),
            shape: ins[0].shape,
        },
        "hypot" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&x, &y)| x.hypot(y))
                .collect(),
            shape: ins[0].shape,
        },
        "fma" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .zip(ins[2].values.iter())
                .map(|((&a, &b), &c)| a.mul_add(b, c))
                .collect(),
            shape: ins[0].shape,
        },
        "reshape" => {
            let sh = arg_usize_vec(args, 1)?;
            if sh.len() != 4 {
                return Err("ref reshape: target shape must be rank-4".into());
            }
            RefTensor {
                values: ins[0].values.clone(),
                shape: [sh[0], sh[1], sh[2], sh[3]],
            }
        }
        "permute" => {
            let axes = arg_usize_vec(args, 1)?;
            if axes.len() != 4 {
                return Err("ref permute: axes must have length 4".into());
            }
            let in_sh = ins[0].shape;
            let out_sh = [in_sh[axes[0]], in_sh[axes[1]], in_sh[axes[2]], in_sh[axes[3]]];
            let mut out = vec![0.0f64; ins[0].values.len()];
            for (f, dst) in out.iter_mut().enumerate() {
                let oc = coords4(out_sh, f);
                let mut ic = [0usize; 4];
                for d in 0..4 {
                    ic[axes[d]] = oc[d];
                }
                *dst = ins[0].values[flat4(in_sh, ic)];
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "slice" => {
            let starts = arg_usize_vec(args, 1)?;
            let ends = arg_usize_vec(args, 2)?;
            let out_sh = [
                ends[0] - starts[0],
                ends[1] - starts[1],
                ends[2] - starts[2],
                ends[3] - starts[3],
            ];
            let mut out = vec![0.0f64; numel(out_sh)];
            for (f, dst) in out.iter_mut().enumerate() {
                let oc = coords4(out_sh, f);
                let ic = [
                    oc[0] + starts[0],
                    oc[1] + starts[1],
                    oc[2] + starts[2],
                    oc[3] + starts[3],
                ];
                *dst = ins[0].values[flat4(ins[0].shape, ic)];
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "selectAxis" => {
            let axis = arg_usize(args, 1)?;
            let indices = arg_usize_vec(args, 2)?;
            let mut out_sh = ins[0].shape;
            out_sh[axis] = indices.len();
            let mut out = vec![0.0f64; numel(out_sh)];
            for (f, dst) in out.iter_mut().enumerate() {
                let mut ic = coords4(out_sh, f);
                ic[axis] = indices[ic[axis]];
                *dst = ins[0].values[flat4(ins[0].shape, ic)];
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "fillLike" => {
            let scalar = arg_f64(args, 1)?;
            RefTensor {
                values: vec![scalar; ins[0].values.len()],
                shape: ins[0].shape,
            }
        }
        "expandLike" => {
            let in_sh = ins[0].shape;
            let out_sh = ins[1].shape;
            let mut out = vec![0.0f64; numel(out_sh)];
            for (f, dst) in out.iter_mut().enumerate() {
                let oc = coords4(out_sh, f);
                let ic = [
                    if in_sh[0] == 1 { 0 } else { oc[0] },
                    if in_sh[1] == 1 { 0 } else { oc[1] },
                    if in_sh[2] == 1 { 0 } else { oc[2] },
                    if in_sh[3] == 1 { 0 } else { oc[3] },
                ];
                *dst = ins[0].values[flat4(in_sh, ic)];
            }
            RefTensor {
                values: out,
                shape: out_sh,
            }
        }
        "lessEqual01" => RefTensor {
            values: ins[0]
                .values
                .iter()
                .zip(ins[1].values.iter())
                .map(|(&a, &b)| if a <= b { 1.0 } else { 0.0 })
                .collect(),
            shape: ins[0].shape,
        },
        "indicesLike" => {
            let axis = arg_usize(args, 1)?;
            let sh = ins[0].shape;
            let mut out = vec![0.0f64; ins[0].values.len()];
            for (f, dst) in out.iter_mut().enumerate() {
                *dst = coords4(sh, f)[axis] as f64;
            }
            RefTensor {
                values: out,
                shape: sh,
            }
        }
        "sumAxis" => axis_reduce(&ins[0].values, ins[0].shape, arg_usize(args, 1)?, "sum"),
        "meanAxis" => axis_reduce(&ins[0].values, ins[0].shape, arg_usize(args, 1)?, "mean"),
        "minAxis" => axis_reduce(&ins[0].values, ins[0].shape, arg_usize(args, 1)?, "min"),
        "maxAxis" => axis_reduce(&ins[0].values, ins[0].shape, arg_usize(args, 1)?, "max"),
        "dot" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let mut acc = 0.0f64;
                for fi in 0..f {
                    acc += ins[0].values[bi * f + fi] * ins[1].values[bi * f + fi];
                }
                out[bi] = acc;
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "l2Norm" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let sq: f64 = ins[0].values[bi * f..(bi + 1) * f]
                    .iter()
                    .map(|&v| v * v)
                    .sum();
                out[bi] = sq.sqrt();
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "l2Distance" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let mut sq = 0.0f64;
                for fi in 0..f {
                    let d = ins[0].values[bi * f + fi] - ins[1].values[bi * f + fi];
                    sq += d * d;
                }
                out[bi] = sq.sqrt();
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "cosineSimilarity" => {
            let eps = arg_opt_f64(args, 1).unwrap_or(1e-12);
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
                for fi in 0..f {
                    let a = ins[0].values[bi * f + fi];
                    let bv = ins[1].values[bi * f + fi];
                    dot += a * bv;
                    na += a * a;
                    nb += bv * bv;
                }
                let denom = (na.sqrt() * nb.sqrt()).max(eps);
                out[bi] = dot / denom;
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "sum" | "mean" | "min" | "max" | "variancePopulation" | "stdPopulation"
        | "varianceSample" | "stdSample" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let sl = &ins[0].values[bi * f..(bi + 1) * f];
                out[bi] = match ctor {
                    "sum" => sl.iter().sum(),
                    "mean" => sl.iter().sum::<f64>() / f as f64,
                    "min" => sl.iter().copied().fold(f64::INFINITY, f64::min),
                    "max" => sl.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    "variancePopulation" | "stdPopulation" => {
                        let m = sl.iter().sum::<f64>() / f as f64;
                        let v = sl.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / f as f64;
                        if ctor == "stdPopulation" { v.sqrt() } else { v }
                    }
                    _ => {
                        let m = sl.iter().sum::<f64>() / f as f64;
                        let v = sl.iter().map(|&x| (x - m) * (x - m)).sum::<f64>()
                            / (f - 1) as f64;
                        if ctor == "stdSample" { v.sqrt() } else { v }
                    }
                };
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "normalize" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b * f];
            for bi in 0..b {
                let sl = &ins[0].values[bi * f..(bi + 1) * f];
                let s: f64 = sl.iter().sum();
                for fi in 0..f {
                    out[bi * f + fi] = sl[fi] / s;
                }
            }
            RefTensor {
                values: out,
                shape: ins[0].shape,
            }
        }
        "entropy" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let sl = &ins[0].values[bi * f..(bi + 1) * f];
                let mut h = 0.0f64;
                for &p in sl {
                    if p > 0.0 {
                        h -= p * p.ln();
                    }
                }
                out[bi] = h;
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        "crossEntropy" | "klDivergence" => {
            let [b, f, _, _] = ins[0].shape;
            let mut out = vec![0.0f64; b];
            for bi in 0..b {
                let mut acc = 0.0f64;
                for fi in 0..f {
                    let p = ins[0].values[bi * f + fi];
                    let q = ins[1].values[bi * f + fi];
                    if p > 0.0 {
                        if ctor == "crossEntropy" {
                            acc -= p * q.ln();
                        } else {
                            acc += p * (p / q).ln();
                        }
                    }
                }
                out[bi] = acc;
            }
            RefTensor {
                values: out,
                shape: [b, 1, 1, 1],
            }
        }
        _ => return Ok(None),
    };
    Ok(Some(out))
}
