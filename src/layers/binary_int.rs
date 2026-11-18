use burn::tensor::{Device, Tensor, TensorData};

use crate::spec::LayerDecl;

use super::{RuntimeLayer, SlotValue, arg_usize};

pub(super) fn try_build(decl: &LayerDecl) -> Result<Option<RuntimeLayer>, String> {
    let args = &decl.args;
    let layer = match decl.constructor.as_str() {
        "add" => RuntimeLayer::Add,
        "sub" => RuntimeLayer::Sub,
        "mul" => RuntimeLayer::Mul,
        "matmul" => RuntimeLayer::Matmul,
        "div" => RuntimeLayer::Div,
        "concat" => RuntimeLayer::Concat(arg_usize(args, 1)?),
        "transpose" => RuntimeLayer::Transpose(arg_usize(args, 1)?, arg_usize(args, 2)?),
        "intAdd" => RuntimeLayer::IntAdd,
        "intSub" => RuntimeLayer::IntSub,
        "intMul" => RuntimeLayer::IntMul,
        "intMatmul" => RuntimeLayer::IntMatmul,
        "intDiv" => RuntimeLayer::IntDiv,
        "intRem" => RuntimeLayer::IntRem,
        "intToFloat" => RuntimeLayer::IntToFloat,
        "intFromFloat" => RuntimeLayer::IntFromFloat,
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

fn checked_div_f32(a: Tensor<4>, b: Tensor<4>) -> Result<Tensor<4>, String> {
    let ad = a.clone().into_data();
    let av = ad.as_slice::<f32>().map_err(|_| "div lhs not f32".to_string())?;
    for (i, &v) in av.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div lhs: non-finite value at index {i}: {v}"));
        }
    }
    let bd = b.clone().into_data();
    let bv = bd.as_slice::<f32>().map_err(|_| "div rhs not f32".to_string())?;
    for (i, &v) in bv.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div rhs: non-finite value at index {i}: {v}"));
        }
        if v == 0.0 {
            return Err(format!("binary div rhs: zero denominator at index {i}"));
        }
    }
    let out = a.div(b);
    let od = out.clone().into_data();
    let ov = od.as_slice::<f32>().map_err(|_| "div output not f32".to_string())?;
    for (i, &v) in ov.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("binary div output: non-finite value at index {i}: {v}"));
        }
    }
    Ok(out)
}

pub(super) fn try_exec(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Add => SlotValue::F32(inputs[0].as_f32("add")?.add(inputs[1].as_f32("add")?)),
        RuntimeLayer::Sub => SlotValue::F32(inputs[0].as_f32("sub")?.sub(inputs[1].as_f32("sub")?)),
        RuntimeLayer::Mul => SlotValue::F32(inputs[0].as_f32("mul")?.mul(inputs[1].as_f32("mul")?)),
        RuntimeLayer::Matmul => SlotValue::F32(
            inputs[0].as_f32("matmul")?.matmul(inputs[1].as_f32("matmul")?),
        ),
        RuntimeLayer::Div => SlotValue::F32(checked_div_f32(
            inputs[0].as_f32("div")?,
            inputs[1].as_f32("div")?,
        )?),
        RuntimeLayer::Concat(dim) => SlotValue::F32(Tensor::cat(
            vec![inputs[0].as_f32("concat")?, inputs[1].as_f32("concat")?],
            *dim,
        )),
        RuntimeLayer::Transpose(d0, d1) => SlotValue::F32(
            inputs[0].as_f32("transpose")?.swap_dims(*d0, *d1),
        ),
        RuntimeLayer::IntAdd
        | RuntimeLayer::IntSub
        | RuntimeLayer::IntMul
        | RuntimeLayer::IntDiv
        | RuntimeLayer::IntRem => {
            let (va, sa) = inputs[0].as_i64("int binary lhs")?;
            let (vb, sb) = inputs[1].as_i64("int binary rhs")?;
            if sa != sb {
                return Err(format!("int binary shape mismatch {sa:?} vs {sb:?}"));
            }
            let mut out = Vec::with_capacity(va.len());
            for (i, (&x, &y)) in va.iter().zip(vb.iter()).enumerate() {
                let res = match layer {
                    RuntimeLayer::IntAdd => x
                        .checked_add(y)
                        .ok_or_else(|| format!("intAdd i64 overflow at {i}")),
                    RuntimeLayer::IntSub => x
                        .checked_sub(y)
                        .ok_or_else(|| format!("intSub i64 overflow at {i}")),
                    RuntimeLayer::IntMul => x
                        .checked_mul(y)
                        .ok_or_else(|| format!("intMul i64 overflow at {i}")),
                    RuntimeLayer::IntDiv => {
                        if y == 0 {
                            Err(format!("intDiv division by zero at {i}"))
                        } else if x == i64::MIN && y == -1 {
                            Err(format!("intDiv i64 overflow at {i}"))
                        } else {
                            Ok(x / y)
                        }
                    }
                    RuntimeLayer::IntRem => {
                        if y == 0 {
                            Err(format!("intRem division by zero at {i}"))
                        } else if x == i64::MIN && y == -1 {
                            Err(format!("intRem i64 overflow at {i}"))
                        } else {
                            Ok(x % y)
                        }
                    }
                    _ => unreachable!(),
                }?;
                out.push(res);
            }
            SlotValue::I64(out, sa)
        }
        RuntimeLayer::IntMatmul => {
            let (va, da) = inputs[0].as_i64("intMatmul lhs")?;
            let (vb, db) = inputs[1].as_i64("intMatmul rhs")?;
            if da[0] != db[0] || da[1] != db[1] || da[3] != db[2] {
                return Err(format!("intMatmul shape mismatch {da:?} @ {db:?}"));
            }
            let (b0n, b1n, m, k, n) = (da[0], da[1], da[2], da[3], db[3]);
            let mut out = Vec::with_capacity(b0n * b1n * m * n);
            for b0 in 0..b0n {
                for b1 in 0..b1n {
                    let batch = b0 * b1n + b1;
                    for mi in 0..m {
                        for ni in 0..n {
                            let mut acc: i64 = 0;
                            for ki in 0..k {
                                let x = va[(batch * m + mi) * k + ki];
                                let y = vb[(batch * k + ki) * n + ni];
                                let prod = x.checked_mul(y).ok_or_else(|| {
                                    format!("intMatmul product i64 overflow at [{b0},{b1},{mi},{ni}]")
                                })?;
                                acc = acc.checked_add(prod).ok_or_else(|| {
                                    format!(
                                        "intMatmul accumulator i64 overflow at [{b0},{b1},{mi},{ni}]"
                                    )
                                })?;
                            }
                            out.push(acc);
                        }
                    }
                }
            }
            SlotValue::I64(out, [b0n, b1n, m, n])
        }
        RuntimeLayer::IntToFloat => {
            let (vi, shape) = inputs[0].as_i64("intToFloat")?;
            let mut vf = Vec::with_capacity(vi.len());
            for (i, &v) in vi.iter().enumerate() {
                if v.unsigned_abs() > 16_777_216 {
                    return Err(format!(
                        "intToFloat: value {v} at index {i} exceeds exact f32 integer range (|x| <= 2^24)"
                    ));
                }
                vf.push(v as f32);
            }
            SlotValue::F32(Tensor::from_data(TensorData::new(vf, shape), device))
        }
        RuntimeLayer::IntFromFloat => {
            let t = inputs[0].as_f32("intFromFloat")?;
            let shape = t.dims();
            let td = t.into_data();
            let vf = td
                .as_slice::<f32>()
                .map_err(|_| "intFromFloat not f32".to_string())?;
            let mut vi = Vec::with_capacity(vf.len());
            for (i, &v) in vf.iter().enumerate() {
                if !v.is_finite() || v < (i64::MIN as f32) || v >= (i64::MAX as f32) {
                    return Err(format!("intFromFloat: out of i64 range at index {i}: {v}"));
                }
                vi.push(v as i64);
            }
            SlotValue::I64(vi, shape)
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
