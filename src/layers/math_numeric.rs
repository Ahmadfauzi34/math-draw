use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::spec::LayerDecl;

use super::{RuntimeLayer, SlotValue, arg_f64, arg_usize, arg_usize_vec};

const MAX_EXACT_F32_COORDINATE: usize = 1 << 24;
const MAX_INDICES_LIKE_AXIS_LENGTH: usize = MAX_EXACT_F32_COORDINATE + 1;

fn validate_finite(t: &Tensor<4>, ctx: &str) -> Result<(), String> {
    let data = t.clone().into_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{ctx}: expected f32 tensor"))?;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{ctx}: non-finite value at index {i}: {v}"));
        }
    }
    Ok(())
}

fn validate_positive(t: &Tensor<4>, ctx: &str) -> Result<(), String> {
    let data = t.clone().into_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{ctx}: expected f32 tensor"))?;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{ctx}: non-finite value at index {i}: {v}"));
        }
    }
    for (i, &v) in values.iter().enumerate() {
        if v <= 0.0 {
            return Err(format!("{ctx}: value must be > 0 at index {i}, got {v}"));
        }
    }
    Ok(())
}

fn checked_f32_output(out: Tensor<4>, ctx: &str) -> Result<Tensor<4>, String> {
    validate_finite(&out, ctx)?;
    Ok(out)
}

fn parse_rank4_shape(vals: &[usize], ctx: &str) -> Result<[usize; 4], String> {
    if vals.len() != 4 {
        return Err(format!("{ctx}: expected 4 dims, got {}", vals.len()));
    }
    let dims = [vals[0], vals[1], vals[2], vals[3]];
    if dims.contains(&0) {
        return Err(format!("{ctx}: zero-sized dimensions are forbidden: {dims:?}"));
    }
    let _ = dims
        .into_iter()
        .try_fold(1usize, |a, b| a.checked_mul(b))
        .ok_or_else(|| format!("{ctx}: element-count overflow for {dims:?}"))?;
    Ok(dims)
}

fn parse_axes4(vals: &[usize]) -> Result<[usize; 4], String> {
    if vals.len() != 4 {
        return Err(format!("permute: expected 4 axes, got {}", vals.len()));
    }
    let mut seen = [false; 4];
    for (pos, &ax) in vals.iter().enumerate() {
        if ax >= 4 {
            return Err(format!("permute: axis[{pos}]={ax} out of range 0..4"));
        }
        if seen[ax] {
            return Err(format!("permute: duplicate axis {ax} at position {pos}"));
        }
        seen[ax] = true;
    }
    Ok([vals[0], vals[1], vals[2], vals[3]])
}

pub(super) fn try_build(decl: &LayerDecl) -> Result<Option<RuntimeLayer>, String> {
    let args = &decl.args;
    let layer = match decl.constructor.as_str() {
        "abs" => RuntimeLayer::Abs,
        "exp" => RuntimeLayer::Exp,
        "log" => RuntimeLayer::Log,
        "clamp" => {
            let min = arg_f64(args, 1)?;
            let max = arg_f64(args, 2)?;
            if !min.is_finite()
                || !max.is_finite()
                || !(min as f32).is_finite()
                || !(max as f32).is_finite()
            {
                return Err(format!(
                    "clamp: bounds must be finite f32-representable, got min={min}, max={max}"
                ));
            }
            if min > max {
                return Err(format!("clamp: min must be <= max, got min={min}, max={max}"));
            }
            RuntimeLayer::Clamp {
                min: min as f32,
                max: max as f32,
            }
        }
        "sin" => RuntimeLayer::Sin,
        "cos" => RuntimeLayer::Cos,
        "tan" => RuntimeLayer::Tan,
        "floor" => RuntimeLayer::Floor,
        "ceil" => RuntimeLayer::Ceil,
        "round" => RuntimeLayer::Round,
        "atan2" => RuntimeLayer::Atan2,
        "powf" => RuntimeLayer::Powf,
        "hypot" => RuntimeLayer::Hypot,
        "fma" => RuntimeLayer::Fma,
        "allFinite" => RuntimeLayer::AllFinite,
        "reshape" => {
            let shape = parse_rank4_shape(&arg_usize_vec(args, 1)?, "reshape")?;
            RuntimeLayer::Reshape(shape)
        }
        "permute" => {
            let axes = parse_axes4(&arg_usize_vec(args, 1)?)?;
            RuntimeLayer::Permute(axes)
        }
        "slice" => {
            let starts = arg_usize_vec(args, 1)?;
            let ends = arg_usize_vec(args, 2)?;
            if starts.len() != 4 || ends.len() != 4 {
                return Err(format!(
                    "slice: starts and ends must each have 4 elements, got {} and {}",
                    starts.len(),
                    ends.len()
                ));
            }
            RuntimeLayer::Slice {
                starts: [starts[0], starts[1], starts[2], starts[3]],
                ends: [ends[0], ends[1], ends[2], ends[3]],
            }
        }
        "selectAxis" => {
            let axis = arg_usize(args, 1)?;
            let indices = arg_usize_vec(args, 2)?;
            if axis >= 4 {
                return Err(format!("selectAxis: axis {axis} out of range 0..4"));
            }
            if indices.is_empty() {
                return Err("selectAxis: indices must not be empty".into());
            }
            for (pos, &idx) in indices.iter().enumerate() {
                if idx > i64::MAX as usize {
                    return Err(format!(
                        "selectAxis: index[{pos}]={idx} exceeds i64::MAX"
                    ));
                }
            }
            RuntimeLayer::SelectAxis { axis, indices }
        }
        "fillLike" => {
            let scalar = arg_f64(args, 1)?;
            if !scalar.is_finite() || !(scalar as f32).is_finite() {
                return Err(format!(
                    "fillLike: scalar must be finite f32-representable, got {scalar}"
                ));
            }
            RuntimeLayer::FillLike(scalar as f32)
        }
        "expandLike" => RuntimeLayer::ExpandLike,
        "sumAxis" => {
            let axis = arg_usize(args, 1)?;
            if axis >= 4 {
                return Err(format!("sumAxis: axis {axis} out of range 0..4"));
            }
            RuntimeLayer::SumAxis(axis)
        }
        "meanAxis" => {
            let axis = arg_usize(args, 1)?;
            if axis >= 4 {
                return Err(format!("meanAxis: axis {axis} out of range 0..4"));
            }
            RuntimeLayer::MeanAxis(axis)
        }
        "minAxis" => {
            let axis = arg_usize(args, 1)?;
            if axis >= 4 {
                return Err(format!("minAxis: axis {axis} out of range 0..4"));
            }
            RuntimeLayer::MinAxis(axis)
        }
        "maxAxis" => {
            let axis = arg_usize(args, 1)?;
            if axis >= 4 {
                return Err(format!("maxAxis: axis {axis} out of range 0..4"));
            }
            RuntimeLayer::MaxAxis(axis)
        }
        "lessEqual01" => RuntimeLayer::LessEqual01,
        "indicesLike" => {
            let axis = arg_usize(args, 1)?;
            if axis >= 4 {
                return Err(format!("indicesLike: axis {axis} out of range 0..4"));
            }
            RuntimeLayer::IndicesLike(axis)
        }
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

fn validate_axis_input(x: &Tensor<4>, axis: usize, ctx: &str) -> Result<[usize; 4], String> {
    if axis >= 4 {
        return Err(format!("{ctx}: axis {axis} out of range 0..4"));
    }
    let dims = x.dims();
    for (ax, &d) in dims.iter().enumerate() {
        if d == 0 {
            return Err(format!("{ctx}: zero-sized dimension at axis {ax}"));
        }
    }
    validate_finite(x, ctx)?;
    Ok(dims)
}

fn checked_keepdim_output(
    out: Tensor<4>,
    in_dims: [usize; 4],
    axis: usize,
    ctx: &str,
) -> Result<Tensor<4>, String> {
    let mut expected = in_dims;
    expected[axis] = 1;
    let actual = out.dims();
    if actual != expected {
        return Err(format!(
            "{ctx}: keepdim invariant failed, expected {expected:?}, got {actual:?}"
        ));
    }
    checked_f32_output(out, ctx)
}

pub(super) fn try_exec(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Abs => {
            let x = inputs[0].as_f32("abs")?;
            validate_finite(&x, "abs input")?;
            SlotValue::F32(checked_f32_output(x.abs(), "abs output")?)
        }
        RuntimeLayer::Exp => {
            let x = inputs[0].as_f32("exp")?;
            validate_finite(&x, "exp input")?;
            SlotValue::F32(checked_f32_output(x.exp(), "exp output")?)
        }
        RuntimeLayer::Log => {
            let x = inputs[0].as_f32("log")?;
            validate_positive(&x, "log input")?;
            SlotValue::F32(checked_f32_output(x.log(), "log output")?)
        }
        RuntimeLayer::Clamp { min, max } => {
            let x = inputs[0].as_f32("clamp")?;
            validate_finite(&x, "clamp input")?;
            SlotValue::F32(checked_f32_output(x.clamp(*min, *max), "clamp output")?)
        }
        RuntimeLayer::Sin => {
            let x = inputs[0].as_f32("sin")?;
            validate_finite(&x, "sin input")?;
            SlotValue::F32(checked_f32_output(x.sin(), "sin output")?)
        }
        RuntimeLayer::Cos => {
            let x = inputs[0].as_f32("cos")?;
            validate_finite(&x, "cos input")?;
            SlotValue::F32(checked_f32_output(x.cos(), "cos output")?)
        }
        RuntimeLayer::Tan => {
            let x = inputs[0].as_f32("tan")?;
            validate_finite(&x, "tan input")?;
            SlotValue::F32(checked_f32_output(x.tan(), "tan output")?)
        }
        RuntimeLayer::Floor => {
            let x = inputs[0].as_f32("floor")?;
            validate_finite(&x, "floor input")?;
            SlotValue::F32(checked_f32_output(x.floor(), "floor output")?)
        }
        RuntimeLayer::Ceil => {
            let x = inputs[0].as_f32("ceil")?;
            validate_finite(&x, "ceil input")?;
            SlotValue::F32(checked_f32_output(x.ceil(), "ceil output")?)
        }
        RuntimeLayer::Round => {
            let x = inputs[0].as_f32("round")?;
            validate_finite(&x, "round input")?;
            SlotValue::F32(checked_f32_output(x.round(), "round output")?)
        }
        RuntimeLayer::Atan2 => {
            let y = inputs[0].as_f32("atan2 y")?;
            let x = inputs[1].as_f32("atan2 x")?;
            if y.dims() != x.dims() {
                return Err(format!(
                    "atan2: shape mismatch {:?} vs {:?}; implicit broadcasting forbidden",
                    y.dims(),
                    x.dims()
                ));
            }
            validate_finite(&y, "atan2 y")?;
            validate_finite(&x, "atan2 x")?;
            SlotValue::F32(checked_f32_output(y.atan2(x), "atan2 output")?)
        }
        RuntimeLayer::Powf => {
            let base = inputs[0].as_f32("powf base")?;
            let exp = inputs[1].as_f32("powf exp")?;
            if base.dims() != exp.dims() {
                return Err(format!(
                    "powf: shape mismatch {:?} vs {:?}; implicit broadcasting forbidden",
                    base.dims(),
                    exp.dims()
                ));
            }
            validate_finite(&base, "powf base")?;
            validate_finite(&exp, "powf exp")?;
            SlotValue::F32(checked_f32_output(base.powf(exp), "powf output")?)
        }
        RuntimeLayer::Hypot => {
            let x = inputs[0].as_f32("hypot x")?;
            let y = inputs[1].as_f32("hypot y")?;
            let dims = x.dims();
            if dims != y.dims() {
                return Err(format!(
                    "hypot: shape mismatch {dims:?} vs {:?}; implicit broadcasting forbidden",
                    y.dims()
                ));
            }
            validate_finite(&x, "hypot x")?;
            validate_finite(&y, "hypot y")?;
            let dx = x.into_data();
            let dy = y.into_data();
            let vx = dx.as_slice::<f32>().map_err(|_| "hypot x: expected f32 tensor")?;
            let vy = dy.as_slice::<f32>().map_err(|_| "hypot y: expected f32 tensor")?;
            let out: Vec<f32> = vx
                .iter()
                .zip(vy.iter())
                .map(|(&a, &b)| {
                    let ax = a.abs();
                    let ay = b.abs();
                    let m = ax.max(ay);
                    let n = ax.min(ay);
                    if m == 0.0 {
                        0.0
                    } else {
                        let r = n / m;
                        m * (1.0 + r * r).sqrt()
                    }
                })
                .collect();
            let t = Tensor::<4>::from_data(TensorData::new(out, dims), device);
            SlotValue::F32(checked_f32_output(t, "hypot output")?)
        }
        RuntimeLayer::Fma => {
            let a = inputs[0].as_f32("fma a")?;
            let b = inputs[1].as_f32("fma b")?;
            let c = inputs[2].as_f32("fma c")?;
            let dims = a.dims();
            if dims != b.dims() || dims != c.dims() {
                return Err(format!(
                    "fma: shape mismatch {:?} vs {:?} vs {:?}; implicit broadcasting forbidden",
                    dims,
                    b.dims(),
                    c.dims()
                ));
            }
            validate_finite(&a, "fma a")?;
            validate_finite(&b, "fma b")?;
            validate_finite(&c, "fma c")?;
            let da = a.into_data();
            let db = b.into_data();
            let dc = c.into_data();
            let va = da.as_slice::<f32>().map_err(|_| "fma a: expected f32 tensor")?;
            let vb = db.as_slice::<f32>().map_err(|_| "fma b: expected f32 tensor")?;
            let vc = dc.as_slice::<f32>().map_err(|_| "fma c: expected f32 tensor")?;
            let out: Vec<f32> = va
                .iter()
                .zip(vb.iter())
                .zip(vc.iter())
                .map(|((&x, &y), &z)| x.mul_add(y, z))
                .collect();
            let t = Tensor::<4>::from_data(TensorData::new(out, dims), device);
            SlotValue::F32(checked_f32_output(t, "fma output")?)
        }
        RuntimeLayer::AllFinite => {
            let x = inputs[0].as_f32("allFinite")?;
            let data = x.into_data();
            let values = data
                .as_slice::<f32>()
                .map_err(|_| "allFinite: expected f32 tensor")?;
            let flag = if values.iter().all(|v| v.is_finite()) {
                1.0f32
            } else {
                0.0f32
            };
            SlotValue::F32(Tensor::<4>::from_data(
                TensorData::new(vec![flag], [1, 1, 1, 1]),
                device,
            ))
        }
        RuntimeLayer::Reshape(target) => {
            let x = inputs[0].as_f32("reshape")?;
            let src_dims = x.dims();
            let src_n = src_dims
                .into_iter()
                .try_fold(1usize, |a, b| a.checked_mul(b))
                .ok_or_else(|| format!("reshape: source numel overflow for {src_dims:?}"))?;
            let dst_n: usize = target.iter().product();
            if src_n != dst_n {
                return Err(format!(
                    "reshape: element-count mismatch {src_dims:?} ({src_n}) vs {target:?} ({dst_n})"
                ));
            }
            SlotValue::F32(x.reshape(*target))
        }
        RuntimeLayer::Permute(axes) => {
            let x = inputs[0].as_f32("permute")?;
            SlotValue::F32(x.permute(*axes))
        }
        RuntimeLayer::Slice { starts, ends } => {
            let x = inputs[0].as_f32("slice")?;
            let dims = x.dims();
            for ax in 0..4 {
                if starts[ax] >= ends[ax] {
                    return Err(format!(
                        "slice: axis {ax} requires start < end, got {}..{}",
                        starts[ax], ends[ax]
                    ));
                }
                if ends[ax] > dims[ax] {
                    return Err(format!(
                        "slice: axis {ax} end {} exceeds dim {}",
                        ends[ax], dims[ax]
                    ));
                }
            }
            SlotValue::F32(x.slice([
                starts[0]..ends[0],
                starts[1]..ends[1],
                starts[2]..ends[2],
                starts[3]..ends[3],
            ]))
        }
        RuntimeLayer::SelectAxis { axis, indices } => {
            let x = inputs[0].as_f32("selectAxis")?;
            let dims = x.dims();
            for (pos, &idx) in indices.iter().enumerate() {
                if idx >= dims[*axis] {
                    return Err(format!(
                        "selectAxis: index[{pos}]={idx} out of bounds for axis {} dim {}",
                        axis, dims[*axis]
                    ));
                }
            }
            let idx_i64: Vec<i64> = indices.iter().map(|&v| v as i64).collect();
            let idx_tensor =
                Tensor::<1, Int>::from_data(TensorData::new(idx_i64, [indices.len()]), device);
            SlotValue::F32(x.select(*axis, idx_tensor))
        }
        RuntimeLayer::FillLike(scalar) => {
            let reference = inputs[0].as_f32("fillLike")?;
            let dims = reference.dims();
            let count = dims
                .into_iter()
                .try_fold(1usize, |a, b| {
                    if b == 0 {
                        return Err(format!("fillLike: zero-sized dim in {dims:?}"));
                    }
                    a.checked_mul(b)
                        .ok_or_else(|| format!("fillLike: numel overflow for {dims:?}"))
                })?;
            let out = Tensor::<4>::from_data(TensorData::new(vec![*scalar; count], dims), device);
            SlotValue::F32(out)
        }
        RuntimeLayer::ExpandLike => {
            let src = inputs[0].as_f32("expandLike source")?;
            let reference = inputs[1].as_f32("expandLike reference")?;
            let sd = src.dims();
            let rd = reference.dims();
            for ax in 0..4 {
                if sd[ax] == 0 || rd[ax] == 0 {
                    return Err(format!(
                        "expandLike: zero-sized dim at axis {ax}: {sd:?} vs {rd:?}"
                    ));
                }
                if sd[ax] != rd[ax] && sd[ax] != 1 {
                    return Err(format!(
                        "expandLike: axis {ax} cannot expand source dim {} to reference dim {}",
                        sd[ax], rd[ax]
                    ));
                }
            }
            if sd == rd {
                SlotValue::F32(src)
            } else {
                let out = src.expand(rd);
                if out.dims() != rd {
                    return Err(format!(
                        "expandLike: shape invariant failed, expected {rd:?}, got {:?}",
                        out.dims()
                    ));
                }
                SlotValue::F32(out)
            }
        }
        RuntimeLayer::SumAxis(axis) => {
            let x = inputs[0].as_f32("sumAxis")?;
            let dims = validate_axis_input(&x, *axis, "sumAxis input")?;
            SlotValue::F32(checked_keepdim_output(
                x.sum_dim(*axis),
                dims,
                *axis,
                "sumAxis output",
            )?)
        }
        RuntimeLayer::MeanAxis(axis) => {
            let x = inputs[0].as_f32("meanAxis")?;
            let dims = validate_axis_input(&x, *axis, "meanAxis input")?;
            SlotValue::F32(checked_keepdim_output(
                x.mean_dim(*axis),
                dims,
                *axis,
                "meanAxis output",
            )?)
        }
        RuntimeLayer::MinAxis(axis) => {
            let x = inputs[0].as_f32("minAxis")?;
            let dims = validate_axis_input(&x, *axis, "minAxis input")?;
            SlotValue::F32(checked_keepdim_output(
                x.min_dim(*axis),
                dims,
                *axis,
                "minAxis output",
            )?)
        }
        RuntimeLayer::MaxAxis(axis) => {
            let x = inputs[0].as_f32("maxAxis")?;
            let dims = validate_axis_input(&x, *axis, "maxAxis input")?;
            SlotValue::F32(checked_keepdim_output(
                x.max_dim(*axis),
                dims,
                *axis,
                "maxAxis output",
            )?)
        }
        RuntimeLayer::LessEqual01 => {
            let lhs = inputs[0].as_f32("lessEqual01 lhs")?;
            let rhs = inputs[1].as_f32("lessEqual01 rhs")?;
            let lshape = lhs.dims();
            let rshape = rhs.dims();
            if lshape != rshape {
                return Err(format!(
                    "lessEqual01: shape mismatch {lshape:?} vs {rshape:?}; implicit broadcasting forbidden"
                ));
            }
            validate_finite(&lhs, "lessEqual01 lhs")?;
            validate_finite(&rhs, "lessEqual01 rhs")?;
            let out = lhs.lower_equal(rhs).float();
            if out.dims() != lshape {
                return Err(format!(
                    "lessEqual01: shape invariant failed, expected {lshape:?}, got {:?}",
                    out.dims()
                ));
            }
            let data = out.clone().into_data();
            let vals = data
                .as_slice::<f32>()
                .map_err(|_| "lessEqual01 output not f32".to_string())?;
            for (i, &v) in vals.iter().enumerate() {
                let bits = v.to_bits();
                if bits != 0.0f32.to_bits() && bits != 1.0f32.to_bits() {
                    return Err(format!(
                        "lessEqual01: non-canonical predicate output at {i}: {v}"
                    ));
                }
            }
            SlotValue::F32(out)
        }
        RuntimeLayer::IndicesLike(axis) => {
            let reference = inputs[0].as_f32("indicesLike")?;
            let shape = reference.dims();
            for (ax, &d) in shape.iter().enumerate() {
                if d == 0 {
                    return Err(format!("indicesLike: zero-sized dim at axis {ax}"));
                }
            }
            let axis_len = shape[*axis];
            if axis_len > MAX_INDICES_LIKE_AXIS_LENGTH {
                return Err(format!(
                    "indicesLike: axis {axis} length {axis_len} exceeds exact-f32 bound {MAX_INDICES_LIKE_AXIS_LENGTH}"
                ));
            }
            let count = shape
                .into_iter()
                .try_fold(1usize, |a, b| a.checked_mul(b))
                .ok_or_else(|| format!("indicesLike: element-count overflow for {shape:?}"))?;
            let stride = shape[(*axis + 1)..]
                .iter()
                .copied()
                .try_fold(1usize, |a, b| a.checked_mul(b))
                .ok_or_else(|| format!("indicesLike: stride overflow for {shape:?}"))?;
            let mut values = Vec::with_capacity(count);
            for flat_idx in 0..count {
                values.push(((flat_idx / stride) % axis_len) as f32);
            }
            let out = Tensor::<4>::from_data(TensorData::new(values, shape), device);
            SlotValue::F32(out)
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
