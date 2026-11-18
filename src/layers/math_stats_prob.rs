use burn::tensor::{Device, Tensor, TensorData};

use crate::spec::LayerDecl;

use super::{RuntimeLayer, SlotValue, arg_opt_f64};

const DEFAULT_COSINE_EPSILON: f64 = 1e-12;
const NORMALIZATION_TOLERANCE: f32 = 1e-5;

fn validate_feature_layout(shape: [usize; 4], ctx: &str) -> Result<(), String> {
    let [batch, features, h, w] = shape;
    if batch == 0 {
        return Err(format!("{ctx}: batch axis must be non-empty"));
    }
    if features == 0 {
        return Err(format!("{ctx}: feature axis 1 must be non-empty"));
    }
    if h != 1 || w != 1 {
        return Err(format!(
            "{ctx}: expected feature-vector layout [B,F,1,1], got {shape:?}"
        ));
    }
    Ok(())
}

fn validate_finite_features(t: &Tensor<4>, ctx: &str) -> Result<([usize; 4], Vec<f32>), String> {
    let shape = t.dims();
    validate_feature_layout(shape, ctx)?;
    let data = t.clone().into_data();
    let values = data
        .try_into_vec::<f32>()
        .map_err(|_| format!("{ctx}: expected f32 tensor"))?;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{ctx}: non-finite value at index {i}: {v}"));
        }
    }
    Ok((shape, values))
}

fn checked_scalar_output(out: Tensor<4>, batch: usize, ctx: &str) -> Result<Tensor<4>, String> {
    let shape = out.dims();
    if shape != [batch, 1, 1, 1] {
        return Err(format!(
            "{ctx}: expected output shape [{batch}, 1, 1, 1], got {shape:?}"
        ));
    }
    let data = out.clone().into_data();
    let values = data
        .as_slice::<f32>()
        .map_err(|_| format!("{ctx}: expected f32 tensor"))?;
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{ctx}: non-finite output at index {i}: {v}"));
        }
    }
    Ok(out)
}

fn validate_feature_pair(
    a: &Tensor<4>,
    b: &Tensor<4>,
    ctx: &str,
) -> Result<[usize; 4], String> {
    let sa = a.dims();
    let sb = b.dims();
    validate_feature_layout(sa, &format!("{ctx} lhs"))?;
    validate_feature_layout(sb, &format!("{ctx} rhs"))?;
    if sa != sb {
        return Err(format!("{ctx}: shape mismatch {sa:?} vs {sb:?}"));
    }
    let _ = validate_finite_features(a, &format!("{ctx} lhs"))?;
    let _ = validate_finite_features(b, &format!("{ctx} rhs"))?;
    Ok(sa)
}

fn batch_masses(values: &[f32], batch: usize, features: usize) -> Vec<f32> {
    (0..batch)
        .map(|b| values[b * features..(b + 1) * features].iter().sum())
        .collect()
}

fn validate_nonnegative_features(t: &Tensor<4>, ctx: &str) -> Result<([usize; 4], Vec<f32>), String> {
    let (shape, values) = validate_finite_features(t, ctx)?;
    for (i, &v) in values.iter().enumerate() {
        if v < 0.0 {
            return Err(format!(
                "{ctx}: probabilities/weights must be >= 0; index {i} has {v}"
            ));
        }
    }
    Ok((shape, values))
}

fn validate_normalized(t: &Tensor<4>, ctx: &str) -> Result<([usize; 4], Vec<f32>), String> {
    let (shape, values) = validate_nonnegative_features(t, ctx)?;
    let masses = batch_masses(&values, shape[0], shape[1]);
    for (b, mass) in masses.into_iter().enumerate() {
        if !mass.is_finite() || (mass - 1.0).abs() > NORMALIZATION_TOLERANCE {
            return Err(format!(
                "{ctx}: batch {b} must sum to 1 within tolerance {NORMALIZATION_TOLERANCE}; got {mass}"
            ));
        }
    }
    Ok((shape, values))
}

fn safe_log_input(input: Tensor<4>) -> Tensor<4> {
    let zero_mask = input.clone().lower_equal_elem(0.0);
    input.mask_fill(zero_mask, 1.0)
}

pub(super) fn try_build(decl: &LayerDecl) -> Result<Option<RuntimeLayer>, String> {
    let args = &decl.args;
    let layer = match decl.constructor.as_str() {
        "dot" => RuntimeLayer::Dot,
        "l2Norm" => RuntimeLayer::L2Norm,
        "l2Distance" => RuntimeLayer::L2Distance,
        "cosineSimilarity" => {
            let eps = arg_opt_f64(args, 1).unwrap_or(DEFAULT_COSINE_EPSILON);
            if !eps.is_finite() || eps <= 0.0 || eps > f32::MAX as f64 {
                return Err(format!(
                    "cosineSimilarity: epsilon must be finite, > 0, and <= f32::MAX, got {eps}"
                ));
            }
            RuntimeLayer::CosineSimilarity(eps as f32)
        }
        "sum" => RuntimeLayer::Sum,
        "mean" => RuntimeLayer::Mean,
        "variancePopulation" => RuntimeLayer::VariancePopulation,
        "stdPopulation" => RuntimeLayer::StdPopulation,
        "varianceSample" => RuntimeLayer::VarianceSample,
        "stdSample" => RuntimeLayer::StdSample,
        "min" => RuntimeLayer::Min,
        "max" => RuntimeLayer::Max,
        "normalize" => RuntimeLayer::Normalize,
        "entropy" => RuntimeLayer::Entropy,
        "crossEntropy" => RuntimeLayer::CrossEntropy,
        "klDivergence" => RuntimeLayer::KlDivergence,
        _ => return Ok(None),
    };
    Ok(Some(layer))
}

pub(super) fn try_exec(
    layer: &RuntimeLayer,
    inputs: &[SlotValue],
    device: &Device,
) -> Result<Option<SlotValue>, String> {
    let val = match layer {
        RuntimeLayer::Dot => {
            let a = inputs[0].as_f32("dot lhs")?;
            let b = inputs[1].as_f32("dot rhs")?;
            let shape = validate_feature_pair(&a, &b, "dot")?;
            let out = (a * b).sum_dim(1);
            SlotValue::F32(checked_scalar_output(out, shape[0], "dot output")?)
        }
        RuntimeLayer::L2Norm => {
            let x = inputs[0].as_f32("l2Norm")?;
            let (shape, _) = validate_finite_features(&x, "l2Norm input")?;
            let out = (x.clone() * x).sum_dim(1).sqrt();
            SlotValue::F32(checked_scalar_output(out, shape[0], "l2Norm output")?)
        }
        RuntimeLayer::L2Distance => {
            let a = inputs[0].as_f32("l2Distance lhs")?;
            let b = inputs[1].as_f32("l2Distance rhs")?;
            let shape = validate_feature_pair(&a, &b, "l2Distance")?;
            let diff = a - b;
            let out = (diff.clone() * diff).sum_dim(1).sqrt();
            SlotValue::F32(checked_scalar_output(out, shape[0], "l2Distance output")?)
        }
        RuntimeLayer::CosineSimilarity(eps) => {
            let a = inputs[0].as_f32("cosineSimilarity lhs")?;
            let b = inputs[1].as_f32("cosineSimilarity rhs")?;
            let shape = validate_feature_pair(&a, &b, "cosineSimilarity")?;
            let dot = (a.clone() * b.clone()).sum_dim(1);
            let norm_a = (a.clone() * a).sum_dim(1).sqrt();
            let norm_b = (b.clone() * b).sum_dim(1).sqrt();
            let denom = (norm_a * norm_b).clamp_min(*eps);
            let out = dot / denom;
            SlotValue::F32(checked_scalar_output(
                out,
                shape[0],
                "cosineSimilarity output",
            )?)
        }
        RuntimeLayer::Sum => {
            let x = inputs[0].as_f32("sum")?;
            let (shape, _) = validate_finite_features(&x, "sum input")?;
            SlotValue::F32(checked_scalar_output(x.sum_dim(1), shape[0], "sum output")?)
        }
        RuntimeLayer::Mean => {
            let x = inputs[0].as_f32("mean")?;
            let (shape, _) = validate_finite_features(&x, "mean input")?;
            SlotValue::F32(checked_scalar_output(
                x.mean_dim(1),
                shape[0],
                "mean output",
            )?)
        }
        RuntimeLayer::Min => {
            let x = inputs[0].as_f32("min")?;
            let (shape, _) = validate_finite_features(&x, "min input")?;
            SlotValue::F32(checked_scalar_output(x.min_dim(1), shape[0], "min output")?)
        }
        RuntimeLayer::Max => {
            let x = inputs[0].as_f32("max")?;
            let (shape, _) = validate_finite_features(&x, "max input")?;
            SlotValue::F32(checked_scalar_output(x.max_dim(1), shape[0], "max output")?)
        }
        RuntimeLayer::VariancePopulation
        | RuntimeLayer::StdPopulation
        | RuntimeLayer::VarianceSample
        | RuntimeLayer::StdSample => {
            let is_std = matches!(
                layer,
                RuntimeLayer::StdPopulation | RuntimeLayer::StdSample
            );
            let ddof: usize = match layer {
                RuntimeLayer::VarianceSample | RuntimeLayer::StdSample => 1,
                _ => 0,
            };
            let ctx = match layer {
                RuntimeLayer::VariancePopulation => "variancePopulation",
                RuntimeLayer::StdPopulation => "stdPopulation",
                RuntimeLayer::VarianceSample => "varianceSample",
                RuntimeLayer::StdSample => "stdSample",
                _ => unreachable!(),
            };
            let x = inputs[0].as_f32(ctx)?;
            let (shape, values) = validate_finite_features(&x, &format!("{ctx} input"))?;
            let (batch, features) = (shape[0], shape[1]);
            if ddof > 0 && features < 2 {
                return Err(format!(
                    "{ctx}: sample variance requires at least 2 features, got {features}"
                ));
            }
            let mut out = Vec::with_capacity(batch);
            for b in 0..batch {
                let row = &values[b * features..(b + 1) * features];
                let n = features as f64;
                let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
                let sq: f64 = row
                    .iter()
                    .map(|&v| {
                        let d = (v as f64) - mean;
                        d * d
                    })
                    .sum();
                let var = sq / (n - ddof as f64);
                let val_f64 = if is_std { var.sqrt() } else { var };
                let val_f32 = val_f64 as f32;
                if !val_f32.is_finite() {
                    return Err(format!(
                        "{ctx} output: non-finite result for batch {b}: {val_f64}"
                    ));
                }
                out.push(val_f32);
            }
            SlotValue::F32(Tensor::from_data(
                TensorData::new(out, [batch, 1, 1, 1]),
                device,
            ))
        }
        RuntimeLayer::Normalize => {
            let x = inputs[0].as_f32("normalize")?;
            let (shape, values) = validate_nonnegative_features(&x, "normalize input")?;
            let masses = batch_masses(&values, shape[0], shape[1]);
            for (b, mass) in masses.into_iter().enumerate() {
                if !mass.is_finite() || mass <= 0.0 {
                    return Err(format!(
                        "normalize: batch {b} must have strictly positive finite mass; got {mass}"
                    ));
                }
            }
            let denom = x.clone().sum_dim(1).repeat_dim(1, shape[1]);
            let out = x / denom;
            let _ = validate_normalized(&out, "normalize output")?;
            SlotValue::F32(out)
        }
        RuntimeLayer::Entropy => {
            let p = inputs[0].as_f32("entropy")?;
            let (shape, _) = validate_normalized(&p, "entropy input")?;
            let safe_p = safe_log_input(p.clone());
            let out = -(p * safe_p.log()).sum_dim(1);
            SlotValue::F32(checked_scalar_output(out, shape[0], "entropy output")?)
        }
        RuntimeLayer::CrossEntropy | RuntimeLayer::KlDivergence => {
            let is_kl = matches!(layer, RuntimeLayer::KlDivergence);
            let ctx = if is_kl { "klDivergence" } else { "crossEntropy" };
            let p = inputs[0].as_f32(&format!("{ctx} P"))?;
            let q = inputs[1].as_f32(&format!("{ctx} Q"))?;
            let (p_shape, p_vals) = validate_normalized(&p, &format!("{ctx} P"))?;
            let (q_shape, q_vals) = validate_normalized(&q, &format!("{ctx} Q"))?;
            if p_shape != q_shape {
                return Err(format!(
                    "{ctx}: distribution shape mismatch {p_shape:?} vs {q_shape:?}"
                ));
            }
            for (i, (&pv, &qv)) in p_vals.iter().zip(q_vals.iter()).enumerate() {
                if pv > 0.0 && qv == 0.0 {
                    return Err(format!(
                        "{ctx}: Q has zero support at index {i} where P is positive ({pv})"
                    ));
                }
            }
            let safe_q = safe_log_input(q);
            let out = if is_kl {
                let safe_p = safe_log_input(p.clone());
                (p * (safe_p.log() - safe_q.log())).sum_dim(1)
            } else {
                -(p * safe_q.log()).sum_dim(1)
            };
            SlotValue::F32(checked_scalar_output(out, p_shape[0], &format!("{ctx} output"))?)
        }
        _ => return Ok(None),
    };
    Ok(Some(val))
}
