use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use std::collections::BTreeMap;

use crate::engine::{SlotValue, eval_spec_with_inputs, execute_forward};
use crate::spec::SpecV1;
use crate::validate::validate_spec;
use crate::verify::{EvalCase, TargetsFile};

use super::{ActiveLayerSlice, apply_flat_weights, discover_trainable_slices};

#[derive(Debug, Clone, Serialize)]
pub struct GdStepRecord {
    pub step: usize,
    pub loss: f64,
    pub grad_norm: f64,
    pub lr_used: f64,
    pub halvings: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SolveGdResult {
    pub ok: bool,
    pub backend: &'static str,
    pub targets_mode: &'static str,
    pub preconditioned: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_trust: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_damping: Option<f64>,
    pub active_layers: Vec<ActiveLayerSlice>,
    pub steps_taken: usize,
    pub backtracking_halvings: usize,
    pub start_loss: f64,
    pub final_loss: f64,
    pub best_weights: Vec<f32>,
    pub best_weights_by_layer: BTreeMap<String, Vec<f32>>,
    pub trajectory: Vec<GdStepRecord>,
    pub optimized_spec: SpecV1,
}

fn hessian_diag_second_order(
    base_spec: &SpecV1,
    slices: &[ActiveLayerSlice],
    cases: &[EvalCase],
    has_explicit: bool,
    w: &[f32],
    f_base: f64,
    h: f64,
) -> Result<Vec<f64>, String> {
    let mut diag = Vec::with_capacity(w.len());
    let mut work = w.to_vec();
    for i in 0..w.len() {
        let orig = w[i];
        work[i] = ((orig as f64) + h) as f32;
        let sp_plus = apply_flat_weights(base_spec, slices, &work);
        let f_plus = eval_loss_only(&sp_plus, cases, has_explicit)?;
        work[i] = ((orig as f64) - h) as f32;
        let sp_minus = apply_flat_weights(base_spec, slices, &work);
        let f_minus = eval_loss_only(&sp_minus, cases, has_explicit)?;
        work[i] = orig;
        diag.push((f_plus - 2.0 * f_base + f_minus) / (h * h));
    }
    Ok(diag)
}

fn classify_hessian_trust(coarse: &[f64], fine: &[f64], tol_abs: f64, tol_rel: f64) -> &'static str {
    let mut max_mag = 0.0f64;
    let mut consistent = true;
    for (&c, &f) in coarse.iter().zip(fine.iter()) {
        if !c.is_finite() || !f.is_finite() {
            return "untrusted";
        }
        let mag = c.abs().max(f.abs());
        if mag > max_mag {
            max_mag = mag;
        }
        if (c - f).abs() > tol_abs + tol_rel * mag {
            consistent = false;
        }
    }
    if max_mag <= tol_abs {
        "zero_within_noise"
    } else if consistent {
        "trusted"
    } else {
        "untrusted"
    }
}

fn compute_preconditioner(
    base_spec: &SpecV1,
    slices: &[ActiveLayerSlice],
    cases: &[EvalCase],
    has_explicit: bool,
    w: &[f32],
    f_base: f64,
    damping: f64,
) -> (&'static str, Vec<f64>) {
    if w.len() > 256 {
        return ("skipped_high_dim", vec![1.0; w.len()]);
    }
    let Ok(coarse) =
        hessian_diag_second_order(base_spec, slices, cases, has_explicit, w, f_base, 1e-2)
    else {
        return ("untrusted", vec![1.0; w.len()]);
    };
    let Ok(fine) =
        hessian_diag_second_order(base_spec, slices, cases, has_explicit, w, f_base, 5e-3)
    else {
        return ("untrusted", vec![1.0; w.len()]);
    };
    let trust = classify_hessian_trust(&coarse, &fine, 1e-2, 0.25);
    if trust == "trusted" {
        let scales = fine
            .iter()
            .map(|&d| 1.0 / d.abs().max(damping))
            .collect();
        (trust, scales)
    } else {
        (trust, vec![1.0; w.len()])
    }
}

fn resolve_gd_targets(
    spec: &SpecV1,
    targets: Option<&TargetsFile>,
) -> Result<(Vec<EvalCase>, &'static str, bool), String> {
    if let Some(tf) = targets {
        if tf.cases.is_empty() {
            return Err("targets.cases must be non-empty".into());
        }
        for (ci, c) in tf.cases.iter().enumerate() {
            if c.expected.is_none() {
                return Err(format!("targets.cases[{ci}] is missing 'expected' array"));
            }
        }
        return Ok((tf.cases.clone(), "explicit", true));
    }

    let y0 = eval_spec_with_inputs(spec, None)?;
    let n = y0.values.len();
    let mode = if n == 1 {
        "scalar_objective"
    } else {
        "zero_default"
    };
    let case = EvalCase {
        inputs: spec.inputs.clone(),
        expected: Some(vec![0.0f32; n]),
    };
    Ok((vec![case], mode, false))
}

pub(super) fn eval_loss_only(
    spec: &SpecV1,
    cases: &[EvalCase],
    has_explicit_targets: bool,
) -> Result<f64, String> {
    let mut total_loss = 0.0f64;
    for case in cases {
        let out = eval_spec_with_inputs(spec, Some(&case.inputs))?;
        let exp = case
            .expected
            .as_ref()
            .ok_or_else(|| "target case is missing 'expected' vector".to_string())?;
        if out.values.len() != exp.len() {
            return Err(format!(
                "target length mismatch: output has {}, expected {}",
                out.values.len(),
                exp.len()
            ));
        }
        if !has_explicit_targets && out.values.len() == 1 {
            total_loss += out.values[0] as f64;
        } else {
            for (&y, &t) in out.values.iter().zip(exp.iter()) {
                let d = (y as f64) - (t as f64);
                total_loss += d * d;
            }
        }
    }
    if !total_loss.is_finite() {
        return Err("non-finite loss".into());
    }
    Ok(total_loss)
}

pub(super) fn eval_loss_and_grad_autodiff(
    spec: &SpecV1,
    slices: &[ActiveLayerSlice],
    cases: &[EvalCase],
    has_explicit_targets: bool,
    total_dim: usize,
) -> Result<(f64, Vec<f32>), String> {
    let grad_layers: Vec<usize> = slices.iter().map(|s| s.layer_index).collect();
    let mut total_loss = 0.0f64;
    let mut total_grad = vec![0.0f32; total_dim];

    let slice_by_layer: BTreeMap<usize, usize> =
        slices.iter().map(|s| (s.layer_index, s.offset)).collect();

    for case in cases {
        let device = Device::flex().autodiff();
        let (slot_val, grad_params) =
            execute_forward(spec, &device, &grad_layers, Some(&case.inputs))?;
        let y = match slot_val {
            SlotValue::F32(t) => t,
            SlotValue::I64(..) => return Err("solve-gd requires f32 differentiable output".into()),
        };
        let exp = case
            .expected
            .as_ref()
            .ok_or_else(|| "target case is missing 'expected' vector".to_string())?;
        let y_dims = y.dims();
        let y_len: usize = y_dims.iter().product();
        if y_len != exp.len() {
            return Err(format!(
                "target length mismatch: output has {y_len}, expected {}",
                exp.len()
            ));
        }

        let loss_tensor = if !has_explicit_targets && y_len == 1 {
            y.reshape([1]).sum()
        } else {
            let exp_t = Tensor::<4>::from_data(TensorData::new(exp.clone(), y_dims), &device);
            (y - exp_t).square().sum()
        };

        let loss_scalar = loss_tensor
            .clone()
            .into_data()
            .try_into_vec::<f32>()
            .map_err(|e| format!("failed reading scalar loss: {e}"))?[0] as f64;
        if !loss_scalar.is_finite() {
            return Err(format!("non-finite autodiff loss: {loss_scalar}"));
        }
        total_loss += loss_scalar;

        let grads = loss_tensor.backward();
        for gp in grad_params {
            let g_vec = gp.tensor.extract_grad(&grads)?;
            if g_vec.is_empty() {
                continue;
            }
            if let Some(&layer_base) = slice_by_layer.get(&gp.layer_idx) {
                for (i, &gv) in g_vec.iter().enumerate() {
                    if !gv.is_finite() {
                        return Err(format!(
                            "non-finite gradient at layer {}[{}]: {gv}",
                            gp.layer_idx,
                            gp.offset + i
                        ));
                    }
                    total_grad[layer_base + gp.offset + i] += gv;
                }
            }
        }
    }

    Ok((total_loss, total_grad))
}

pub fn solve_gd_autodiff(
    spec: &SpecV1,
    targets: Option<&TargetsFile>,
    layer_filter: Option<usize>,
    steps: usize,
    lr: f64,
    precondition: bool,
) -> Result<SolveGdResult, String> {
    if steps == 0 {
        return Err("solve-gd: steps must be >= 1".into());
    }
    if !(lr.is_finite() && lr > 0.0) {
        return Err(format!(
            "solve-gd: lr must be finite and > 0, got {lr}"
        ));
    }
    let v = validate_spec(spec, false);
    if !v.valid {
        return Err(format!("invalid spec: {:?}", v.code));
    }
    let (slices, mut cur_w) = discover_trainable_slices(spec, layer_filter)?;
    let total_dim = cur_w.len();
    let (cases, targets_mode, has_explicit) = resolve_gd_targets(spec, targets)?;

    let mut cur_spec = apply_flat_weights(spec, &slices, &cur_w);
    let (mut cur_loss, mut cur_grad) =
        eval_loss_and_grad_autodiff(&cur_spec, &slices, &cases, has_explicit, total_dim)?;
    let start_loss = cur_loss;

    let damping = 1e-2f64;
    let mut hessian_trust: Option<&'static str> = None;
    let mut precond_scales = vec![1.0f64; total_dim];
    if precondition {
        let (trust, scales) =
            compute_preconditioner(spec, &slices, &cases, has_explicit, &cur_w, cur_loss, damping);
        hessian_trust = Some(trust);
        precond_scales = scales;
    }

    let mut traj = Vec::new();
    let mut total_halvings = 0usize;
    let max_halvings = 8usize;
    let tol = 1e-6f64;

    for step in 0..steps {
        let grad_norm_sq: f64 = cur_grad.iter().map(|&g| (g as f64) * (g as f64)).sum();
        let grad_norm = grad_norm_sq.sqrt();
        if grad_norm <= tol {
            traj.push(GdStepRecord {
                step,
                loss: cur_loss,
                grad_norm,
                lr_used: 0.0,
                halvings: 0,
            });
            break;
        }

        if precondition && step > 0 && step % 5 == 0 {
            let (trust, scales) = compute_preconditioner(
                spec,
                &slices,
                &cases,
                has_explicit,
                &cur_w,
                cur_loss,
                damping,
            );
            hessian_trust = Some(trust);
            precond_scales = scales;
        }

        let mut step_lr = lr;
        let mut accepted = None;
        let mut halvings_used = 0usize;

        for h in 0..=max_halvings {
            let cand_w: Vec<f32> = cur_w
                .iter()
                .zip(cur_grad.iter())
                .zip(precond_scales.iter())
                .map(|((&w, &g), &sc)| (w as f64 - step_lr * sc * (g as f64)) as f32)
                .collect();
            if cand_w.iter().any(|w| !w.is_finite()) {
                step_lr *= 0.5;
                continue;
            }
            let cand_spec = apply_flat_weights(spec, &slices, &cand_w);
            if let Ok(cand_loss) = eval_loss_only(&cand_spec, &cases, has_explicit)
                && cand_loss <= cur_loss + 1e-12
            {
                accepted = Some((cand_w, cand_spec, cand_loss));
                halvings_used = h;
                break;
            }
            step_lr *= 0.5;
        }

        let Some((next_w, next_spec, next_loss)) = accepted else {
            break;
        };
        total_halvings += halvings_used;
        traj.push(GdStepRecord {
            step,
            loss: cur_loss,
            grad_norm,
            lr_used: step_lr,
            halvings: halvings_used,
        });

        cur_w = next_w;
        cur_spec = next_spec;
        let (_recomp_loss, next_grad) =
            eval_loss_and_grad_autodiff(&cur_spec, &slices, &cases, has_explicit, total_dim)?;
        cur_loss = next_loss;
        cur_grad = next_grad;
    }

    let final_norm = cur_grad
        .iter()
        .map(|&g| (g as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    traj.push(GdStepRecord {
        step: traj.len(),
        loss: cur_loss,
        grad_norm: final_norm,
        lr_used: 0.0,
        halvings: 0,
    });

    let mut best_by_layer = BTreeMap::new();
    for sl in &slices {
        best_by_layer.insert(
            sl.layer_index.to_string(),
            cur_w[sl.offset..sl.offset + sl.length].to_vec(),
        );
    }

    Ok(SolveGdResult {
        ok: cur_loss <= start_loss + 1e-12,
        backend: "burn_flex_autodiff",
        targets_mode,
        preconditioned: precondition,
        hessian_trust,
        active_damping: if precondition { Some(damping) } else { None },
        active_layers: slices,
        steps_taken: traj.len().saturating_sub(1),
        backtracking_halvings: total_halvings,
        start_loss,
        final_loss: cur_loss,
        best_weights: cur_w,
        best_weights_by_layer: best_by_layer,
        trajectory: traj,
        optimized_spec: cur_spec,
    })
}
