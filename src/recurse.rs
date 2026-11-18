use burn::tensor::{Device, Tensor, TensorData};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet, VecDeque};

use crate::catalog::expected_weight_count;
use crate::engine::{eval_spec_with_inputs, execute_forward, get_layer_weight_len, SlotValue};
use crate::rewrite::{apply_rewrite, SIMPLIFY_ORDER};
use crate::spec::{SpecV1, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{EvalCase, TargetsFile, default_cases_from_spec, verify_specs};

#[derive(Debug, Clone, Serialize)]
pub struct SearchNodeSummary {
    pub id: String,
    pub depth: usize,
    pub parent_id: Option<String>,
    pub transform_chain: Vec<String>,
    pub steps_count: usize,
    pub active_params: usize,
    pub equivalent: bool,
    pub max_abs_diff: f64,
    pub state_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub ok: bool,
    pub baseline_sha256: String,
    pub best_node_id: String,
    pub best_sha256: String,
    pub baseline_steps: usize,
    pub best_steps: usize,
    pub baseline_params: usize,
    pub best_params: usize,
    pub visited_nodes: usize,
    pub transform_chain: Vec<String>,
    pub nodes: Vec<SearchNodeSummary>,
    pub best_spec: SpecV1,
}

fn count_active_params(spec: &SpecV1) -> usize {
    let used: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    used.into_iter()
        .filter_map(|li| spec.layers.get(li))
        .filter_map(|decl| expected_weight_count(&decl.constructor, &decl.args))
        .sum()
}

pub fn search_graph_tree(
    base: &SpecV1,
    max_depth: usize,
    cases_override: Option<&[EvalCase]>,
    abs_tol: f64,
    rel_tol: f64,
) -> Result<SearchResult, String> {
    let v = validate_spec(base, false);
    if !v.valid {
        return Err(format!("baseline spec invalid: {:?}", v.code));
    }

    let eval_cases: Vec<EvalCase> = match cases_override {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => default_cases_from_spec(base),
    };

    let base_sha = sha256_json(base);
    let base_steps = base.steps.len();
    let base_params = count_active_params(base);

    let mut nodes = vec![SearchNodeSummary {
        id: "root".to_string(),
        depth: 0,
        parent_id: None,
        transform_chain: vec![],
        steps_count: base_steps,
        active_params: base_params,
        equivalent: true,
        max_abs_diff: 0.0,
        state_sha256: base_sha.clone(),
    }];

    let mut best_id = "root".to_string();
    let mut best_sha = base_sha.clone();
    let mut best_spec = base.clone();
    let mut best_steps = base_steps;
    let mut best_params = base_params;
    let mut best_chain = Vec::new();

    let mut seen_shas = HashSet::from([base_sha.clone()]);
    let mut queue: VecDeque<(String, usize, Vec<String>, SpecV1)> = VecDeque::new();
    queue.push_back(("root".to_string(), 0, vec![], base.clone()));

    let max_total_nodes = 64usize;
    let mut counter = 0usize;

    while let Some((parent_id, depth, chain, cur_spec)) = queue.pop_front() {
        if depth >= max_depth || nodes.len() >= max_total_nodes {
            continue;
        }
        for &rule in SIMPLIFY_ORDER {
            if nodes.len() >= max_total_nodes {
                break;
            }
            let Ok(cand) = apply_rewrite(rule, &cur_spec) else {
                continue;
            };
            let cand_sha = sha256_json(&cand);
            if !seen_shas.insert(cand_sha.clone()) {
                continue;
            }
            let cand_valid = validate_spec(&cand, false);
            if !cand_valid.valid {
                continue;
            }
            let ver = verify_specs(base, &cand, &eval_cases, abs_tol, rel_tol);
            counter += 1;
            let node_id = format!("n{counter}");
            let mut next_chain = chain.clone();
            next_chain.push(rule.to_string());

            let steps_cnt = cand.steps.len();
            let params_cnt = count_active_params(&cand);

            nodes.push(SearchNodeSummary {
                id: node_id.clone(),
                depth: depth + 1,
                parent_id: Some(parent_id.clone()),
                transform_chain: next_chain.clone(),
                steps_count: steps_cnt,
                active_params: params_cnt,
                equivalent: ver.equivalent,
                max_abs_diff: ver.max_abs_diff,
                state_sha256: cand_sha.clone(),
            });

            if ver.equivalent {
                let better = steps_cnt < best_steps
                    || (steps_cnt == best_steps && params_cnt < best_params);
                if better {
                    best_id = node_id.clone();
                    best_sha = cand_sha;
                    best_spec = cand.clone();
                    best_steps = steps_cnt;
                    best_params = params_cnt;
                    best_chain = next_chain.clone();
                }
                queue.push_back((node_id, depth + 1, next_chain, cand));
            }
        }
    }

    Ok(SearchResult {
        ok: true,
        baseline_sha256: base_sha,
        best_node_id: best_id,
        best_sha256: best_sha,
        baseline_steps: base_steps,
        best_steps,
        baseline_params: base_params,
        best_params,
        visited_nodes: nodes.len(),
        transform_chain: best_chain,
        nodes,
        best_spec,
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct ActiveLayerSlice {
    pub layer_index: usize,
    pub constructor: String,
    pub offset: usize,
    pub length: usize,
}

fn default_weights_for_layer(spec: &SpecV1, layer_idx: usize) -> Vec<f32> {
    let decl = &spec.layers[layer_idx];
    let len = get_layer_weight_len(decl);
    if let Some(existing) = spec.weights.get(&layer_idx.to_string())
        && existing.len() == len
    {
        return existing.clone();
    }
    match decl.constructor.as_str() {
        "batchNorm" | "groupNorm" | "instanceNorm" | "layerNorm" => {
            let mut v = vec![0.0f32; len];
            for x in v.iter_mut().take(len / 2) {
                *x = 1.0;
            }
            v
        }
        "rmsNorm" => vec![1.0f32; len],
        "prelu" => {
            let alpha = decl.args.get(2).and_then(|v| v.as_f64()).unwrap_or(0.25) as f32;
            vec![alpha; len]
        }
        _ => vec![0.0f32; len],
    }
}

pub fn discover_trainable_slices(
    spec: &SpecV1,
    layer_filter: Option<usize>,
) -> Result<(Vec<ActiveLayerSlice>, Vec<f32>), String> {
    let used_layers: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    let mut slices = Vec::new();
    let mut flat = Vec::new();

    for li in 0..spec.layers.len() {
        if let Some(target_li) = layer_filter
            && li != target_li
        {
            continue;
        }
        if !used_layers.contains(&li) && layer_filter.is_none() {
            continue;
        }
        let decl = &spec.layers[li];
        let len = get_layer_weight_len(decl);
        if len == 0 || decl.constructor == "embedding" {
            continue;
        }
        let w = default_weights_for_layer(spec, li);
        slices.push(ActiveLayerSlice {
            layer_index: li,
            constructor: decl.constructor.clone(),
            offset: flat.len(),
            length: len,
        });
        flat.extend(w);
    }

    if slices.is_empty() {
        if let Some(idx) = layer_filter {
            return Err(format!("layer {idx} has no differentiable trainable weights"));
        }
        return Err("spec has no active trainable weighted layers".into());
    }
    Ok((slices, flat))
}

pub fn apply_flat_weights(spec: &SpecV1, slices: &[ActiveLayerSlice], flat: &[f32]) -> SpecV1 {
    let mut s = spec.clone();
    for sl in slices {
        s.weights.insert(
            sl.layer_index.to_string(),
            flat[sl.offset..sl.offset + sl.length].to_vec(),
        );
    }
    s
}

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
    let Ok(coarse) = hessian_diag_second_order(base_spec, slices, cases, has_explicit, w, f_base, 1e-2) else {
        return ("untrusted", vec![1.0; w.len()]);
    };
    let Ok(fine) = hessian_diag_second_order(base_spec, slices, cases, has_explicit, w, f_base, 5e-3) else {
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

fn eval_loss_only(
    spec: &SpecV1,
    cases: &[EvalCase],
    has_explicit_targets: bool,
) -> Result<f64, String> {
    let mut total_loss = 0.0f64;
    for case in cases {
        let out = eval_spec_with_inputs(spec, Some(&case.inputs))?;
        let exp = case.expected.as_ref().unwrap();
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

fn eval_loss_and_grad_autodiff(
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
        let exp = case.expected.as_ref().unwrap();
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
            let exp_t =
                Tensor::<4>::from_data(TensorData::new(exp.clone(), y_dims), &device);
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

    let final_norm = cur_grad.iter().map(|&g| (g as f64).powi(2)).sum::<f64>().sqrt();
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

#[derive(Debug, Clone, Serialize)]
pub struct FitTrajectoryItem {
    #[serde(rename = "gen")]
    pub generation: usize,
    pub best_loss: f64,
    pub mean_loss: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub ok: bool,
    pub dim: usize,
    pub sensitivity_guided: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub frozen_layers: Vec<usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub layer_sensitivity: BTreeMap<String, f64>,
    pub active_layers: Vec<ActiveLayerSlice>,
    pub generations: usize,
    pub pop: usize,
    pub sigma: f64,
    pub lr: f64,
    pub seed: u64,
    pub start_loss: f64,
    pub final_loss: f64,
    pub best_weights: Vec<f32>,
    pub best_weights_by_layer: BTreeMap<String, Vec<f32>>,
    pub trajectory: Vec<FitTrajectoryItem>,
    pub optimized_spec: SpecV1,
}

struct EsRng(u64);
impl EsRng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        if x == 0 {
            x = 0x517cc1b727220a95;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = ((self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn compute_layer_sensitivities(
    base_spec: &SpecV1,
    slices: &[ActiveLayerSlice],
    cases: &[EvalCase],
    init_w: &[f32],
) -> BTreeMap<String, f64> {
    let start_spec = apply_flat_weights(base_spec, slices, init_w);
    let mut out = BTreeMap::new();

    if let Ok((_loss, grad)) =
        eval_loss_and_grad_autodiff(&start_spec, slices, cases, true, init_w.len())
    {
        for sl in slices {
            let max_g = grad[sl.offset..sl.offset + sl.length]
                .iter()
                .map(|&g| (g as f64).abs())
                .fold(0.0f64, f64::max);
            out.insert(sl.layer_index.to_string(), max_g);
        }
        return out;
    }

    // Fallback: finite differences if autodiff is unavailable for a mixed graph
    let eps = 1e-3f64;
    let mut work = init_w.to_vec();
    for sl in slices {
        let mut max_g = 0.0f64;
        for d in sl.offset..sl.offset + sl.length {
            let orig = init_w[d];
            work[d] = ((orig as f64) + eps) as f32;
            let lp = eval_loss_only(&apply_flat_weights(base_spec, slices, &work), cases, true)
                .unwrap_or(0.0);
            work[d] = ((orig as f64) - eps) as f32;
            let lm = eval_loss_only(&apply_flat_weights(base_spec, slices, &work), cases, true)
                .unwrap_or(0.0);
            work[d] = orig;
            let g = ((lp - lm) / (2.0 * eps)).abs();
            if g > max_g {
                max_g = g;
            }
        }
        out.insert(sl.layer_index.to_string(), max_g);
    }
    out
}

pub fn fit_es(
    spec: &SpecV1,
    targets: &TargetsFile,
    generations: usize,
    pop: usize,
    sigma: f64,
    lr: f64,
    seed: u64,
    sensitivity_guided: bool,
    freeze_threshold: f64,
) -> Result<FitResult, String> {
    let v = validate_spec(spec, false);
    if !v.valid {
        return Err(format!("invalid spec: {:?}", v.code));
    }
    if targets.cases.is_empty() {
        return Err("fit: targets.cases must be non-empty".into());
    }
    let (slices, init_w) = discover_trainable_slices(spec, None)?;
    let dim = init_w.len();
    let pairs = (pop / 2).max(2);
    let actual_pop = pairs * 2;
    let mut rng = EsRng(seed);

    let start_spec = apply_flat_weights(spec, &slices, &init_w);
    let start_loss = eval_loss_only(&start_spec, &targets.cases, true)?;

    let mut frozen_layers = Vec::new();
    let mut layer_sensitivity = BTreeMap::new();
    let mut active_mask = vec![true; dim];

    if sensitivity_guided {
        layer_sensitivity = compute_layer_sensitivities(spec, &slices, &targets.cases, &init_w);
        if slices.len() > 1 {
            let mut candidate_frozen: Vec<usize> = Vec::new();
            for sl in &slices {
                let s = layer_sensitivity
                    .get(&sl.layer_index.to_string())
                    .copied()
                    .unwrap_or(0.0);
                if s < freeze_threshold {
                    candidate_frozen.push(sl.layer_index);
                }
            }
            if candidate_frozen.len() < slices.len() {
                frozen_layers = candidate_frozen;
                let frozen_set: HashSet<usize> = frozen_layers.iter().copied().collect();
                for sl in &slices {
                    if frozen_set.contains(&sl.layer_index) {
                        for flag in &mut active_mask[sl.offset..sl.offset + sl.length] {
                            *flag = false;
                        }
                    }
                }
            }
        }
    }

    let mut best_w = init_w.clone();
    let mut best_loss = start_loss;
    let mut mean_w: Vec<f64> = init_w.iter().map(|&x| x as f64).collect();
    let mut traj = Vec::with_capacity(generations);

    for generation in 0..generations {
        let mut epsilons = Vec::with_capacity(actual_pop);
        let mut losses = Vec::with_capacity(actual_pop);

        for _ in 0..pairs {
            let eps: Vec<f64> = (0..dim)
                .map(|d| if active_mask[d] { rng.next_normal() } else { 0.0 })
                .collect();
            for &sign in &[1.0f64, -1.0f64] {
                let cand: Vec<f32> = mean_w
                    .iter()
                    .zip(eps.iter())
                    .map(|(&m, &e)| (m + sign * sigma * e) as f32)
                    .collect();
                let cand_spec = apply_flat_weights(spec, &slices, &cand);
                let l = eval_loss_only(&cand_spec, &targets.cases, true).unwrap_or(1e12);
                if l < best_loss {
                    best_loss = l;
                    best_w = cand;
                }
                let signed_eps: Vec<f64> = eps.iter().map(|&e| sign * e).collect();
                epsilons.push(signed_eps);
                losses.push(l);
            }
        }

        // Fitness shaping (centered rank / standardized negative loss)
        let mean_l: f64 = losses.iter().sum::<f64>() / (losses.len() as f64);
        let var_l: f64 = losses
            .iter()
            .map(|&l| (l - mean_l).powi(2))
            .sum::<f64>()
            / (losses.len() as f64);
        let std_l = var_l.sqrt().max(1e-8);

        for d in 0..dim {
            if !active_mask[d] {
                continue;
            }
            let mut grad_est = 0.0f64;
            for p in 0..actual_pop {
                let adv = -(losses[p] - mean_l) / std_l;
                grad_est += adv * epsilons[p][d];
            }
            mean_w[d] += lr * grad_est / ((actual_pop as f64) * sigma);
        }

        traj.push(FitTrajectoryItem {
            generation,
            best_loss,
            mean_loss: mean_l,
        });
    }

    let opt_spec = apply_flat_weights(spec, &slices, &best_w);
    let mut best_by_layer = BTreeMap::new();
    for sl in &slices {
        best_by_layer.insert(
            sl.layer_index.to_string(),
            best_w[sl.offset..sl.offset + sl.length].to_vec(),
        );
    }

    Ok(FitResult {
        ok: best_loss <= start_loss + 1e-12,
        dim,
        sensitivity_guided,
        frozen_layers,
        layer_sensitivity,
        active_layers: slices,
        generations,
        pop: actual_pop,
        sigma,
        lr,
        seed,
        start_loss,
        final_loss: best_loss,
        best_weights: best_w,
        best_weights_by_layer: best_by_layer,
        trajectory: traj,
        optimized_spec: opt_spec,
    })
}
