use serde::Serialize;
use std::collections::{BTreeMap, HashSet};

use crate::spec::SpecV1;
use crate::validate::validate_spec;
use crate::verify::{EvalCase, TargetsFile};

use super::solve_gd::{eval_loss_and_grad_autodiff, eval_loss_only};
use super::{ActiveLayerSlice, apply_flat_weights, discover_trainable_slices};

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
        let u1 =
            ((self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12);
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
