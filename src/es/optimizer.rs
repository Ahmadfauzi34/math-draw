use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

use crate::spec::SpecV1;
use crate::validate::validate_spec;
use crate::verify::{EvalCase, TargetsFile};

use super::diag::{compute_flags, diversity, mean_std, vec_l2_norm, vec_l2_norm_f64};
use super::rng::{AnyRng, RngKind};
use super::solve_gd::{eval_loss_and_grad_autodiff, eval_loss_only};
use super::{ActiveLayerSlice, apply_flat_weights, discover_trainable_slices};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EsStrategyKind {
    OpenEs,
    MuLambda,
}

impl EsStrategyKind {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "openes" | "open-es" | "openes_antithetic" => Ok(Self::OpenEs),
            "mu-lambda" | "mu_lambda" | "mulambda" => Ok(Self::MuLambda),
            other => Err(format!(
                "unknown ES strategy '{other}' (expected: openes | mu-lambda)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::OpenEs => "openes_antithetic",
            Self::MuLambda => "mu_lambda",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitEsConfig {
    pub generations: usize,
    pub pop: usize,
    pub mu: Option<usize>,
    pub sigma: f64,
    pub lr: f64,
    pub seed: u64,
    pub strategy: EsStrategyKind,
    pub rng: RngKind,
    pub sensitivity_guided: bool,
    pub freeze_threshold: f64,
}

impl Default for FitEsConfig {
    fn default() -> Self {
        Self {
            generations: 40,
            pop: 16,
            mu: None,
            sigma: 0.2,
            lr: 0.2,
            seed: 42,
            strategy: EsStrategyKind::OpenEs,
            rng: RngKind::Xorshift64,
            sensitivity_guided: false,
            freeze_threshold: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FitTrajectoryItem {
    #[serde(rename = "gen")]
    pub generation: usize,
    pub best_loss: f64,
    pub worst_loss: f64,
    pub mean_loss: f64,
    pub std_loss: f64,
    pub improvement: f64,
    pub stagnation: usize,
    pub diversity: f64,
    pub mean_norm: f64,
    pub best_norm: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FitResult {
    pub ok: bool,
    pub strategy: String,
    pub rng: String,
    pub dim: usize,
    pub sensitivity_guided: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub frozen_layers: Vec<usize>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub layer_sensitivity: BTreeMap<String, f64>,
    pub active_layers: Vec<ActiveLayerSlice>,
    pub generations: usize,
    pub pop: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mu: Option<usize>,
    pub sigma: f64,
    pub lr: f64,
    pub seed: u64,
    pub start_loss: f64,
    pub final_loss: f64,
    pub stagnation: usize,
    pub final_diversity: f64,
    pub flags: Vec<String>,
    pub best_weights: Vec<f32>,
    pub best_weights_by_layer: BTreeMap<String, Vec<f32>>,
    pub trajectory: Vec<FitTrajectoryItem>,
    pub optimized_spec: SpecV1,
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

fn validate_fit_config(config: &FitEsConfig) -> Result<Option<usize>, String> {
    if config.generations == 0 {
        return Err("fit: generations must be >= 1".into());
    }
    if config.pop < 2 {
        return Err(format!("fit: pop must be >= 2, got {}", config.pop));
    }
    if !config.sigma.is_finite() || config.sigma <= 0.0 {
        return Err(format!(
            "fit: sigma must be finite and > 0, got {}",
            config.sigma
        ));
    }
    if !config.freeze_threshold.is_finite() || config.freeze_threshold < 0.0 {
        return Err(format!(
            "fit: freeze_threshold must be finite and >= 0, got {}",
            config.freeze_threshold
        ));
    }
    match config.strategy {
        EsStrategyKind::OpenEs => {
            if !config.pop.is_multiple_of(2) {
                return Err(format!(
                    "fit: openes requires an even pop >= 2 (antithetic pairs), got {}",
                    config.pop
                ));
            }
            if !config.lr.is_finite() || config.lr <= 0.0 {
                return Err(format!(
                    "fit: openes lr must be finite and > 0, got {}",
                    config.lr
                ));
            }
            if config.mu.is_some() {
                return Err("fit: --mu is only valid for mu-lambda strategy".into());
            }
            Ok(None)
        }
        EsStrategyKind::MuLambda => {
            let mu = config.mu.unwrap_or_else(|| (config.pop / 4).max(1));
            if mu == 0 || mu > config.pop {
                return Err(format!(
                    "fit: mu-lambda requires 1 <= mu <= pop, got mu={mu}, pop={}",
                    config.pop
                ));
            }
            Ok(Some(mu))
        }
    }
}

/// Configurable Evolutionary Strategy parameter optimizer supporting OpenES (antithetic)
/// and `(mu, lambda)` elitist selection across 4 deterministic PRNGs, with full per-generation
/// population diagnostics and fail-closed hyperparameter checks.
pub fn fit_es_with_config(
    spec: &SpecV1,
    targets: &TargetsFile,
    config: &FitEsConfig,
) -> Result<FitResult, String> {
    let resolved_mu = validate_fit_config(config)?;
    let v = validate_spec(spec, false);
    if !v.valid {
        return Err(format!("invalid spec: {:?}", v.code));
    }
    if targets.cases.is_empty() {
        return Err("fit: targets.cases must be non-empty".into());
    }
    let (slices, init_w) = discover_trainable_slices(spec, None)?;
    let dim = init_w.len();
    let mut rng = AnyRng::new(config.rng, config.seed);

    let start_spec = apply_flat_weights(spec, &slices, &init_w);
    let start_loss = eval_loss_only(&start_spec, &targets.cases, true)?;

    let mut frozen_layers = Vec::new();
    let mut layer_sensitivity = BTreeMap::new();
    let mut active_mask = vec![true; dim];

    if config.sensitivity_guided {
        layer_sensitivity = compute_layer_sensitivities(spec, &slices, &targets.cases, &init_w);
        if slices.len() > 1 {
            let mut candidate_frozen: Vec<usize> = Vec::new();
            for sl in &slices {
                let s = layer_sensitivity
                    .get(&sl.layer_index.to_string())
                    .copied()
                    .unwrap_or(0.0);
                if s < config.freeze_threshold {
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
    let mut parents: Vec<Vec<f32>> = match resolved_mu {
        Some(mu) => vec![init_w.clone(); mu],
        None => Vec::new(),
    };
    let mut traj = Vec::with_capacity(config.generations);
    let mut stagnation = 0usize;
    let mut last_diversity = 0.0f64;
    let mut last_flags: Vec<String> = Vec::new();

    for generation in 0..config.generations {
        let prev_best_loss = best_loss;
        let mut cands: Vec<Vec<f32>> = Vec::with_capacity(config.pop);
        let mut losses: Vec<f64> = Vec::with_capacity(config.pop);
        let mut non_finite_count = 0usize;

        match config.strategy {
            EsStrategyKind::OpenEs => {
                let pairs = config.pop / 2;
                let mut epsilons = Vec::with_capacity(config.pop);
                for _ in 0..pairs {
                    let eps: Vec<f64> = (0..dim)
                        .map(|d| {
                            if active_mask[d] {
                                rng.next_normal_f64()
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    for &sign in &[1.0f64, -1.0f64] {
                        let cand: Vec<f32> = mean_w
                            .iter()
                            .zip(eps.iter())
                            .map(|(&m, &e)| (m + sign * config.sigma * e) as f32)
                            .collect();
                        let cand_spec = apply_flat_weights(spec, &slices, &cand);
                        let l = match eval_loss_only(&cand_spec, &targets.cases, true) {
                            Ok(val) if val.is_finite() => val,
                            _ => {
                                non_finite_count += 1;
                                1e12
                            }
                        };
                        if l < best_loss {
                            best_loss = l;
                            best_w = cand.clone();
                        }
                        let signed_eps: Vec<f64> = eps.iter().map(|&e| sign * e).collect();
                        epsilons.push(signed_eps);
                        cands.push(cand);
                        losses.push(l);
                    }
                }

                let (mean_l, std_raw) = mean_std(&losses);
                let std_l = std_raw.max(1e-8);
                if non_finite_count == 0 {
                    for d in 0..dim {
                        if !active_mask[d] {
                            continue;
                        }
                        let mut grad_est = 0.0f64;
                        for p in 0..config.pop {
                            let adv = -(losses[p] - mean_l) / std_l;
                            grad_est += adv * epsilons[p][d];
                        }
                        mean_w[d] +=
                            config.lr * grad_est / ((config.pop as f64) * config.sigma);
                    }
                }
            }
            EsStrategyKind::MuLambda => {
                let mu = resolved_mu
                    .ok_or_else(|| "internal error: missing validated mu for mu-lambda".to_string())?;
                let mut scored: Vec<(f64, Vec<f32>)> = Vec::with_capacity(config.pop);
                for i in 0..config.pop {
                    let parent = &parents[i % mu];
                    let cand: Vec<f32> = parent
                        .iter()
                        .enumerate()
                        .map(|(d, &v)| {
                            if active_mask[d] {
                                ((v as f64) + rng.next_normal_f64() * config.sigma) as f32
                            } else {
                                v
                            }
                        })
                        .collect();
                    let cand_spec = apply_flat_weights(spec, &slices, &cand);
                    let l = match eval_loss_only(&cand_spec, &targets.cases, true) {
                        Ok(val) if val.is_finite() => val,
                        _ => {
                            non_finite_count += 1;
                            1e12
                        }
                    };
                    if l < best_loss {
                        best_loss = l;
                        best_w = cand.clone();
                    }
                    cands.push(cand.clone());
                    losses.push(l);
                    scored.push((l, cand));
                }

                if non_finite_count == 0 {
                    scored.sort_by(|a, b| {
                        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    parents = scored.into_iter().take(mu).map(|(_, c)| c).collect();
                    for (d, slot) in mean_w.iter_mut().enumerate().take(dim) {
                        let sum: f64 = parents.iter().map(|p| p[d] as f64).sum();
                        *slot = sum / (parents.len() as f64);
                    }
                }
            }
        }

        let (mean_l, std_l) = mean_std(&losses);
        let worst_l = losses
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let improvement = prev_best_loss - best_loss;
        if best_loss + 1e-12 < prev_best_loss {
            stagnation = 0;
        } else {
            stagnation += 1;
        }
        let div = diversity(&cands);
        let mean_norm = vec_l2_norm_f64(&mean_w);
        let best_norm = vec_l2_norm(&best_w);
        let flags = compute_flags(div, stagnation, mean_norm, non_finite_count);
        last_diversity = div;
        last_flags = flags.clone();

        traj.push(FitTrajectoryItem {
            generation,
            best_loss,
            worst_loss: worst_l,
            mean_loss: mean_l,
            std_loss: std_l,
            improvement,
            stagnation,
            diversity: div,
            mean_norm,
            best_norm,
            flags,
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
        strategy: config.strategy.name().to_string(),
        rng: config.rng.name().to_string(),
        dim,
        sensitivity_guided: config.sensitivity_guided,
        frozen_layers,
        layer_sensitivity,
        active_layers: slices,
        generations: config.generations,
        pop: config.pop,
        mu: resolved_mu,
        sigma: config.sigma,
        lr: if config.strategy == EsStrategyKind::OpenEs {
            config.lr
        } else {
            0.0
        },
        seed: config.seed,
        start_loss,
        final_loss: best_loss,
        stagnation,
        final_diversity: last_diversity,
        flags: last_flags,
        best_weights: best_w,
        best_weights_by_layer: best_by_layer,
        trajectory: traj,
        optimized_spec: opt_spec,
    })
}

/// Backward-compatible OpenES + Xorshift64 entry point preserved for existing callers.
#[allow(clippy::too_many_arguments)]
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
    let config = FitEsConfig {
        generations,
        pop,
        mu: None,
        sigma,
        lr,
        seed,
        strategy: EsStrategyKind::OpenEs,
        rng: RngKind::Xorshift64,
        sensitivity_guided,
        freeze_threshold,
    };
    fit_es_with_config(spec, targets, &config)
}
