use serde::Serialize;
use std::collections::HashSet;
use std::time::Instant;

use crate::catalog::expected_weight_count;
use crate::engine::PreparedGraph;
use crate::spec::{SpecV1, sha256_f32_slice, sha256_json};
use crate::validate::validate_spec;
use crate::verify::{default_cases_from_spec, verify_specs};

#[derive(Debug, Clone, Serialize)]
pub struct ProfileReport {
    pub ok: bool,
    pub spec_sha256: String,
    pub warmup: usize,
    pub iters: usize,
    pub compile_ms: f64,
    pub min_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub steps_count: usize,
    pub total_params: usize,
    pub output_shape: [usize; 4],
    pub outputs_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparePerfReport {
    pub ok: bool,
    pub outputs_match: bool,
    pub max_abs_diff: f64,
    pub speedup: f64,
    pub median_delta_ms: f64,
    pub steps_delta: isize,
    pub params_delta: isize,
    pub base: ProfileReport,
    pub candidate: ProfileReport,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn count_active_params(spec: &SpecV1) -> usize {
    let used: HashSet<usize> = spec.steps.iter().map(|s| s.layer).collect();
    used.into_iter()
        .filter_map(|li| spec.layers.get(li))
        .filter_map(|d| expected_weight_count(&d.constructor, &d.args))
        .sum()
}

pub fn profile_spec(spec: &SpecV1, warmup: usize, iters: usize) -> Result<ProfileReport, String> {
    if iters == 0 {
        return Err("profile requires iters >= 1".into());
    }
    let val = validate_spec(spec, false);
    if !val.valid {
        return Err(
            serde_json::to_string(&val).unwrap_or_else(|_| format!("invalid spec: {:?}", val.code)),
        );
    }

    let t_compile = Instant::now();
    let graph = PreparedGraph::compile(spec)?;
    let compile_ms = t_compile.elapsed().as_secs_f64() * 1000.0;

    for _ in 0..warmup {
        let _ = graph.run_once(spec, None)?;
    }

    let mut samples_ms = Vec::with_capacity(iters);
    let mut last_out = None;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = graph.run_once(spec, None)?;
        samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_out = Some(out);
    }

    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let n = samples_ms.len();
    let min_ms = samples_ms[0];
    let max_ms = samples_ms[n - 1];
    let mean_ms = samples_ms.iter().sum::<f64>() / (n as f64);
    let median_ms = if n % 2 == 1 {
        samples_ms[n / 2]
    } else {
        0.5 * (samples_ms[n / 2 - 1] + samples_ms[n / 2])
    };
    let p95_idx = (((n as f64) * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    let p95_ms = samples_ms[p95_idx];

    let out = last_out.expect("iters >= 1");
    let outputs_sha256 = sha256_f32_slice(&out.values);

    Ok(ProfileReport {
        ok: true,
        spec_sha256: sha256_json(spec),
        warmup,
        iters,
        compile_ms: round3(compile_ms),
        min_ms: round3(min_ms),
        median_ms: round3(median_ms),
        p95_ms: round3(p95_ms),
        mean_ms: round3(mean_ms),
        max_ms: round3(max_ms),
        steps_count: spec.steps.len(),
        total_params: count_active_params(spec),
        output_shape: out.shape,
        outputs_sha256,
    })
}

pub fn compare_specs_perf(
    base: &SpecV1,
    candidate: &SpecV1,
    warmup: usize,
    iters: usize,
) -> Result<ComparePerfReport, String> {
    let base_rep = profile_spec(base, warmup, iters)?;
    let cand_rep = profile_spec(candidate, warmup, iters)?;

    let cases = default_cases_from_spec(base);
    let eq = verify_specs(base, candidate, &cases, 1e-5, 1e-5);
    let speedup = if cand_rep.median_ms > 0.0 {
        round3(base_rep.median_ms / cand_rep.median_ms)
    } else {
        1.0
    };
    let median_delta_ms = round3(cand_rep.median_ms - base_rep.median_ms);
    let steps_delta = (cand_rep.steps_count as isize) - (base_rep.steps_count as isize);
    let params_delta = (cand_rep.total_params as isize) - (base_rep.total_params as isize);

    Ok(ComparePerfReport {
        ok: true,
        outputs_match: eq.equivalent,
        max_abs_diff: eq.max_abs_diff,
        speedup,
        median_delta_ms,
        steps_delta,
        params_delta,
        base: base_rep,
        candidate: cand_rep,
    })
}
