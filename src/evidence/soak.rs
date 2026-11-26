use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Instant;

use crate::es::rng::{AnyRng, RngKind};
use crate::runtime::engine::PreparedGraph;
use crate::spec::{SpecV1, sha256_f32_slice};

#[derive(Debug, Clone, Serialize)]
pub struct SoakReport {
    pub ok: bool,
    pub runs: usize,
    pub identical: bool,
    pub sha256: String,
    pub output_shape: [usize; 4],
    pub output_f32_count: usize,
    pub first_mismatch_index: Option<usize>,
    pub total_ms: f64,
    pub ms_per_run: f64,
}

/// Long-run determinism proof: compiles `spec` and executes `runs` fresh
/// evaluations on `Device::flex()`, verifying bit-identical SHA-256 output digests.
pub fn soak_spec(spec: &SpecV1, runs: usize) -> Result<SoakReport, String> {
    if runs < 2 {
        return Err(format!("soak: runs must be >= 2, got {runs}"));
    }
    let graph = PreparedGraph::compile(spec)?;

    let mut ref_hash: Option<String> = None;
    let mut first_mismatch: Option<usize> = None;
    let mut out_shape = [0usize; 4];
    let mut out_len = 0usize;

    let t0 = Instant::now();
    for i in 0..runs {
        let out = graph.run_once(spec, None)?;
        out_shape = out.shape;
        out_len = out.values.len();
        if !out.values.iter().all(|v| v.is_finite()) {
            return Err(format!("soak: non-finite output detected at run {i}"));
        }
        let h = sha256_f32_slice(&out.values);
        match &ref_hash {
            None => ref_hash = Some(h),
            Some(expected) => {
                if &h != expected && first_mismatch.is_none() {
                    first_mismatch = Some(i);
                }
            }
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let identical = first_mismatch.is_none();

    Ok(SoakReport {
        ok: identical,
        runs,
        identical,
        sha256: ref_hash.unwrap_or_default(),
        output_shape: out_shape,
        output_f32_count: out_len,
        first_mismatch_index: first_mismatch,
        total_ms: (total_ms * 1000.0).round() / 1000.0,
        ms_per_run: ((total_ms / runs as f64) * 1000.0).round() / 1000.0,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrngManifest {
    pub ok: bool,
    pub algorithm: String,
    pub seed: u64,
    pub count: usize,
    pub sha256: String,
    pub preview: Vec<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrngVerifyReport {
    pub ok: bool,
    pub verified: bool,
    pub algorithm: String,
    pub seed: u64,
    pub count: usize,
    pub expected_sha256: String,
    pub actual_sha256: String,
}

fn generate_stream_digest(kind: RngKind, seed: u64, count: usize) -> Result<(String, Vec<f32>), String> {
    if count == 0 {
        return Err("prng-check: count must be >= 1, got 0".into());
    }
    const MAX_PRNG_COUNT: usize = 10_000_000;
    if count > MAX_PRNG_COUNT {
        return Err(format!(
            "prng-check: count {count} exceeds maximum allowed {MAX_PRNG_COUNT}"
        ));
    }
    let mut rng = AnyRng::new(kind, seed);
    let mut hasher = Sha256::new();
    let mut preview = Vec::with_capacity(count.min(5));
    for i in 0..count {
        let v = rng.uniform();
        let norm = if v == 0.0 { 0.0f32 } else { v };
        hasher.update(norm.to_le_bytes());
        if i < 5 {
            preview.push(norm);
        }
    }
    let sha256 = format!("sha256:{:x}", hasher.finalize());
    Ok((sha256, preview))
}

/// Certify a deterministic PRNG stream by hashing `count` uniform `f32` draws.
pub fn certify_prng_stream(kind: RngKind, seed: u64, count: usize) -> Result<PrngManifest, String> {
    let (sha256, preview) = generate_stream_digest(kind, seed, count)?;
    Ok(PrngManifest {
        ok: true,
        algorithm: kind.name().to_string(),
        seed,
        count,
        sha256,
        preview,
    })
}

/// Re-generate and verify a previously certified PRNG stream manifest.
pub fn verify_prng_manifest(
    algorithm: &str,
    seed: u64,
    count: usize,
    expected_sha256: &str,
) -> Result<PrngVerifyReport, String> {
    let kind = RngKind::parse(algorithm)?;
    let (actual_sha256, _) = generate_stream_digest(kind, seed, count)?;
    let verified = actual_sha256 == expected_sha256;
    Ok(PrngVerifyReport {
        ok: verified,
        verified,
        algorithm: kind.name().to_string(),
        seed,
        count,
        expected_sha256: expected_sha256.to_string(),
        actual_sha256,
    })
}
