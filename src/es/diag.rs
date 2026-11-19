/// Population mean and population standard deviation.
/// Returns `(NaN, NaN)` for an empty slice.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let m = values.iter().sum::<f64>() / (n as f64);
    let var = values
        .iter()
        .map(|&x| {
            let d = x - m;
            d * d
        })
        .sum::<f64>()
        / (n as f64);
    (m, var.sqrt())
}

/// Average per-dimension standard deviation across population candidates.
///
/// Returns `0.0` when `cands.len() < 2` or `dim == 0`, and `f64::NAN` on ragged inputs
/// rather than panicking.
pub fn diversity(cands: &[Vec<f32>]) -> f64 {
    let n = cands.len();
    if n < 2 {
        return 0.0;
    }
    let dim = cands[0].len();
    if dim == 0 {
        return 0.0;
    }
    if cands.iter().any(|c| c.len() != dim) {
        return f64::NAN;
    }

    let mut sum_std = 0.0f64;
    for d in 0..dim {
        let mut s = 0.0f64;
        for c in cands {
            s += c[d] as f64;
        }
        let m = s / (n as f64);
        let mut var = 0.0f64;
        for c in cands {
            let diff = (c[d] as f64) - m;
            var += diff * diff;
        }
        sum_std += (var / (n as f64)).sqrt();
    }
    sum_std / (dim as f64)
}

/// Euclidean L2 norm of an `f32` parameter vector.
pub fn vec_l2_norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt()
}

/// Euclidean L2 norm of an `f64` parameter vector.
pub fn vec_l2_norm_f64(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Compute deterministic ES health/diagnostic flags for a generation.
pub fn compute_flags(
    diversity_val: f64,
    stagnation: usize,
    mean_norm: f64,
    non_finite_count: usize,
) -> Vec<String> {
    let mut flags = Vec::new();
    if non_finite_count > 0 {
        flags.push("non_finite_candidate".to_string());
    }
    if !diversity_val.is_finite() || diversity_val < 1e-6 {
        flags.push("low_diversity".to_string());
    }
    if stagnation >= 10 {
        flags.push("stagnant".to_string());
    }
    if !mean_norm.is_finite() || mean_norm > 1e4 {
        flags.push("diverging_norm".to_string());
    }
    flags
}
