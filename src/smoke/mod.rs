mod advanced_checks;
mod core_checks;

use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Clone, Serialize)]
pub struct SmokeCheck {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct SmokeReport {
    pub ok: bool,
    pub passed: usize,
    pub total: usize,
    pub checks: Vec<SmokeCheck>,
}

pub(crate) fn run_check<F: FnOnce() -> Result<String, String>>(
    name: &'static str,
    f: F,
) -> SmokeCheck {
    let t0 = Instant::now();
    match f() {
        Ok(detail) => SmokeCheck {
            name,
            ok: true,
            detail,
            elapsed_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        },
        Err(detail) => SmokeCheck {
            name,
            ok: false,
            detail,
            elapsed_ms: (t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0,
        },
    }
}

pub(crate) fn approx_eq_slice(actual: &[f32], expected: &[f32], tol: f32) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!("len {} != {}", actual.len(), expected.len()));
    }
    for (i, (&a, &b)) in actual.iter().zip(expected.iter()).enumerate() {
        if (a - b).abs() > tol {
            return Err(format!("mismatch at [{i}]: got {a}, expected {b}"));
        }
    }
    Ok(())
}

pub fn run_smoke_suite() -> SmokeReport {
    let mut checks = core_checks::collect_core_checks();
    checks.extend(advanced_checks::collect_advanced_checks());

    let passed = checks.iter().filter(|c| c.ok).count();
    let total = checks.len();
    SmokeReport {
        ok: passed == total,
        passed,
        total,
        checks,
    }
}
