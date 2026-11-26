use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

use crate::catalog::is_int_constructor;
use crate::engine::eval_spec_with_inputs;
use crate::spec::SpecV1;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalCase {
    pub inputs: BTreeMap<String, Vec<Value>>,
    #[serde(default)]
    pub expected: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TargetsFile {
    pub cases: Vec<EvalCase>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseDiffDetail {
    pub case_index: usize,
    pub ok: bool,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VerifyReport {
    pub ok: bool,
    pub equivalent: bool,
    pub verdict: &'static str,
    pub cases: usize,
    pub max_abs_diff: f64,
    pub max_rel_diff: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub case_details: Vec<CaseDiffDetail>,
}

pub struct FuzzConfig {
    pub count: usize,
    pub seed: u64,
    pub lo: f64,
    pub hi: f64,
}

impl Default for FuzzConfig {
    fn default() -> Self {
        Self {
            count: 8,
            seed: 0x9E37_79B9_7F4A_7C15,
            lo: -2.0,
            hi: 2.0,
        }
    }
}

struct XorShift64(u64);
impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        if x == 0 {
            x = 0x9E37_79B9_7F4A_7C15;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) / ((1u64 << 53) as f64)
    }
}

pub fn generate_fuzz_cases(spec: &SpecV1, cfg: &FuzzConfig) -> Vec<EvalCase> {
    let mut rng = XorShift64(cfg.seed);

    // Detect embedding vocab bounds and int ports so generated fuzz inputs stay in domain
    let mut slot_vocab: BTreeMap<usize, usize> = BTreeMap::new();
    let mut int_slots = vec![false; spec.num_slots];
    for st in &spec.steps {
        if let Some(layer) = spec.layers.get(st.layer) {
            if layer.constructor == "embedding"
                && let Some(vocab) = layer.args.get(1).and_then(|v| v.as_u64())
                && let Some(&in_s) = st.slots.first()
            {
                let v = vocab as usize;
                slot_vocab
                    .entry(in_s)
                    .and_modify(|cur| *cur = (*cur).min(v))
                    .or_insert(v);
            } else if is_int_constructor(&layer.constructor) && layer.constructor != "intFromFloat" {
                for &in_s in st.slots.iter().take(st.slots.len().saturating_sub(1)) {
                    if in_s < int_slots.len() {
                        int_slots[in_s] = true;
                    }
                }
            }
        }
    }

    let mut cases = Vec::with_capacity(cfg.count);
    for c_idx in 0..cfg.count {
        let mut inputs = BTreeMap::new();
        for port in &spec.ports {
            let n: usize = port.shape.iter().product();
            if c_idx == 0
                && let Some(existing) = spec.inputs.get(&port.slot.to_string())
                && existing.len() == n
            {
                inputs.insert(port.slot.to_string(), existing.clone());
                continue;
            }
            let mut arr = Vec::with_capacity(n);
            for _ in 0..n {
                if let Some(&vocab) = slot_vocab.get(&port.slot) {
                    let tok = (rng.next_u64() as usize) % vocab.max(1);
                    arr.push(Value::from(tok));
                } else if port.slot < int_slots.len() && int_slots[port.slot] {
                    let span = (cfg.hi - cfg.lo).max(2.0).round() as i64;
                    let val = (rng.next_u64() % (span as u64 + 1)) as i64 + (cfg.lo.round() as i64);
                    arr.push(Value::from(val));
                } else {
                    let v = cfg.lo + rng.next_f64() * (cfg.hi - cfg.lo);
                    // Round slightly for cleaner JSON reproducibility
                    let rounded = (v * 1e5).round() / 1e5;
                    arr.push(Value::from(rounded));
                }
            }
            inputs.insert(port.slot.to_string(), arr);
        }
        cases.push(EvalCase {
            inputs,
            expected: None,
        });
    }
    cases
}

pub fn default_cases_from_spec(spec: &SpecV1) -> Vec<EvalCase> {
    if spec.ports.iter().all(|p| spec.inputs.contains_key(&p.slot.to_string())) {
        vec![EvalCase {
            inputs: spec.inputs.clone(),
            expected: None,
        }]
    } else {
        generate_fuzz_cases(
            spec,
            &FuzzConfig {
                count: 4,
                ..Default::default()
            },
        )
    }
}

pub fn verify_specs(
    base: &SpecV1,
    cand: &SpecV1,
    cases: &[EvalCase],
    abs_tol: f64,
    rel_tol: f64,
) -> VerifyReport {
    if !(abs_tol.is_finite() && abs_tol >= 0.0) || !(rel_tol.is_finite() && rel_tol >= 0.0) {
        return VerifyReport {
            ok: false,
            equivalent: false,
            verdict: "REJECT",
            cases: cases.len(),
            max_abs_diff: f64::INFINITY,
            max_rel_diff: f64::INFINITY,
            abs_tol,
            rel_tol,
            case_details: vec![CaseDiffDetail {
                case_index: 0,
                ok: false,
                max_abs_diff: f64::INFINITY,
                max_rel_diff: f64::INFINITY,
                error: Some(format!(
                    "invalid tolerance: abs_tol ({abs_tol}) and rel_tol ({rel_tol}) must be finite and >= 0"
                )),
            }],
        };
    }
    let mut details = Vec::with_capacity(cases.len());
    let mut global_abs = 0.0f64;
    let mut global_rel = 0.0f64;
    let mut all_ok = !cases.is_empty();

    for (ci, case) in cases.iter().enumerate() {
        let r_base = eval_spec_with_inputs(base, Some(&case.inputs));
        let r_cand = eval_spec_with_inputs(cand, Some(&case.inputs));
        match (r_base, r_cand) {
            (Ok(ob), Ok(oc)) => {
                if ob.shape != oc.shape || ob.values.len() != oc.values.len() {
                    all_ok = false;
                    details.push(CaseDiffDetail {
                        case_index: ci,
                        ok: false,
                        max_abs_diff: f64::INFINITY,
                        max_rel_diff: f64::INFINITY,
                        error: Some(format!(
                            "output shape mismatch: base {:?} vs candidate {:?}",
                            ob.shape, oc.shape
                        )),
                    });
                    continue;
                }
                let mut c_abs = 0.0f64;
                let mut c_rel = 0.0f64;
                let mut case_ok = true;
                for (&a, &b) in ob.values.iter().zip(oc.values.iter()) {
                    let ad = (a as f64 - b as f64).abs();
                    let denom = (a as f64).abs().max((b as f64).abs()).max(1e-12);
                    let rd = ad / denom;
                    if ad > c_abs {
                        c_abs = ad;
                    }
                    if rd > c_rel {
                        c_rel = rd;
                    }
                    if !(ad <= abs_tol || rd <= rel_tol) {
                        case_ok = false;
                    }
                }
                if c_abs > global_abs {
                    global_abs = c_abs;
                }
                if c_rel > global_rel {
                    global_rel = c_rel;
                }
                if !case_ok {
                    all_ok = false;
                }
                details.push(CaseDiffDetail {
                    case_index: ci,
                    ok: case_ok,
                    max_abs_diff: c_abs,
                    max_rel_diff: c_rel,
                    error: None,
                });
            }
            (Err(eb), _) => {
                all_ok = false;
                details.push(CaseDiffDetail {
                    case_index: ci,
                    ok: false,
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    error: Some(format!("baseline execution error: {eb}")),
                });
            }
            (_, Err(ec)) => {
                all_ok = false;
                details.push(CaseDiffDetail {
                    case_index: ci,
                    ok: false,
                    max_abs_diff: f64::INFINITY,
                    max_rel_diff: f64::INFINITY,
                    error: Some(format!("candidate execution error: {ec}")),
                });
            }
        }
    }

    VerifyReport {
        ok: all_ok,
        equivalent: all_ok,
        verdict: if all_ok { "ACCEPT" } else { "REJECT" },
        cases: cases.len(),
        max_abs_diff: global_abs,
        max_rel_diff: global_rel,
        abs_tol,
        rel_tol,
        case_details: details,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ZoomLevelRecord {
    pub depth: usize,
    pub radius: f64,
    pub max_abs_error: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ZoomFuzzReport {
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verdict: Option<&'static str>,
    pub seed: u64,
    pub zoom_depth: usize,
    pub samples_per_level: usize,
    pub initial_max_abs_error: f64,
    pub zoomed_max_abs_error: f64,
    pub zoomed_max_rel_error: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub abs_tol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_tol: Option<f64>,
    pub amplification_ratio: Option<f64>,
    pub worst_case_input: BTreeMap<String, Vec<Value>>,
    pub zoom_history: Vec<ZoomLevelRecord>,
}

/// Recursive Adversarial Interval Narrowing (Zoom Fuzzing):
/// Sweeps initial fuzz cases at depth 0, then iteratively narrows a box (`radius *= 0.5`)
/// around the input case that maximizes `||Base(x) - Cand(x)||_inf`.
///
/// Exploratory mode (no tolerance gate). Use [`zoom_fuzz_specs_with_tol`] when enforcing
/// fail-closed `abs_tol` / `rel_tol` acceptance thresholds.
pub fn zoom_fuzz_specs(
    base: &SpecV1,
    cand: &SpecV1,
    cfg: &FuzzConfig,
    zoom_depth: usize,
) -> Result<ZoomFuzzReport, String> {
    zoom_fuzz_specs_with_tol(base, cand, cfg, zoom_depth, None, None)
}

/// Adversarial Zoom Fuzzing with optional fail-closed `abs_tol` / `rel_tol` acceptance gating.
///
/// When `abs_tol` and/or `rel_tol` is `Some(_)`, any evaluated sample violating the specified
/// tolerance produces `ok: false` and `verdict: Some("REJECT")`.
pub fn zoom_fuzz_specs_with_tol(
    base: &SpecV1,
    cand: &SpecV1,
    cfg: &FuzzConfig,
    zoom_depth: usize,
    abs_tol: Option<f64>,
    rel_tol: Option<f64>,
) -> Result<ZoomFuzzReport, String> {
    if cfg.count == 0 {
        return Err("zoom_fuzz: fuzz count must be >= 1".into());
    }
    if !(cfg.lo.is_finite() && cfg.hi.is_finite()) {
        return Err(format!(
            "zoom_fuzz: lo ({}) and hi ({}) must be finite",
            cfg.lo, cfg.hi
        ));
    }
    if cfg.lo > cfg.hi {
        return Err(format!(
            "zoom_fuzz: invalid fuzz range lo ({}) > hi ({})",
            cfg.lo, cfg.hi
        ));
    }
    if let Some(at) = abs_tol
        && !(at.is_finite() && at >= 0.0)
    {
        return Err(format!(
            "zoom_fuzz: abs_tol ({at}) must be finite and >= 0"
        ));
    }
    if let Some(rt) = rel_tol
        && !(rt.is_finite() && rt >= 0.0)
    {
        return Err(format!(
            "zoom_fuzz: rel_tol ({rt}) must be finite and >= 0"
        ));
    }
    let g_base = crate::engine::PreparedGraph::compile(base)?;
    let g_cand = crate::engine::PreparedGraph::compile(cand)?;
    let i64_mask = crate::engine::port_is_i64_mask(base);
    let samples_per_level = cfg.count.max(4);
    let init_cfg = FuzzConfig {
        count: samples_per_level,
        seed: cfg.seed,
        lo: cfg.lo,
        hi: cfg.hi,
    };
    let initial = generate_fuzz_cases(base, &init_cfg);
    if initial.is_empty() {
        return Err("zoom_fuzz: failed to generate initial cases".into());
    }

    let eval_sample = |inputs: &BTreeMap<String, Vec<Value>>| -> Result<(f64, f64, bool), String> {
        let ob = g_base.run_once(base, Some(inputs))?;
        let oc = g_cand.run_once(cand, Some(inputs))?;
        if ob.shape != oc.shape || ob.values.len() != oc.values.len() {
            return Ok((
                f64::INFINITY,
                f64::INFINITY,
                abs_tol.is_some() || rel_tol.is_some(),
            ));
        }
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut violated = false;
        for (&a, &b) in ob.values.iter().zip(oc.values.iter()) {
            let ad = ((a as f64) - (b as f64)).abs();
            let denom = (a as f64).abs().max((b as f64).abs()).max(1e-12);
            let rd = ad / denom;
            if !ad.is_finite() || !rd.is_finite() {
                max_abs = f64::INFINITY;
                max_rel = f64::INFINITY;
                if abs_tol.is_some() || rel_tol.is_some() {
                    violated = true;
                }
                continue;
            }
            if ad > max_abs {
                max_abs = ad;
            }
            if rd > max_rel {
                max_rel = rd;
            }
            let elem_ok = match (abs_tol, rel_tol) {
                (Some(at), Some(rt)) => ad <= at || rd <= rt,
                (Some(at), None) => ad <= at,
                (None, Some(rt)) => rd <= rt,
                (None, None) => true,
            };
            if !elem_ok {
                violated = true;
            }
        }
        Ok((max_abs, max_rel, violated))
    };

    let rel_only = abs_tol.is_none() && rel_tol.is_some();
    let mut best_case = initial[0].inputs.clone();
    let mut best_score = -1.0f64;
    let mut global_max_abs = 0.0f64;
    let mut global_max_rel = 0.0f64;
    let mut any_violated = false;

    for c in &initial {
        let (ad, rd, violated) = eval_sample(&c.inputs)?;
        if ad > global_max_abs {
            global_max_abs = ad;
        }
        if rd > global_max_rel {
            global_max_rel = rd;
        }
        if violated {
            any_violated = true;
        }
        let score = if rel_only { rd } else { ad };
        if score > best_score {
            best_score = score;
            best_case = c.inputs.clone();
        }
    }

    let initial_max = global_max_abs;
    let mut history = vec![ZoomLevelRecord {
        depth: 0,
        radius: (cfg.hi - cfg.lo) / 2.0,
        max_abs_error: initial_max,
    }];

    let mut rng = XorShift64(cfg.seed ^ 0xA5A5_5A5A_F0F0_0F0F);
    let mut radius = (cfg.hi - cfg.lo) / 4.0;

    for depth in 1..=zoom_depth {
        for _ in 0..samples_per_level {
            let mut trial = BTreeMap::new();
            for port in &base.ports {
                let k = port.slot.to_string();
                let Some(center) = best_case.get(&k) else {
                    continue;
                };
                if port.slot < i64_mask.len() && i64_mask[port.slot] {
                    trial.insert(k, center.clone());
                } else {
                    let mut next_vals = Vec::with_capacity(center.len());
                    for val in center {
                        let c_f = val.as_f64().unwrap_or(0.0);
                        let jitter = (rng.next_f64() * 2.0 - 1.0) * radius;
                        let clamped = (c_f + jitter).clamp(cfg.lo, cfg.hi);
                        let rounded = (clamped * 1e6).round() / 1e6;
                        next_vals.push(Value::from(rounded));
                    }
                    trial.insert(k, next_vals);
                }
            }
            if let Ok((ad, rd, violated)) = eval_sample(&trial) {
                if ad > global_max_abs {
                    global_max_abs = ad;
                }
                if rd > global_max_rel {
                    global_max_rel = rd;
                }
                if violated {
                    any_violated = true;
                }
                let score = if rel_only { rd } else { ad };
                if score > best_score {
                    best_score = score;
                    best_case = trial;
                }
            }
        }
        history.push(ZoomLevelRecord {
            depth,
            radius,
            max_abs_error: global_max_abs,
        });
        radius *= 0.5;
    }

    let zoomed_max = global_max_abs;
    let amp = if initial_max > 0.0 && initial_max.is_finite() && zoomed_max.is_finite() {
        Some(((zoomed_max / initial_max) * 1e4).round() / 1e4)
    } else {
        None
    };

    let gated = abs_tol.is_some() || rel_tol.is_some();
    let passed = if gated {
        !any_violated && zoomed_max.is_finite() && global_max_rel.is_finite()
    } else {
        true
    };
    let verdict = if gated {
        Some(if passed { "ACCEPT" } else { "REJECT" })
    } else {
        None
    };

    Ok(ZoomFuzzReport {
        ok: passed,
        verdict,
        seed: cfg.seed,
        zoom_depth,
        samples_per_level,
        initial_max_abs_error: initial_max,
        zoomed_max_abs_error: zoomed_max,
        zoomed_max_rel_error: global_max_rel,
        abs_tol,
        rel_tol,
        amplification_ratio: amp,
        worst_case_input: best_case,
        zoom_history: history,
    })
}

