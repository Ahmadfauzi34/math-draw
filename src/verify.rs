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
