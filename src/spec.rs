use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use crate::composer::{self, ExpansionReceipt};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerDecl {
    pub constructor: String,
    #[serde(default)]
    pub args: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StepDecl {
    pub kind: String,
    pub layer: usize,
    pub slots: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortDecl {
    pub slot: usize,
    #[serde(default = "default_role")]
    pub role: String,
    pub shape: [usize; 4],
    #[serde(default = "default_layout")]
    pub layout: String,
}

fn default_role() -> String {
    "observation".to_string()
}

fn default_layout() -> String {
    "feature_axis1_singleton".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SpecV1 {
    pub num_slots: usize,
    #[serde(default)]
    pub layers: Vec<LayerDecl>,
    #[serde(default)]
    pub steps: Vec<StepDecl>,
    pub output_slot: usize,
    #[serde(default)]
    pub ports: Vec<PortDecl>,
    #[serde(default)]
    pub weights: BTreeMap<String, Vec<f32>>,
    #[serde(default)]
    pub inputs: BTreeMap<String, Vec<serde_json::Value>>,
}

impl SpecV1 {
    pub fn port_by_slot(&self, slot: usize) -> Option<&PortDecl> {
        self.ports.iter().find(|p| p.slot == slot)
    }

    pub fn parse_f32_input(&self, slot: usize) -> Result<Vec<f32>, String> {
        let key = slot.to_string();
        let raw = self
            .inputs
            .get(&key)
            .ok_or_else(|| format!("missing inputs for port slot {slot}"))?;
        parse_f32_values(raw, &format!("inputs[{slot}]"))
    }

    pub fn parse_i64_input(&self, slot: usize) -> Result<Vec<i64>, String> {
        let key = slot.to_string();
        let raw = self
            .inputs
            .get(&key)
            .ok_or_else(|| format!("missing inputs for port slot {slot}"))?;
        parse_i64_values(raw, &format!("inputs[{slot}]"))
    }
}

pub fn parse_f32_values(raw: &[serde_json::Value], context: &str) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (idx, val) in raw.iter().enumerate() {
        let num = val
            .as_f64()
            .ok_or_else(|| format!("{context}[{idx}]: expected finite number, got {val}"))?;
        if !num.is_finite() {
            return Err(format!("{context}[{idx}]: non-finite number {val}"));
        }
        let f = num as f32;
        if !f.is_finite() {
            return Err(format!(
                "{context}[{idx}]: overflows f32 to non-finite ({val})"
            ));
        }
        out.push(f);
    }
    Ok(out)
}

pub fn parse_i64_values(raw: &[serde_json::Value], context: &str) -> Result<Vec<i64>, String> {
    let mut out = Vec::with_capacity(raw.len());
    for (idx, val) in raw.iter().enumerate() {
        if let Some(i) = val.as_i64() {
            out.push(i);
        } else if let Some(f) = val.as_f64() {
            if !f.is_finite() || f.fract() != 0.0 || f < (i64::MIN as f64) || f > (i64::MAX as f64)
            {
                return Err(format!(
                    "{context}[{idx}]: expected exact i64 integer, got {val}"
                ));
            }
            out.push(f as i64);
        } else {
            return Err(format!("{context}[{idx}]: expected integer, got {val}"));
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct LoadedSpec {
    pub spec: SpecV1,
    pub was_v2: bool,
    pub expansion: Option<ExpansionReceipt>,
}

pub fn load_spec_file(path: &Path) -> Result<LoadedSpec, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read spec '{}': {e}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    load_spec_str(&content, base_dir)
}

pub fn load_spec_str(json_str: &str, base_dir: &Path) -> Result<LoadedSpec, String> {
    let raw: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| format!("invalid JSON: {e}"))?;
    load_spec_value(&raw, base_dir)
}

pub fn load_spec_value(raw: &serde_json::Value, base_dir: &Path) -> Result<LoadedSpec, String> {
    if composer::is_recursive_spec_v2(raw) {
        let (compiled, receipt) = composer::compile_recursive_spec(raw, base_dir)?;
        Ok(LoadedSpec {
            spec: compiled,
            was_v2: true,
            expansion: Some(receipt),
        })
    } else {
        let spec: SpecV1 = serde_json::from_value(raw.clone())
            .map_err(|e| format!("invalid v1 spec structure: {e}"))?;
        Ok(LoadedSpec {
            spec,
            was_v2: false,
            expansion: None,
        })
    }
}

pub fn sha256_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_default();
    let digest = Sha256::digest(&bytes);
    format!("sha256:{digest:x}")
}

pub fn sha256_f32_slice(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        let canonical = if *v == 0.0 { 0.0f32 } else { *v };
        hasher.update(canonical.to_le_bytes());
    }
    format!("sha256:{:x}", hasher.finalize())
}

pub fn is_canonical_uint_key(s: &str) -> bool {
    if s == "0" {
        return true;
    }
    let bytes = s.as_bytes();
    !bytes.is_empty() && bytes[0] >= b'1' && bytes[0] <= b'9' && bytes.iter().all(|b| b.is_ascii_digit())
}
