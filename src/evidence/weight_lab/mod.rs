pub mod certify;
pub mod linalg;
pub mod selftest;

pub use certify::*;
pub use linalg::*;
pub use selftest::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WeightRequires {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spectral_radius_lt: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub orthonormal_tol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub det_positive: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeightProbe {
    #[serde(rename = "in", alias = "input")]
    pub input: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub want: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WeightSet {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default = "default_weight_kind")]
    pub kind: String,
    #[serde(rename = "W")]
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    #[serde(default)]
    pub probes: Vec<WeightProbe>,
    #[serde(default)]
    pub probe_tol: Option<f64>,
    #[serde(default)]
    pub requires: WeightRequires,
}

fn default_weight_kind() -> String {
    "map".to_string()
}

impl WeightSet {
    pub fn validate_dims(&self) -> Result<(usize, usize), String> {
        let out_dim = self.w.len();
        if out_dim == 0 {
            return Err("W must be a non-empty 2D matrix".into());
        }
        let in_dim = self.w[0].len();
        if in_dim == 0 || !self.w.iter().all(|r| r.len() == in_dim) {
            return Err("all rows of W must have equal non-zero length".into());
        }
        if self.b.len() != out_dim {
            return Err(format!(
                "b length ({}) must match W row count out_dim ({out_dim})",
                self.b.len()
            ));
        }
        Ok((out_dim, in_dim))
    }
}
