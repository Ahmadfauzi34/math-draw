pub mod catalog;
pub mod cli;
pub mod error;
pub mod es;
pub mod evidence;
pub mod graph;
pub mod harness;
pub mod layers;
pub mod runtime;
pub mod smoke;
pub mod spec;

// Backward-compatible root module facades (matching ai_agent_builder style)
pub use es as recurse;
pub use evidence::{
    bundle, diff_ref, geom3, gradcheck, linear_equiv, quant, soak, verify, weight_lab,
};
pub use graph::{composer, grad_spec, liveness, mutate, rewrite, shard, validate};
pub use runtime::{cost_model, direct_math, engine, profile, rollout, trace};
