# src/evidence/geom3/ — 3D Geometry Domain Pack

## Purpose

Provides deterministic 3D affine transform builders, hierarchical scene-tree compilation, multi-joint 3D inverse kinematics (`solve_ik`), and Burn `Device::flex()`-backed 3D-to-2D single and batched projection (`math-draw geom`).

## Ownership

- Pure `f64` 4×4/3×3 math, Euler/quaternion/Rodrigues/repeat transform builders, `normal_mat3`, `look_at4`, `perspective4`, and Sutherland–Hodgman `clip_triangle_near`: [`math.rs`](math.rs)
- Spec builders (`build_chain_spec`, `compile_scene_tree`), Polyak-stepped IK solver (`solve_ik`), single & block-diagonal batched (`BATCH_MAX = 63`) projection (`project_model`), and 17-check selftest (`run_geom_selftest`): [`mod.rs`](mod.rs)

## Local Contracts

- **Canonical Conventions**: Angles are in degrees; coordinates are right-handed with active column-vector transforms (`v' = M v`); internal matrices are row-major `f64`; Burn `linear(4, 4, false)` weights are emitted column-major `f32`.
- **Documented Perspective-Divide Boundary**: Linear MVP transforms execute on Burn `Device::flex()` returning homogeneous clip coordinates `[x, y, z, w]`. Perspective divide (`x/w, y/w, z/w`) and viewport mapping stay host-side, flagging any finite vertex with `w <= 0` as `behind: true` and failing closed (`Err`) if clip, NDC, or screen coordinates overflow to non-finite values (`NaN`/`Inf`).
- **Fail-Closed Validation**: Reject non-finite inputs or intermediate/final matrix/dehomogenize/`f32`-weight overflows (`mat4_mul`/`compose_chain_checked`, `repeat`, `normal_mat3`, `perspective4`, `dehomogenize`), zero rotation axes, zero quaternions, singular or near-singular normal matrices, invalid `near`/`far`/`fov_deg`/`aspect`, `repeat` depth `> 1024`, scene-tree cycles/duplicate IDs/unknown parents/depth `> 512`, out-of-range triangle indices, invalid joint limits (`min_deg > max_deg`), and IK engine verification divergence (`solve_ik` requires both `converged` and finite `engine_max_abs_diff` within tolerance).
- **Batching Limit**: Block-diagonal projection packs at most `BATCH_MAX = 63` vertices per `linear(4N, 4N, false)` chunk (`(4 * 63)^2 = 63,504 <= 65,535` weights).

## Work Guidance

- Keep `math.rs` completely independent of Burn; only `mod.rs` invokes `PreparedGraph` for engine-backed verification and projection.
- Preserve bit-exact / `<= 1e-6` agreement between single and batched projection modes.

## Verification

- `cargo test --locked --test integration geom3_builders_scene_tree_ik_projection_batch_and_cli`
- `./target/release/math-draw geom selftest` (17/17 checks)
- `./target/release/math-draw smoke` (check #33)
