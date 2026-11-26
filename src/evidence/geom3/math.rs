use serde_json::Value;

pub type Mat4 = [f64; 16];
pub type Mat3 = [f64; 9];
pub type Vec3 = [f64; 3];
pub type Vec4 = [f64; 4];

pub fn deg2rad(d: f64) -> f64 {
    d * std::f64::consts::PI / 180.0
}

pub fn check_finite(v: f64, ctx: &str) -> Result<f64, String> {
    if !v.is_finite() {
        return Err(format!("geom3: non-finite value in {ctx}"));
    }
    Ok(v)
}

pub fn check_vec3_finite(v: Vec3, ctx: &str) -> Result<Vec3, String> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(format!("geom3: non-finite 3-vector in {ctx}"));
    }
    Ok(v)
}

pub fn mat4_identity() -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// Row-major 4x4 matrix product `C = A * B`. Applies `B` first, then `A`.
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [0.0f64; 16];
    for r in 0..4 {
        for col in 0..4 {
            let mut s = 0.0f64;
            for k in 0..4 {
                s += a[r * 4 + k] * b[k * 4 + col];
            }
            c[r * 4 + col] = s;
        }
    }
    c
}

fn rot_x_rad(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn rot_y_rad(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

fn rot_z_rad(a: f64) -> Mat4 {
    let (s, c) = a.sin_cos();
    [
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

pub fn rot_x4(deg: f64) -> Result<Mat4, String> {
    check_finite(deg, "rot_x4(degrees)")?;
    Ok(rot_x_rad(deg2rad(deg)))
}

pub fn rot_y4(deg: f64) -> Result<Mat4, String> {
    check_finite(deg, "rot_y4(degrees)")?;
    Ok(rot_y_rad(deg2rad(deg)))
}

pub fn rot_z4(deg: f64) -> Result<Mat4, String> {
    check_finite(deg, "rot_z4(degrees)")?;
    Ok(rot_z_rad(deg2rad(deg)))
}

pub fn rot_axis_angle4(ax: f64, ay: f64, az: f64, deg: f64) -> Result<Mat4, String> {
    check_vec3_finite([ax, ay, az], "rot_axis_angle4(axis)")?;
    check_finite(deg, "rot_axis_angle4(degrees)")?;
    let n = (ax * ax + ay * ay + az * az).sqrt();
    if !n.is_finite() || n <= 0.0 {
        return Err("geom3: rot_axis_angle4 axis must be non-zero".into());
    }
    let (x, y, z) = (ax / n, ay / n, az / n);
    let (s, c) = deg2rad(deg).sin_cos();
    let t = 1.0 - c;
    Ok([
        t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0,
        t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0,
        t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

fn parse_euler_order(order: &str) -> Result<[char; 3], String> {
    let chars: Vec<char> = order.to_ascii_uppercase().chars().collect();
    if chars.len() != 3 {
        return Err(format!(
            "geom3: euler order must be a permutation of XYZ, got '{order}'"
        ));
    }
    let mut seen = [false; 3];
    for &ch in &chars {
        let idx = match ch {
            'X' => 0,
            'Y' => 1,
            'Z' => 2,
            _ => {
                return Err(format!(
                    "geom3: euler order must be a permutation of XYZ, got '{order}'"
                ));
            }
        };
        if seen[idx] {
            return Err(format!(
                "geom3: euler order must be a permutation of XYZ, got '{order}'"
            ));
        }
        seen[idx] = true;
    }
    Ok([chars[0], chars[1], chars[2]])
}

/// Euler angles in degrees -> row-major 4x4. Default order `"ZYX"` yields `Rz(yaw) * Ry(pitch) * Rx(roll)`.
pub fn euler4(roll: f64, pitch: f64, yaw: f64, order: &str) -> Result<Mat4, String> {
    check_vec3_finite([roll, pitch, yaw], "euler4(roll,pitch,yaw)")?;
    let seq = parse_euler_order(order)?;
    let rx = rot_x_rad(deg2rad(roll));
    let ry = rot_y_rad(deg2rad(pitch));
    let rz = rot_z_rad(deg2rad(yaw));
    let pick = |ch: char| match ch {
        'X' => &rx,
        'Y' => &ry,
        _ => &rz,
    };
    let m12 = mat4_mul(pick(seq[1]), pick(seq[2]));
    Ok(mat4_mul(pick(seq[0]), &m12))
}

fn quat_mul(a: Vec4, b: Vec4) -> Vec4 {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn axis_quat(x: f64, y: f64, z: f64, deg: f64) -> Vec4 {
    let (s, c) = (deg2rad(deg) * 0.5).sin_cos();
    [c, x * s, y * s, z * s]
}

pub fn quat_from_euler(roll: f64, pitch: f64, yaw: f64, order: &str) -> Result<Vec4, String> {
    check_vec3_finite([roll, pitch, yaw], "quat_from_euler")?;
    let seq = parse_euler_order(order)?;
    let qx = axis_quat(1.0, 0.0, 0.0, roll);
    let qy = axis_quat(0.0, 1.0, 0.0, pitch);
    let qz = axis_quat(0.0, 0.0, 1.0, yaw);
    let pick = |ch: char| match ch {
        'X' => qx,
        'Y' => qy,
        _ => qz,
    };
    Ok(quat_mul(pick(seq[0]), quat_mul(pick(seq[1]), pick(seq[2]))))
}

pub fn quat_to_mat4(w: f64, x: f64, y: f64, z: f64) -> Result<Mat4, String> {
    if ![w, x, y, z].iter().all(|v| v.is_finite()) {
        return Err("geom3: quat_to_mat4 received non-finite component".into());
    }
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if !n.is_finite() || n <= 0.0 {
        return Err("geom3: quat_to_mat4 zero quaternion".into());
    }
    let (w, x, y, z) = (w / n, x / n, y / n, z / n);
    let (x2, y2, z2) = (x + x, y + y, z + z);
    let (xx, xy, xz) = (x * x2, x * y2, x * z2);
    let (yy, yz, zz) = (y * y2, y * z2, z * z2);
    let (wx, wy, wz) = (w * x2, w * y2, w * z2);
    Ok([
        1.0 - (yy + zz), xy - wz, xz + wy, 0.0,
        xy + wz, 1.0 - (xx + zz), yz - wx, 0.0,
        xz - wy, yz + wx, 1.0 - (xx + yy), 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn translate4(tx: f64, ty: f64, tz: f64) -> Result<Mat4, String> {
    check_vec3_finite([tx, ty, tz], "translate4")?;
    Ok([
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0,
    ])
}

pub fn scale4(sx: f64, sy: f64, sz: f64) -> Result<Mat4, String> {
    check_vec3_finite([sx, sy, sz], "scale4")?;
    Ok([
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// Convert row-major 4x4 to column-major `f32` weights for Burn `linear(4, 4, false)`.
pub fn mat4_to_col_major_f32(m: &Mat4) -> Vec<f32> {
    let mut w = vec![0.0f32; 16];
    for r in 0..4 {
        for c in 0..4 {
            w[c * 4 + r] = m[r * 4 + c] as f32;
        }
    }
    w
}

pub fn homogenize(v: Vec3) -> Vec4 {
    [v[0], v[1], v[2], 1.0]
}

pub fn dehomogenize(v: Vec4) -> Result<Vec3, String> {
    if !v[3].is_finite() || v[3] == 0.0 {
        return Err("geom3: dehomogenize division by zero or non-finite w".into());
    }
    Ok([v[0] / v[3], v[1] / v[3], v[2] / v[3]])
}

pub fn apply_mat4_clip(m: &Mat4, v: Vec3) -> Vec4 {
    let [x, y, z, w] = homogenize(v);
    let mut o = [0.0f64; 4];
    for r in 0..4 {
        o[r] = m[r * 4] * x + m[r * 4 + 1] * y + m[r * 4 + 2] * z + m[r * 4 + 3] * w;
    }
    o
}

pub fn apply_mat4(m: &Mat4, v: Vec3) -> Result<Vec3, String> {
    dehomogenize(apply_mat4_clip(m, v))
}

/// Compose a chain of row-major 4x4 matrices (first op applied first: `acc = m * acc`).
pub fn compose_chain(mats: &[Mat4]) -> Mat4 {
    let mut acc = mat4_identity();
    for m in mats {
        acc = mat4_mul(m, &acc);
    }
    acc
}

fn mat3_inverse(m: &Mat3) -> Result<Mat3, String> {
    let [a, b, c, d, e, f, g, h, i] = *m;
    let ca = e * i - f * h;
    let cb = f * g - d * i;
    let cc = d * h - e * g;
    let det = a * ca + b * cb + c * cc;
    if !det.is_finite() || det.abs() <= 1e-18 {
        return Err("geom3: normal_mat3 singular upper-3x3 matrix".into());
    }
    Ok([
        ca / det, (c * h - b * i) / det, (b * f - c * e) / det,
        cb / det, (a * i - c * g) / det, (c * d - a * f) / det,
        cc / det, (b * g - a * h) / det, (a * e - b * d) / det,
    ])
}

/// Inverse-transpose of the upper 3x3 of a row-major 4x4 model matrix.
pub fn normal_mat3(model4: &Mat4) -> Result<Mat3, String> {
    let m3 = [
        model4[0], model4[1], model4[2],
        model4[4], model4[5], model4[6],
        model4[8], model4[9], model4[10],
    ];
    let inv = mat3_inverse(&m3)?;
    Ok([
        inv[0], inv[3], inv[6],
        inv[1], inv[4], inv[7],
        inv[2], inv[5], inv[8],
    ])
}

pub fn apply_mat3(m3: &Mat3, v: Vec3) -> Vec3 {
    [
        m3[0] * v[0] + m3[1] * v[1] + m3[2] * v[2],
        m3[3] * v[0] + m3[4] * v[1] + m3[5] * v[2],
        m3[6] * v[0] + m3[7] * v[1] + m3[8] * v[2],
    ]
}

fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_norm(v: Vec3, ctx: &str) -> Result<Vec3, String> {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if !n.is_finite() || n <= 0.0 {
        return Err(format!("geom3: {ctx} produced zero-length vector"));
    }
    Ok([v[0] / n, v[1] / n, v[2] / n])
}

/// Camera view matrix (world -> camera), row-major, right-handed active convention.
pub fn look_at4(eye: Vec3, target: Vec3, up: Vec3) -> Result<Mat4, String> {
    check_vec3_finite(eye, "look_at4(eye)")?;
    check_vec3_finite(target, "look_at4(target)")?;
    check_vec3_finite(up, "look_at4(up)")?;
    let z = vec3_norm(vec3_sub(eye, target), "look_at4(eye - target)")?;
    let x = vec3_norm(vec3_cross(up, z), "look_at4(up x z)")?;
    let y = vec3_cross(z, x);
    Ok([
        x[0], x[1], x[2], -vec3_dot(x, eye),
        y[0], y[1], y[2], -vec3_dot(y, eye),
        z[0], z[1], z[2], -vec3_dot(z, eye),
        0.0, 0.0, 0.0, 1.0,
    ])
}

/// OpenGL-style perspective projection matrix (row-major, active).
pub fn perspective4(fov_deg: f64, aspect: f64, near: f64, far: f64) -> Result<Mat4, String> {
    check_finite(fov_deg, "perspective4(fov_deg)")?;
    check_finite(aspect, "perspective4(aspect)")?;
    check_finite(near, "perspective4(near)")?;
    check_finite(far, "perspective4(far)")?;
    if fov_deg <= 0.0 || fov_deg >= 180.0 {
        return Err(format!(
            "geom3: perspective4 fov_deg must be in (0, 180), got {fov_deg}"
        ));
    }
    if aspect <= 0.0 {
        return Err(format!(
            "geom3: perspective4 aspect must be > 0, got {aspect}"
        ));
    }
    if near <= 0.0 || far <= near {
        return Err(format!(
            "geom3: perspective4 requires 0 < near < far, got near={near}, far={far}"
        ));
    }
    let f = 1.0 / (deg2rad(fov_deg) * 0.5).tan();
    Ok([
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), (2.0 * far * near) / (near - far),
        0.0, 0.0, -1.0, 0.0,
    ])
}

#[derive(Debug, Clone)]
pub struct ClipNearResult {
    pub triangles: Vec<[Vec4; 3]>,
    pub spans_camera: bool,
}

/// Sutherland–Hodgman near-plane triangle clipping in homogeneous clip space (`z + w >= 0`).
pub fn clip_triangle_near(c0: Vec4, c1: Vec4, c2: Vec4) -> ClipNearResult {
    let verts = [c0, c1, c2];
    if verts
        .iter()
        .any(|v| !v.iter().all(|x| x.is_finite()) || v[3] <= 0.0)
    {
        return ClipNearResult {
            triangles: Vec::new(),
            spans_camera: true,
        };
    }
    let inside = |p: Vec4| p[2] + p[3] >= 0.0;
    let intersect = |a: Vec4, b: Vec4| -> Vec4 {
        let denom = (b[2] - a[2]) + (b[3] - a[3]);
        let t = -(a[3] + a[2]) / denom;
        [
            a[0] + t * (b[0] - a[0]),
            a[1] + t * (b[1] - a[1]),
            a[2] + t * (b[2] - a[2]),
            a[3] + t * (b[3] - a[3]),
        ]
    };

    let mut poly = Vec::with_capacity(4);
    for i in 0..3 {
        let curr = verts[i];
        let next = verts[(i + 1) % 3];
        let c_in = inside(curr);
        let n_in = inside(next);
        if c_in && n_in {
            poly.push(next);
        } else if c_in && !n_in {
            poly.push(intersect(curr, next));
        } else if !c_in && n_in {
            poly.push(intersect(curr, next));
            poly.push(next);
        }
    }

    let mut triangles = Vec::new();
    let mut i = 1;
    while i + 1 < poly.len() {
        triangles.push([poly[0], poly[i], poly[i + 1]]);
        i += 1;
    }
    ClipNearResult {
        triangles,
        spans_camera: false,
    }
}

pub fn parse_vec3_value(val: &Value, ctx: &str) -> Result<Vec3, String> {
    let arr = val
        .as_array()
        .ok_or_else(|| format!("geom3: {ctx} must be [x, y, z] array"))?;
    if arr.len() != 3 {
        return Err(format!("geom3: {ctx} must have length 3, got {}", arr.len()));
    }
    let mut out = [0.0f64; 3];
    for (i, v) in arr.iter().enumerate() {
        let f = v
            .as_f64()
            .ok_or_else(|| format!("geom3: {ctx}[{i}] must be a finite number"))?;
        if !f.is_finite() {
            return Err(format!("geom3: {ctx}[{i}] is non-finite"));
        }
        out[i] = f;
    }
    Ok(out)
}

fn opt_f64_field(obj: &serde_json::Map<String, Value>, key: &str, default: f64) -> Result<f64, String> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let f = v
                .as_f64()
                .ok_or_else(|| format!("geom3: field '{key}' must be a number"))?;
            check_finite(f, key)
        }
    }
}

/// Convert a JSON transform op object into a row-major 4x4 matrix.
pub fn op_to_mat4(val: &Value) -> Result<Mat4, String> {
    op_to_mat4_inner(val, 0)
}

fn op_to_mat4_inner(val: &Value, nest: usize) -> Result<Mat4, String> {
    if nest > 32 {
        return Err("geom3: repeat nesting depth exceeded maximum (32)".into());
    }
    let obj = val
        .as_object()
        .ok_or_else(|| "geom3: transform op must be a JSON object".to_string())?;
    let op = obj
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "geom3: transform op missing string field 'op'".to_string())?;

    match op {
        "rotate_x" => rot_x4(opt_f64_field(obj, "degrees", 0.0)?),
        "rotate_y" => rot_y4(opt_f64_field(obj, "degrees", 0.0)?),
        "rotate_z" => rot_z4(opt_f64_field(obj, "degrees", 0.0)?),
        "rotate_axis" => {
            let axis_val = obj
                .get("axis")
                .ok_or_else(|| "geom3: rotate_axis missing 'axis'".to_string())?;
            let [ax, ay, az] = parse_vec3_value(axis_val, "rotate_axis.axis")?;
            let deg = opt_f64_field(obj, "degrees", 0.0)?;
            rot_axis_angle4(ax, ay, az, deg)
        }
        "euler" => {
            let roll = opt_f64_field(obj, "roll", 0.0)?;
            let pitch = opt_f64_field(obj, "pitch", 0.0)?;
            let yaw = opt_f64_field(obj, "yaw", 0.0)?;
            let order = obj.get("order").and_then(|v| v.as_str()).unwrap_or("ZYX");
            euler4(roll, pitch, yaw, order)
        }
        "quaternion" => {
            let w = opt_f64_field(obj, "w", 1.0)?;
            let x = opt_f64_field(obj, "x", 0.0)?;
            let y = opt_f64_field(obj, "y", 0.0)?;
            let z = opt_f64_field(obj, "z", 0.0)?;
            quat_to_mat4(w, x, y, z)
        }
        "translate" => {
            let by_val = obj
                .get("by")
                .ok_or_else(|| "geom3: translate missing 'by'".to_string())?;
            let [tx, ty, tz] = parse_vec3_value(by_val, "translate.by")?;
            translate4(tx, ty, tz)
        }
        "scale" => {
            let by_val = obj
                .get("by")
                .ok_or_else(|| "geom3: scale missing 'by'".to_string())?;
            let [sx, sy, sz] = parse_vec3_value(by_val, "scale.by")?;
            scale4(sx, sy, sz)
        }
        "matrix" => {
            let arr = obj
                .get("m")
                .and_then(|v| v.as_array())
                .ok_or_else(|| "geom3: 'matrix' op needs 'm': 16 numbers (row-major)".to_string())?;
            if arr.len() != 16 {
                return Err(format!(
                    "geom3: 'matrix' op needs 16 numbers, got {}",
                    arr.len()
                ));
            }
            let mut m = [0.0f64; 16];
            for (i, v) in arr.iter().enumerate() {
                let f = v
                    .as_f64()
                    .ok_or_else(|| format!("geom3: matrix.m[{i}] must be a number"))?;
                m[i] = check_finite(f, "matrix.m")?;
            }
            Ok(m)
        }
        "repeat" => {
            let depth_val = obj
                .get("depth")
                .or_else(|| obj.get("steps"))
                .or_else(|| obj.get("count"));
            let depth = match depth_val {
                None => 1usize,
                Some(v) => {
                    let u = v.as_u64().ok_or_else(|| {
                        format!("geom3: 'repeat' op needs integer depth in 0..=1024, got {v}")
                    })?;
                    if u > 1024 {
                        return Err(format!(
                            "geom3: 'repeat' op depth {u} exceeds maximum 1024"
                        ));
                    }
                    u as usize
                }
            };
            let sub_mats: Vec<Mat4> = if let Some(chain_val) = obj.get("chain") {
                let arr = chain_val
                    .as_array()
                    .ok_or_else(|| "geom3: repeat.chain must be an array".to_string())?;
                arr.iter()
                    .map(|item| op_to_mat4_inner(item, nest + 1))
                    .collect::<Result<Vec<_>, _>>()?
            } else if let Some(inner) = obj.get("inner") {
                vec![op_to_mat4_inner(inner, nest + 1)?]
            } else {
                Vec::new()
            };
            let step_m = compose_chain(&sub_mats);
            let mut acc = mat4_identity();
            for _ in 0..depth {
                acc = mat4_mul(&step_m, &acc);
            }
            Ok(acc)
        }
        other => Err(format!("geom3: unknown geom op '{other}'")),
    }
}
