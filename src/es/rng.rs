use serde::{Deserialize, Serialize};
use std::f32::consts::PI as PI_F32;
use std::f64::consts::PI as PI_F64;

/// Legacy 64-bit xorshift PRNG (default for backward-compatible `fit_es` streams).
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        if x == 0 {
            x = 0x517cc1b727220a95;
        }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    pub fn next_normal_f64(&mut self) -> f64 {
        let u1 =
            ((self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)).clamp(1e-12, 1.0 - 1e-12);
        let u2 = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI_F64 * u2).cos()
    }
}

/// Deterministic Mulberry32 PRNG (bit-identical to `ai_agent_builder::es::rng::Rng`).
#[derive(Debug, Clone)]
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9 } else { seed },
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        t ^ (t >> 14)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    /// Standard normal `N(0, 1)` via Box-Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI_F32 * u2;
        r * theta.cos()
    }
}

/// PCG-XSH-RR 64/32 (O'Neill 2014).
///
/// Published pcg-c-basic test vector: `Pcg32::new(42, 54).next_u32() == 0xa15c02b7`.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULT: u64 = 6_364_136_223_846_793_005;

    pub fn new(seed: u64, seq: u64) -> Self {
        let mut r = Self {
            state: 0,
            inc: (seq << 1) | 1,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn from_seed_u32(seed: u32) -> Self {
        Self::new(seed as u64, 0x853c_49e6_748f_ea9b)
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    /// Standard normal `N(0, 1)` via Box-Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI_F32 * u2;
        r * theta.cos()
    }
}

/// Philox4x32-10 (Salmon-Moraes-Dror-Shaw 2011, Random123) counter-based PRNG.
///
/// Published 10-round vector: `key=[0, 0], counter=[0, 0, 0, 0]` ->
/// `[0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]`.
#[derive(Debug, Clone)]
pub struct Philox4x32 {
    key: [u32; 2],
    counter: [u32; 4],
    buf: [u32; 4],
    idx: usize,
}

impl Philox4x32 {
    const M0: u32 = 0xD251_1F53;
    const M1: u32 = 0xCD9E_8D57;
    const W0: u32 = 0x9E37_79B9;
    const W1: u32 = 0xBB67_AE85;

    fn round(c: &mut [u32; 4], k: &[u32; 2]) {
        let p0 = (c[0] as u64) * (Self::M0 as u64);
        let p1 = (c[2] as u64) * (Self::M1 as u64);
        let (hi0, lo0) = ((p0 >> 32) as u32, p0 as u32);
        let (hi1, lo1) = ((p1 >> 32) as u32, p1 as u32);
        c[0] = hi1 ^ c[1] ^ k[0];
        c[1] = lo1;
        c[2] = hi0 ^ c[3] ^ k[1];
        c[3] = lo0;
    }

    fn philox(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
        let mut c = counter;
        let mut k = key;
        for _ in 0..10 {
            Self::round(&mut c, &k);
            k[0] = k[0].wrapping_add(Self::W0);
            k[1] = k[1].wrapping_add(Self::W1);
        }
        c
    }

    fn bump_counter(counter: &mut [u32; 4]) {
        let mut carry: u64 = 1;
        for w in counter.iter_mut() {
            let s = (*w as u64) + carry;
            *w = s as u32;
            carry = s >> 32;
            if carry == 0 {
                break;
            }
        }
    }

    pub fn with_key(key: [u32; 2]) -> Self {
        Self::with_key_counter(key, [0; 4])
    }

    pub fn with_key_counter(key: [u32; 2], counter: [u32; 4]) -> Self {
        Self {
            key,
            counter,
            buf: [0; 4],
            idx: 4,
        }
    }

    pub fn new(seed: u32) -> Self {
        fn splitmix32(x: u32) -> u32 {
            let mut z = x.wrapping_add(0x9E37_79B9);
            z = (z ^ (z >> 16)).wrapping_mul(0x21F0_AAAD);
            z = (z ^ (z >> 15)).wrapping_mul(0x735A_2D97);
            z ^ (z >> 15)
        }
        Self::with_key([splitmix32(seed), splitmix32(seed ^ 0x9E37_79B9)])
    }

    /// Jump to 64-bit low counter stream (`[lo, hi, 0, 0]`).
    pub fn seek(&mut self, ctr_lo: u64) {
        self.counter = [ctr_lo as u32, (ctr_lo >> 32) as u32, 0, 0];
        self.idx = 4;
    }

    /// Jump to an explicit 128-bit counter.
    pub fn seek_counter(&mut self, counter: [u32; 4]) {
        self.counter = counter;
        self.idx = 4;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 4 {
            self.buf = Self::philox(self.counter, self.key);
            Self::bump_counter(&mut self.counter);
            self.idx = 0;
        }
        let v = self.buf[self.idx];
        self.idx += 1;
        v
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f32 {
        ((self.next_u32() >> 8) as f32) * (1.0 / 16_777_216.0)
    }

    /// Standard normal `N(0, 1)` via Box-Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0 * PI_F32 * u2;
        r * theta.cos()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RngKind {
    Xorshift64,
    Mulberry32,
    Pcg32,
    Philox4x32,
}

impl RngKind {
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "xorshift64" | "xorshift" => Ok(Self::Xorshift64),
            "mulberry32" | "mulberry" => Ok(Self::Mulberry32),
            "pcg32" | "pcg" => Ok(Self::Pcg32),
            "philox4x32" | "philox" => Ok(Self::Philox4x32),
            other => Err(format!(
                "unknown rng '{other}' (expected: xorshift64 | mulberry32 | pcg32 | philox4x32)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Xorshift64 => "xorshift64",
            Self::Mulberry32 => "mulberry32",
            Self::Pcg32 => "pcg32",
            Self::Philox4x32 => "philox4x32",
        }
    }
}

#[derive(Debug, Clone)]
pub enum AnyRng {
    Xorshift64(Xorshift64),
    Mulberry32(Mulberry32),
    Pcg32(Pcg32),
    Philox4x32(Philox4x32),
}

impl AnyRng {
    pub fn new(kind: RngKind, seed: u64) -> Self {
        let seed32 = seed as u32;
        match kind {
            RngKind::Xorshift64 => Self::Xorshift64(Xorshift64::new(seed)),
            RngKind::Mulberry32 => Self::Mulberry32(Mulberry32::new(seed32)),
            RngKind::Pcg32 => Self::Pcg32(Pcg32::from_seed_u32(seed32)),
            RngKind::Philox4x32 => Self::Philox4x32(Philox4x32::new(seed32)),
        }
    }

    pub fn kind(&self) -> RngKind {
        match self {
            Self::Xorshift64(_) => RngKind::Xorshift64,
            Self::Mulberry32(_) => RngKind::Mulberry32,
            Self::Pcg32(_) => RngKind::Pcg32,
            Self::Philox4x32(_) => RngKind::Philox4x32,
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        match self {
            Self::Xorshift64(r) => r.next_u32(),
            Self::Mulberry32(r) => r.next_u32(),
            Self::Pcg32(r) => r.next_u32(),
            Self::Philox4x32(r) => r.next_u32(),
        }
    }

    pub fn uniform(&mut self) -> f32 {
        match self {
            Self::Xorshift64(r) => r.uniform(),
            Self::Mulberry32(r) => r.uniform(),
            Self::Pcg32(r) => r.uniform(),
            Self::Philox4x32(r) => r.uniform(),
        }
    }

    pub fn gaussian_f32(&mut self) -> f32 {
        match self {
            Self::Xorshift64(r) => r.next_normal_f64() as f32,
            Self::Mulberry32(r) => r.gaussian(),
            Self::Pcg32(r) => r.gaussian(),
            Self::Philox4x32(r) => r.gaussian(),
        }
    }

    pub fn next_normal_f64(&mut self) -> f64 {
        match self {
            Self::Xorshift64(r) => r.next_normal_f64(),
            Self::Mulberry32(r) => r.gaussian() as f64,
            Self::Pcg32(r) => r.gaussian() as f64,
            Self::Philox4x32(r) => r.gaussian() as f64,
        }
    }

    /// Counter-stream seek (supported only for `Philox4x32`; fails closed otherwise).
    pub fn philox_seek(&mut self, ctr_lo: u64) -> Result<(), String> {
        match self {
            Self::Philox4x32(r) => {
                r.seek(ctr_lo);
                Ok(())
            }
            other => Err(format!(
                "philox_seek only valid for philox4x32, got {}",
                other.kind().name()
            )),
        }
    }
}
