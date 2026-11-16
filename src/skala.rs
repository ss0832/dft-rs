// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Evaluation of the Skala deep-learned exchange-correlation functional.
//!
//! Skala (Microsoft Research, MIT licence) predicts an enhancement factor of
//! the LDA exchange energy density from meta-GGA features and from a
//! non-local message passing between an atom and the grid points of its
//! atomic grid. This module evaluates such a model and the derivatives of
//! its energy with respect to the densities; it contains no trained
//! parameters. Every size of the network (feature width, number and width
//! of the non-local layers, maximum angular momentum, correlation order,
//! depth of the perceptrons) and every constant is read from the model
//! file, so a model with other hyperparameters is evaluated by the same
//! code.
//!
//! Per grid point `p` of atom `a(p)`:
//!
//! ```text
//! x   = (rho_a, rho_b, |grad rho_a|^2, |grad rho_b|^2, tau_a, tau_b, |grad rho|^2)
//! f   = ln(|x| + eps)                      and its spin-swapped copy f'
//! h   = [MLP(f) + MLP(f')] / 2             spin-symmetric features
//! per non-local layer:
//!   g      = SiLU(W h + b)
//!   C_a    = sum_{p in a} w^atom_p n_l (g W_l)_c R_c(r_p) Y_lm(r_p)
//!   C_a   <- contraction(linear(C_a))      products up to the correlation order
//!   m_p    = R(r_p) sum_lm n_l Y_lm(r_p) (C_a W'_l)
//!   h     <- SiLU(W'' [h, exp(-rho) SiLU(W''' m + b)] + b)
//! F   = s sigmoid(MLP_out(h) / s)
//! e   = F C_x (max(rho_a, 0)^(4/3) + max(rho_b, 0)^(4/3)),  E = sum_p w_p e_p
//! ```
//!
//! with `r_p` the vector from the atom to the point, the radial functions
//! `R_c(r) = 2 r^2 exp(-r^2/t_c) / (3 t_c (pi t_c)^(3/2))`, `n_0 = 1`,
//! `n_l = (2l+1)^(-1/2)`, the unpartitioned atomic quadrature weights
//! `w^atom` and the partitioned weights `w`. The real spherical harmonics
//! are stored as homogeneous polynomials of the unit vector and the coupling
//! tensors of the contraction as plain arrays, so their conventions travel
//! with the model file. The derivatives are a hand-written reverse pass.
//!
//! Model file: the magic bytes `DFTRS-SKALA\0`, a little-endian `u32`
//! version and tensor count, then per tensor a `u32` name length, the UTF-8
//! name, a `u32` rank, the `u64` dimensions and the row-major `f64` data.
//! Tensor names:
//!
//! ```text
//! feature_epsilon, distance_epsilon, sigmoid_scale, lda_prefactor   [1]
//! input.{i}.weight [out, in], input.{i}.bias [out]        i = 0, 1, ...
//! radial.temperatures [channels]
//! harmonics.{l} [2l+1, (l+1)(l+2)/2]                      l = 0, ..., lmax
//! nonlocal.{n}.pre_down.{weight,bias}                     width -> channels
//! nonlocal.{n}.down.{l}, .coarse.{l}, .up.{l}             [channels, channels]
//! nonlocal.{n}.contraction.{l}.coupling.{nu}   [2l+1, D, ..., D (nu times), paths]
//! nonlocal.{n}.contraction.{l}.weights.{nu}    [paths, channels]
//! nonlocal.{n}.post_up.{weight,bias}, nonlocal.{n}.concat.{weight,bias}
//! output.{i}.weight, output.{i}.bias                      the last one maps to 1
//! ```
//!
//! with `D = (lmax + 1)^2`. The monomials of `harmonics.{l}` are
//! `x^a y^b z^c`, `a + b + c = l`, with `a` descending and then `b`
//! descending.

use crate::dual::{DualN, Scalar};
use crate::error::{DftError, Result};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

mod excited;
pub mod spinor;
pub use excited::{skala_lagrangian_gradient, skala_third_contraction};

/// Number of semilocal input features.
const FEATURES: usize = 7;
/// Feature permutation of the alpha-beta exchange.
const SWAP: [usize; FEATURES] = [1, 0, 3, 2, 5, 4, 6];
const MAGIC: &[u8; 12] = b"DFTRS-SKALA\0";
const FORMAT_VERSION: u32 = 1;

/// A named array of the model file.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

/// The tensors of a model file by name.
pub type Tensors = BTreeMap<String, Tensor>;

/// Write a model file.
pub fn write_tensors(path: &Path, tensors: &Tensors) -> Result<()> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for (name, tensor) in tensors {
        if tensor.shape.iter().product::<usize>() != tensor.data.len() {
            return Err(DftError::InvalidInput(format!(
                "tensor `{name}` has {} values for the shape {:?}",
                tensor.data.len(),
                tensor.shape
            )));
        }
        bytes.extend_from_slice(&(name.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dimension in &tensor.shape {
            bytes.extend_from_slice(&(*dimension as u64).to_le_bytes());
        }
        for value in &tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Option<&'a [u8]> {
        let end = self.cursor.checked_add(count)?;
        let slice = self.bytes.get(self.cursor..end)?;
        self.cursor = end;
        Some(slice)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}

fn parse_tensors(reader: &mut Reader<'_>) -> Option<Tensors> {
    if reader.take(MAGIC.len())? != MAGIC.as_slice() || reader.u32()? != FORMAT_VERSION {
        return None;
    }
    let count = reader.u32()?;
    let mut tensors = Tensors::new();
    for _ in 0..count {
        let length = reader.u32()? as usize;
        let name = std::str::from_utf8(reader.take(length)?).ok()?.to_string();
        let rank = reader.u32()? as usize;
        let mut shape = Vec::new();
        for _ in 0..rank {
            shape.push(usize::try_from(reader.u64()?).ok()?);
        }
        let size = shape
            .iter()
            .try_fold(1usize, |total, d| total.checked_mul(*d))?;
        let raw = reader.take(size.checked_mul(8)?)?;
        let data = raw
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().expect("eight bytes")))
            .collect();
        tensors.insert(name, Tensor { shape, data });
    }
    (reader.cursor == reader.bytes.len()).then_some(tensors)
}

/// Read a model file.
pub fn read_tensors(path: &Path) -> Result<Tensors> {
    let bytes = std::fs::read(path)?;
    let mut reader = Reader {
        bytes: &bytes,
        cursor: 0,
    };
    parse_tensors(&mut reader).ok_or_else(|| {
        DftError::InvalidInput(format!(
            "{}: not a version {FORMAT_VERSION} Skala model file",
            path.display()
        ))
    })
}

fn invalid(message: String) -> DftError {
    DftError::InvalidInput(format!("Skala model: {message}"))
}

fn tensor<'a>(tensors: &'a Tensors, name: &str, shape: &[usize]) -> Result<&'a Tensor> {
    let found = tensors
        .get(name)
        .ok_or_else(|| invalid(format!("tensor `{name}` is missing")))?;
    if found.shape != shape {
        return Err(invalid(format!(
            "tensor `{name}` has the shape {:?}, expected {shape:?}",
            found.shape
        )));
    }
    if found.data.iter().any(|v| !v.is_finite()) {
        return Err(invalid(format!("tensor `{name}` has a non-finite value")));
    }
    Ok(found)
}

/// Numbers the network is evaluated with: reals, or reals that carry
/// directional derivatives, which turn the reverse pass into products of
/// the second derivatives of the energy with changes of the densities.
pub trait Number: Scalar + Send + Sync {
    fn is_zero(self) -> bool;
}

impl Number for f64 {
    fn is_zero(self) -> bool {
        self == 0.0
    }
}

/// A real with one directional derivative.
pub(crate) type Tangent = DualN<f64, 1>;

impl<S: Number, const N: usize> Number for DualN<S, N> {
    fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.iter().all(|v| v.is_zero())
    }
}

impl<const W: usize> Number for crate::dual::Taylor<W> {
    fn is_zero(self) -> bool {
        self.c.iter().all(|v| *v == 0.0)
    }
}

fn zeros<S: Number>(len: usize) -> Vec<S> {
    vec![S::from_f64(0.0); len]
}

fn logistic<S: Number>(x: S) -> S {
    (S::from_f64(1.0) + (-x).exp()).recip()
}

fn silu<S: Number>(x: S) -> S {
    x * logistic(x)
}

fn silu_derivative<S: Number>(x: S) -> S {
    let s = logistic(x);
    s * (S::from_f64(1.0) + x * (S::from_f64(1.0) - s))
}

fn dot<S: Number>(a: [S; 3], b: [S; 3]) -> S {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn shell_norm(l: usize) -> f64 {
    if l == 0 {
        1.0
    } else {
        1.0 / ((2 * l + 1) as f64).sqrt()
    }
}

/// `y = W x + b` on a batch stored row by row.
#[derive(Clone, Debug)]
struct Linear {
    inputs: usize,
    outputs: usize,
    weight: Vec<f64>,
    bias: Vec<f64>,
}

impl Linear {
    fn load(tensors: &Tensors, prefix: &str) -> Result<Self> {
        let name = format!("{prefix}.weight");
        let weight = tensors
            .get(&name)
            .ok_or_else(|| invalid(format!("tensor `{name}` is missing")))?;
        if weight.shape.len() != 2 || weight.shape[0] == 0 || weight.shape[1] == 0 {
            return Err(invalid(format!(
                "tensor `{name}` must be a non-empty matrix, got the shape {:?}",
                weight.shape
            )));
        }
        let (outputs, inputs) = (weight.shape[0], weight.shape[1]);
        let weight = tensor(tensors, &name, &[outputs, inputs])?;
        let bias = tensor(tensors, &format!("{prefix}.bias"), &[outputs])?;
        Ok(Self {
            inputs,
            outputs,
            weight: weight.data.clone(),
            bias: bias.data.clone(),
        })
    }

    fn forward<S: Number>(&self, x: &[S]) -> Vec<S> {
        let mut out = zeros::<S>(x.len() / self.inputs * self.outputs);
        out.par_chunks_mut(self.outputs)
            .zip(x.par_chunks(self.inputs))
            .for_each(|(row, input)| {
                for (j, value) in row.iter_mut().enumerate() {
                    let weights = &self.weight[j * self.inputs..(j + 1) * self.inputs];
                    let mut sum = S::from_f64(0.0);
                    for (w, v) in weights.iter().zip(input) {
                        sum = sum + v.scale(*w);
                    }
                    *value = S::from_f64(self.bias[j]) + sum;
                }
            });
        out
    }

    /// Adjoint of the input batch from the adjoint of the output batch.
    fn backward<S: Number>(&self, adjoint: &[S]) -> Vec<S> {
        let mut out = zeros::<S>(adjoint.len() / self.outputs * self.inputs);
        out.par_chunks_mut(self.inputs)
            .zip(adjoint.par_chunks(self.outputs))
            .for_each(|(row, upstream)| {
                for (j, g) in upstream.iter().enumerate() {
                    if g.is_zero() {
                        continue;
                    }
                    let weights = &self.weight[j * self.inputs..(j + 1) * self.inputs];
                    for (value, w) in row.iter_mut().zip(weights) {
                        *value = *value + g.scale(*w);
                    }
                }
            });
        out
    }
}

/// A stack of linear layers, each followed by SiLU: output and the
/// pre-activations of every layer.
fn forward_stack<S: Number>(layers: &[Linear], input: Vec<S>) -> (Vec<S>, Vec<Vec<S>>) {
    let mut z = input;
    let mut tape = Vec::with_capacity(layers.len());
    for layer in layers {
        let pre = layer.forward(&z);
        z = pre.par_iter().map(|v| silu(*v)).collect();
        tape.push(pre);
    }
    (z, tape)
}

fn backward_stack<S: Number>(layers: &[Linear], tape: &[Vec<S>], adjoint: Vec<S>) -> Vec<S> {
    let mut g = adjoint;
    for (layer, pre) in layers.iter().zip(tape).rev() {
        g.par_iter_mut()
            .zip(pre.par_iter())
            .for_each(|(value, a)| *value = *value * silu_derivative(*a));
        g = layer.backward(&g);
    }
    g
}

/// Symmetric contraction onto one output angular momentum: the coupling
/// tensors of every order folded with their channel weights,
/// `orders[nu - 1][channel][M][i_1 ... i_nu]`.
#[derive(Clone, Debug)]
struct Contraction {
    dim: usize,
    orders: Vec<Vec<f64>>,
}

impl Contraction {
    fn load(
        tensors: &Tensors,
        prefix: &str,
        dim: usize,
        nsph: usize,
        channels: usize,
    ) -> Result<Self> {
        let mut orders = Vec::new();
        loop {
            let nu = orders.len() + 1;
            let coupling_name = format!("{prefix}.coupling.{nu}");
            if !tensors.contains_key(&coupling_name) {
                break;
            }
            let weights_name = format!("{prefix}.weights.{nu}");
            let paths = tensors
                .get(&weights_name)
                .and_then(|t| t.shape.first().copied())
                .ok_or_else(|| invalid(format!("tensor `{weights_name}` is missing")))?;
            let weights = tensor(tensors, &weights_name, &[paths, channels])?;
            let mut shape = vec![dim];
            shape.extend(std::iter::repeat_n(nsph, nu));
            shape.push(paths);
            let coupling = tensor(tensors, &coupling_name, &shape)?;
            let block = dim * nsph.pow(nu as u32);
            let mut folded = vec![0.0_f64; channels * block];
            for c in 0..channels {
                for j in 0..block {
                    let mut sum = 0.0;
                    for k in 0..paths {
                        sum += coupling.data[j * paths + k] * weights.data[k * channels + c];
                    }
                    folded[c * block + j] = sum;
                }
            }
            orders.push(folded);
        }
        if orders.is_empty() {
            return Err(invalid(format!("`{prefix}` has no coupling tensor")));
        }
        Ok(Self { dim, orders })
    }

    /// Output `[dim]` of one channel and the tensors `T_nu` (coupling plus
    /// the contraction of the next order) that the reverse pass needs:
    /// `out_M = sum_i (U1 + sum_j (U2 + sum_k U3 x_k) x_j) x_i`.
    fn forward<S: Number>(&self, channel: usize, x: &[S]) -> (Vec<S>, Vec<Vec<S>>) {
        let d = x.len();
        let correlation = self.orders.len();
        let mut tensors: Vec<Vec<S>> = vec![Vec::new(); correlation];
        let mut carry: Vec<S> = Vec::new();
        for nu in (1..=correlation).rev() {
            let block = self.dim * d.pow(nu as u32);
            let mut t: Vec<S> = self.orders[nu - 1][channel * block..(channel + 1) * block]
                .iter()
                .map(|v| S::from_f64(*v))
                .collect();
            if nu < correlation {
                for (a, b) in t.iter_mut().zip(&carry) {
                    *a = *a + *b;
                }
            }
            carry = t
                .chunks(d)
                .map(|row| {
                    let mut sum = S::from_f64(0.0);
                    for (a, b) in row.iter().zip(x) {
                        sum = sum + *a * *b;
                    }
                    sum
                })
                .collect();
            tensors[nu - 1] = t;
        }
        (carry, tensors)
    }

    /// Adjoint of `x` from the adjoint of the output.
    fn backward<S: Number>(tensors: &[Vec<S>], x: &[S], output_adjoint: &[S]) -> Vec<S> {
        let d = x.len();
        let mut gx = zeros::<S>(d);
        let mut adjoint = output_adjoint.to_vec();
        for t in tensors {
            let mut lifted = zeros::<S>(t.len());
            for (j, a) in adjoint.iter().enumerate() {
                if a.is_zero() {
                    continue;
                }
                for i in 0..d {
                    gx[i] = gx[i] + *a * t[j * d + i];
                    lifted[j * d + i] = *a * x[i];
                }
            }
            adjoint = lifted;
        }
        gx
    }
}

/// Intermediates of one non-local layer kept for the reverse pass.
struct Tape<S> {
    pre_down: Vec<S>,
    coarse: Vec<S>,
    pre_post: Vec<S>,
    pre_concat: Vec<S>,
}

/// One non-local layer.
#[derive(Clone, Debug)]
struct NonLocal {
    channels: usize,
    pre_down: Linear,
    down: Vec<Vec<f64>>,
    coarse: Vec<Vec<f64>>,
    contractions: Vec<Contraction>,
    up: Vec<Vec<f64>>,
    post_up: Linear,
    concat: Linear,
}

impl NonLocal {
    fn load(
        tensors: &Tensors,
        prefix: &str,
        width: usize,
        channels: usize,
        shells: usize,
    ) -> Result<Self> {
        let pre_down = Linear::load(tensors, &format!("{prefix}.pre_down"))?;
        let post_up = Linear::load(tensors, &format!("{prefix}.post_up"))?;
        let concat = Linear::load(tensors, &format!("{prefix}.concat"))?;
        if pre_down.inputs != width
            || pre_down.outputs != channels
            || post_up.inputs != channels
            || concat.inputs != width + post_up.outputs
            || concat.outputs != width
        {
            return Err(invalid(format!(
                "`{prefix}` does not map {width} features through {channels} channels back to {width} features"
            )));
        }
        let square = |kind: &str| -> Result<Vec<Vec<f64>>> {
            (0..shells)
                .map(|l| {
                    tensor(
                        tensors,
                        &format!("{prefix}.{kind}.{l}"),
                        &[channels, channels],
                    )
                    .map(|t| t.data.clone())
                })
                .collect()
        };
        let nsph = shells * shells;
        let contractions = (0..shells)
            .map(|l| {
                Contraction::load(
                    tensors,
                    &format!("{prefix}.contraction.{l}"),
                    2 * l + 1,
                    nsph,
                    channels,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            channels,
            pre_down,
            down: square("down")?,
            coarse: square("coarse")?,
            contractions,
            up: square("up")?,
            post_up,
            concat,
        })
    }

    fn shells(&self) -> usize {
        self.down.len()
    }

    /// Components `i = l^2 + m` of one channel of an atomic block
    /// `[(l, channel, m)]`.
    fn packed<S: Number>(&self, block: &[S], channel: usize) -> Vec<S> {
        let mut x = Vec::with_capacity(self.shells() * self.shells());
        for l in 0..self.shells() {
            let dim = 2 * l + 1;
            let start = self.channels * l * l + channel * dim;
            x.extend_from_slice(&block[start..start + dim]);
        }
        x
    }

    /// Atomic features after the channel mixing: `[atom][(l, channel, m)]`.
    fn gather<S: Number>(
        &self,
        pre_down: &[S],
        radial: &[f64],
        angular: &[f64],
        input: &SkalaInput<'_, S>,
    ) -> Vec<S> {
        let channels = self.channels;
        let nsph = self.shells() * self.shells();
        let hidden = channels * nsph;
        let natoms = input.centers.len();
        // One atom at a time, its points in their order: the sums do not
        // depend on the thread schedule.
        let raw: Vec<S> = input
            .members()
            .par_iter()
            .map(|points| {
                let mut block = zeros::<S>(hidden);
                for &p in points {
                    let g: Vec<S> = pre_down[p * channels..(p + 1) * channels]
                        .iter()
                        .map(|a| silu(*a))
                        .collect();
                    for (l, matrix) in self.down.iter().enumerate() {
                        let dim = 2 * l + 1;
                        let scale = input.atomic_weights[p] * shell_norm(l);
                        for c in 0..channels {
                            let mut q = S::from_f64(0.0);
                            for (u, gu) in g.iter().enumerate() {
                                q = q + gu.scale(matrix[u * channels + c]);
                            }
                            let factor = q.scale(scale).scale(radial[p * channels + c]);
                            for m in 0..dim {
                                let slot = channels * l * l + c * dim + m;
                                block[slot] =
                                    block[slot] + factor.scale(angular[p * nsph + l * l + m]);
                            }
                        }
                    }
                }
                block
            })
            .collect::<Vec<Vec<S>>>()
            .concat();
        let mut mixed = zeros::<S>(natoms * hidden);
        for atom in 0..natoms {
            for (l, matrix) in self.coarse.iter().enumerate() {
                let dim = 2 * l + 1;
                let offset = atom * hidden + channels * l * l;
                for u in 0..channels {
                    for v in 0..channels {
                        let w = matrix[u * channels + v];
                        for m in 0..dim {
                            mixed[offset + v * dim + m] =
                                mixed[offset + v * dim + m] + raw[offset + u * dim + m].scale(w);
                        }
                    }
                }
            }
        }
        mixed
    }

    /// Messages `[point][channel]` from the atomic features.
    fn scatter<S: Number>(
        &self,
        coarse: &[S],
        radial: &[f64],
        angular: &[f64],
        input: &SkalaInput<'_, S>,
    ) -> Vec<S> {
        let channels = self.channels;
        let nsph = self.shells() * self.shells();
        let hidden = channels * nsph;
        let natoms = input.centers.len();
        // table[atom][(l, m)][w] = n_l sum_u contracted[(l, u, m)] W'_l[u, w]
        let mut table = zeros::<S>(natoms * nsph * channels);
        for atom in 0..natoms {
            let block = &coarse[atom * hidden..(atom + 1) * hidden];
            for c in 0..channels {
                let x = self.packed(block, c);
                for (l, contraction) in self.contractions.iter().enumerate() {
                    let (out, _) = contraction.forward(c, &x);
                    let norm = shell_norm(l);
                    for (m, value) in out.iter().enumerate() {
                        let row = (atom * nsph + l * l + m) * channels;
                        for w in 0..channels {
                            table[row + w] = table[row + w]
                                + value.scale(norm).scale(self.up[l][c * channels + w]);
                        }
                    }
                }
            }
        }
        let mut messages = zeros::<S>(input.points.len() * channels);
        messages
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(p, row)| {
                let base = input.atom_of[p] * nsph;
                for s in 0..nsph {
                    let y = angular[p * nsph + s];
                    if y == 0.0 {
                        continue;
                    }
                    for (w, value) in row.iter_mut().enumerate() {
                        *value = *value + table[(base + s) * channels + w].scale(y);
                    }
                }
                for (w, value) in row.iter_mut().enumerate() {
                    *value = value.scale(radial[p * channels + w]);
                }
            });
        messages
    }

    /// Reverse pass of one layer: the adjoint of its input features from the
    /// adjoint of its output; `gate_adjoint` collects `dE/d exp(-rho)`.
    #[allow(clippy::too_many_arguments)]
    fn backward<S: Number>(
        &self,
        tape: &Tape<S>,
        output_adjoint: Vec<S>,
        gate: &[S],
        gate_adjoint: &mut [S],
        radial: &[f64],
        angular: &[f64],
        input: &SkalaInput<'_, S>,
        width: usize,
    ) -> Vec<S> {
        let channels = self.channels;
        let nsph = self.shells() * self.shells();
        let hidden = channels * nsph;
        let natoms = input.centers.len();
        let n = input.points.len();
        let extra = self.post_up.outputs;
        let mut g = output_adjoint;
        g.par_iter_mut()
            .zip(tape.pre_concat.par_iter())
            .for_each(|(value, a)| *value = *value * silu_derivative(*a));
        let joined = self.concat.backward(&g);
        let mut below = zeros::<S>(n * width);
        let mut post = zeros::<S>(n * extra);
        below
            .par_chunks_mut(width)
            .zip(post.par_chunks_mut(extra))
            .zip(gate_adjoint.par_iter_mut())
            .enumerate()
            .for_each(|(p, ((b, q), ga))| {
                let row = &joined[p * (width + extra)..(p + 1) * (width + extra)];
                b.copy_from_slice(&row[..width]);
                let pre = &tape.pre_post[p * extra..(p + 1) * extra];
                for ((value, upstream), a) in q.iter_mut().zip(&row[width..]).zip(pre) {
                    *ga = *ga + *upstream * silu(*a);
                    *value = *upstream * gate[p] * silu_derivative(*a);
                }
            });
        let message_adjoint = self.post_up.backward(&post);
        // Adjoint of the table of `scatter`.
        let table_adjoint: Vec<S> = input
            .members()
            .par_iter()
            .map(|points| {
                let mut block = zeros::<S>(nsph * channels);
                for &p in points {
                    for s in 0..nsph {
                        let y = angular[p * nsph + s];
                        if y == 0.0 {
                            continue;
                        }
                        for w in 0..channels {
                            block[s * channels + w] = block[s * channels + w]
                                + message_adjoint[p * channels + w]
                                    .scale(radial[p * channels + w])
                                    .scale(y);
                        }
                    }
                }
                block
            })
            .collect::<Vec<Vec<S>>>()
            .concat();
        // Through the contraction and the channel mixing of every atom.
        let mut raw_adjoint = zeros::<S>(natoms * hidden);
        for atom in 0..natoms {
            let block = &tape.coarse[atom * hidden..(atom + 1) * hidden];
            let mut mixed_adjoint = zeros::<S>(hidden);
            for c in 0..channels {
                let x = self.packed(block, c);
                let mut gx = zeros::<S>(nsph);
                for (l, contraction) in self.contractions.iter().enumerate() {
                    let norm = shell_norm(l);
                    let output_adjoint: Vec<S> = (0..2 * l + 1)
                        .map(|m| {
                            let row = (atom * nsph + l * l + m) * channels;
                            let mut sum = S::from_f64(0.0);
                            for w in 0..channels {
                                sum = sum
                                    + table_adjoint[row + w].scale(self.up[l][c * channels + w]);
                            }
                            sum.scale(norm)
                        })
                        .collect();
                    let (_, tensors) = contraction.forward(c, &x);
                    let partial = Contraction::backward(&tensors, &x, &output_adjoint);
                    for (total, value) in gx.iter_mut().zip(&partial) {
                        *total = *total + *value;
                    }
                }
                for l in 0..self.shells() {
                    let dim = 2 * l + 1;
                    for m in 0..dim {
                        mixed_adjoint[channels * l * l + c * dim + m] = gx[l * l + m];
                    }
                }
            }
            for (l, matrix) in self.coarse.iter().enumerate() {
                let dim = 2 * l + 1;
                let offset = channels * l * l;
                for u in 0..channels {
                    for v in 0..channels {
                        let w = matrix[u * channels + v];
                        for m in 0..dim {
                            let slot = atom * hidden + offset + u * dim + m;
                            raw_adjoint[slot] =
                                raw_adjoint[slot] + mixed_adjoint[offset + v * dim + m].scale(w);
                        }
                    }
                }
            }
        }
        // Back to the points through `gather`.
        let mut down_adjoint = zeros::<S>(n * channels);
        down_adjoint
            .par_chunks_mut(channels)
            .enumerate()
            .for_each(|(p, row)| {
                let atom = input.atom_of[p];
                let block = &raw_adjoint[atom * hidden..(atom + 1) * hidden];
                for (l, matrix) in self.down.iter().enumerate() {
                    let dim = 2 * l + 1;
                    let scale = input.atomic_weights[p] * shell_norm(l);
                    for c in 0..channels {
                        let mut projected = S::from_f64(0.0);
                        for m in 0..dim {
                            projected = projected
                                + block[channels * l * l + c * dim + m]
                                    .scale(angular[p * nsph + l * l + m]);
                        }
                        let factor = projected.scale(scale * radial[p * channels + c]);
                        if factor.is_zero() {
                            continue;
                        }
                        for (u, value) in row.iter_mut().enumerate() {
                            *value = *value + factor.scale(matrix[u * channels + c]);
                        }
                    }
                }
                let pre = &tape.pre_down[p * channels..(p + 1) * channels];
                for (value, a) in row.iter_mut().zip(pre) {
                    *value = *value * silu_derivative(*a);
                }
            });
        let from_down = self.pre_down.backward(&down_adjoint);
        below
            .par_iter_mut()
            .zip(from_down.par_iter())
            .for_each(|(value, extra)| *value = *value + *extra);
        below
    }
}

/// Densities on an atom-centred molecular grid. The densities are reals, or
/// reals with a directional derivative ([`Number`]).
pub struct SkalaInput<'a, S = f64> {
    /// Owning atom of every grid point.
    pub atom_of: &'a [usize],
    pub points: &'a [[f64; 3]],
    /// Atomic positions.
    pub centers: &'a [[f64; 3]],
    /// Partitioned quadrature weights (integration of the energy density).
    pub weights: &'a [S],
    /// Atomic quadrature weights without the partition factor.
    pub atomic_weights: &'a [f64],
    /// Spin densities, their gradients and kinetic energy densities
    /// `tau = (1/2) sum_i |grad phi_i|^2`.
    pub rho: [&'a [S]; 2],
    pub gradient: [&'a [[S; 3]]; 2],
    pub tau: [&'a [S]; 2],
}

impl<S> SkalaInput<'_, S> {
    /// Grid points of every atom, in their order.
    fn members(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.centers.len()];
        for (p, atom) in self.atom_of.iter().enumerate() {
            out[*atom].push(p);
        }
        out
    }
}

/// Energy and its derivatives with respect to the densities at every point
/// (the quadrature weights are included in the derivatives).
#[derive(Clone, Debug)]
pub struct SkalaOutput<S = f64> {
    pub energy: S,
    /// Energy density per volume at every point.
    pub energy_density: Vec<S>,
    pub d_rho: [Vec<S>; 2],
    pub d_gradient: [Vec<[S; 3]>; 2],
    pub d_tau: [Vec<S>; 2],
}

/// A Skala model: all weights and constants of one model file.
#[derive(Clone, Debug)]
pub struct SkalaModel {
    feature_epsilon: f64,
    distance_epsilon: f64,
    sigmoid_scale: f64,
    prefactor: f64,
    input: Vec<Linear>,
    temperatures: Vec<f64>,
    harmonics: Vec<Vec<f64>>,
    layers: Vec<NonLocal>,
    hidden: Vec<Linear>,
    last: Linear,
}

impl SkalaModel {
    /// Load a model file.
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_tensors(&read_tensors(path)?)
    }

    /// Build a model from named tensors; every size is taken from their
    /// shapes and checked for consistency.
    pub fn from_tensors(tensors: &Tensors) -> Result<Self> {
        let scalar = |name: &str| -> Result<f64> { Ok(tensor(tensors, name, &[1])?.data[0]) };
        let stack = |prefix: &str| -> Result<Vec<Linear>> {
            let mut out = Vec::new();
            while tensors.contains_key(&format!("{prefix}.{}.weight", out.len())) {
                out.push(Linear::load(tensors, &format!("{prefix}.{}", out.len()))?);
            }
            Ok(out)
        };
        let chained = |layers: &[Linear], first: usize| -> Option<usize> {
            let mut size = first;
            for layer in layers {
                if layer.inputs != size {
                    return None;
                }
                size = layer.outputs;
            }
            Some(size)
        };
        let input = stack("input")?;
        let width = chained(&input, FEATURES)
            .filter(|_| !input.is_empty())
            .ok_or_else(|| {
                invalid(format!(
                    "the input layers must form a chain starting from {FEATURES} features"
                ))
            })?;
        let mut hidden = stack("output")?;
        let last = hidden
            .pop()
            .ok_or_else(|| invalid("the output layers are missing".into()))?;
        if chained(&hidden, width) != Some(last.inputs) || last.outputs != 1 {
            return Err(invalid(format!(
                "the output layers must form a chain from {width} features to one value"
            )));
        }
        let channels = tensors
            .get("radial.temperatures")
            .map_or(0, |t| t.data.len());
        let temperatures = tensor(tensors, "radial.temperatures", &[channels])?
            .data
            .clone();
        if channels == 0 || temperatures.iter().any(|t| *t <= 0.0) {
            return Err(invalid("radial.temperatures needs positive entries".into()));
        }
        let mut harmonics = Vec::new();
        while tensors.contains_key(&format!("harmonics.{}", harmonics.len())) {
            let l = harmonics.len();
            let block = tensor(
                tensors,
                &format!("harmonics.{l}"),
                &[2 * l + 1, (l + 1) * (l + 2) / 2],
            )?;
            harmonics.push(block.data.clone());
        }
        if harmonics.is_empty() {
            return Err(invalid("harmonics.0 is missing".into()));
        }
        let mut layers = Vec::new();
        while tensors.contains_key(&format!("nonlocal.{}.pre_down.weight", layers.len())) {
            layers.push(NonLocal::load(
                tensors,
                &format!("nonlocal.{}", layers.len()),
                width,
                channels,
                harmonics.len(),
            )?);
        }
        let model = Self {
            feature_epsilon: scalar("feature_epsilon")?,
            distance_epsilon: scalar("distance_epsilon")?,
            sigmoid_scale: scalar("sigmoid_scale")?,
            prefactor: scalar("lda_prefactor")?,
            input,
            temperatures,
            harmonics,
            layers,
            hidden,
            last,
        };
        if !(model.feature_epsilon > 0.0 && model.sigmoid_scale > 0.0)
            || model.distance_epsilon < 0.0
        {
            return Err(invalid(
                "feature_epsilon and sigmoid_scale must be positive and distance_epsilon non-negative"
                    .into(),
            ));
        }
        Ok(model)
    }

    /// Feature width of the network.
    pub fn width(&self) -> usize {
        self.input.last().map_or(0, |layer| layer.outputs)
    }

    /// The sizes read from the model file.
    pub fn description(&self) -> String {
        let correlation = self
            .layers
            .first()
            .map_or(0, |layer| layer.contractions[0].orders.len());
        format!(
            "Skala model: {} features, {} input layers, {} non-local layers ({} channels, lmax {}, correlation order {}), {} output layers, sigmoid scale {}, feature epsilon {:e}",
            self.width(),
            self.input.len(),
            self.layers.len(),
            self.temperatures.len(),
            self.harmonics.len() - 1,
            correlation,
            self.hidden.len() + 1,
            self.sigmoid_scale,
            self.feature_epsilon
        )
    }

    fn fill_harmonics(&self, unit: [f64; 3], out: &mut [f64]) {
        for (l, coefficients) in self.harmonics.iter().enumerate() {
            let count = (l + 1) * (l + 2) / 2;
            let mut monomials = Vec::with_capacity(count);
            for a in (0..=l).rev() {
                for b in (0..=l - a).rev() {
                    let c = l - a - b;
                    monomials.push(
                        unit[0].powi(a as i32) * unit[1].powi(b as i32) * unit[2].powi(c as i32),
                    );
                }
            }
            for m in 0..2 * l + 1 {
                out[l * l + m] = coefficients[m * count..(m + 1) * count]
                    .iter()
                    .zip(&monomials)
                    .map(|(c, v)| c * v)
                    .sum();
            }
        }
    }

    /// Energy and, when `derivatives` is set, its derivatives with respect
    /// to the spin densities, their gradients and the kinetic energy
    /// densities at every grid point.
    pub fn evaluate<S: Number>(
        &self,
        input: &SkalaInput<'_, S>,
        derivatives: bool,
    ) -> Result<SkalaOutput<S>> {
        let n = input.points.len();
        let natoms = input.centers.len();
        let lengths = [
            input.atom_of.len(),
            input.weights.len(),
            input.atomic_weights.len(),
            input.rho[0].len(),
            input.rho[1].len(),
            input.gradient[0].len(),
            input.gradient[1].len(),
            input.tau[0].len(),
            input.tau[1].len(),
        ];
        if lengths.iter().any(|len| *len != n) || input.atom_of.iter().any(|a| *a >= natoms) {
            return Err(DftError::InvalidInput(
                "Skala input arrays need one entry per grid point and atom indices inside the list of centres"
                    .into(),
            ));
        }
        let width = self.width();
        let channels = self.temperatures.len();
        let nsph = self.harmonics.len() * self.harmonics.len();
        let eps = self.feature_epsilon;
        let zero = S::from_f64(0.0);
        let one = S::from_f64(1.0);

        // Semilocal features and the spin-symmetric input perceptron.
        let mut raw = zeros::<S>(n * FEATURES);
        raw.par_chunks_mut(FEATURES).enumerate().for_each(|(p, x)| {
            let (a, b) = (input.gradient[0][p], input.gradient[1][p]);
            let total = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
            x[0] = input.rho[0][p];
            x[1] = input.rho[1][p];
            x[2] = dot(a, a);
            x[3] = dot(b, b);
            x[4] = input.tau[0][p];
            x[5] = input.tau[1][p];
            x[6] = dot(total, total);
        });
        let logs: Vec<S> = raw
            .par_iter()
            .map(|x| {
                let magnitude = if x.value() < 0.0 { -*x } else { *x };
                (magnitude + S::from_f64(eps)).ln()
            })
            .collect();
        let mut swapped = zeros::<S>(n * FEATURES);
        swapped
            .par_chunks_mut(FEATURES)
            .zip(logs.par_chunks(FEATURES))
            .for_each(|(row, f)| {
                for (value, source) in row.iter_mut().zip(SWAP) {
                    *value = f[source];
                }
            });
        let (z_direct, tape_direct) = forward_stack(&self.input, logs);
        let (z_swapped, tape_swapped) = forward_stack(&self.input, swapped);
        let mut h: Vec<S> = z_direct
            .par_iter()
            .zip(z_swapped.par_iter())
            .map(|(a, b)| (*a + *b).scale(0.5))
            .collect();

        // Radial functions and harmonics of the point seen from its atom.
        let mut radial = vec![0.0_f64; n * channels];
        let mut angular = vec![0.0_f64; n * nsph];
        radial
            .par_chunks_mut(channels)
            .zip(angular.par_chunks_mut(nsph))
            .enumerate()
            .for_each(|(p, (r, y))| {
                let centre = input.centers[input.atom_of[p]];
                let point = input.points[p];
                let d = [
                    point[0] - centre[0],
                    point[1] - centre[1],
                    point[2] - centre[2],
                ];
                let squared = dot(d, d) + 3.0 * self.distance_epsilon;
                let length = squared.sqrt();
                for (value, t) in r.iter_mut().zip(&self.temperatures) {
                    *value = 2.0 / (3.0 * t * (std::f64::consts::PI * t).powf(1.5))
                        * (-squared / t).exp()
                        * squared;
                }
                self.fill_harmonics([d[0] / length, d[1] / length, d[2] / length], y);
            });
        let gate: Vec<S> = (0..n)
            .map(|p| (-(input.rho[0][p] + input.rho[1][p])).exp())
            .collect();

        // Non-local layers.
        let mut tapes: Vec<Tape<S>> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let pre_down = layer.pre_down.forward(&h);
            let coarse = layer.gather(&pre_down, &radial, &angular, input);
            let messages = layer.scatter(&coarse, &radial, &angular, input);
            let pre_post = layer.post_up.forward(&messages);
            let extra = layer.post_up.outputs;
            let mut joined = zeros::<S>(n * (width + extra));
            joined
                .par_chunks_mut(width + extra)
                .enumerate()
                .for_each(|(p, row)| {
                    row[..width].copy_from_slice(&h[p * width..(p + 1) * width]);
                    let pre = &pre_post[p * extra..(p + 1) * extra];
                    for (value, a) in row[width..].iter_mut().zip(pre) {
                        *value = gate[p] * silu(*a);
                    }
                });
            let pre_concat = layer.concat.forward(&joined);
            h = pre_concat.par_iter().map(|a| silu(*a)).collect();
            tapes.push(Tape {
                pre_down,
                coarse,
                pre_post,
                pre_concat,
            });
        }

        // Enhancement factor and energy.
        let (z, tape_output) = forward_stack(&self.hidden, h);
        let y = self.last.forward(&z);
        let scale = S::from_f64(self.sigmoid_scale);
        let power = |rho: S| {
            if rho.value() > 0.0 {
                rho.powf(4.0 / 3.0)
            } else {
                zero
            }
        };
        let local: Vec<S> = (0..n)
            .map(|p| (power(input.rho[0][p]) + power(input.rho[1][p])).scale(self.prefactor))
            .collect();
        let enhancement: Vec<S> = y.iter().map(|v| scale * logistic(*v / scale)).collect();
        let energy_density: Vec<S> = enhancement
            .iter()
            .zip(&local)
            .map(|(f, e)| *f * *e)
            .collect();
        let mut energy = zero;
        for (e, w) in energy_density.iter().zip(input.weights) {
            energy = energy + *e * *w;
        }
        let mut output = SkalaOutput {
            energy,
            energy_density,
            d_rho: [Vec::new(), Vec::new()],
            d_gradient: [Vec::new(), Vec::new()],
            d_tau: [Vec::new(), Vec::new()],
        };
        if !derivatives {
            return Ok(output);
        }

        // Reverse pass.
        let head: Vec<S> = (0..n)
            .map(|p| {
                let s = logistic(y[p] / scale);
                local[p] * input.weights[p] * s * (one - s)
            })
            .collect();
        let mut gh = backward_stack(&self.hidden, &tape_output, self.last.backward(&head));
        let mut gate_adjoint = zeros::<S>(n);
        for (layer, tape) in self.layers.iter().zip(&tapes).rev() {
            gh = layer.backward(
                tape,
                gh,
                &gate,
                &mut gate_adjoint,
                &radial,
                &angular,
                input,
                width,
            );
        }
        let half: Vec<S> = gh.par_iter().map(|v| v.scale(0.5)).collect();
        let g_direct = backward_stack(&self.input, &tape_direct, half.clone());
        let g_swapped = backward_stack(&self.input, &tape_swapped, half);
        let mut d_rho = [zeros::<S>(n), zeros::<S>(n)];
        let mut d_tau = [zeros::<S>(n), zeros::<S>(n)];
        let mut d_gradient = [vec![[zero; 3]; n], vec![[zero; 3]; n]];
        for p in 0..n {
            let mut gf = [zero; FEATURES];
            for i in 0..FEATURES {
                gf[i] = gf[i] + g_direct[p * FEATURES + i];
                gf[SWAP[i]] = gf[SWAP[i]] + g_swapped[p * FEATURES + i];
            }
            let mut gx = [zero; FEATURES];
            for i in 0..FEATURES {
                let x = raw[p * FEATURES + i];
                gx[i] = if x.value() > 0.0 {
                    gf[i] / (x + S::from_f64(eps))
                } else if x.value() < 0.0 {
                    (-gf[i]) / (-x + S::from_f64(eps))
                } else {
                    zero
                };
            }
            let gated = gate[p] * gate_adjoint[p];
            for spin in 0..2 {
                let rho = input.rho[spin][p];
                let exchange = if rho.value() > 0.0 {
                    (enhancement[p] * input.weights[p])
                        .scale(self.prefactor)
                        .scale(4.0 / 3.0)
                        * rho.cbrt()
                } else {
                    zero
                };
                d_rho[spin][p] = gx[spin] + exchange - gated;
                d_tau[spin][p] = gx[4 + spin];
            }
            let (a, b) = (input.gradient[0][p], input.gradient[1][p]);
            for axis in 0..3 {
                let total = a[axis] + b[axis];
                d_gradient[0][p][axis] = (gx[2] * a[axis]).scale(2.0) + (gx[6] * total).scale(2.0);
                d_gradient[1][p][axis] = (gx[3] * b[axis]).scale(2.0) + (gx[6] * total).scale(2.0);
            }
        }
        output.d_rho = d_rho;
        output.d_tau = d_tau;
        output.d_gradient = d_gradient;
        Ok(output)
    }
}

/// Sizes of a model with pseudo-random weights.
#[derive(Clone, Copy, Debug)]
pub struct SkalaSizes {
    /// Feature width of the perceptrons.
    pub width: usize,
    /// Channels of the non-local layers (also the number of radial functions).
    pub channels: usize,
    /// Number of angular momentum shells, `lmax + 1`.
    pub shells: usize,
    /// Correlation order of the symmetric contraction.
    pub correlation: usize,
    /// Number of non-local layers.
    pub layers: usize,
    /// Width of the message entering the concatenation.
    pub post: usize,
}

/// Deterministic values in `[-0.5, 0.5)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// The tensors of a model of the given sizes with pseudo-random weights.
///
/// This is not a physical functional: it exists to exercise an integration
/// and the derivative code with any set of hyperparameters. The harmonics of
/// l = 0 and l = 1 are the true ones (1 and the unit vector); higher shells
/// and the coupling tensors are arbitrary.
pub fn random_tensors(seed: u64, sizes: SkalaSizes) -> Tensors {
    let mut rng = Lcg(seed);
    let mut tensors = Tensors::new();
    let mut random = |name: String, shape: Vec<usize>, scale: f64| {
        let size: usize = shape.iter().product();
        let data = (0..size).map(|_| scale * rng.next()).collect();
        tensors.insert(name, Tensor { shape, data });
    };
    let SkalaSizes {
        width,
        channels,
        shells,
        correlation,
        layers,
        post,
    } = sizes;
    let nsph = shells * shells;
    random("input.0.weight".into(), vec![width, FEATURES], 0.8);
    random("input.0.bias".into(), vec![width], 0.4);
    random("input.1.weight".into(), vec![width, width], 1.2);
    random("input.1.bias".into(), vec![width], 0.4);
    for n in 0..layers {
        random(
            format!("nonlocal.{n}.pre_down.weight"),
            vec![channels, width],
            1.2,
        );
        random(format!("nonlocal.{n}.pre_down.bias"), vec![channels], 0.4);
        for l in 0..shells {
            random(
                format!("nonlocal.{n}.down.{l}"),
                vec![channels, channels],
                2.0,
            );
            random(
                format!("nonlocal.{n}.coarse.{l}"),
                vec![channels, channels],
                1.5,
            );
            random(
                format!("nonlocal.{n}.up.{l}"),
                vec![channels, channels],
                2.0,
            );
            for nu in 1..=correlation {
                let mut shape = vec![2 * l + 1];
                shape.extend(std::iter::repeat_n(nsph, nu));
                shape.push(2);
                random(
                    format!("nonlocal.{n}.contraction.{l}.coupling.{nu}"),
                    shape,
                    1.5,
                );
                random(
                    format!("nonlocal.{n}.contraction.{l}.weights.{nu}"),
                    vec![2, channels],
                    2.0,
                );
            }
        }
        random(
            format!("nonlocal.{n}.post_up.weight"),
            vec![post, channels],
            3.0,
        );
        random(format!("nonlocal.{n}.post_up.bias"), vec![post], 0.4);
        random(
            format!("nonlocal.{n}.concat.weight"),
            vec![width, width + post],
            1.2,
        );
        random(format!("nonlocal.{n}.concat.bias"), vec![width], 0.4);
    }
    random("output.0.weight".into(), vec![width, width], 1.2);
    random("output.0.bias".into(), vec![width], 0.4);
    random("output.1.weight".into(), vec![1, width], 2.0);
    random("output.1.bias".into(), vec![1], 0.4);
    for l in 2..shells {
        random(
            format!("harmonics.{l}"),
            vec![2 * l + 1, (l + 1) * (l + 2) / 2],
            2.0,
        );
    }
    let mut fixed = |name: &str, shape: Vec<usize>, data: Vec<f64>| {
        tensors.insert(name.to_string(), Tensor { shape, data });
    };
    fixed("harmonics.0", vec![1, 1], vec![1.0]);
    if shells > 1 {
        fixed(
            "harmonics.1",
            vec![3, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
    }
    fixed(
        "radial.temperatures",
        vec![channels],
        (0..channels).map(|k| 0.8 + 0.6 * k as f64).collect(),
    );
    fixed("feature_epsilon", vec![1], vec![1e-5]);
    fixed("distance_epsilon", vec![1], vec![1e-20]);
    fixed("sigmoid_scale", vec![1], vec![2.0]);
    fixed("lda_prefactor", vec![1], vec![-0.9305257363491001]);
    tensors
}

/// Spin densities, their gradients and kinetic energy densities on a grid.
type Densities = ([Vec<f64>; 2], [Vec<[f64; 3]>; 2], [Vec<f64>; 2]);

/// Batches of grid points handled by one task.
pub(crate) fn batch_ranges(n: usize) -> Vec<(usize, usize)> {
    const BATCH: usize = 256;
    (0..n)
        .step_by(BATCH)
        .map(|start| (start, (start + BATCH).min(n)))
        .collect()
}

/// Densities of the symmetric spin matrices `p` on the molecular grid; the
/// map is linear, so it also gives the response to a change of the matrices.
pub(crate) fn grid_densities(
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    p: [&crate::linalg::Matrix; 2],
    ranges: &[(usize, usize)],
) -> Densities {
    use crate::grid::eval::eval_aos_full;
    use crate::linalg::Matrix;
    let n = grid.len();
    let nao = mb.nao_sph;
    let batches: Vec<Densities> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let batch = eval_aos_full(mb, &grid.points[start..end], true, false);
            let g = batch.grad.as_ref().expect("gradients were requested");
            let npts = end - start;
            let mut rho = [vec![0.0_f64; npts], vec![0.0_f64; npts]];
            let mut gradient = [vec![[0.0_f64; 3]; npts], vec![[0.0_f64; 3]; npts]];
            let mut tau = [vec![0.0_f64; npts], vec![0.0_f64; npts]];
            for spin in 0..2 {
                let phip = batch.phi.matmul_seq(p[spin]);
                let gp: [Matrix; 3] = std::array::from_fn(|axis| g[axis].matmul_seq(p[spin]));
                for i in 0..npts {
                    let mut value = 0.0;
                    for mu in 0..nao {
                        value += phip[(i, mu)] * batch.phi[(i, mu)];
                    }
                    rho[spin][i] = value;
                    let mut kinetic = 0.0;
                    for axis in 0..3 {
                        let mut slope = 0.0;
                        for mu in 0..nao {
                            slope += phip[(i, mu)] * g[axis][(i, mu)];
                            kinetic += gp[axis][(i, mu)] * g[axis][(i, mu)];
                        }
                        gradient[spin][i][axis] = 2.0 * slope;
                    }
                    tau[spin][i] = 0.5 * kinetic;
                }
            }
            (rho, gradient, tau)
        })
        .collect();
    let mut rho = [Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut gradient = [Vec::with_capacity(n), Vec::with_capacity(n)];
    let mut tau = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for (r, g, t) in batches {
        for spin in 0..2 {
            rho[spin].extend_from_slice(&r[spin]);
            gradient[spin].extend_from_slice(&g[spin]);
            tau[spin].extend_from_slice(&t[spin]);
        }
    }
    (rho, gradient, tau)
}

pub(crate) fn grid_geometry(
    mol: &crate::system::Molecule,
    grid: &crate::grid::MolecularGrid,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let points = grid.points.iter().map(|point| point.to_array()).collect();
    let centers = mol
        .atoms
        .iter()
        .map(|atom| atom.position.to_array())
        .collect();
    (points, centers)
}

/// Densities of the spin density matrices `p` on the molecular grid and the
/// model output with its derivatives.
fn evaluate_on_grid(
    model: &SkalaModel,
    mol: &crate::system::Molecule,
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    p: [&crate::linalg::Matrix; 2],
    ranges: &[(usize, usize)],
) -> Result<SkalaOutput> {
    let (rho, gradient, tau) = grid_densities(mb, grid, p, ranges);
    let (points, centers) = grid_geometry(mol, grid);
    model.evaluate(
        &SkalaInput {
            atom_of: &grid.atom_of,
            points: &points,
            centers: &centers,
            weights: &grid.weights,
            atomic_weights: &grid.raw_weights,
            rho: [&rho[0], &rho[1]],
            gradient: [&gradient[0], &gradient[1]],
            tau: [&tau[0], &tau[1]],
        },
        true,
    )
}

/// Nuclear gradient of the Skala energy at fixed density matrices.
///
/// With `g_B(p)` the derivative of the energy with respect to the centres of
/// the basis functions of atom `B` through the densities at point `p`
/// (`-2 sum_{mu in B} P_mu,nu [dE/drho d phi_mu phi_nu + dE/dgrad rho .
/// (d grad phi_mu phi_nu + d phi_mu grad phi_nu) + (1/2) dE/dtau d grad
/// phi_mu . grad phi_nu]`), the points of atom `A` ride with it, which by
/// translational invariance moves every other basis function the opposite
/// way: `dE/dR_A = sum_{p not in A} g_A(p) - sum_{p in A} sum_{B != A} g_B(p)`
/// plus the response of the partition weights. The vectors from an atom to
/// its own grid points, which the non-local layers use, do not change.
pub fn skala_gradient(
    model: &SkalaModel,
    mol: &crate::system::Molecule,
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    p: [&crate::linalg::Matrix; 2],
) -> Result<Vec<crate::math::Vec3>> {
    let ranges = batch_ranges(grid.len());
    let out = evaluate_on_grid(model, mol, mb, grid, p, &ranges)?;
    Ok(gradient_contraction(
        mol,
        mb,
        grid,
        &ranges,
        p,
        [&out.d_rho[0], &out.d_rho[1]],
        [&out.d_gradient[0], &out.d_gradient[1]],
        [&out.d_tau[0], &out.d_tau[1]],
        &out.energy_density,
    ))
}

/// Nuclear derivative, at fixed symmetric spin matrices `p`, of
/// `sum_p [a rho + b . grad rho + c tau + e w]` with `(a, b, c)` =
/// `(d_rho, d_gradient, d_tau)` and `e = energy_density` held fixed: the
/// basis functions, the grid points riding their atoms and the partition
/// weights move.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_contraction(
    mol: &crate::system::Molecule,
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    ranges: &[(usize, usize)],
    p: [&crate::linalg::Matrix; 2],
    d_rho: [&[f64]; 2],
    d_gradient: [&[[f64; 3]]; 2],
    d_tau: [&[f64]; 2],
    energy_density: &[f64],
) -> Vec<crate::math::Vec3> {
    use crate::grid::eval::eval_aos_full;
    use crate::linalg::Matrix;
    use crate::math::Vec3;
    /// Packed second-derivative index (xx, xy, xz, yy, yz, zz).
    const HESSIAN: [[usize; 3]; 3] = [[0, 1, 2], [1, 3, 4], [2, 4, 5]];
    let nao = mb.nao_sph;
    let natoms = mol.atoms.len();
    let ao_atom = mb.sph_ao_atoms();
    let centers: Vec<Vec3> = mol.atoms.iter().map(|atom| atom.position).collect();
    let aij = crate::grid::MolecularGrid::becke_aij(mol);
    let partials: Vec<Vec<Vec3>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let pts = &grid.points[start..end];
            let batch = eval_aos_full(mb, pts, true, true);
            let g = batch.grad.as_ref().expect("gradients were requested");
            let h = batch
                .hess
                .as_ref()
                .expect("second derivatives were requested");
            let npts = end - start;
            let mut acc = vec![Vec3::zero(); natoms];
            for spin in 0..2 {
                let x = batch.phi.matmul_seq(p[spin]);
                let y: [Matrix; 3] = std::array::from_fn(|axis| g[axis].matmul_seq(p[spin]));
                for i in 0..npts {
                    let point = start + i;
                    let owner = grid.atom_of[point];
                    let dr = d_rho[spin][point];
                    let dg = d_gradient[spin][point];
                    let dt = d_tau[spin][point];
                    for mu in 0..nao {
                        let atom = ao_atom[mu];
                        if atom == owner {
                            continue;
                        }
                        let mut c = [0.0_f64; 3];
                        for (ax, value) in c.iter_mut().enumerate() {
                            let mut term = dr * g[ax][(i, mu)] * x[(i, mu)];
                            let mut kinetic = 0.0;
                            for u in 0..3 {
                                let second = h[HESSIAN[ax][u]][(i, mu)];
                                term +=
                                    dg[u] * (second * x[(i, mu)] + g[ax][(i, mu)] * y[u][(i, mu)]);
                                kinetic += second * y[u][(i, mu)];
                            }
                            *value = -2.0 * (term + 0.5 * dt * kinetic);
                        }
                        let c = Vec3::new(c[0], c[1], c[2]);
                        acc[atom] += c;
                        acc[owner] -= c;
                    }
                }
            }
            // Response of the partition weights, including the motion of
            // the point with its own atom.
            for i in 0..npts {
                let point = start + i;
                let owner = grid.atom_of[point];
                let scale = grid.raw_weights[point] * energy_density[point];
                let (_, dwb) =
                    crate::grid::becke_weight_gradient(&centers, &aij, natoms, owner, pts[i]);
                let mut sum = Vec3::zero();
                for (atom, d) in dwb.iter().enumerate() {
                    acc[atom] += *d * scale;
                    sum += *d;
                }
                acc[owner] -= sum * scale;
            }
            acc
        })
        .collect();
    let mut total = vec![Vec3::zero(); natoms];
    for partial in partials {
        for (target, value) in total.iter_mut().zip(partial) {
            *target += value;
        }
    }
    total
}

/// Kohn-Sham matrices from derivatives of an energy with respect to the
/// densities at the grid points:
/// `V^s_mn = sum_p [a_p phi_m phi_n + b_p . grad(phi_m phi_n)
/// + (1/2) c_p grad phi_m . grad phi_n]`.
pub(crate) fn assemble(
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    ranges: &[(usize, usize)],
    d_rho: [&[f64]; 2],
    d_gradient: [&[[f64; 3]]; 2],
    d_tau: [&[f64]; 2],
) -> [crate::linalg::Matrix; 2] {
    use crate::grid::eval::eval_aos_full;
    use crate::linalg::Matrix;
    let nao = mb.nao_sph;
    // V = phi^T A + A^T phi + sum_axis G^T (c/2) G with
    // A = (a / 2) phi + b . G.
    let partials: Vec<[Matrix; 2]> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let batch = eval_aos_full(mb, &grid.points[start..end], true, false);
            let g = batch.grad.as_ref().expect("gradients were requested");
            let npts = end - start;
            std::array::from_fn(|spin| {
                let mut a = Matrix::zeros(npts, nao);
                let mut scaled: [Matrix; 3] = std::array::from_fn(|_| Matrix::zeros(npts, nao));
                for i in 0..npts {
                    let point = start + i;
                    let dr = d_rho[spin][point];
                    let dg = d_gradient[spin][point];
                    let dt = d_tau[spin][point];
                    for nu in 0..nao {
                        a[(i, nu)] = 0.5 * dr * batch.phi[(i, nu)]
                            + dg[0] * g[0][(i, nu)]
                            + dg[1] * g[1][(i, nu)]
                            + dg[2] * g[2][(i, nu)];
                        for axis in 0..3 {
                            scaled[axis][(i, nu)] = 0.5 * dt * g[axis][(i, nu)];
                        }
                    }
                }
                let half = batch.phi.transpose_matmul_seq(&a);
                let mut v = Matrix::zeros(nao, nao);
                for mu in 0..nao {
                    for nu in 0..nao {
                        v[(mu, nu)] = half[(mu, nu)] + half[(nu, mu)];
                    }
                }
                for axis in 0..3 {
                    let kinetic = g[axis].transpose_matmul_seq(&scaled[axis]);
                    for (target, value) in v.as_mut_slice().iter_mut().zip(kinetic.as_slice()) {
                        *target += *value;
                    }
                }
                v
            })
        })
        .collect();
    let mut total = [Matrix::zeros(nao, nao), Matrix::zeros(nao, nao)];
    for partial in partials {
        for spin in 0..2 {
            for (target, value) in total[spin]
                .as_mut_slice()
                .iter_mut()
                .zip(partial[spin].as_slice())
            {
                *target += *value;
            }
        }
    }
    total
}

/// Skala energy and Kohn-Sham matrices of the spin density matrices `p` on
/// the molecular grid:
/// `V^s_mn = sum_p [dE/drho_s phi_m phi_n + dE/dgrad rho_s . grad(phi_m phi_n)
/// + (1/2) dE/dtau_s grad phi_m . grad phi_n]`.
pub fn skala_fock(
    model: &SkalaModel,
    mol: &crate::system::Molecule,
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    p: [&crate::linalg::Matrix; 2],
) -> Result<(f64, [crate::linalg::Matrix; 2])> {
    let ranges = batch_ranges(grid.len());
    let out = evaluate_on_grid(model, mol, mb, grid, p, &ranges)?;
    let matrices = assemble(
        mb,
        grid,
        &ranges,
        [&out.d_rho[0], &out.d_rho[1]],
        [&out.d_gradient[0], &out.d_gradient[1]],
        [&out.d_tau[0], &out.d_tau[1]],
    );
    Ok((out.energy, matrices))
}

/// Response of the Skala Kohn-Sham matrices to a change `dp` of the spin
/// density matrices, `d/dt V[P + t dP]` at `t = 0`: the second derivatives
/// of the energy contracted with the density response.
///
/// The reverse pass that gives the first derivatives is run on numbers that
/// carry the directional derivative along the density response, so the
/// product is exact for the non-local layers as well and costs about two
/// evaluations of the potential. Only the symmetric part of `dp` changes the
/// densities.
pub fn skala_fxc_contraction(
    model: &SkalaModel,
    mol: &crate::system::Molecule,
    mb: &crate::basis::MolBasis,
    grid: &crate::grid::MolecularGrid,
    p: [&crate::linalg::Matrix; 2],
    dp: [&crate::linalg::Matrix; 2],
) -> Result<[crate::linalg::Matrix; 2]> {
    use crate::linalg::Matrix;
    let ranges = batch_ranges(grid.len());
    let symmetric: [Matrix; 2] = std::array::from_fn(|spin| {
        let source = dp[spin];
        let mut out = Matrix::zeros(source.rows, source.cols);
        for i in 0..source.rows {
            for j in 0..source.cols {
                out[(i, j)] = 0.5 * (source[(i, j)] + source[(j, i)]);
            }
        }
        out
    });
    let (rho, gradient, tau) = grid_densities(mb, grid, p, &ranges);
    let (d_rho, d_gradient, d_tau) =
        grid_densities(mb, grid, [&symmetric[0], &symmetric[1]], &ranges);
    let joined = |value: f64, tangent: f64| Tangent {
        re: value,
        im: [tangent],
    };
    let scalars = |values: &[f64], tangents: &[f64]| -> Vec<Tangent> {
        values
            .iter()
            .zip(tangents)
            .map(|(v, t)| joined(*v, *t))
            .collect()
    };
    let rho_t: [Vec<Tangent>; 2] = std::array::from_fn(|s| scalars(&rho[s], &d_rho[s]));
    let tau_t: [Vec<Tangent>; 2] = std::array::from_fn(|s| scalars(&tau[s], &d_tau[s]));
    let gradient_t: [Vec<[Tangent; 3]>; 2] = std::array::from_fn(|s| {
        gradient[s]
            .iter()
            .zip(&d_gradient[s])
            .map(|(v, t)| std::array::from_fn(|axis| joined(v[axis], t[axis])))
            .collect()
    });
    let (points, centers) = grid_geometry(mol, grid);
    let weights: Vec<Tangent> = grid.weights.iter().map(|w| joined(*w, 0.0)).collect();
    let out = model.evaluate(
        &SkalaInput {
            atom_of: &grid.atom_of,
            points: &points,
            centers: &centers,
            weights: &weights,
            atomic_weights: &grid.raw_weights,
            rho: [&rho_t[0], &rho_t[1]],
            gradient: [&gradient_t[0], &gradient_t[1]],
            tau: [&tau_t[0], &tau_t[1]],
        },
        true,
    )?;
    let slopes = |values: &[Tangent]| -> Vec<f64> { values.iter().map(|v| v.im[0]).collect() };
    let v_rho = [slopes(&out.d_rho[0]), slopes(&out.d_rho[1])];
    let v_tau = [slopes(&out.d_tau[0]), slopes(&out.d_tau[1])];
    let v_gradient: [Vec<[f64; 3]>; 2] = std::array::from_fn(|s| {
        out.d_gradient[s]
            .iter()
            .map(|g| [g[0].im[0], g[1].im[0], g[2].im[0]])
            .collect()
    });
    Ok(assemble(
        mb,
        grid,
        &ranges,
        [&v_rho[0], &v_rho[1]],
        [&v_gradient[0], &v_gradient[1]],
        [&v_tau[0], &v_tau[1]],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: SkalaSizes = SkalaSizes {
        width: 10,
        channels: 3,
        shells: 2,
        correlation: 2,
        layers: 1,
        post: 3,
    };
    const LARGE: SkalaSizes = SkalaSizes {
        width: 12,
        channels: 4,
        shells: 3,
        correlation: 3,
        layers: 2,
        post: 5,
    };

    /// Densities on a small two-atom grid.
    #[derive(Clone)]
    struct Fields {
        atom_of: Vec<usize>,
        points: Vec<[f64; 3]>,
        centers: Vec<[f64; 3]>,
        weights: Vec<f64>,
        atomic_weights: Vec<f64>,
        rho: [Vec<f64>; 2],
        gradient: [Vec<[f64; 3]>; 2],
        tau: [Vec<f64>; 2],
    }

    impl Fields {
        fn new(seed: u64, per_atom: usize) -> Self {
            let mut rng = Lcg(seed);
            let centers = vec![[0.0, 0.0, 0.0], [1.4, 0.3, -0.2]];
            let n = 2 * per_atom;
            let mut out = Self {
                atom_of: Vec::new(),
                points: Vec::new(),
                centers: centers.clone(),
                weights: Vec::new(),
                atomic_weights: Vec::new(),
                rho: [Vec::new(), Vec::new()],
                gradient: [Vec::new(), Vec::new()],
                tau: [Vec::new(), Vec::new()],
            };
            for p in 0..n {
                let atom = p / per_atom;
                out.atom_of.push(atom);
                out.points.push([
                    centers[atom][0] + 2.4 * rng.next(),
                    centers[atom][1] + 2.4 * rng.next(),
                    centers[atom][2] + 2.4 * rng.next(),
                ]);
                let atomic = 0.6 + rng.next();
                out.atomic_weights.push(atomic);
                out.weights.push(atomic * (0.5 + 0.4 * rng.next()));
                for spin in 0..2 {
                    out.rho[spin].push(0.5 + 0.4 * rng.next());
                    out.gradient[spin].push([0.6 * rng.next(), 0.6 * rng.next(), 0.6 * rng.next()]);
                    out.tau[spin].push(0.35 + 0.3 * rng.next());
                }
            }
            out
        }

        fn input(&self) -> SkalaInput<'_> {
            SkalaInput {
                atom_of: &self.atom_of,
                points: &self.points,
                centers: &self.centers,
                weights: &self.weights,
                atomic_weights: &self.atomic_weights,
                rho: [&self.rho[0], &self.rho[1]],
                gradient: [&self.gradient[0], &self.gradient[1]],
                tau: [&self.tau[0], &self.tau[1]],
            }
        }

        /// The fields moved by `t` along `direction`.
        fn moved(&self, direction: &Fields, t: f64) -> Self {
            let mut out = self.clone();
            for spin in 0..2 {
                for p in 0..self.points.len() {
                    out.rho[spin][p] += t * direction.rho[spin][p];
                    out.tau[spin][p] += t * direction.tau[spin][p];
                    for axis in 0..3 {
                        out.gradient[spin][p][axis] += t * direction.gradient[spin][p][axis];
                    }
                }
            }
            out
        }

        fn swapped(&self) -> Self {
            let mut out = self.clone();
            out.rho.swap(0, 1);
            out.gradient.swap(0, 1);
            out.tau.swap(0, 1);
            out
        }
    }

    /// Log-log least squares `log|e| = log C + p log h`: exponent, R^2.
    fn power_fit(h: &[f64], errors: &[f64]) -> (f64, f64) {
        let x: Vec<f64> = h.iter().map(|v| v.ln()).collect();
        let y: Vec<f64> = errors.iter().map(|e| e.abs().ln()).collect();
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
        (sxy / sxx, sxy * sxy / (sxx * syy))
    }

    #[test]
    fn derivatives_pass_a_four_step_fit_for_two_architectures() {
        for (label, sizes) in [("small", SMALL), ("large", LARGE)] {
            let model = SkalaModel::from_tensors(&random_tensors(11, sizes)).unwrap();
            println!("{label}: {}", model.description());
            let fields = Fields::new(5, 9);
            let direction = Fields::new(23, 9);
            let reference = model.evaluate(&fields.input(), true).unwrap();
            let mut analytic = 0.0;
            for spin in 0..2 {
                for p in 0..fields.points.len() {
                    analytic += reference.d_rho[spin][p] * direction.rho[spin][p]
                        + reference.d_tau[spin][p] * direction.tau[spin][p]
                        + dot(reference.d_gradient[spin][p], direction.gradient[spin][p]);
                }
            }
            let energy = |t: f64| {
                model
                    .evaluate(&fields.moved(&direction, t).input(), false)
                    .unwrap()
                    .energy
            };
            let steps: [f64; 4] = [0.04, 0.02, 0.01, 0.005];
            let differences: Vec<f64> = steps
                .iter()
                .map(|&t| (energy(t) - energy(-t)) / (2.0 * t))
                .collect();
            let errors: Vec<f64> = differences.iter().map(|d| d - analytic).collect();
            let (p, r2) = power_fit(&steps, &errors);
            let richardson = (4.0 * differences[3] - differences[2]) / 3.0 - analytic;
            println!(
                "{label}: energy {:.12}, directional derivative {analytic:.12e}, errors {errors:?}, t 0.005..0.04, p {p:.6}, R^2 {r2:.8}, Richardson difference {richardson:e}",
                reference.energy
            );
            assert!((p - 2.0).abs() < 0.05 && r2 > 0.9999, "{label}");
            assert!(
                richardson.abs() < 1e-6 * analytic.abs().max(1e-3),
                "{label}"
            );
        }
    }

    #[test]
    fn energy_is_spin_symmetric_and_derivatives_swap() {
        let model = SkalaModel::from_tensors(&random_tensors(3, LARGE)).unwrap();
        let fields = Fields::new(7, 8);
        let direct = model.evaluate(&fields.input(), true).unwrap();
        let swapped = model.evaluate(&fields.swapped().input(), true).unwrap();
        println!(
            "energy {:.14}, swapped {:.14}",
            direct.energy, swapped.energy
        );
        assert!((direct.energy - swapped.energy).abs() < 1e-13 * direct.energy.abs().max(1.0));
        for p in 0..fields.points.len() {
            for spin in 0..2 {
                assert!((direct.d_rho[spin][p] - swapped.d_rho[1 - spin][p]).abs() < 1e-12);
                assert!((direct.d_tau[spin][p] - swapped.d_tau[1 - spin][p]).abs() < 1e-12);
                for axis in 0..3 {
                    assert!(
                        (direct.d_gradient[spin][p][axis] - swapped.d_gradient[1 - spin][p][axis])
                            .abs()
                            < 1e-12
                    );
                }
            }
        }
    }

    #[test]
    fn message_passing_couples_the_points_of_one_atom_only() {
        let model = SkalaModel::from_tensors(&random_tensors(17, LARGE)).unwrap();
        let fields = Fields::new(9, 8);
        let base = model.evaluate(&fields.input(), false).unwrap();
        // Change the density at the first point of the first atom.
        let mut changed = fields.clone();
        changed.rho[0][0] += 0.05;
        let other = model.evaluate(&changed.input(), false).unwrap();
        let ratio = |p: usize, out: &SkalaOutput| -> f64 {
            // The enhancement factor: energy density over its local part.
            out.energy_density[p]
                / (fields.rho[0][p].powf(4.0 / 3.0) + fields.rho[1][p].powf(4.0 / 3.0))
        };
        let same_atom = (1..8)
            .map(|p| (ratio(p, &other) - ratio(p, &base)).abs())
            .fold(0.0_f64, f64::max);
        let other_atom = (8..16)
            .map(|p| (other.energy_density[p] - base.energy_density[p]).abs())
            .fold(0.0_f64, f64::max);
        println!("largest change on the same atom {same_atom:e}, on the other atom {other_atom:e}");
        assert!(same_atom > 1e-8);
        assert_eq!(other_atom, 0.0);
        // The enhancement factor stays inside the scaled sigmoid.
        for p in 0..16 {
            let factor = ratio(p, &base) / -0.9305257363491001;
            assert!(factor > 0.0 && factor < 2.0, "point {p}: {factor}");
        }
    }

    #[test]
    fn vanishing_density_has_no_energy_and_finite_derivatives() {
        let model = SkalaModel::from_tensors(&random_tensors(29, SMALL)).unwrap();
        let mut fields = Fields::new(13, 6);
        for spin in 0..2 {
            for p in 0..fields.points.len() {
                fields.rho[spin][p] = 0.0;
                fields.tau[spin][p] = 0.0;
                fields.gradient[spin][p] = [0.0; 3];
            }
        }
        let out = model.evaluate(&fields.input(), true).unwrap();
        assert_eq!(out.energy, 0.0);
        assert!(out
            .d_rho
            .iter()
            .flatten()
            .chain(out.d_tau.iter().flatten())
            .all(|v| v.is_finite()));
        assert!(out.d_gradient.iter().flatten().flatten().all(|v| *v == 0.0));
    }

    #[test]
    fn model_files_round_trip_and_reject_inconsistent_sizes() {
        let tensors = random_tensors(41, LARGE);
        let path = std::env::temp_dir().join(format!("dft_rs_skala_{}.model", std::process::id()));
        write_tensors(&path, &tensors).unwrap();
        let back = read_tensors(&path).unwrap();
        assert_eq!(back, tensors);
        let loaded = SkalaModel::load(&path).unwrap();
        let fields = Fields::new(3, 5);
        let direct = SkalaModel::from_tensors(&tensors).unwrap();
        assert_eq!(
            loaded.evaluate(&fields.input(), false).unwrap().energy,
            direct.evaluate(&fields.input(), false).unwrap().energy
        );
        // A truncated file is refused.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 5]).unwrap();
        assert!(read_tensors(&path).is_err());
        let _ = std::fs::remove_file(&path);
        // A missing tensor and a mismatched width are named.
        let mut missing = tensors.clone();
        missing.remove("nonlocal.1.up.2");
        let error = SkalaModel::from_tensors(&missing).unwrap_err().to_string();
        assert!(error.contains("nonlocal.1.up.2"), "{error}");
        let mut wrong = tensors.clone();
        wrong.insert(
            "output.1.weight".into(),
            Tensor {
                shape: vec![1, 5],
                data: vec![0.1; 5],
            },
        );
        assert!(SkalaModel::from_tensors(&wrong).is_err());
        // Sizes come from the file: the two synthetic architectures differ.
        let small = SkalaModel::from_tensors(&random_tensors(41, SMALL)).unwrap();
        assert_eq!((small.width(), direct.width()), (10, 12));
    }

    #[test]
    fn kohn_sham_matrices_pass_a_four_step_fit() {
        use crate::basis::{data, MolBasis};
        use crate::grid::{GridOptions, MolecularGrid};
        use crate::linalg::Matrix;
        use crate::math::Vec3;
        use crate::system::{Atom, Molecule};
        let molecule = Molecule::new(vec![
            Atom {
                z: 8,
                position: Vec3::new(0.0, 0.0, 0.0),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, 1.43, 1.11),
            },
            Atom {
                z: 1,
                position: Vec3::new(0.0, -1.43, 1.11),
            },
        ]);
        let basis = MolBasis::build(&molecule, &data::load("sto-3g").unwrap()).unwrap();
        let grid = MolecularGrid::build(
            &molecule,
            &GridOptions {
                level: 1,
                ..GridOptions::default()
            },
        )
        .unwrap();
        let model = SkalaModel::from_tensors(&random_tensors(53, LARGE)).unwrap();
        let nao = basis.nao_sph;
        let mut rng = Lcg(77);
        let mut symmetric = |diagonal: f64, scale: f64| {
            let mut m = Matrix::zeros(nao, nao);
            for i in 0..nao {
                for j in 0..=i {
                    let value = scale * rng.next() + if i == j { diagonal } else { 0.0 };
                    m[(i, j)] = value;
                    m[(j, i)] = value;
                }
            }
            m
        };
        // Positive definite density matrices and a symmetric direction.
        let p = [symmetric(0.5, 0.1), symmetric(0.4, 0.1)];
        let d = [symmetric(0.0, 0.2), symmetric(0.0, 0.2)];
        let (energy, v) = skala_fock(&model, &molecule, &basis, &grid, [&p[0], &p[1]]).unwrap();
        let analytic: f64 = (0..2).map(|s| v[s].frobenius_dot(&d[s])).sum();
        let moved = |t: f64| -> f64 {
            let shifted: Vec<Matrix> = (0..2)
                .map(|s| {
                    let mut m = p[s].clone();
                    for (a, b) in m.as_mut_slice().iter_mut().zip(d[s].as_slice()) {
                        *a += t * b;
                    }
                    m
                })
                .collect();
            skala_fock(&model, &molecule, &basis, &grid, [&shifted[0], &shifted[1]])
                .unwrap()
                .0
        };
        let steps: [f64; 4] = [0.04, 0.02, 0.01, 0.005];
        let differences: Vec<f64> = steps
            .iter()
            .map(|&t| (moved(t) - moved(-t)) / (2.0 * t))
            .collect();
        let errors: Vec<f64> = differences.iter().map(|x| x - analytic).collect();
        let (order, r2) = power_fit(&steps, &errors);
        let richardson = (4.0 * differences[3] - differences[2]) / 3.0 - analytic;
        println!(
            "Kohn-Sham matrices on {} grid points: energy {energy:.12}, tr(V dP) {analytic:.12e}, errors {errors:?}, t 0.005..0.04, p {order:.6}, R^2 {r2:.8}, Richardson difference {richardson:e}",
            grid.len()
        );
        assert!((order - 2.0).abs() < 0.05 && r2 > 0.9999);
        assert!(richardson.abs() < 1e-6 * analytic.abs().max(1e-3));
        for matrix in &v {
            for i in 0..nao {
                for j in 0..nao {
                    assert!((matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-12);
                }
            }
        }
        // The kernel contraction is the derivative of the matrices along a
        // change of the density matrices.
        let change = [symmetric(0.0, 0.05), symmetric(0.0, 0.05)];
        let reference = [symmetric(0.35, 0.04), symmetric(0.3, 0.04)];
        let kernel = skala_fxc_contraction(
            &model,
            &molecule,
            &basis,
            &grid,
            [&reference[0], &reference[1]],
            [&change[0], &change[1]],
        )
        .unwrap();
        let shifted = |h: f64| -> [Matrix; 2] {
            let moved: [Matrix; 2] = std::array::from_fn(|spin| {
                let mut m = reference[spin].clone();
                for (value, step) in m.as_mut_slice().iter_mut().zip(change[spin].as_slice()) {
                    *value += h * step;
                }
                m
            });
            skala_fock(&model, &molecule, &basis, &grid, [&moved[0], &moved[1]])
                .unwrap()
                .1
        };
        let kernel_steps = [0.08, 0.04, 0.02, 0.01];
        let kernel_errors: Vec<f64> = kernel_steps
            .iter()
            .map(|&h| {
                let (plus, minus) = (shifted(h), shifted(-h));
                let mut worst = 0.0_f64;
                for spin in 0..2 {
                    for ((a, b), k) in plus[spin]
                        .as_slice()
                        .iter()
                        .zip(minus[spin].as_slice())
                        .zip(kernel[spin].as_slice())
                    {
                        worst = worst.max(((a - b) / (2.0 * h) - k).abs());
                    }
                }
                worst
            })
            .collect();
        let (kernel_p, kernel_r2) = power_fit(&kernel_steps, &kernel_errors);
        let mut kernel_asymmetry = 0.0_f64;
        let mut kernel_scale = 0.0_f64;
        for spin in 0..2 {
            for i in 0..nao {
                for j in 0..nao {
                    kernel_asymmetry =
                        kernel_asymmetry.max((kernel[spin][(i, j)] - kernel[spin][(j, i)]).abs());
                    kernel_scale = kernel_scale.max(kernel[spin][(i, j)].abs());
                }
            }
        }
        println!(
            "kernel contraction: steps {kernel_steps:?} largest matrix-element errors {kernel_errors:?} p={kernel_p:.6} R2={kernel_r2:.8}; largest element {kernel_scale:e}, asymmetry {kernel_asymmetry:e}"
        );
        assert!((kernel_p - 2.0).abs() < 0.1 && kernel_r2 > 0.999);
        assert!(kernel_asymmetry < 1e-10 * kernel_scale.max(1.0));
    }
}
