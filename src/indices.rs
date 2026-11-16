// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Index kind is checked statically; space ownership and range are checked
//! at runtime. Rebuilding a space creates a new identity even at equal size.
//! IDs are process-local capabilities, not serialized persistent identifiers.
//!
//! ```compile_fail
//! use dft_rs::indices::{Atom, AtomicOrbital, IndexSpace};
//! let atoms = IndexSpace::<Atom>::new(2).unwrap();
//! let aos = IndexSpace::<AtomicOrbital>::new(2).unwrap();
//! aos.resolve(atoms.id(0).unwrap()).unwrap();
//! ```

use crate::{DftError, Result};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SPACE: AtomicU64 = AtomicU64::new(1);

macro_rules! index_kinds {
    ($(($kind:ident, $id:ident)),+ $(,)?) => {$(
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $kind;
        pub type $id = TypedId<$kind>;
    )+};
}
index_kinds!(
    (Atom, AtomId),
    (AtomicOrbital, AoId),
    (Shell, ShellId),
    (AuxiliaryOrbital, AuxAoId),
    (MolecularOrbital, MoId),
    (AlphaOrbital, AlphaOrbitalId),
    (BetaOrbital, BetaOrbitalId),
    (Spinor, SpinorId)
);

/// A spinor is an orbital with both components, not an alpha or beta orbital.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpinComponent {
    Alpha,
    Beta,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TypedId<K> {
    owner: u64,
    index: usize,
    kind: PhantomData<K>,
}

/// A clone denotes the same immutable space; constructing a new space denotes
/// a new owner/generation. No raw integer can be cast into a valid typed ID.
#[derive(Clone, Debug)]
pub struct IndexSpace<K> {
    owner: u64,
    len: usize,
    kind: PhantomData<K>,
}
impl<K> IndexSpace<K> {
    pub fn new(len: usize) -> Result<Self> {
        let owner = NEXT_SPACE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| DftError::InvalidInput("index-space identity counter exhausted".into()))?;
        Ok(Self {
            owner,
            len,
            kind: PhantomData,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn id(&self, index: usize) -> Result<TypedId<K>> {
        if index >= self.len {
            return Err(DftError::InvalidInput(format!(
                "index {index} is outside a space of length {}",
                self.len
            )));
        }
        Ok(TypedId {
            owner: self.owner,
            index,
            kind: PhantomData,
        })
    }
    /// Resolve once at a checked boundary, before the numerical kernel loop.
    pub fn resolve(&self, id: TypedId<K>) -> Result<usize> {
        if id.owner != self.owner || id.index >= self.len {
            return Err(DftError::InvalidInput(
                "index belongs to a different owner or generation".into(),
            ));
        }
        Ok(id.index)
    }
}
