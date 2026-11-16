// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! Named calculation controls shared by every surface.
//!
//! Python keyword arguments, CLI `--key value` flags and saved JSON requests
//! are all reduced to one ordered key/value list. Keys are normalized to
//! `snake_case` (`--grid-level` and `grid_level` are the same control).
//! Each getter marks its key as consumed, and [`Controls::finish`] refuses any
//! control that no part of the calculation read, naming it, so a misspelled or
//! inapplicable option can never be silently ignored.

use super::value::Value;
use crate::math::Vec3;
use crate::{DftError, Result};
use std::cell::RefCell;

#[derive(Clone, Debug, Default)]
pub struct Controls {
    entries: Vec<(String, Value)>,
    used: RefCell<Vec<bool>>,
}

pub fn normalize_key(key: &str) -> String {
    key.trim_start_matches('-')
        .replace('-', "_")
        .to_ascii_lowercase()
}

impl Controls {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from a JSON-like object. Duplicate keys after normalization are
    /// refused.
    pub fn from_value(value: &Value) -> Result<Self> {
        let Value::Object(entries) = value else {
            return Err(DftError::InvalidInput(
                "calculation controls must be a key/value object".into(),
            ));
        };
        let mut controls = Self::new();
        for (key, value) in entries {
            controls.insert(key, value.clone())?;
        }
        Ok(controls)
    }

    pub fn insert(&mut self, key: &str, value: Value) -> Result<()> {
        let key = normalize_key(key);
        if key.is_empty() {
            return Err(DftError::InvalidInput("empty control name".into()));
        }
        if self.entries.iter().any(|(k, _)| *k == key) {
            return Err(DftError::InvalidInput(format!(
                "control `{key}` was given more than once"
            )));
        }
        self.entries.push((key, value));
        self.used.borrow_mut().push(false);
        Ok(())
    }

    /// Parse CLI arguments of the form `--key value`, `--key=value` or a bare
    /// `--flag` (boolean true). Values that read as JSON scalars or arrays
    /// keep that type; `a,b,c` becomes a numeric array and `a,b;c,d` a list
    /// of rows; anything else is a string.
    pub fn from_cli_args(args: &[String]) -> Result<Self> {
        let mut controls = Self::new();
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if !arg.starts_with("--") || arg.len() < 3 {
                return Err(DftError::InvalidInput(format!(
                    "unexpected argument `{arg}`; controls are given as --name value"
                )));
            }
            let (key, value) = if let Some((k, v)) = arg.split_once('=') {
                (k.to_string(), parse_cli_value(v))
            } else if i + 1 < args.len() && !is_flag(&args[i + 1]) {
                i += 1;
                (arg.clone(), parse_cli_value(&args[i]))
            } else {
                (arg.clone(), Value::Bool(true))
            };
            controls.insert(&key, value)?;
            i += 1;
        }
        Ok(controls)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.index(key).is_some()
    }

    /// A feature-local view that preserves consumption state and leaves the
    /// original request available for provenance and serialization.
    pub(crate) fn excluding(&self, names: &[&str]) -> Self {
        let used = self.used.borrow();
        let mut filtered = Self::new();
        for ((key, value), &consumed) in self.entries.iter().zip(used.iter()) {
            if !names.contains(&key.as_str()) {
                filtered.entries.push((key.clone(), value.clone()));
                filtered.used.get_mut().push(consumed);
            }
        }
        filtered
    }

    fn index(&self, key: &str) -> Option<usize> {
        let key = normalize_key(key);
        self.entries.iter().position(|(k, _)| *k == key)
    }

    /// Inspect a capability selector without consuming the requested control.
    pub(crate) fn peek(&self, key: &str) -> Option<&Value> {
        self.index(key).map(|i| &self.entries[i].1)
    }

    /// Inspect a string selector without accepting it on behalf of a handler.
    /// Unlike `opt_str`, selectors never accept numeric aliases.
    pub(crate) fn peek_str(&self, key: &str) -> Result<Option<&str>> {
        match self.peek(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Str(value)) => Ok(Some(value)),
            Some(value) => Err(Self::type_error(key, "a string selector", value)),
        }
    }

    /// Begin a fresh feature request after enclosing controls are stripped.
    /// Reset bookkeeping without cloning model, matrix or geometry payloads.
    pub(crate) fn reset_usage(&self) {
        self.used.borrow_mut().fill(false);
    }

    /// Refuse a selector that no reference-specific handler has accepted.
    pub(crate) fn require_consumed(&self, key: &str, context: &str) -> Result<()> {
        if let Some(i) = self.index(key) {
            if !self.used.borrow()[i] && !matches!(self.entries[i].1, Value::Null) {
                return Err(DftError::Unsupported {
                    feature: context.into(),
                    missing: format!(
                        "a handler for control `{key}`={}; the requested model cannot be replaced by the default",
                        self.entries[i].1.to_json()
                    ),
                });
            }
        }
        Ok(())
    }

    /// Raw access; marks the key consumed.
    pub fn take(&self, key: &str) -> Option<&Value> {
        let i = self.index(key)?;
        self.used.borrow_mut()[i] = true;
        Some(&self.entries[i].1)
    }

    fn type_error(key: &str, expected: &str, value: &Value) -> DftError {
        DftError::InvalidInput(format!(
            "control `{}` must be {expected}, got {}",
            normalize_key(key),
            value.to_json()
        ))
    }

    pub fn opt_f64(&self, key: &str) -> Result<Option<f64>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) if x.is_finite() => Ok(Some(x)),
                _ => Err(Self::type_error(key, "a finite number", v)),
            },
        }
    }

    pub fn f64(&self, key: &str, default: f64) -> Result<f64> {
        Ok(self.opt_f64(key)?.unwrap_or(default))
    }

    pub fn opt_i64(&self, key: &str) -> Result<Option<i64>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_i64()
                .map(Some)
                .ok_or_else(|| Self::type_error(key, "an integer", v)),
        }
    }

    pub fn opt_usize(&self, key: &str) -> Result<Option<usize>> {
        match self.opt_i64(key)? {
            None => Ok(None),
            Some(v) if v >= 0 => Ok(Some(v as usize)),
            Some(v) => Err(Self::type_error(
                key,
                "a non-negative integer",
                &Value::Int(v),
            )),
        }
    }

    /// Global resource options share Python/CLI's strict integer contract;
    /// legacy feature controls retain their existing integral-float grammar.
    pub(crate) fn opt_strict_usize(&self, key: &str) -> Result<Option<usize>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Int(value)) => usize::try_from(*value).map(Some).map_err(|_| {
                Self::type_error(
                    key,
                    "a non-negative integer representable as usize",
                    &Value::Int(*value),
                )
            }),
            Some(value) => Err(Self::type_error(key, "a non-negative integer", value)),
        }
    }

    pub fn usize(&self, key: &str, default: usize) -> Result<usize> {
        let value = self.opt_usize(key)?.unwrap_or(default);
        Ok(
            if matches!(
                normalize_key(key).as_str(),
                "incore_mb" | "pt2_memory_mb" | "td_memory_mb"
            ) {
                let capped = crate::resources::component_limit_mb(value);
                if value > 0 && capped == 0 {
                    if let Some(usage) =
                        crate::resources::current_usage().filter(|usage| usage.memory_mb.is_some())
                    {
                        let needed = usage.reserved_bytes.saturating_add(crate::resources::MIB);
                        return Err(DftError::ResourceLimit {
                            operation: "remaining component workspace",
                            required_mb: needed / crate::resources::MIB
                                + usize::from(needed % crate::resources::MIB != 0),
                            limit_mb: usage.memory_mb.expect("memory budget checked"),
                        });
                    }
                }
                capped
            } else {
                value
            },
        )
    }

    pub fn bool(&self, key: &str, default: bool) -> Result<bool> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(default),
            Some(v) => match v {
                Value::Bool(b) => Ok(*b),
                Value::Str(s) if s.eq_ignore_ascii_case("true") => Ok(true),
                Value::Str(s) if s.eq_ignore_ascii_case("false") => Ok(false),
                _ => Err(Self::type_error(key, "true or false", v)),
            },
        }
    }

    pub fn opt_str(&self, key: &str) -> Result<Option<String>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Str(s)) => Ok(Some(s.clone())),
            // A bare number given where a name is expected (for example a
            // basis named "321") is still a name.
            Some(Value::Int(v)) => Ok(Some(v.to_string())),
            Some(v) => Err(Self::type_error(key, "a string", v)),
        }
    }

    pub fn str(&self, key: &str, default: &str) -> Result<String> {
        Ok(self.opt_str(key)?.unwrap_or_else(|| default.to_string()))
    }

    pub fn opt_f64_list(&self, key: &str) -> Result<Option<Vec<f64>>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                let list = match v {
                    Value::Array(items) => {
                        items.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()
                    }
                    other => other.as_f64().map(|x| vec![x]),
                };
                match list {
                    Some(list) if list.iter().all(|x| x.is_finite()) => Ok(Some(list)),
                    _ => Err(Self::type_error(key, "a list of finite numbers", v)),
                }
            }
        }
    }

    pub fn opt_vec3(&self, key: &str) -> Result<Option<Vec3>> {
        match self.opt_f64_list(key)? {
            None => Ok(None),
            Some(v) if v.len() == 3 => Ok(Some(Vec3::new(v[0], v[1], v[2]))),
            Some(v) => Err(Self::type_error(
                key,
                "three numbers",
                &Value::Array(v.into_iter().map(Value::Float).collect()),
            )),
        }
    }

    pub fn opt_usize_list(&self, key: &str) -> Result<Option<Vec<usize>>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                let list = match v {
                    Value::Array(items) => items
                        .iter()
                        .map(|x| x.as_i64().filter(|&i| i >= 0).map(|i| i as usize))
                        .collect::<Option<Vec<_>>>(),
                    other => other.as_i64().filter(|&i| i >= 0).map(|i| vec![i as usize]),
                };
                list.map(Some)
                    .ok_or_else(|| Self::type_error(key, "a list of non-negative integers", v))
            }
        }
    }

    /// A list of numeric rows, for example a 3x3 cell or per-atom vectors.
    pub fn opt_rows(&self, key: &str) -> Result<Option<Vec<Vec<f64>>>> {
        match self.take(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => {
                let rows = v.as_array().and_then(|items| {
                    items
                        .iter()
                        .map(|row| {
                            row.as_array()?
                                .iter()
                                .map(|x| x.as_f64().filter(|x| x.is_finite()))
                                .collect::<Option<Vec<_>>>()
                        })
                        .collect::<Option<Vec<_>>>()
                });
                rows.map(Some)
                    .ok_or_else(|| Self::type_error(key, "a list of numeric rows", v))
            }
        }
    }

    /// Refuse every control that was supplied but never read. Unknown names
    /// are input errors; known controls unavailable in this context are
    /// unsupported capabilities. The shared unit registry owns known names.
    pub fn finish(&self, context: &str) -> Result<()> {
        let used = self.used.borrow();
        let unused: Vec<&str> = self
            .entries
            .iter()
            .zip(used.iter())
            .filter(|(_, &u)| !u)
            .map(|((k, _), _)| k.as_str())
            .collect();
        let unknown: Vec<_> = unused
            .iter()
            .copied()
            .filter(|name| {
                !super::units::CONTROL_UNITS
                    .iter()
                    .any(|(key, _)| key == name)
            })
            .collect();
        if !unknown.is_empty() {
            return Err(DftError::InvalidInput(format!(
                "{context} does not accept unknown controls: {}",
                unknown.join(", ")
            )));
        }
        if !unused.is_empty() {
            return Err(DftError::Unsupported {
                feature: context.into(),
                missing: format!(
                    "support for controls in this context: {}",
                    unused.join(", ")
                ),
            });
        }
        Ok(())
    }

    /// The supplied controls, for provenance records.
    pub fn to_value(&self) -> Value {
        Value::Object(self.entries.clone())
    }
}

fn is_flag(arg: &str) -> bool {
    // A negative number is a value, not a flag.
    arg.starts_with("--") && arg.len() > 2
}

fn parse_scalar(text: &str) -> Option<Value> {
    let t = text.trim();
    if let Ok(v) = t.parse::<i64>() {
        return Some(Value::Int(v));
    }
    t.parse::<f64>()
        .ok()
        .filter(|x| x.is_finite())
        .map(Value::Float)
}

/// Interpret one CLI value.
pub fn parse_cli_value(text: &str) -> Value {
    let t = text.trim();
    match t {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        _ => {}
    }
    if t.starts_with('[') || t.starts_with('{') {
        if let Ok(v) = Value::parse_json(t) {
            return v;
        }
    }
    if let Some(v) = parse_scalar(t) {
        return v;
    }
    if t.contains(';') {
        let rows: Option<Vec<Value>> = t
            .split(';')
            .map(|row| {
                row.split(',')
                    .map(parse_scalar)
                    .collect::<Option<Vec<_>>>()
                    .map(Value::Array)
            })
            .collect();
        if let Some(rows) = rows {
            return Value::Array(rows);
        }
    }
    if t.contains(',') {
        if let Some(items) = t.split(',').map(parse_scalar).collect::<Option<Vec<_>>>() {
            return Value::Array(items);
        }
    }
    Value::Str(t.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn cli_values_are_typed_and_keys_normalized() {
        let c = Controls::from_cli_args(&args(&[
            "--xc",
            "pbe",
            "--grid-level",
            "4",
            "--field",
            "0,0,-0.01",
            "--cell",
            "5,0,0;0,5,0;0,0,5",
            "--unrestricted",
            "--conv-tol=1e-10",
        ]))
        .unwrap();
        assert_eq!(c.str("xc", "hf").unwrap(), "pbe");
        assert_eq!(c.usize("grid_level", 3).unwrap(), 4);
        assert_eq!(
            c.opt_vec3("field").unwrap(),
            Some(Vec3::new(0.0, 0.0, -0.01))
        );
        assert_eq!(c.opt_rows("cell").unwrap().unwrap()[2], vec![0.0, 0.0, 5.0]);
        assert!(c.bool("unrestricted", false).unwrap());
        assert_eq!(c.f64("conv_tol", 1e-9).unwrap(), 1e-10);
        c.finish("test").unwrap();
    }

    #[test]
    fn unused_and_duplicate_controls_are_refused_by_name() {
        let c = Controls::from_cli_args(&args(&["--xc", "pbe", "--grid-levl", "4"])).unwrap();
        c.str("xc", "hf").unwrap();
        let error = c.finish("energy").unwrap_err().to_string();
        assert!(error.contains("grid_levl"), "{error}");
        assert!(Controls::from_cli_args(&args(&["--xc", "a", "--xc", "b"])).is_err());
        assert!(
            Controls::from_cli_args(&args(&["--grid-level", "1", "--grid_level", "2"])).is_err()
        );
    }

    #[test]
    fn wrong_types_are_refused() {
        let c =
            Controls::from_cli_args(&args(&["--grid-level", "fine", "--field", "1,2"])).unwrap();
        assert!(c.usize("grid_level", 3).is_err());
        assert!(c.opt_vec3("field").is_err());
        let negative = Controls::from_cli_args(&args(&["--charge", "-1"])).unwrap();
        assert_eq!(negative.opt_i64("charge").unwrap(), Some(-1));
    }

    #[test]
    fn capability_inspection_does_not_consume_controls() {
        let c = Controls::from_cli_args(&args(&["--reference", "roks"])).unwrap();
        assert_eq!(c.peek("--Reference").and_then(Value::as_str), Some("roks"));
        assert!(c.peek("absent").is_none());
        assert!(c
            .finish("preflight")
            .unwrap_err()
            .to_string()
            .contains("reference"));
        assert_eq!(c.opt_str("reference").unwrap().as_deref(), Some("roks"));
        c.finish("selected").unwrap();
    }

    #[test]
    fn known_inapplicable_controls_are_unsupported_but_typos_are_invalid() {
        let known = Controls::from_cli_args(&args(&["--xc", "pbe"])).unwrap();
        assert!(matches!(
            known.finish("data model"),
            Err(DftError::Unsupported { .. })
        ));
        let mixed = Controls::from_cli_args(&args(&["--xc", "pbe", "--grid-levl", "4"])).unwrap();
        assert!(matches!(
            mixed.finish("data model"),
            Err(DftError::InvalidInput(_))
        ));
        let wrong_type = Controls::from_cli_args(&args(&["--grid-level", "fine"])).unwrap();
        assert!(matches!(
            wrong_type.usize("grid_level", 1),
            Err(DftError::InvalidInput(_))
        ));
        known.str("xc", "hf").unwrap();
        known.finish("accepted model").unwrap();
    }

    #[test]
    fn selectors_must_be_strings_and_inspection_never_accepts_them() {
        let selected = Controls::from_cli_args(&args(&["--reference", "RoKs"])).unwrap();
        assert_eq!(selected.peek_str("reference").unwrap(), Some("RoKs"));
        assert!(matches!(
            selected.require_consumed("reference", "setup"),
            Err(DftError::Unsupported { .. })
        ));
        let copied = selected.excluding(&["cpu_threads"]);
        assert!(copied.require_consumed("reference", "setup").is_err());
        selected.str("reference", "auto").unwrap();
        selected.require_consumed("reference", "setup").unwrap();
        selected
            .excluding(&[])
            .require_consumed("reference", "setup")
            .unwrap();
        for value in [Value::Int(1), Value::Bool(true), Value::Array(Vec::new())] {
            let controls = Controls::from_value(&Value::object().with("reference", value)).unwrap();
            assert!(matches!(
                controls.peek_str("reference"),
                Err(DftError::InvalidInput(_))
            ));
        }
        let null = Controls::from_value(&Value::object().with("reference", Value::Null)).unwrap();
        assert_eq!(null.peek_str("reference").unwrap(), None);
        null.require_consumed("reference", "setup").unwrap();
        Controls::new()
            .require_consumed("reference", "setup")
            .unwrap();
    }

    #[test]
    fn a_new_dispatch_cannot_reuse_an_old_selector_acceptance() {
        let controls = Controls::from_cli_args(&args(&[
            "--reference",
            "roks",
            "--grid-level",
            "1",
            "--cpu-threads",
            "1",
        ]))
        .unwrap();
        let before = controls.to_value();
        controls.str("reference", "auto").unwrap();
        controls.usize("grid_level", 3).unwrap();
        controls.opt_strict_usize("cpu_threads").unwrap();
        controls
            .require_consumed("reference", "first setup")
            .unwrap();
        let feature = controls.excluding(&["cpu_threads"]);
        feature.reset_usage();
        let error = feature
            .require_consumed("reference", "next setup")
            .unwrap_err();
        assert!(matches!(error, DftError::Unsupported { .. }));
        let pending = feature.finish("next setup").unwrap_err().to_string();
        assert!(pending.contains("reference"));
        assert!(pending.contains("grid_level"));
        assert!(!pending.contains("cpu_threads"));
        assert_eq!(controls.to_value(), before);
        assert!(feature
            .excluding(&["cpu_threads"])
            .require_consumed("reference", "filtered setup")
            .is_err());
        feature.str("reference", "auto").unwrap();
        feature.usize("grid_level", 3).unwrap();
        feature.finish("accepted next setup").unwrap();
        controls
            .finish("outer envelope consumption retained")
            .unwrap();
    }
}
