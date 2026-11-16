// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 ss0832
// This file is part of the dft-rs library.

//! A small self-describing value tree with a JSON reader and writer.
//!
//! Every calculation routed through the common driver returns its physical
//! values, units, conventions and diagnostics as a [`Value`] tree. The CLI
//! prints it as text or JSON, the Python extension converts it into plain
//! dictionaries and lists, and saved results can be read back, so all
//! surfaces share one representation without an extra dependency.
//!
//! Conventions: object keys keep their insertion order; complex numbers are
//! written as two-element `[real, imaginary]` arrays; non-finite floats are
//! written as `null` (JSON has no NaN or infinity) and a report that can
//! produce them says so in its warnings.

use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::response::complex::Complex;
use crate::{DftError, Result};
use std::fmt::Write as _;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn object() -> Self {
        Self::Object(Vec::new())
    }

    /// Insert or replace `key` in an object. Panics on a non-object, which
    /// is a programming error rather than a runtime condition.
    pub fn set(&mut self, key: &str, value: impl ToValue) {
        let Self::Object(entries) = self else {
            panic!("Value::set on a non-object");
        };
        let value = value.to_value();
        if let Some(entry) = entries.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value;
        } else {
            entries.push((key.to_string(), value));
        }
    }

    /// Builder form of [`Self::set`].
    pub fn with(mut self, key: &str, value: impl ToValue) -> Self {
        self.set(key, value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Self::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Float(v) if v.fract() == 0.0 && v.abs() < 9.0e15 => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    pub fn entries(&self) -> &[(String, Value)] {
        match self {
            Self::Object(entries) => entries,
            _ => &[],
        }
    }

    /// Flatten a numeric array of any nesting depth into row-major data and
    /// its shape. Returns `None` for ragged or non-numeric arrays.
    pub fn numeric_array(&self) -> Option<(Vec<usize>, Vec<f64>)> {
        match self {
            Self::Float(_) | Self::Int(_) => Some((Vec::new(), vec![self.as_f64()?])),
            Self::Array(items) => {
                let mut shape = None;
                let mut data = Vec::new();
                for item in items {
                    let (inner, values) = item.numeric_array()?;
                    if shape.get_or_insert_with(|| inner.clone()) != &inner {
                        return None;
                    }
                    data.extend(values);
                }
                let mut full = vec![items.len()];
                full.extend(shape.unwrap_or_default());
                Some((full, data))
            }
            _ => None,
        }
    }

    /// Compact JSON text.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out, None, 0);
        out
    }

    /// Indented JSON text (two spaces per level); numeric leaf arrays stay
    /// on one line so tensors remain readable.
    pub fn to_json_pretty(&self) -> String {
        let mut out = String::new();
        self.write_json(&mut out, Some(2), 0);
        out.push('\n');
        out
    }

    fn write_json(&self, out: &mut String, indent: Option<usize>, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
            Self::Int(v) => {
                let _ = write!(out, "{v}");
            }
            Self::Float(v) => write_float(out, *v),
            Self::Str(v) => write_string(out, v),
            Self::Array(items) => {
                let leaf = items
                    .iter()
                    .all(|v| !matches!(v, Self::Array(_) | Self::Object(_)));
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if let (Some(step), false) = (indent, leaf) {
                        newline(out, step * (depth + 1));
                    }
                    item.write_json(out, indent, depth + 1);
                }
                if let (Some(step), false, false) = (indent, leaf, items.is_empty()) {
                    newline(out, step * depth);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (i, (key, value)) in entries.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    if let Some(step) = indent {
                        newline(out, step * (depth + 1));
                    }
                    write_string(out, key);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    value.write_json(out, indent, depth + 1);
                }
                if let (Some(step), false) = (indent, entries.is_empty()) {
                    newline(out, step * depth);
                }
                out.push('}');
            }
        }
    }

    /// Parse JSON text. Integers without a fraction or exponent become
    /// [`Value::Int`]; every other number is [`Value::Float`].
    pub fn parse_json(text: &str) -> Result<Self> {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        parser.skip_space();
        let value = parser.value(0)?;
        parser.skip_space();
        if parser.pos != parser.bytes.len() {
            return Err(parser.error("trailing characters after the JSON value"));
        }
        Ok(value)
    }
}

fn newline(out: &mut String, spaces: usize) {
    out.push('\n');
    out.extend(std::iter::repeat_n(' ', spaces));
}

fn write_float(out: &mut String, v: f64) {
    if !v.is_finite() {
        out.push_str("null");
    } else if v == v.trunc() && v.abs() < 1e15 {
        let _ = write!(out, "{v:.1}");
    } else {
        // Shortest round-trip representation.
        let _ = write!(out, "{v:?}");
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &str) -> DftError {
        DftError::InvalidInput(format!("JSON parse error at byte {}: {message}", self.pos))
    }

    fn skip_space(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, literal: &str) -> Result<()> {
        if self.bytes[self.pos..].starts_with(literal.as_bytes()) {
            self.pos += literal.len();
            Ok(())
        } else {
            Err(self.error(&format!("expected `{literal}`")))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        if depth > 256 {
            return Err(self.error("nesting deeper than 256 levels"));
        }
        match self.bytes.get(self.pos) {
            None => Err(self.error("unexpected end of input")),
            Some(b'n') => self.expect("null").map(|_| Value::Null),
            Some(b't') => self.expect("true").map(|_| Value::Bool(true)),
            Some(b'f') => self.expect("false").map(|_| Value::Bool(false)),
            Some(b'"') => self.string().map(Value::Str),
            Some(b'[') => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_space();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                loop {
                    self.skip_space();
                    items.push(self.value(depth + 1)?);
                    self.skip_space();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b']') => {
                            self.pos += 1;
                            return Ok(Value::Array(items));
                        }
                        _ => return Err(self.error("expected `,` or `]`")),
                    }
                }
            }
            Some(b'{') => {
                self.pos += 1;
                let mut entries: Vec<(String, Value)> = Vec::new();
                self.skip_space();
                if self.bytes.get(self.pos) == Some(&b'}') {
                    self.pos += 1;
                    return Ok(Value::Object(entries));
                }
                loop {
                    self.skip_space();
                    if self.bytes.get(self.pos) != Some(&b'"') {
                        return Err(self.error("expected an object key"));
                    }
                    let key = self.string()?;
                    self.skip_space();
                    self.expect(":")?;
                    self.skip_space();
                    let value = self.value(depth + 1)?;
                    if entries.iter().any(|(k, _)| *k == key) {
                        return Err(self.error(&format!("duplicate key `{key}`")));
                    }
                    entries.push((key, value));
                    self.skip_space();
                    match self.bytes.get(self.pos) {
                        Some(b',') => self.pos += 1,
                        Some(b'}') => {
                            self.pos += 1;
                            return Ok(Value::Object(entries));
                        }
                        _ => return Err(self.error("expected `,` or `}`")),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    fn string(&mut self) -> Result<String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let Some(&byte) = self.bytes.get(self.pos) else {
                return Err(self.error("unterminated string"));
            };
            match byte {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    let escape = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| self.error("unterminated escape"))?;
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xd800..0xdc00).contains(&code) {
                                self.expect("\\u")?;
                                let low = self.hex4()?;
                                if !(0xdc00..0xe000).contains(&low) {
                                    return Err(self.error("invalid surrogate pair"));
                                }
                                code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                            }
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| self.error("invalid unicode escape"))?,
                            );
                        }
                        _ => return Err(self.error("invalid escape")),
                    }
                }
                _ => {
                    let rest = std::str::from_utf8(&self.bytes[self.pos..])
                        .map_err(|_| self.error("invalid UTF-8"))?;
                    let c = rest.chars().next().unwrap();
                    if (c as u32) < 0x20 {
                        return Err(self.error("control character in string"));
                    }
                    out.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| self.error("invalid \\u escape"))?;
        self.pos += 4;
        Ok(digits)
    }

    fn number(&mut self) -> Result<Value> {
        let start = self.pos;
        let mut integer = true;
        while let Some(&b) = self.bytes.get(self.pos) {
            match b {
                b'0'..=b'9' | b'-' | b'+' => {}
                b'.' | b'e' | b'E' => integer = false,
                _ => break,
            }
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("");
        if text.is_empty() {
            return Err(self.error("unexpected character"));
        }
        if integer {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Value::Int(v));
            }
        }
        text.parse::<f64>()
            .map(Value::Float)
            .map_err(|_| self.error(&format!("invalid number `{text}`")))
    }
}

/// Conversion into the common value tree.
pub trait ToValue {
    fn to_value(&self) -> Value;
}

impl ToValue for Value {
    fn to_value(&self) -> Value {
        self.clone()
    }
}
impl ToValue for f64 {
    fn to_value(&self) -> Value {
        Value::Float(*self)
    }
}
impl ToValue for bool {
    fn to_value(&self) -> Value {
        Value::Bool(*self)
    }
}
macro_rules! int_value {
    ($($t:ty),*) => {$(
        impl ToValue for $t {
            fn to_value(&self) -> Value {
                Value::Int(*self as i64)
            }
        }
    )*};
}
int_value!(i32, i64, u8, u32, usize);
impl ToValue for str {
    fn to_value(&self) -> Value {
        Value::Str(self.to_string())
    }
}
impl ToValue for String {
    fn to_value(&self) -> Value {
        Value::Str(self.clone())
    }
}
impl<T: ToValue + ?Sized> ToValue for &T {
    fn to_value(&self) -> Value {
        (**self).to_value()
    }
}
impl<T: ToValue> ToValue for Option<T> {
    fn to_value(&self) -> Value {
        self.as_ref().map_or(Value::Null, ToValue::to_value)
    }
}
impl<T: ToValue> ToValue for [T] {
    fn to_value(&self) -> Value {
        Value::Array(self.iter().map(ToValue::to_value).collect())
    }
}
impl<T: ToValue> ToValue for Vec<T> {
    fn to_value(&self) -> Value {
        self.as_slice().to_value()
    }
}
impl<T: ToValue, const N: usize> ToValue for [T; N] {
    fn to_value(&self) -> Value {
        self.as_slice().to_value()
    }
}
impl ToValue for Vec3 {
    fn to_value(&self) -> Value {
        self.to_array().to_value()
    }
}
impl ToValue for Complex {
    fn to_value(&self) -> Value {
        [self.re, self.im].to_value()
    }
}
impl ToValue for Matrix {
    fn to_value(&self) -> Value {
        Value::Array(
            (0..self.rows)
                .map(|r| Value::Array((0..self.cols).map(|c| Value::Float(self[(r, c)])).collect()))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_preserves_structure_and_numbers() {
        let value = Value::object()
            .with("energy", -1.1167593073964255)
            .with("count", 3usize)
            .with("label", "H2 \"test\"\n")
            .with("tensor", vec![[1.0, 2.5e-17], [-3.0, 4.0]])
            .with("missing", Option::<f64>::None)
            .with("complex", Complex::new(0.5, -0.25))
            .with("flag", true);
        for text in [value.to_json(), value.to_json_pretty()] {
            assert_eq!(Value::parse_json(&text).unwrap(), value);
        }
        let (shape, data) = value.get("tensor").unwrap().numeric_array().unwrap();
        assert_eq!(shape, vec![2, 2]);
        assert_eq!(data, vec![1.0, 2.5e-17, -3.0, 4.0]);
    }

    #[test]
    fn floats_round_trip_bit_exactly() {
        for v in [0.1, -1.0 / 3.0, 6.02214076e23, 1e-300, 123456789.0, -0.0] {
            let text = Value::Float(v).to_json();
            let back = Value::parse_json(&text).unwrap().as_f64().unwrap();
            assert_eq!(back.to_bits(), v.to_bits(), "{v} -> {text}");
        }
        assert_eq!(Value::Float(f64::NAN).to_json(), "null");
    }

    #[test]
    fn malformed_json_is_refused() {
        for text in ["{", "[1,]", "{\"a\":1,\"a\":2}", "01x", "\"\\q\"", "[1] 2"] {
            assert!(Value::parse_json(text).is_err(), "{text}");
        }
        assert_eq!(
            Value::parse_json("\"\\u00e9\\ud83d\\ude00\"").unwrap(),
            Value::Str("\u{e9}\u{1f600}".into())
        );
    }
}
