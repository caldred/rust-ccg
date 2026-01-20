//! Card attribute system for game-specific properties.
//!
//! Cards have attributes like "power", "toughness", "cost", etc.
//! These are game-specific - the engine doesn't interpret them.
//!
//! ## AttributeValue Types
//!
//! - `Int`: Numbers (power, cost, health)
//! - `Bool`: Flags (flying, haste)
//! - `Text`: Strings (subtypes, names)
//! - `IntList`: Number lists (mana cost breakdown)
//! - `TextList`: String lists (keywords, types)

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

/// Key for accessing card attributes.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AttributeKey(pub String);

impl AttributeKey {
    /// Create a new attribute key.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }
}

impl From<&str> for AttributeKey {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for AttributeKey {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Value for a card attribute.
///
/// Supports multiple types to handle different game needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Integer value (power, toughness, cost).
    Int(i64),
    /// Boolean flag (flying, haste).
    Bool(bool),
    /// Text value (subtype, keyword as string).
    Text(String),
    /// List of integers (mana cost breakdown: [colorless, white, blue, ...]).
    IntList(Vec<i64>),
    /// List of strings (keywords, types).
    TextList(Vec<String>),
}

impl AttributeValue {
    /// Get as integer if this is an Int value.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as bool if this is a Bool value.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            AttributeValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string reference if this is a Text value.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            AttributeValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get as int list reference if this is an IntList value.
    #[must_use]
    pub fn as_int_list(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::IntList(v) => Some(v),
            _ => None,
        }
    }

    /// Get as text list reference if this is a TextList value.
    #[must_use]
    pub fn as_text_list(&self) -> Option<&[String]> {
        match self {
            AttributeValue::TextList(v) => Some(v),
            _ => None,
        }
    }
}

// Convenient From implementations
impl From<i64> for AttributeValue {
    fn from(v: i64) -> Self {
        AttributeValue::Int(v)
    }
}

impl From<i32> for AttributeValue {
    fn from(v: i32) -> Self {
        AttributeValue::Int(v as i64)
    }
}

impl From<bool> for AttributeValue {
    fn from(v: bool) -> Self {
        AttributeValue::Bool(v)
    }
}

impl From<String> for AttributeValue {
    fn from(v: String) -> Self {
        AttributeValue::Text(v)
    }
}

impl From<&str> for AttributeValue {
    fn from(v: &str) -> Self {
        AttributeValue::Text(v.to_string())
    }
}

impl From<Vec<i64>> for AttributeValue {
    fn from(v: Vec<i64>) -> Self {
        AttributeValue::IntList(v)
    }
}

impl From<Vec<String>> for AttributeValue {
    fn from(v: Vec<String>) -> Self {
        AttributeValue::TextList(v)
    }
}

/// Collection of attributes.
pub type Attributes = FxHashMap<AttributeKey, AttributeValue>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribute_key() {
        let key1 = AttributeKey::new("power");
        let key2: AttributeKey = "power".into();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_attribute_value_int() {
        let val = AttributeValue::Int(5);
        assert_eq!(val.as_int(), Some(5));
        assert_eq!(val.as_bool(), None);
    }

    #[test]
    fn test_attribute_value_bool() {
        let val = AttributeValue::Bool(true);
        assert_eq!(val.as_bool(), Some(true));
        assert_eq!(val.as_int(), None);
    }

    #[test]
    fn test_attribute_value_text() {
        let val = AttributeValue::Text("flying".to_string());
        assert_eq!(val.as_text(), Some("flying"));
    }

    #[test]
    fn test_attribute_value_from() {
        let int: AttributeValue = 42i32.into();
        assert_eq!(int.as_int(), Some(42));

        let boolean: AttributeValue = true.into();
        assert_eq!(boolean.as_bool(), Some(true));

        let text: AttributeValue = "keyword".into();
        assert_eq!(text.as_text(), Some("keyword"));
    }

    #[test]
    fn test_attributes_map() {
        let mut attrs = Attributes::default();
        attrs.insert("power".into(), 3i32.into());
        attrs.insert("flying".into(), true.into());

        assert_eq!(
            attrs.get(&"power".into()).and_then(|v| v.as_int()),
            Some(3)
        );
        assert_eq!(
            attrs.get(&"flying".into()).and_then(|v| v.as_bool()),
            Some(true)
        );
    }
}
