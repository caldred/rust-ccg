//! Core type bindings for Python.

use pyo3::prelude::*;

use crate::core::{Action, PlayerId, TemplateId};

/// Python wrapper for PlayerId.
#[pyclass(name = "PlayerId")]
#[derive(Clone, Debug)]
pub struct PyPlayerId(pub PlayerId);

#[pymethods]
impl PyPlayerId {
    /// Create a new player ID.
    #[new]
    fn new(id: u8) -> Self {
        Self(PlayerId::new(id))
    }

    /// Get the player index (0-based).
    fn index(&self) -> usize {
        self.0.index()
    }

    /// Get the raw ID value.
    #[getter]
    fn id(&self) -> u8 {
        self.0 .0
    }

    fn __repr__(&self) -> String {
        format!("PlayerId({})", self.0 .0)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        self.0 .0 as u64
    }
}

/// Python wrapper for Action.
#[pyclass(name = "Action")]
#[derive(Clone, Debug)]
pub struct PyAction(pub Action);

#[pymethods]
impl PyAction {
    /// Create a new action with the given template ID.
    #[new]
    fn new(template_id: u16) -> Self {
        Self(Action::new(TemplateId::new(template_id)))
    }

    /// Get the template ID.
    #[getter]
    fn template_id(&self) -> u16 {
        self.0.template.0
    }

    /// Get the number of pointers (entity targets).
    fn pointer_count(&self) -> usize {
        self.0.pointer_count()
    }

    fn __repr__(&self) -> String {
        format!("Action(template={})", self.0.template.0)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.0.template.0.hash(&mut hasher);
        for ptr in &self.0.pointers {
            ptr.0.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Python wrapper for TemplateId.
#[pyclass(name = "TemplateId")]
#[derive(Clone, Debug)]
pub struct PyTemplateId(pub TemplateId);

#[pymethods]
impl PyTemplateId {
    #[new]
    fn new(id: u16) -> Self {
        Self(TemplateId::new(id))
    }

    #[getter]
    fn id(&self) -> u16 {
        self.0 .0
    }

    fn __repr__(&self) -> String {
        format!("TemplateId({})", self.0 .0)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn __hash__(&self) -> u64 {
        self.0 .0 as u64
    }
}
