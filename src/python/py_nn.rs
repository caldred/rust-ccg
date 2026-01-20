//! Neural network type bindings for Python.

use numpy::PyArray1;
use pyo3::prelude::*;

use crate::nn::{EncodedState, PolicyValueNetwork};

use super::py_core::PyPlayerId;

/// Python wrapper for EncodedState.
#[pyclass(name = "EncodedState")]
#[derive(Clone, Debug)]
pub struct PyEncodedState(pub EncodedState);

#[pymethods]
impl PyEncodedState {
    /// Create a new encoded state from tensor data and shape.
    #[new]
    fn new(tensor: Vec<f32>, shape: Vec<usize>) -> Self {
        Self(EncodedState::new(tensor, shape))
    }

    /// Create a zero-filled encoded state with the given shape.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        Self(EncodedState::zeros(shape))
    }

    /// Get the tensor data as a list.
    #[getter]
    fn tensor(&self) -> Vec<f32> {
        self.0.tensor.clone()
    }

    /// Get the tensor shape.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape.clone()
    }

    /// Get the total number of elements.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Convert tensor to numpy array (flat).
    fn to_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.0.tensor)
    }

    /// Get a value at the given flat index.
    fn get(&self, index: usize) -> Option<f32> {
        self.0.get(index)
    }

    /// Set a value at the given flat index.
    fn set(&mut self, index: usize, value: f32) {
        self.0.set(index, value);
    }

    fn __repr__(&self) -> String {
        format!("EncodedState(shape={:?}, len={})", self.0.shape, self.0.len())
    }
}

/// Python-implemented PolicyValueNetwork wrapper.
///
/// Allows Python callables to be used as neural networks from Rust.
#[pyclass(name = "PolicyValueNetwork")]
pub struct PyPolicyValueNetwork {
    callback: PyObject,
    action_space_size: usize,
    player_count: usize,
}

#[pymethods]
impl PyPolicyValueNetwork {
    /// Create a new network wrapper from a Python callable.
    ///
    /// The callable should accept an EncodedState and return a tuple of:
    /// - policy: List[float] of length action_space_size
    /// - values: List[float] of length player_count (per-player values)
    #[new]
    #[pyo3(signature = (callback, action_space_size, player_count = 2))]
    fn new(callback: PyObject, action_space_size: usize, player_count: usize) -> Self {
        Self {
            callback,
            action_space_size,
            player_count,
        }
    }

    /// Call the network on an encoded state.
    fn predict(&self, py: Python<'_>, encoded: &PyEncodedState) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let result = self.callback.call1(py, (encoded.clone(),))?;
        let (policy, value): (Vec<f32>, Vec<f32>) = result.extract(py)?;
        Ok((policy, value))
    }

    #[getter]
    fn action_space_size(&self) -> usize {
        self.action_space_size
    }

    #[getter]
    fn player_count(&self) -> usize {
        self.player_count
    }
}

impl PyPolicyValueNetwork {
    /// Return a fallback prediction when Python callback fails.
    fn fallback_prediction(&self) -> (Vec<f32>, Vec<f32>) {
        let policy = vec![1.0 / self.action_space_size as f32; self.action_space_size];
        let values = vec![0.0; self.player_count];
        (policy, values)
    }
}

// Implement the Rust trait for the Python wrapper
impl PolicyValueNetwork for PyPolicyValueNetwork {
    fn predict(&self, encoded: &EncodedState) -> (Vec<f32>, Vec<f32>) {
        Python::with_gil(|py| {
            let py_encoded = PyEncodedState(encoded.clone());
            match self.callback.call1(py, (py_encoded,)) {
                Ok(result) => match result.extract::<(Vec<f32>, Vec<f32>)>(py) {
                    Ok((policy, value)) => (policy, value),
                    Err(e) => {
                        eprintln!("PyPolicyValueNetwork: failed to extract result: {}", e);
                        self.fallback_prediction()
                    }
                },
                Err(e) => {
                    eprintln!("PyPolicyValueNetwork: predict() call failed: {}", e);
                    self.fallback_prediction()
                }
            }
        })
    }
}

// SAFETY: PyPolicyValueNetwork is Send + Sync because:
// 1. All Python interactions go through Python::with_gil()
// 2. PyObject internally uses reference counting that's safe across threads
//    when accessed through the GIL
// 3. The other fields (action_space_size, player_count) are plain data
//
// INVARIANT: Any new methods that access self.callback MUST use with_gil().
unsafe impl Send for PyPolicyValueNetwork {}
unsafe impl Sync for PyPolicyValueNetwork {}

/// Uniform policy baseline (for testing).
#[pyclass(name = "UniformPolicy")]
#[derive(Clone)]
pub struct PyUniformPolicy {
    action_space_size: usize,
    player_count: usize,
}

#[pymethods]
impl PyUniformPolicy {
    #[new]
    #[pyo3(signature = (action_space_size, player_count = 2))]
    fn new(action_space_size: usize, player_count: usize) -> Self {
        Self {
            action_space_size,
            player_count,
        }
    }

    /// Get uniform policy and zero values.
    fn predict(&self, _encoded: &PyEncodedState) -> (Vec<f32>, Vec<f32>) {
        let policy = vec![1.0 / self.action_space_size as f32; self.action_space_size];
        let values = vec![0.0; self.player_count];
        (policy, values)
    }
}

/// Simple state encoder that produces a flat vector (for testing).
#[pyclass(name = "SimpleEncoder")]
#[derive(Clone)]
pub struct PySimpleEncoder {
    player_count: usize,
    features_per_player: usize,
}

#[pymethods]
impl PySimpleEncoder {
    #[new]
    #[pyo3(signature = (player_count = 2, features_per_player = 5))]
    fn new(player_count: usize, features_per_player: usize) -> Self {
        Self {
            player_count,
            features_per_player,
        }
    }

    /// Get the output shape.
    fn output_shape(&self) -> Vec<usize> {
        vec![self.player_count * self.features_per_player]
    }

    #[getter]
    fn player_count(&self) -> usize {
        self.player_count
    }

    /// Encode a game state from a player's perspective.
    ///
    /// This is a placeholder that returns zeros - use the Rust encoder
    /// through SimpleGameWorker for actual encoding.
    fn encode(&self, _perspective: &PyPlayerId) -> PyEncodedState {
        PyEncodedState::zeros(self.output_shape())
    }
}
