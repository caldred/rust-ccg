//! Training data type bindings for Python.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::core::PlayerId;
use crate::training::{ExperienceBuffer, Step, TrainingSample, Trajectory};

use super::py_core::{PyAction, PyPlayerId};
use super::py_nn::PyEncodedState;

/// Python wrapper for Step.
#[pyclass(name = "Step")]
#[derive(Clone)]
pub struct PyStep(pub Step);

#[pymethods]
impl PyStep {
    /// Get the encoded state at this step.
    #[getter]
    fn encoded_state(&self) -> PyEncodedState {
        PyEncodedState(self.0.encoded_state.clone())
    }

    /// Get the MCTS action probabilities.
    #[getter]
    fn action_probs(&self) -> Vec<(PyAction, f64)> {
        self.0
            .action_probs
            .iter()
            .map(|(a, p)| (PyAction(a.clone()), *p))
            .collect()
    }

    /// Get the action that was taken.
    #[getter]
    fn action_taken(&self) -> PyAction {
        PyAction(self.0.action_taken.clone())
    }

    /// Get the player who acted.
    #[getter]
    fn player(&self) -> PyPlayerId {
        PyPlayerId(self.0.player)
    }

    /// Get the move number within the game.
    #[getter]
    fn move_number(&self) -> usize {
        self.0.move_number
    }

    fn __repr__(&self) -> String {
        format!(
            "Step(move={}, player={}, action={})",
            self.0.move_number, self.0.player.0, self.0.action_taken.template.0
        )
    }
}

/// Python wrapper for Trajectory.
#[pyclass(name = "Trajectory")]
#[derive(Clone)]
pub struct PyTrajectory(pub Trajectory);

#[pymethods]
impl PyTrajectory {
    /// Get all steps in this trajectory.
    #[getter]
    fn steps(&self) -> Vec<PyStep> {
        self.0.steps.iter().map(|s| PyStep(s.clone())).collect()
    }

    /// Get the game outcome for each player.
    ///
    /// Returns a dict mapping player index to reward (1.0 for win, 0.0 for loss, 0.5 for draw).
    #[getter]
    fn outcome(&self) -> Vec<(usize, f64)> {
        (0..self.0.outcome.player_count())
            .map(|i| (i, self.0.outcome[PlayerId::new(i as u8)]))
            .collect()
    }

    /// Get the number of moves in the game.
    #[getter]
    fn game_length(&self) -> usize {
        self.0.game_length
    }

    /// Get the RNG seed used for this game.
    #[getter]
    fn seed(&self) -> u64 {
        self.0.seed
    }

    /// Get the number of steps.
    fn __len__(&self) -> usize {
        self.0.len()
    }

    /// Get steps for a specific player.
    fn player_steps(&self, player: &PyPlayerId) -> Vec<PyStep> {
        self.0
            .player_steps(player.0)
            .map(|s| PyStep(s.clone()))
            .collect()
    }

    /// Convert to training samples.
    fn to_training_samples(&self) -> Vec<PyTrainingSample> {
        self.0
            .to_training_samples()
            .into_iter()
            .map(PyTrainingSample)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Trajectory(length={}, steps={}, seed={})",
            self.0.game_length,
            self.0.steps.len(),
            self.0.seed
        )
    }
}

/// Python wrapper for TrainingSample.
#[pyclass(name = "TrainingSample")]
#[derive(Clone)]
pub struct PyTrainingSample(pub TrainingSample);

#[pymethods]
impl PyTrainingSample {
    /// Get the encoded state.
    #[getter]
    fn state(&self) -> PyEncodedState {
        PyEncodedState(self.0.state.clone())
    }

    /// Get the target policy (MCTS visit distribution).
    #[getter]
    fn policy(&self) -> Vec<f32> {
        self.0.policy.clone()
    }

    /// Get the target value (game outcome from this player's perspective).
    #[getter]
    fn value(&self) -> f32 {
        self.0.value
    }

    /// Get the player whose perspective this sample is from.
    #[getter]
    fn player(&self) -> PyPlayerId {
        PyPlayerId(self.0.player)
    }

    /// Get state tensor as numpy array.
    fn state_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.0.state.tensor)
    }

    /// Get policy as numpy array.
    fn policy_numpy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice_bound(py, &self.0.policy)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrainingSample(player={}, value={:.2}, state_dim={})",
            self.0.player.0,
            self.0.value,
            self.0.state.len()
        )
    }
}

/// Python wrapper for ExperienceBuffer.
#[pyclass(name = "ExperienceBuffer")]
pub struct PyExperienceBuffer {
    inner: ExperienceBuffer,
}

#[pymethods]
impl PyExperienceBuffer {
    /// Create a new experience buffer with maximum capacity.
    #[new]
    fn new(max_trajectories: usize) -> Self {
        Self {
            inner: ExperienceBuffer::new(max_trajectories),
        }
    }

    /// Add a trajectory to the buffer.
    fn push(&mut self, trajectory: &PyTrajectory) {
        self.inner.push(trajectory.0.clone());
    }

    /// Get the number of trajectories in the buffer.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the maximum capacity.
    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Clear all trajectories.
    fn clear(&mut self) {
        self.inner.clear();
    }

    /// Get all training samples from all trajectories.
    fn to_training_samples(&self) -> Vec<PyTrainingSample> {
        self.inner
            .to_training_samples()
            .into_iter()
            .map(PyTrainingSample)
            .collect()
    }

    /// Sample a random batch of training samples.
    fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<PyTrainingSample> {
        self.inner
            .sample_batch(batch_size, seed)
            .into_iter()
            .map(PyTrainingSample)
            .collect()
    }

    /// Get all samples as batched numpy arrays for efficient training.
    ///
    /// Returns (states, policies, values) as numpy arrays:
    /// - states: [N, state_dim] float32
    /// - policies: [N, action_dim] float32
    /// - values: [N] float32
    fn to_numpy_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
    )> {
        let samples = self.inner.to_training_samples();

        if samples.is_empty() {
            // Return empty arrays with correct shapes
            let states = PyArray2::zeros_bound(py, [0, 0], false);
            let policies = PyArray2::zeros_bound(py, [0, 0], false);
            let values = PyArray1::zeros_bound(py, [0], false);
            return Ok((states, policies, values));
        }

        let n = samples.len();
        let state_dim = samples[0].state.len();
        // Find max policy length (can vary if legal actions vary)
        let policy_dim = samples.iter().map(|s| s.policy.len()).max().unwrap_or(0);

        // Validate all states have consistent dimensions
        for (i, sample) in samples.iter().enumerate() {
            if sample.state.len() != state_dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Inconsistent state dimension at sample {}: expected {}, got {}",
                    i,
                    state_dim,
                    sample.state.len()
                )));
            }
        }

        // Flatten all data, padding policies to max length
        let mut states_flat: Vec<f32> = Vec::with_capacity(n * state_dim);
        let mut policies_flat: Vec<f32> = Vec::with_capacity(n * policy_dim);
        let mut values_vec: Vec<f32> = Vec::with_capacity(n);

        for sample in &samples {
            states_flat.extend_from_slice(&sample.state.tensor);
            // Pad policy to policy_dim with zeros
            policies_flat.extend_from_slice(&sample.policy);
            policies_flat.extend(std::iter::repeat(0.0f32).take(policy_dim - sample.policy.len()));
            values_vec.push(sample.value);
        }

        // Create numpy arrays using from_vec_bound and reshape
        let states = PyArray1::from_vec_bound(py, states_flat)
            .reshape([n, state_dim])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let policies = PyArray1::from_vec_bound(py, policies_flat)
            .reshape([n, policy_dim])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let values = PyArray1::from_vec_bound(py, values_vec);

        Ok((states, policies, values))
    }

    fn __repr__(&self) -> String {
        format!(
            "ExperienceBuffer(len={}, capacity={})",
            self.inner.len(),
            self.inner.capacity()
        )
    }

    /// Iterate over trajectories in the buffer.
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyTrajectoryIterator>> {
        let trajectories: Vec<PyTrajectory> = slf
            .inner
            .iter()
            .map(|t| PyTrajectory(t.clone()))
            .collect();
        Py::new(
            slf.py(),
            PyTrajectoryIterator {
                trajectories,
                index: 0,
            },
        )
    }
}

/// Iterator over trajectories in an ExperienceBuffer.
#[pyclass]
pub struct PyTrajectoryIterator {
    trajectories: Vec<PyTrajectory>,
    index: usize,
}

#[pymethods]
impl PyTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyTrajectory> {
        if slf.index < slf.trajectories.len() {
            let traj = slf.trajectories[slf.index].clone();
            slf.index += 1;
            Some(traj)
        } else {
            None
        }
    }
}
