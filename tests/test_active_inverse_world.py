import unittest

import numpy as np

from active_inverse_core import svd_diagnostics
from active_inverse_world import (
    build_problem,
    inverse_update,
    measure_view_jacobians,
    selected_sensor_residual,
)


class ActiveInverseWorldTests(unittest.TestCase):
    def test_bounded_view_is_rank_deficient_and_inverse_step_reduces_observed_error(self):
        problem = build_problem(
            seed=2,
            operator_dim=4,
            camera_views=5,
            image_size=8,
            features_per_view=2,
            hidden_magnitude=0.12,
            min_truth_change=1e-4,
            device="cpu",
        )
        z0 = np.zeros(problem.basis.shape[1], dtype=np.float64)
        initial_view = problem.camera_views // 2

        _pred, jacobians = measure_view_jacobians(problem, z0, epsilon=0.01)
        J0 = jacobians[[initial_view]].reshape(-1, problem.basis.shape[1])
        diag = svd_diagnostics(J0, parameter_dim=problem.basis.shape[1])
        self.assertLess(diag["rank"], problem.basis.shape[1])

        before = selected_sensor_residual(problem, z0, [initial_view])
        z1, _info = inverse_update(
            problem,
            z0,
            [initial_view],
            epsilon=0.01,
            damping=1e-6,
            trust_radius=0.20,
        )
        after = selected_sensor_residual(problem, z1, [initial_view])
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
