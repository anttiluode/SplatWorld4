import unittest

import numpy as np

from active_inverse_core import (
    damped_least_squares,
    material_tangent_basis,
    select_active_view,
    svd_diagnostics,
)


class ActiveInverseCoreTests(unittest.TestCase):
    def test_material_tangent_basis_removes_common_softmax_gauge(self):
        basis = material_tangent_basis(6)
        self.assertEqual(basis.shape, (6, 5))
        np.testing.assert_allclose(basis.T @ basis, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(np.ones(6) @ basis, np.zeros(5), atol=1e-12)

    def test_damped_least_squares_reduces_measured_residual(self):
        J = np.array(
            [
                [1.0, 0.0],
                [0.0, 0.5],
                [1.0, 1.0],
            ]
        )
        true_step = np.array([0.30, -0.20])
        residual = J @ true_step
        before = np.linalg.norm(residual)
        step = damped_least_squares(
            J,
            residual,
            damping=1e-10,
            trust_radius=10.0,
        )
        after = np.linalg.norm(residual - J @ step)
        self.assertLess(after, before * 1e-6)

    def test_svd_diagnostics_returns_exact_null_space_when_rank_deficient(self):
        J = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        )
        diag = svd_diagnostics(J, parameter_dim=3, relative_tol=1e-10, weak_count=1)
        self.assertEqual(diag["rank"], 2)
        self.assertEqual(diag["weak_basis"].shape, (3, 1))
        weak = diag["weak_basis"][:, 0]
        self.assertLess(np.linalg.norm(J @ weak), 1e-12)
        self.assertGreater(abs(weak[2]), 0.999999)

    def test_active_view_targets_currently_unobservable_direction(self):
        # View 0 observes parameter coordinates 0 and 1 but not coordinate 2.
        view_jacobians = np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.10], [0.0, 0.0, 0.00]],
                [[0.0, 0.0, 2.00], [0.0, 0.0, 0.00]],
                [[1.0, 1.0, 0.00], [0.0, 1.0, 0.00]],
            ]
        )
        chosen, scores, diag = select_active_view(
            view_jacobians,
            observed_views=[0],
            relative_tol=1e-10,
            weak_count=1,
        )
        self.assertEqual(diag["rank"], 2)
        self.assertEqual(chosen, 2)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[2], scores[3])


if __name__ == "__main__":
    unittest.main()
