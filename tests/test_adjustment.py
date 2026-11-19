"""Enotski testi za modul geodetske izravnave po MNK (src/adjustment.py)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Omogoči zagon testa neposredno iz korenske mape ali iz mape tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adjustment import (
    AdjustmentResult,
    fit_cone_vertical,
    fit_cylinder_vertical,
    fit_plane,
    fit_sphere,
)


class TestGeodeticAdjustment(unittest.TestCase):
    """Preverjanje natančnosti in konvergence MNK izravnave za vsa 4 geometrijska telesa."""

    def test_fit_sphere_geodetic_coords_and_sigma0(self) -> None:
        """Sintetična krogla v velikih geodetskih koordinatah s šumom 0.0005 m."""
        rng = np.random.default_rng(42)
        n = 4000
        true_x0 = 100298.5
        true_y0 = 461293.5
        true_z0 = 200.1
        true_r = 0.10
        noise_std = 0.0005

        # Enakomerno porazdeljene točke po površini krogle
        u = rng.uniform(-1.0, 1.0, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        sin_theta = np.sqrt(1.0 - u * u)

        nx = sin_theta * np.cos(phi)
        ny = sin_theta * np.sin(phi)
        nz = u

        # Gaussov šum v radialni (pravokotni) smeri s std = 0.0005 m
        radial_noise = rng.normal(0.0, noise_std, size=n)
        r_noisy = true_r + radial_noise

        points = np.column_stack(
            (
                true_x0 + r_noisy * nx,
                true_y0 + r_noisy * ny,
                true_z0 + r_noisy * nz,
            )
        ).astype(np.float64)

        res = fit_sphere(points)
        self.assertIsInstance(res, AdjustmentResult)
        self.assertTrue(res.converged)
        self.assertEqual(res.n_points, n)
        self.assertEqual(res.n_unknowns, 4)
        self.assertEqual(res.degrees_of_freedom, n - 4)
        self.assertEqual(res.Q_dd.shape, (4, 4))
        self.assertEqual(res.residuals.shape, (n,))

        self.assertAlmostEqual(res.params["x0"], true_x0, delta=0.0001)
        self.assertAlmostEqual(res.params["y0"], true_y0, delta=0.0001)
        self.assertAlmostEqual(res.params["z0"], true_z0, delta=0.0001)
        self.assertAlmostEqual(res.params["R"], true_r, delta=0.0001)

        # Aposteriorni standardni odklon mora natančno oceniti vneseni šum (0.0005 m)
        self.assertAlmostEqual(res.sigma0, noise_std, delta=0.00005)
        self.assertAlmostEqual(res.rmse, res.sigma0, delta=0.00002)

        # Preveri še ročne začetne približke (zamaknjene za 2 cm)
        res_manual = fit_sphere(
            points,
            initial_params={
                "x0": true_x0 + 0.02,
                "y0": true_y0 - 0.02,
                "z0": true_z0 + 0.015,
                "R": true_r + 0.03,
            },
        )
        self.assertTrue(res_manual.converged)
        for key in ("x0", "y0", "z0", "R"):
            self.assertAlmostEqual(res_manual.params[key], res.params[key], places=10)

    def test_fit_sphere_chunk_vs_point_by_point_equivalence(self) -> None:
        """Akumulacija po blokih (chunk_size=5000) mora biti matematično identična zanki po 1 točki."""
        rng = np.random.default_rng(7)
        n = 250
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        center = np.array([100298.5, 461293.5, 200.1], dtype=np.float64)
        r_noisy = 0.25 + rng.normal(0.0, 0.0005, size=n)
        points = center + dirs * r_noisy[:, np.newaxis]

        res_single = fit_sphere(points, chunk_size=1)
        res_chunk = fit_sphere(points, chunk_size=64)

        for key in ("x0", "y0", "z0", "R"):
            self.assertAlmostEqual(res_single.params[key], res_chunk.params[key], places=12)
        self.assertAlmostEqual(res_single.sigma0, res_chunk.sigma0, places=12)
        self.assertTrue(np.allclose(res_single.Q_dd, res_chunk.Q_dd, rtol=1e-11, atol=1e-12))

    def test_fit_plane_tilted_and_vertical_wall(self) -> None:
        """Preveri izravnavo poševne ravnine in navpične stene (brez singularnosti)."""
        rng = np.random.default_rng(101)
        n = 3000
        noise_std = 0.0005
        origin = np.array([100298.5, 461293.5, 200.1], dtype=np.float64)

        # 1) Poševna ravnina
        n_true = np.array([0.3, -0.4, 0.8660254037844386], dtype=np.float64)
        n_true /= np.linalg.norm(n_true)
        # Dve ortogonalni bazi v ravnini
        e1 = np.array([1.0, 0.0, -n_true[0] / n_true[2]], dtype=np.float64)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_true, e1)

        s = rng.uniform(-2.0, 2.0, size=n)
        t = rng.uniform(-2.0, 2.0, size=n)
        perp_noise = rng.normal(0.0, noise_std, size=n)

        pts_tilted = (
            origin
            + s[:, np.newaxis] * e1
            + t[:, np.newaxis] * e2
            + perp_noise[:, np.newaxis] * n_true
        )

        res = fit_plane(pts_tilted)
        self.assertTrue(res.converged)
        self.assertEqual(res.n_unknowns, 3)
        self.assertEqual(res.degrees_of_freedom, n - 3)

        n_est = np.array([res.params["nx"], res.params["ny"], res.params["nz"]])
        self.assertAlmostEqual(float(np.linalg.norm(n_est)), 1.0, places=12)
        # Smer normale (do predznaka)
        self.assertGreater(abs(float(np.dot(n_est, n_true))), 0.9999999)
        # Odmik ravnine: n_est . origin + d ~ 0
        self.assertAlmostEqual(
            float(np.dot(n_est, origin) + res.params["d"]), 0.0, delta=0.0001
        )
        self.assertAlmostEqual(res.sigma0, noise_std, delta=0.00005)

        # 2) Popolnoma navpična stena (nx = 1, ny = 0, nz = 0)
        y_wall = origin[1] + rng.uniform(-3.0, 3.0, size=n)
        z_wall = origin[2] + rng.uniform(-1.5, 1.5, size=n)
        x_wall = origin[0] + rng.normal(0.0, noise_std, size=n)
        pts_wall = np.column_stack((x_wall, y_wall, z_wall))

        res_wall = fit_plane(pts_wall)
        self.assertTrue(res_wall.converged)
        self.assertEqual(res_wall.extra["fixed_axis"], "x")
        self.assertAlmostEqual(abs(res_wall.params["nx"]), 1.0, delta=1e-4)
        self.assertAlmostEqual(res_wall.params["ny"], 0.0, delta=1e-4)
        self.assertAlmostEqual(res_wall.params["nz"], 0.0, delta=1e-4)
        self.assertAlmostEqual(res_wall.sigma0, noise_std, delta=0.00005)

    def test_fit_cylinder_vertical(self) -> None:
        """Preveri izravnavo pokončnega valja v velikih geodetskih koordinatah."""
        rng = np.random.default_rng(202)
        n = 3000
        true_x0 = 100298.5
        true_y0 = 461293.5
        true_r = 0.25
        noise_std = 0.0005

        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        z = 200.0 + rng.uniform(0.0, 1.5, size=n)
        r_noisy = true_r + rng.normal(0.0, noise_std, size=n)

        points = np.column_stack(
            (
                true_x0 + r_noisy * np.cos(theta),
                true_y0 + r_noisy * np.sin(theta),
                z,
            )
        ).astype(np.float64)

        res = fit_cylinder_vertical(points)
        self.assertTrue(res.converged)
        self.assertEqual(res.n_unknowns, 3)
        self.assertAlmostEqual(res.params["x0"], true_x0, delta=0.0001)
        self.assertAlmostEqual(res.params["y0"], true_y0, delta=0.0001)
        self.assertAlmostEqual(res.params["R"], true_r, delta=0.0001)
        self.assertAlmostEqual(res.sigma0, noise_std, delta=0.00005)

    def test_fit_cone_vertical(self) -> None:
        """Preveri izravnavo pokončnega stožca v velikih geodetskih koordinatah."""
        rng = np.random.default_rng(303)
        n = 3500
        true_x0 = 100298.5
        true_y0 = 461293.5
        true_z_vrh = 201.2
        true_k = 0.5  # tan(alpha)
        noise_std = 0.0005

        z = rng.uniform(200.0, 200.9, size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r_exact = true_k * np.abs(z - true_z_vrh)

        # Pravokotni šum na plašč stožca: enotska normala je (cos(a)*cos(th), cos(a)*sin(th), sin(a))
        cos_a = 1.0 / np.sqrt(1.0 + true_k * true_k)
        sin_a = true_k * cos_a
        perp_noise = rng.normal(0.0, noise_std, size=n)

        x = true_x0 + r_exact * np.cos(theta) + perp_noise * cos_a * np.cos(theta)
        y = true_y0 + r_exact * np.sin(theta) + perp_noise * cos_a * np.sin(theta)
        z_noisy = z + perp_noise * sin_a

        points = np.column_stack((x, y, z_noisy)).astype(np.float64)

        res = fit_cone_vertical(points)
        self.assertTrue(res.converged)
        self.assertEqual(res.n_unknowns, 4)
        self.assertAlmostEqual(res.params["x0"], true_x0, delta=0.0002)
        self.assertAlmostEqual(res.params["y0"], true_y0, delta=0.0002)
        self.assertAlmostEqual(res.params["z_vrh"], true_z_vrh, delta=0.0005)
        self.assertAlmostEqual(res.params["k"], true_k, delta=0.0005)
        self.assertAlmostEqual(res.sigma0, noise_std, delta=0.00005)


if __name__ == "__main__":
    unittest.main()
