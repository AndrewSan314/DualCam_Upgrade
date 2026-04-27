from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_vendor_main():
    path = (
        Path(__file__).resolve().parents[1]
        / "vendor"
        / "pose_judgement_optimization"
        / "main.py"
    )
    spec = importlib.util.spec_from_file_location("vendor_pose_judgement_main_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VendorVisibilityTests(unittest.TestCase):
    def test_joint_in_front_of_mesh_plane_is_visible(self) -> None:
        vendor = _load_vendor_main()
        verts = np.asarray(
            [
                [-0.2, -0.2, 2.0],
                [0.2, -0.2, 2.0],
                [0.0, 0.2, 2.0],
            ],
            dtype=float,
        )
        visibility = vendor.compute_visibility_from_mesh_vertices(
            {"joint": [0.0, 0.0, 1.8]},
            verts,
            faces=np.asarray([[0, 1, 2]], dtype=int),
            grid_size=64,
            occlusion_tau=0.02,
        )

        self.assertTrue(visibility["joint"])

    def test_joint_behind_mesh_plane_is_occluded(self) -> None:
        vendor = _load_vendor_main()
        verts = np.asarray(
            [
                [-0.2, -0.2, 2.0],
                [0.2, -0.2, 2.0],
                [0.0, 0.2, 2.0],
            ],
            dtype=float,
        )
        visibility = vendor.compute_visibility_from_mesh_vertices(
            {"joint": [0.0, 0.0, 2.2]},
            verts,
            faces=np.asarray([[0, 1, 2]], dtype=int),
            grid_size=64,
            occlusion_tau=0.02,
        )

        self.assertFalse(visibility["joint"])

    def test_joint_outside_projected_buffer_is_visible_not_hard_occluded(self) -> None:
        vendor = _load_vendor_main()
        verts = np.asarray(
            [
                [-0.2, -0.2, 2.0],
                [0.2, -0.2, 2.0],
                [0.0, 0.2, 2.0],
            ],
            dtype=float,
        )
        visibility = vendor.compute_visibility_from_mesh_vertices(
            {"joint": [5.0, 5.0, 2.2]},
            verts,
            faces=np.asarray([[0, 1, 2]], dtype=int),
            grid_size=64,
            occlusion_tau=0.02,
        )

        self.assertTrue(visibility["joint"])

    def test_normalized_fallback_classifies_front_and_back_on_same_ray(self) -> None:
        vendor = _load_vendor_main()
        verts = np.asarray(
            [
                [0.8, -0.2, 2.0],
                [1.2, -0.2, 2.0],
                [1.0, 0.2, 2.0],
            ],
            dtype=float,
        )
        visibility = vendor.compute_visibility_from_mesh_vertices(
            {
                "front": [0.9, 0.0, 1.8],
                "back": [1.1, 0.0, 2.2],
            },
            verts,
            faces=np.asarray([[0, 1, 2]], dtype=int),
            grid_size=64,
            occlusion_tau=0.02,
        )

        self.assertTrue(visibility["front"])
        self.assertFalse(visibility["back"])

    def test_visibility_uses_camera_ray_depth_not_raw_xy_or_abs_z(self) -> None:
        vendor = _load_vendor_main()
        verts = np.asarray(
            [
                [0.9, -0.1, 1.0],
                [1.1, -0.1, 1.0],
                [1.0, 0.1, 1.0],
            ],
            dtype=float,
        )
        faces = np.asarray([[0, 1, 2]], dtype=int)
        joints = {
            "behind_same_ray": [2.0, 0.0, 2.0],
            "front_same_ray": [0.5, 0.0, 0.5],
            "behind_negative_z": [-2.0, 0.0, -2.0],
        }

        visibility = vendor.compute_visibility_from_mesh_vertices(
            joints,
            verts,
            faces=faces,
            grid_size=64,
            occlusion_tau=0.01,
        )

        self.assertFalse(visibility["behind_same_ray"])
        self.assertTrue(visibility["front_same_ray"])
        self.assertTrue(visibility["behind_negative_z"])

    def test_visibility_supports_intrinsics_payload(self) -> None:
        vendor = _load_vendor_main()
        camera = {
            "K": np.asarray(
                [
                    [100.0, 0.0, 50.0],
                    [0.0, 100.0, 50.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        }
        verts = np.asarray(
            [
                [-0.1, -0.1, 1.0],
                [0.1, -0.1, 1.0],
                [0.0, 0.1, 1.0],
            ],
            dtype=float,
        )
        faces = np.asarray([[0, 1, 2]], dtype=int)
        visibility = vendor.compute_visibility_from_mesh_vertices(
            {"joint": [0.0, 0.0, 2.0]},
            verts,
            faces=faces,
            camera=camera,
            image_size=(100, 100),
            grid_size=64,
            occlusion_tau=0.01,
        )

        self.assertFalse(visibility["joint"])


if __name__ == "__main__":
    unittest.main()
