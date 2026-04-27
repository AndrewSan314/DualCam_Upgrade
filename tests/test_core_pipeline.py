from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pose_pipeline.executor import force_unified_output
from pose_pipeline.naming import make_dual_output_paths, make_unified_output_path
from pose_pipeline.pipelines.judgement import _output_mode_for_judgement, _verts_by_cam_for_frame
from pose_pipeline.pipelines.judgement_alignment import align_sequence_umeyama
from pose_pipeline.pipelines.judgement_losses import (
    bone_length_loss,
    compute_reference_bone_lengths,
)
from pose_pipeline.pipelines.judgement_weights import build_view_weights
from pose_pipeline.pipelines.learnable_smplify import _update_fused_after_side_refine
from pose_pipeline.schema import load_pose_pkl, save_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.validation import validate_state
from pose_pipeline.visualization.render_alignment import (
    align_pose_to_render_reference,
    apply_render_alignment_transform,
    load_render_alignment_transform,
    save_render_alignment_transform,
)
from pose_pipeline.visualization.waveform import angle_between


class CorePipelineTests(unittest.TestCase):
    def test_naming_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = make_dual_output_paths(root, ["R", "L"])
            unified = make_unified_output_path(root, ["R", "L", "J"])

            self.assertEqual(left.name, "left_RL.pkl")
            self.assertEqual(right.name, "right_RL.pkl")
            self.assertEqual(unified.name, "unify_RLJ.pkl")
            self.assertTrue((root / "intermediate").exists())

    def test_save_load_pose_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pose.pkl"
            data = {
                "poses_3d": np.zeros((2, 3, 3), dtype=np.float32),
                "poses_2d": None,
                "confidence": np.ones((2, 3), dtype=np.float32),
                "smpl_params": None,
                "camera": None,
                "joint_names": ["a", "b", "c"],
                "skeleton_edges": [(0, 1), (1, 2)],
                "source": {"view": "unified", "pipeline_history": ["R"], "input_files": []},
                "metadata": {"created_by": "test"},
            }

            save_pose_pkl(data, path)
            loaded = load_pose_pkl(path)

            self.assertEqual(loaded["poses_3d"].shape, (2, 3, 3))
            self.assertEqual(loaded["joint_names"], ["a", "b", "c"])

    def test_angle_between(self) -> None:
        angles = angle_between(
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        )

        np.testing.assert_allclose(angles, [90.0, 180.0], atol=1e-5)

    def test_bone_length_loss_zero_for_reference(self) -> None:
        pose = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )
        edges = [(0, 1)]
        ref = compute_reference_bone_lengths(pose, edges)

        self.assertAlmostEqual(float(bone_length_loss(pose, edges, ref)), 0.0, places=6)

    def test_align_sequence_umeyama_recovers_base(self) -> None:
        base = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        candidate = base * 2.0 + np.asarray([3.0, -2.0, 1.0], dtype=np.float32)

        aligned, diagnostics = align_sequence_umeyama(candidate, base)

        self.assertIn("scale", diagnostics)
        np.testing.assert_allclose(aligned, base, atol=1e-5)

    def test_render_alignment_returns_aligned_copy_only(self) -> None:
        base = np.asarray(
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.2], [-0.5, 0.0, -0.1]],
                [[0.1, 0.0, 0.0], [0.1, 1.0, 0.0], [0.6, 0.0, 0.2], [-0.4, 0.0, -0.1]],
            ],
            dtype=np.float32,
        )
        theta = np.radians(25.0)
        rotation_y = np.asarray(
            [
                [np.cos(theta), 0.0, np.sin(theta)],
                [0.0, 1.0, 0.0],
                [-np.sin(theta), 0.0, np.cos(theta)],
            ],
            dtype=np.float32,
        )
        pose = 1.2 * (base @ rotation_y.T) + np.asarray([2.0, -1.0, 0.5])
        original = pose.copy()

        aligned, diagnostics = align_pose_to_render_reference(pose, base)

        self.assertIn("scale", diagnostics)
        np.testing.assert_allclose(aligned, base, atol=1e-5)
        np.testing.assert_allclose(pose, original, atol=0.0)

    def test_render_alignment_transform_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "render_transform.json"
            transform = {
                "scale": 2.0,
                "rotation": np.eye(3),
                "translation": np.asarray([1.0, -2.0, 3.0]),
            }
            pose = np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)

            save_render_alignment_transform(transform, path, {"source": "test"})
            loaded = load_render_alignment_transform(path)
            aligned = apply_render_alignment_transform(pose, loaded)

            np.testing.assert_allclose(aligned, [[[3.0, 2.0, 9.0]]], atol=1e-6)

    def test_build_view_weights_shapes(self) -> None:
        base = np.zeros((3, 2, 3), dtype=np.float32)
        weights, diagnostics = build_view_weights(
            {
                "left": {"confidence": np.ones((3, 2), dtype=np.float32)},
                "right": {"confidence": np.ones((3, 2), dtype=np.float32) * 0.5},
            },
            source_view=[],
            left_candidate=base,
            right_candidate=base,
            base_pose=base,
            joint_names=["a", "b"],
            config={"judgement_weight_smoothing_window": 1},
        )

        self.assertEqual(weights["left"].shape, (3, 2))
        self.assertEqual(weights["right"].shape, (3, 2))
        self.assertEqual(weights["base"].shape, (3, 2))
        self.assertGreater(diagnostics["left_mean"], diagnostics["right_mean"])

    def test_force_unified_output_from_dual(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pose_data = {
                "joint_names": ["a", "b"],
                "left": {
                    "poses_3d": np.ones((2, 2, 3), dtype=np.float32),
                    "confidence": np.ones((2, 2), dtype=np.float32),
                    "metadata": {},
                },
                "right": {
                    "poses_3d": np.ones((2, 2, 3), dtype=np.float32) * 3.0,
                    "confidence": np.ones((2, 2), dtype=np.float32) * 0.5,
                    "metadata": {},
                },
                "fused": {"poses_3d": None, "confidence": None, "metadata": {}},
                "logs": [],
            }
            state = PipelineState(
                mode="dual",
                output_dir=output_dir,
                history=["L"],
                pose_data=pose_data,
            )

            unified = force_unified_output(state)
            validate_state(unified)

            self.assertEqual(unified.mode, "unified")
            self.assertEqual(unified.unified_pkl, output_dir / "intermediate" / "unify_L.pkl")
            self.assertTrue(unified.unified_pkl.exists())
            np.testing.assert_allclose(unified.pose_data["fused"]["poses_3d"], 2.0)

    def test_judgement_dual_output_policy(self) -> None:
        state = PipelineState(mode="dual")

        self.assertEqual(
            _output_mode_for_judgement(state, {"__remaining_sequence": "R"}),
            "dual",
        )
        self.assertEqual(
            _output_mode_for_judgement(state, {"__remaining_sequence": "L"}),
            "unified",
        )
        self.assertEqual(
            _output_mode_for_judgement(
                state,
                {
                    "__remaining_sequence": "R",
                    "judgement_output_mode_when_dual": "unified",
                },
            ),
            "unified",
        )

    def test_l_after_j_updates_fused_with_judgement_weights(self) -> None:
        pose_data = {
            "left": {
                "poses_3d": np.ones((2, 2, 3), dtype=np.float32) * 10.0,
                "confidence": np.ones((2, 2), dtype=np.float32),
            },
            "right": {
                "poses_3d": np.ones((2, 2, 3), dtype=np.float32) * 20.0,
                "confidence": np.ones((2, 2), dtype=np.float32),
            },
            "fused": {
                "poses_3d": np.ones((2, 2, 3), dtype=np.float32) * 100.0,
                "confidence": np.ones((2, 2), dtype=np.float32),
                "metadata": {
                    "pose_judgement_optimization": {
                        "diagnostics": {
                            "view_weights": {
                                "left_mean": 0.2,
                                "right_mean": 0.3,
                                "base_mean": 0.5,
                            }
                        }
                    }
                },
            },
        }

        status = _update_fused_after_side_refine(pose_data, {"learnable_fused_update": "auto"})

        self.assertEqual(status, "recomputed_from_left_right_with_judgement_weights")
        np.testing.assert_allclose(pose_data["fused"]["poses_3d"], 58.0, atol=1e-6)

    def test_judgement_mesh_payload_includes_faces_camera_and_image_size(self) -> None:
        raw_person = {
            "verts": np.zeros((2, 4, 3), dtype=np.float32),
            "faces": np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        }
        side = {
            "raw_person": raw_person,
            "camera_intrinsics": {"fx": 100.0, "fy": 110.0, "cx": 50.0, "cy": 60.0},
            "video_info": {"width": 1920, "height": 1080},
        }
        payload = _verts_by_cam_for_frame({"left": side, "right": side}, 1)

        self.assertIsNotNone(payload)
        cam1 = payload["camera1"]
        self.assertEqual(cam1["vertices"].shape, (4, 3))
        self.assertEqual(cam1["faces"].shape, (2, 3))
        np.testing.assert_allclose(
            cam1["camera"]["K"],
            [[100.0, 0.0, 50.0], [0.0, 110.0, 60.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(cam1["image_size"], (1920, 1080))


if __name__ == "__main__":
    unittest.main()
