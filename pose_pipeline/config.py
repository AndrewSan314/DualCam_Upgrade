from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT.parent / "AnhVTM-20260426T073026Z-3-001" / "AnhVTM"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

SUPPORTED_SEQUENCES = (
    "RJL",
    "LJR",
    "LJRL",
    "LRJ",
    "LRJL",
    "RLJ",
    "RLJL",
    "JLR",
    "JLRL",
    "JRL",
)

PIPELINE_LABELS = {
    "R": "Pose Refinement Optimization",
    "J": "Pose Judgement Optimization",
    "L": "Learnable SMPLify",
}

BODY25_JOINT_NAMES = [
    "nose",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "mid_hip",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

SMPL24_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

BODY25_SKELETON_EDGES = [
    ("nose", "neck"),
    ("nose", "right_eye"),
    ("right_eye", "right_ear"),
    ("nose", "left_eye"),
    ("left_eye", "left_ear"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "mid_hip"),
    ("mid_hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("mid_hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_heel"),
    ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_heel"),
    ("right_big_toe", "right_small_toe"),
]

SMPL24_SKELETON_EDGES = [
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
]

SKELETON_EDGES = BODY25_SKELETON_EDGES
FALLBACK_SKELETON_EDGES = SMPL24_SKELETON_EDGES
