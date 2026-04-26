from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import numpy as np


SMPL_JOINT_NAMES = [
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

BODY13_JOINT_NAMES = [
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_hand",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
]

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

SMPL_PARENTS = np.asarray(
    [
        -1,
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        12,
        12,
        12,
        13,
        14,
        16,
        17,
        18,
        19,
        20,
        21,
    ],
    dtype=int,
)

SMPL_REST_OFFSETS = np.asarray(
    [
        [0.000, 0.000, 0.000],
        [0.090, -0.090, 0.000],
        [-0.090, -0.090, 0.000],
        [0.000, 0.120, 0.000],
        [0.000, -0.430, 0.020],
        [0.000, -0.430, 0.020],
        [0.000, 0.120, 0.000],
        [0.000, -0.420, -0.010],
        [0.000, -0.420, -0.010],
        [0.000, 0.120, 0.000],
        [0.000, -0.080, 0.120],
        [0.000, -0.080, 0.120],
        [0.000, 0.130, 0.000],
        [0.070, 0.040, 0.000],
        [-0.070, 0.040, 0.000],
        [0.000, 0.160, 0.020],
        [0.150, 0.020, 0.000],
        [-0.150, 0.020, 0.000],
        [0.280, 0.000, 0.000],
        [-0.280, 0.000, 0.000],
        [0.250, 0.000, 0.000],
        [-0.250, 0.000, 0.000],
        [0.080, 0.000, 0.000],
        [-0.080, 0.000, 0.000],
    ],
    dtype=float,
)


@dataclass
class PoseSequence:
    source_path: Path
    frames: Dict[int | str, Dict[str, np.ndarray]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None
    raw_person: Any = None

    @property
    def frame_ids(self) -> list[int | str]:
        from pose_analysis import sort_frame_ids

        return sort_frame_ids(self.frames)


def load_pose_sequence(
    path: str | Path,
    person_id: int = 0,
    smpl_model_dir: str | Path | None = None,
    gender: str = "neutral",
) -> PoseSequence:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot find input PKL: {source_path}")

    data = _load_pickle(source_path)
    person = _select_person(data, person_id)

    explicit_joints = _extract_explicit_joints(person)
    if explicit_joints is not None:
        frame_ids = _extract_frame_ids(person, len(explicit_joints))
        frames = _array_to_frames(
            explicit_joints,
            frame_ids,
            _extract_joint_names(person, explicit_joints.shape[1]),
        )
        representation = "explicit_joints"
    else:
        poses = _extract_pose_array(person)
        if poses is None:
            raise ValueError(
                f"{source_path} does not contain a supported joint array or SMPL pose array."
            )

        frame_ids = _extract_frame_ids(person, len(poses))
        joints = _joints_from_smpl_model_if_available(
            poses=poses,
            person=person,
            smpl_model_dir=smpl_model_dir,
            gender=gender,
        )
        if joints is None:
            joints = _joints_from_pose_surrogate(
                poses=poses,
                trans=_extract_trans_array(person, len(poses)),
            )
            representation = "smpl_pose_kinematic_surrogate"
        else:
            representation = "smpl_pose_smpl_model"
        frames = _array_to_frames(joints, frame_ids)

    return PoseSequence(
        source_path=source_path,
        frames=frames,
        metadata={
            "representation": representation,
            "person_id": person_id,
            "frame_count": len(frames),
            "smpl_model_dir": str(smpl_model_dir) if smpl_model_dir else None,
        },
        raw_data=data,
        raw_person=person,
    )


def _load_pickle(path: Path) -> Any:
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def _select_person(data: Any, person_id: int) -> Any:
    if isinstance(data, (list, tuple)):
        return data[person_id]
    if isinstance(data, Mapping):
        if person_id in data:
            return data[person_id]
        text_key = str(person_id)
        if text_key in data:
            return data[text_key]
        if any(
            key in data
            for key in ("pose", "poses", "joints", "joints3d", "keypoints3d")
        ):
            return data
        for key in ("results", "people", "persons", "tracks"):
            value = data.get(key)
            if isinstance(value, (list, tuple)):
                return value[person_id]
            if isinstance(value, Mapping):
                return _select_person(value, person_id)
    raise ValueError(f"Cannot select person_id={person_id} from this PKL structure.")


def _extract_frame_ids(person: Any, length: int) -> list[int | str]:
    if isinstance(person, Mapping):
        for key in ("frame_ids", "frame_id", "frames", "img_frame_ids"):
            if key in person:
                values = np.asarray(person[key]).tolist()
                if not isinstance(values, list):
                    values = [values]
                if len(values) == length:
                    return [
                        int(v) if isinstance(v, (np.integer, int)) else str(v)
                        for v in values
                    ]
    return list(range(length))


def _extract_explicit_joints(person: Any) -> np.ndarray | None:
    if isinstance(person, Mapping):
        for key in (
            "joints3d",
            "joints_3d",
            "joints",
            "keypoints3d",
            "keypoints_3d",
            "pred_joints",
            "smpl_joints",
        ):
            if key in person:
                array = np.asarray(person[key], dtype=float)
                if array.ndim == 3 and array.shape[-1] >= 3:
                    return array[..., :3]
        if _looks_like_frame_joint_dict(person):
            return _frame_joint_dict_to_array(person)
    return None


def _extract_joint_names(person: Any, count: int) -> list[str] | None:
    if isinstance(person, Mapping):
        for key in ("joint_names", "joints_name", "keypoint_names", "names"):
            if key in person:
                names = [str(name) for name in person[key]]
                if len(names) == count:
                    return names
    if count == len(BODY13_JOINT_NAMES):
        return BODY13_JOINT_NAMES
    if count == len(BODY25_JOINT_NAMES):
        return BODY25_JOINT_NAMES
    if count == len(SMPL_JOINT_NAMES):
        return SMPL_JOINT_NAMES
    return None


def _extract_pose_array(person: Any) -> np.ndarray | None:
    if isinstance(person, Mapping):
        for key in ("pose", "poses", "smpl_pose", "thetas", "theta"):
            if key in person:
                array = np.asarray(person[key], dtype=float)
                if array.ndim == 2 and array.shape[1] >= 24 * 3:
                    return array[:, : 24 * 3]
                if array.ndim == 3 and array.shape[1:] == (24, 3):
                    return array.reshape(array.shape[0], 24 * 3)
    return None


def _extract_trans_array(person: Any, length: int) -> np.ndarray:
    if isinstance(person, Mapping):
        for key in ("trans", "transl", "translation", "cam_trans"):
            if key in person:
                array = np.asarray(person[key], dtype=float)
                if array.ndim == 2 and array.shape[0] == length and array.shape[1] >= 3:
                    return array[:, :3]
    return np.zeros((length, 3), dtype=float)


def _extract_betas(person: Any, length: int) -> np.ndarray | None:
    if isinstance(person, Mapping):
        for key in ("betas", "shape", "shapes"):
            if key in person:
                betas = np.asarray(person[key], dtype=float)
                if betas.ndim == 1:
                    return np.repeat(betas[np.newaxis, :10], length, axis=0)
                if betas.ndim == 2 and betas.shape[0] == length:
                    return betas[:, :10]
                if betas.ndim == 2 and betas.shape[0] == 1:
                    return np.repeat(betas[:, :10], length, axis=0)
    return None


def _joints_from_smpl_model_if_available(
    poses: np.ndarray,
    person: Any,
    smpl_model_dir: str | Path | None,
    gender: str,
) -> np.ndarray | None:
    if smpl_model_dir is None:
        return None
    try:
        import torch
        import smplx
    except Exception:
        return None

    model_dir = Path(smpl_model_dir)
    if not model_dir.exists():
        return None

    length = poses.shape[0]
    betas = _extract_betas(person, length)
    if betas is None:
        betas = np.zeros((length, 10), dtype=float)
    trans = _extract_trans_array(person, length)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender=gender,
        ext="pkl",
        batch_size=length,
    ).to(device)

    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(betas[:, :10], dtype=torch.float32, device=device),
            global_orient=torch.as_tensor(
                poses[:, :3], dtype=torch.float32, device=device
            ),
            body_pose=torch.as_tensor(
                poses[:, 3:72], dtype=torch.float32, device=device
            ),
            transl=torch.as_tensor(trans, dtype=torch.float32, device=device),
        )
    joints = output.joints.detach().cpu().numpy()
    return joints[:, : len(SMPL_JOINT_NAMES), :3]


def _joints_from_pose_surrogate(poses: np.ndarray, trans: np.ndarray) -> np.ndarray:
    length = poses.shape[0]
    pose_vectors = poses[:, : 24 * 3].reshape(length, 24, 3)
    all_coords = np.zeros((length, 24, 3), dtype=float)

    for frame_index, rotvecs in enumerate(pose_vectors):
        rotations = np.zeros((24, 3, 3), dtype=float)
        coords = np.zeros((24, 3), dtype=float)
        for joint_index in range(24):
            parent = SMPL_PARENTS[joint_index]
            local_rotation = _rodrigues(rotvecs[joint_index])
            if parent == -1:
                rotations[joint_index] = local_rotation
                coords[joint_index] = trans[frame_index]
            else:
                rotations[joint_index] = rotations[parent] @ local_rotation
                coords[joint_index] = (
                    coords[parent] + rotations[parent] @ SMPL_REST_OFFSETS[joint_index]
                )
        all_coords[frame_index] = coords
    return all_coords


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3, dtype=float)
    axis = rotvec / theta
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def _array_to_frames(
    joints: np.ndarray,
    frame_ids: list[int | str],
    joint_names: list[str] | None = None,
) -> Dict[int | str, Dict[str, np.ndarray]]:
    frames: Dict[int | str, Dict[str, np.ndarray]] = {}
    for frame_id, coords in zip(frame_ids, joints):
        names = joint_names
        if names is None:
            names = _extract_joint_names({}, coords.shape[0])
        if names is None:
            names = [f"joint_{idx:02d}" for idx in range(coords.shape[0])]
        frames[frame_id] = {
            "frame_id": frame_id,
            **{
                name: np.asarray(coords[index, :3], dtype=float)
                for index, name in enumerate(names)
            },
        }
    return frames


def _looks_like_frame_joint_dict(value: Mapping[Any, Any]) -> bool:
    if not value:
        return False
    first_value = next(iter(value.values()))
    return isinstance(first_value, Mapping) and all(
        np.asarray(point).shape[-1] >= 3 for point in first_value.values()
    )


def _frame_joint_dict_to_array(value: Mapping[Any, Mapping[str, Any]]) -> np.ndarray:
    frame_keys = sorted(value)
    first_frame = value[frame_keys[0]]
    joint_names = list(first_frame)
    frames = []
    for frame_key in frame_keys:
        frames.append(
            [
                np.asarray(value[frame_key][name], dtype=float)[:3]
                for name in joint_names
            ]
        )
    return np.asarray(frames, dtype=float)
