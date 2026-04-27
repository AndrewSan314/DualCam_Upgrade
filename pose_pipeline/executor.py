from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from pose_pipeline.naming import make_unified_output_path
from pose_pipeline.pipelines.judgement import run_J
from pose_pipeline.pipelines.learnable_smplify import run_L
from pose_pipeline.pipelines.refinement import run_R
from pose_pipeline.schema import pose_data_view_to_standard_schema, save_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.validation import validate_state


PIPELINE_FUNCS: dict[str, Callable[[PipelineState, dict[str, Any]], PipelineState]] = {
    "R": run_R,
    "J": run_J,
    "L": run_L,
}


def run_pipeline_sequence(
    sequence: str, state: PipelineState, config: dict[str, Any]
) -> PipelineState:
    state.snapshots["input"] = _select_final_pose(state).copy()
    for index, step in enumerate(sequence):
        if step not in PIPELINE_FUNCS:
            raise ValueError(f"Unknown pipeline step: {step}")

        print(f"  Dang chay pipeline {step} (mode={state.mode})...", flush=True)
        step_start = time.perf_counter()
        before_mode = state.mode
        step_config = dict(config)
        step_config["__remaining_sequence"] = sequence[index + 1 :]
        step_config["__current_step"] = step
        state = PIPELINE_FUNCS[step](state, step_config)
        validate_state(state)
        elapsed = time.perf_counter() - step_start
        _record_transition(state, step, before_mode, state.mode, elapsed)
        state.snapshots[step] = _select_final_pose(state).copy()
        print(
            f"  Xong pipeline {step} sau {elapsed:.2f}s "
            f"(mode={state.mode})",
            flush=True,
        )
        print_state_files(state)
    state = force_unified_output(state)
    validate_state(state)
    return state


def print_state_files(state: PipelineState) -> None:
    if state.mode == "dual":
        print(f"    left: {state.left_pkl}", flush=True)
        print(f"    right: {state.right_pkl}", flush=True)
    else:
        print(f"    unified: {state.unified_pkl}", flush=True)


def _select_final_pose(state: PipelineState):
    if state.pose_data is None:
        raise ValueError("PipelineState.pose_data is required by the executor")
    fused = state.pose_data["fused"].get("poses_3d")
    if fused is not None:
        return fused
    return (state.pose_data["left"]["poses_3d"] + state.pose_data["right"]["poses_3d"]) / 2.0


def force_unified_output(state: PipelineState) -> PipelineState:
    if state.mode == "unified":
        return state
    if state.pose_data is None:
        raise ValueError("PipelineState.pose_data is required by force_unified_output")
    if state.output_dir is None:
        raise ValueError("PipelineState.output_dir is required by force_unified_output")
    if not state.history:
        raise ValueError("Cannot force unified output before any pipeline step has run")

    left = np.asarray(state.pose_data["left"]["poses_3d"], dtype=np.float32)
    right = np.asarray(state.pose_data["right"]["poses_3d"], dtype=np.float32)
    frame_count = min(left.shape[0], right.shape[0])
    state.pose_data["fused"]["poses_3d"] = ((left[:frame_count] + right[:frame_count]) / 2.0).astype(np.float32)

    left_conf = state.pose_data["left"].get("confidence")
    right_conf = state.pose_data["right"].get("confidence")
    if left_conf is not None and right_conf is not None:
        state.pose_data["fused"]["confidence"] = (
            (
                np.asarray(left_conf[:frame_count], dtype=np.float32)
                + np.asarray(right_conf[:frame_count], dtype=np.float32)
            )
            / 2.0
        ).astype(np.float32)

    state.pose_data["fused"].setdefault("metadata", {})["force_unify"] = {
        "method": "left_right_average",
        "reason": "final sequence output must be unified for render/waveform/benchmark",
        "frame_count": int(frame_count),
    }
    state.pose_data["logs"].append("executor: forced final dual state to unified output")
    before_mode = state.mode

    output_path = make_unified_output_path(state.output_dir, state.history)
    save_pose_pkl(
        pose_data_view_to_standard_schema(
            state.pose_data,
            "unified",
            state.history,
            created_by="pose_pipeline.executor.force_unified_output",
        ),
        output_path,
    )
    state.mode = "unified"
    state.unified_pkl = output_path
    state.artifacts["".join(state.history)] = output_path
    _record_transition(state, "force_unify", before_mode, state.mode, 0.0)
    return state


def _record_transition(
    state: PipelineState,
    step: str,
    before_mode: str,
    after_mode: str,
    seconds: float,
) -> None:
    state.metadata.setdefault("transitions", []).append(
        {
            "step": step,
            "before_mode": before_mode,
            "after_mode": after_mode,
            "seconds": round(float(seconds), 3),
            "history": list(state.history),
            "artifacts": {key: str(path) for key, path in state.artifacts.items()},
        }
    )
