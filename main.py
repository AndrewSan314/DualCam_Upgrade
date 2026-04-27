from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pose_pipeline.benchmark.evaluator import evaluate_benchmark
from pose_pipeline.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, SUPPORTED_SEQUENCES
from pose_pipeline.executor import run_pipeline_sequence as run_state_pipeline_sequence
from pose_pipeline.io_utils.input_loader import load_inputs
from pose_pipeline.schema import load_pose_pkl
from pose_pipeline.state import PipelineState
from pose_pipeline.visualization.render_alignment import (
    apply_render_alignment_transform,
    default_render_alignment_cache_path,
    estimate_render_alignment_transform,
    load_render_alignment_transform,
    save_render_alignment_transform,
    transform_diagnostics,
)
from pose_pipeline.visualization.video_composer import compose_output_video
from pose_pipeline.visualization.waveform import draw_waveform_analysis


def main() -> int:
    configure_console()
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    config = vars(args)
    start = time.perf_counter()

    input_dir, benchmark_path, sequences = resolve_console_args(args)
    print(f"Đang đọc input từ: {Path(input_dir).resolve()}", flush=True)
    pose_data = load_inputs(input_dir)
    print(
        "Đã đọc input: "
        f"{pose_data['metadata']['frame_count']} frame, "
        f"{pose_data['metadata']['joint_count']} joint.",
        flush=True,
    )
    ensure_output_dirs(output_dir)
    prepare_default_render_alignment_reference(args, pose_data, input_dir, output_dir, config)

    sequence_results = []
    for sequence in sequences:
        seq_start = time.perf_counter()
        print(f"\nBắt đầu sequence {sequence}", flush=True)
        current = clone_initial_pose_data(pose_data)
        state = PipelineState.from_pose_data(
            current,
            input_dir=Path(input_dir).resolve(),
            output_dir=output_dir,
            benchmark_path=Path(benchmark_path).resolve() if benchmark_path else None,
        )
        state = run_state_pipeline_sequence(sequence, state, config)
        current = state.pose_data
        if current is None:
            raise ValueError("Pipeline executor finished without pose_data")
        pose_snapshots = state.snapshots
        final_pose = select_final_pose(current)
        render_pose = select_render_pose(final_pose, pose_snapshots, args.render_stage)
        render_alignment = None
        if not args.skip_video:
            render_pose, render_alignment = align_render_pose_if_requested(
                render_pose,
                final_pose,
                pose_snapshots,
                args.render_align_stage,
                args.render_align_reference,
                args.render_align_transform,
                args.render_align_save_transform,
                args.render_align_cache,
                getattr(args, "default_render_reference_used", False),
            )

        video_path = None
        figure_paths = []
        benchmark_result = None
        if not args.skip_video:
            print("Đang tạo video ghép...", flush=True)
            video_path = compose_output_video(
                current["left"]["video_path"],
                current["right"]["video_path"],
                render_pose,
                current["joint_names"],
                output_dir / "videos" / f"{sequence}_cam_left_right_3D_poses.mp4",
                render_zoom=float(args.render_zoom),
                render_view=str(args.render_view),
                render_yaw_deg=float(args.render_yaw_deg),
                render_pitch_deg=float(args.render_pitch_deg),
                render_y_up=render_up_axis_to_bool(args.render_up_axis),
            )
            print(f"Đã tạo video: {video_path}", flush=True)
        if not args.skip_waveform:
            print("Đang vẽ waveform analysis...", flush=True)
            figure_paths = draw_waveform_analysis(
                final_pose,
                current["joint_names"],
                output_dir / "figures" / sequence,
            )
            print(f"Đã vẽ {len(figure_paths)} biểu đồ.", flush=True)
        if benchmark_path:
            print("Đang đánh giá benchmark...", flush=True)
            benchmark_result = evaluate_benchmark(
                final_pose,
                benchmark_path,
                output_dir / "logs" / f"{sequence}_benchmark_result.json",
            )

        sequence_results.append(
            {
                "sequence": sequence,
                "seconds": round(time.perf_counter() - seq_start, 3),
                "video": str(video_path) if video_path else None,
                "render_stage": args.render_stage,
                "render_view": args.render_view,
                "render_yaw_deg": args.render_yaw_deg,
                "render_pitch_deg": args.render_pitch_deg,
                "render_up_axis": args.render_up_axis,
                "render_alignment": render_alignment,
                "figures": [str(path) for path in figure_paths],
                "benchmark": benchmark_result,
                "metadata": {
                    "left": current["left"].get("metadata", {}),
                    "right": current["right"].get("metadata", {}),
                    "fused": current["fused"].get("metadata", {}),
                    "state_mode": state.mode,
                    "history": state.history,
                    "artifacts": {key: str(path) for key, path in state.artifacts.items()},
                    "transitions": state.metadata.get("transitions", []),
                },
                "logs": current["logs"],
            }
        )

    run_log = {
        "input_dir": str(Path(input_dir).resolve()),
        "sequences": sequence_results,
        "total_seconds": round(time.perf_counter() - start, 3),
    }
    log_path = output_dir / "logs" / "run_log.json"
    log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")

    print_summary(run_log, log_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-camera 3D pose pipeline using vendored original modules."
    )
    parser.add_argument("--input-dir", default=None, help="Folder containing cam_left/right mp4 and left/right pkl.")
    parser.add_argument("--benchmark", default=None, help="Benchmark PKL path, or $ to skip.")
    parser.add_argument("--sequence", default=None, help="One sequence such as RJL. Empty means all supported.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--skip-waveform", action="store_true")
    parser.add_argument("--render-zoom", type=float, default=1.25)
    parser.add_argument("--render-view", default="front", choices=("front", "side", "top", "orbit"))
    parser.add_argument("--render-yaw-deg", type=float, default=45.0)
    parser.add_argument("--render-pitch-deg", type=float, default=55.0)
    parser.add_argument("--render-up-axis", default="auto", choices=("auto", "y_up", "y_down"))
    parser.add_argument("--render-stage", default="final", choices=("final", "input", "R", "J", "L"))
    parser.add_argument("--render-align-stage", default="none", choices=("none", "input", "R", "J", "L", "final"))
    parser.add_argument(
        "--render-align-reference",
        default=None,
        help=(
            "Optional standard pose PKL whose coordinate frame is used only for rendering. "
            "If omitted, the run-local iter1 reference is generated under the output directory."
        ),
    )
    parser.add_argument(
        "--render-align-transform",
        default=None,
        help=(
            "JSON or run_log.json containing a fixed render similarity transform. "
            "If the file does not exist and a render reference/stage is provided, it is created."
        ),
    )
    parser.add_argument(
        "--render-align-save-transform",
        default=None,
        help="Save the estimated render alignment transform to this JSON path.",
    )
    parser.add_argument(
        "--render-align-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When --render-align-reference is used, cache the estimated transform beside "
            "that reference and reuse it on later runs."
        ),
    )
    parser.add_argument(
        "--default-render-align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use an iteration-1 R prepass as the default render reference when no "
            "explicit render alignment option is set. If the reference PKL is missing, "
            "it is generated once under the output directory and reused."
        ),
    )
    parser.add_argument("--max-judgement-frames", type=int, default=None)
    parser.add_argument("--judgement-log-interval", type=int, default=10)
    parser.add_argument(
        "--judgement-mode",
        default="safe_fusion",
        choices=("metadata_only", "safe_fusion", "temporal_multiview_optimize", "full_legacy"),
    )
    parser.add_argument(
        "--judgement-coordinate-alignment",
        default="sequence_umeyama",
        choices=("none", "root_scale", "sequence_umeyama"),
    )
    parser.add_argument("--judgement-window-size", type=int, default=32)
    parser.add_argument("--judgement-stride", type=int, default=8)
    parser.add_argument("--judgement-iters", type=int, default=80)
    parser.add_argument("--judgement-lr", type=float, default=0.03)
    parser.add_argument("--judgement-lambda-data", type=float, default=1.0)
    parser.add_argument("--judgement-lambda-prior", type=float, default=0.4)
    parser.add_argument("--judgement-lambda-bone", type=float, default=8.0)
    parser.add_argument("--judgement-lambda-temp", type=float, default=0.3)
    parser.add_argument("--judgement-lambda-acc", type=float, default=1.5)
    parser.add_argument("--judgement-lambda-floor", type=float, default=3.0)
    parser.add_argument("--judgement-lambda-contact", type=float, default=2.0)
    parser.add_argument("--judgement-camera-disagreement-threshold-m", type=float, default=0.25)
    parser.add_argument("--judgement-min-base-prior-weight", type=float, default=0.2)
    parser.add_argument("--judgement-max-base-prior-weight", type=float, default=1.0)
    parser.add_argument("--judgement-safe-max-joint-shift-m", type=float, default=0.12)
    parser.add_argument("--judgement-temporal-smoothing-window", type=int, default=5)
    parser.add_argument("--judgement-temporal-smoothing-alpha", type=float, default=0.65)
    parser.add_argument(
        "--judgement-temporal-smoothing-target",
        default="correction",
        choices=("correction", "pose"),
    )
    parser.add_argument("--judgement-temporal-median-window", type=int, default=3)
    parser.add_argument("--judgement-weight-smoothing-window", type=int, default=5)
    parser.add_argument("--judgement-weight-smoothing-alpha", type=float, default=0.65)
    parser.add_argument("--judgement-spike-repair", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judgement-spike-acceleration-ratio-vs-base", type=float, default=1.8)
    parser.add_argument("--judgement-spike-acceleration-m-per-frame2", type=float, default=0.12)
    parser.add_argument("--judgement-spike-repair-alpha", type=float, default=0.85)
    parser.add_argument("--judgement-spike-repair-passes", type=int, default=2)
    parser.add_argument("--judgement-floor-axis", type=int, default=2)
    parser.add_argument("--judgement-floor-value", type=float, default=0.0)
    parser.add_argument("--judgement-max-bone-deviation-ratio", type=float, default=0.20)
    parser.add_argument("--judgement-max-joint-velocity-m-per-frame", type=float, default=0.35)
    parser.add_argument("--judgement-max-joint-acceleration-m-per-frame2", type=float, default=0.45)
    parser.add_argument("--judgement-max-mean-acceleration-ratio-vs-base", type=float, default=1.35)
    parser.add_argument("--judgement-max-acceleration-ratio-vs-base", type=float, default=2.25)
    parser.add_argument("--judgement-pose-update", default="off", choices=("off", "blend", "full"))
    parser.add_argument(
        "--judgement-output-mode-when-dual",
        default="auto",
        choices=("auto", "dual", "unified"),
        help="When J receives dual left/right input, keep dual if R remains by default.",
    )
    parser.add_argument("--judgement-blend-alpha", type=float, default=0.2)
    parser.add_argument("--judgement-max-joint-shift-m", type=float, default=0.15)
    parser.add_argument("--judgement-regularization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judgement-regularization-lambda", type=float, default=10.0)
    parser.add_argument("--judgement-vendor-max-joint-move-m", type=float, default=0.10)
    parser.add_argument(
        "--judgement-vendor-reject-excessive-displacement",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--soft-tail-temperature", type=float, default=0.05)
    parser.add_argument("--soft-tail-weight", type=float, default=1.0)
    parser.add_argument("--learnable-smplify-src", default=None)
    parser.add_argument("--learnable-checkpoint", default=None)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument(
        "--learnable-fused-update",
        default="auto",
        choices=("auto", "average", "judgement_weights", "off"),
        help="How L updates fused output after refining left/right raw SMPL.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-refiner", default="smooth", choices=("smooth", "none"))
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--opencap-iterations", type=int, default=20)
    parser.add_argument("--opencap-height-m", type=float, default=1.7)
    parser.add_argument("--opencap-mass-kg", type=float, default=70.0)
    parser.add_argument("--opencap-sex", default="m", choices=("m", "f"))
    parser.add_argument("--opencap-activity", default="other")
    parser.add_argument("--opencap-max-frames", type=int, default=None)
    return parser.parse_args()


def resolve_console_args(args: argparse.Namespace) -> tuple[str, str | None, list[str]]:
    if args.no_prompt:
        input_dir = args.input_dir or str(DEFAULT_INPUT_DIR)
        benchmark = None if args.benchmark in (None, "$", "") else args.benchmark
        sequence_text = args.sequence or "RJL"
    else:
        print("Chương trình có các pipeline sau:")
        print("    (0) đọc hai file video và hai file pkl tương ứng")
        print("    (1) Pose Refinement Optimization")
        print("    (2) Pose Judgement Optimization")
        print("    (3) Learnable SMPLify")
        print("    (4) tạo video tổng hợp")
        print("    (5) vẽ waveform analysis")
        benchmark_text = input("Benchmark path (nhập $ để bỏ qua): ").strip()
        print("Cac cach thuc hien ba pipeline (1), (2), (3) nhu sau:")
        print("    RJL.  (0) => (1) => (2) => (3)")
        print("    LJR.  (0) => (3) => (2) => (1)")
        print("    LJRL. (0) => (3) => (2) => (1) => (3)")
        print("    LRJ.  (0) => (3) => (1) => (2)")
        print("    LRJL. (0) => (3) => (1) => (2) => (3)")
        print("    RLJ.  (0) => (1) => (3) => (2)")
        print("    RLJL. (0) => (1) => (3) => (2) => (3)")
        print("    JLR.  (0) => (2) => (3) => (1)")
        print("    JLRL. (0) => (2) => (3) => (1) => (3)")
        print("    JRL.  (0) => (2) => (1) => (3)")
        input_text = input(
            f"Input folder [default: {DEFAULT_INPUT_DIR}]: "
        ).strip()
        sequence_text = input(
            "Thứ tự pipeline RJL/LJR/LJRL/LRJ/LRJL/RLJ/RLJL/JLR/JLRL/JRL "
            "(Enter để chạy tất cả): "
        ).strip()
        input_dir = str(DEFAULT_INPUT_DIR) if input_text in ("", "$") else input_text
        benchmark = None if benchmark_text in ("", "$") else benchmark_text

    if not sequence_text:
        sequences = list(SUPPORTED_SEQUENCES)
        print(
            "Bạn đã nhấn Enter, chương trình sẽ chạy tất cả sequence: "
            + ", ".join(sequences),
            flush=True,
        )
    else:
        sequences = [sequence_text.upper()]
    for sequence in sequences:
        if sequence not in SUPPORTED_SEQUENCES and not set(sequence).issubset(set("RJL")):
            raise ValueError(f"Unsupported sequence: {sequence}")
    return input_dir, benchmark, sequences


def ensure_output_dirs(output_dir: Path) -> None:
    for child in ("intermediate", "videos", "figures", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def prepare_default_render_alignment_reference(
    args: argparse.Namespace,
    pose_data: dict[str, Any],
    input_dir: str,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    args.default_render_reference_used = False
    if not _should_use_default_render_reference(args):
        return

    reference_path = ensure_iter1_render_reference(pose_data, input_dir, output_dir, config)
    args.render_align_reference = str(reference_path)
    args.default_render_reference_used = True


def _should_use_default_render_reference(args: argparse.Namespace) -> bool:
    if args.skip_video or not args.default_render_align:
        return False
    if args.render_align_reference or args.render_align_transform:
        return False
    return args.render_align_stage == "none"


def ensure_iter1_render_reference(
    pose_data: dict[str, Any],
    input_dir: str,
    output_dir: Path,
    config: dict[str, Any],
) -> Path:
    reference_output_dir = iter1_render_reference_output_dir(output_dir)
    reference_path = iter1_render_reference_path(output_dir)
    metadata_path = iter1_render_reference_metadata_path(output_dir)
    expected_metadata = iter1_render_reference_metadata(input_dir, config)
    if reference_path.exists() and _json_file_matches(metadata_path, expected_metadata):
        print(f"Render align: dùng iter1 reference có sẵn: {reference_path}", flush=True)
        return reference_path
    if reference_path.exists():
        print(
            "Render align: iter1 reference cũ không khớp input/config hiện tại, "
            "sẽ tạo lại.",
            flush=True,
        )

    print(
        "Render align: chưa có iter1 reference, tự chạy prepass R với "
        f"opencap_iterations=1 tại {reference_output_dir}",
        flush=True,
    )
    ensure_output_dirs(reference_output_dir)
    cache_path = default_render_alignment_cache_path(reference_path)
    if cache_path.exists():
        cache_path.unlink()

    prepass_config = dict(config)
    prepass_config["output_dir"] = str(reference_output_dir)
    prepass_config["opencap_iterations"] = 1
    prepass_config["benchmark"] = None
    prepass_state = PipelineState.from_pose_data(
        clone_initial_pose_data(pose_data),
        input_dir=Path(input_dir).resolve(),
        output_dir=reference_output_dir,
        benchmark_path=None,
    )
    prepass_state = run_state_pipeline_sequence("R", prepass_state, prepass_config)
    if prepass_state.unified_pkl != reference_path:
        raise RuntimeError(
            "Iter1 render reference was created at an unexpected path: "
            f"{prepass_state.unified_pkl}; expected {reference_path}"
        )
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    metadata_path.write_text(
        json.dumps(expected_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Render align: đã tạo iter1 reference: {reference_path}", flush=True)
    return reference_path


def iter1_render_reference_output_dir(output_dir: Path) -> Path:
    return Path(output_dir).resolve() / "render_reference_iter1"


def iter1_render_reference_path(output_dir: Path) -> Path:
    return iter1_render_reference_output_dir(output_dir) / "intermediate" / "unify_R.pkl"


def iter1_render_reference_metadata_path(output_dir: Path) -> Path:
    return iter1_render_reference_output_dir(output_dir) / "reference_meta.json"


def iter1_render_reference_metadata(input_dir: str, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(input_dir).expanduser().resolve()
    return {
        "schema": "pose_pipeline.iter1_render_reference.v1",
        "input_dir": str(root),
        "left_pkl": _file_fingerprint(root / "left.pkl"),
        "right_pkl": _file_fingerprint(root / "right.pkl"),
        "opencap_iterations": 1,
        "opencap_max_frames": config.get("opencap_max_frames"),
        "opencap_sex": config.get("opencap_sex"),
        "opencap_height_m": config.get("opencap_height_m"),
        "opencap_mass_kg": config.get("opencap_mass_kg"),
        "opencap_activity": config.get("opencap_activity"),
    }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _json_file_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def select_final_pose(pose_data: dict[str, Any]):
    if pose_data["fused"]["poses_3d"] is not None:
        return pose_data["fused"]["poses_3d"]
    return (pose_data["left"]["poses_3d"] + pose_data["right"]["poses_3d"]) / 2.0


def select_render_pose(final_pose, snapshots: dict[str, Any], render_stage: str):
    if render_stage == "final":
        return final_pose
    if render_stage not in snapshots:
        available = ", ".join(["final", *snapshots.keys()])
        raise ValueError(f"Render stage {render_stage!r} is not available. Available: {available}")
    return snapshots[render_stage]


def align_render_pose_if_requested(
    render_pose,
    final_pose,
    snapshots: dict[str, Any],
    align_stage: str,
    reference_path: str | None,
    transform_path: str | None,
    save_transform_path: str | None,
    use_reference_cache: bool,
    used_default_reference: bool,
):
    load_transform_path, save_target_path = _resolve_render_transform_paths(
        reference_path,
        transform_path,
        save_transform_path,
        use_reference_cache,
    )
    if load_transform_path is not None:
        transform = load_render_alignment_transform(load_transform_path)
        aligned = apply_render_alignment_transform(render_pose, transform)
        diagnostics = transform_diagnostics(transform)
        diagnostics["source"] = str(load_transform_path.resolve())
        diagnostics["mode"] = "loaded_fixed_transform"
        diagnostics["default_reference"] = used_default_reference
        return aligned, diagnostics

    reference_pose = None
    source = None
    if reference_path:
        reference_pose = load_pose_pkl(Path(reference_path))["poses_3d"]
        source = str(Path(reference_path).resolve())
    elif align_stage != "none":
        reference_pose = final_pose if align_stage == "final" else snapshots.get(align_stage)
        if reference_pose is None:
            available = ", ".join(["none", "final", *snapshots.keys()])
            raise ValueError(
                f"Render align stage {align_stage!r} is not available. Available: {available}"
            )
        source = align_stage

    if reference_pose is None:
        return render_pose, None

    transform, diagnostics = estimate_render_alignment_transform(render_pose, reference_pose)
    diagnostics["source"] = source
    diagnostics["default_reference"] = used_default_reference
    if transform is None:
        return render_pose, diagnostics

    if save_target_path is not None:
        save_path = save_render_alignment_transform(transform, save_target_path, diagnostics)
        diagnostics["transform_cache"] = str(save_path.resolve())

    aligned = apply_render_alignment_transform(render_pose, transform)
    return aligned, diagnostics


def _resolve_render_transform_paths(
    reference_path: str | None,
    transform_path: str | None,
    save_transform_path: str | None,
    use_reference_cache: bool,
) -> tuple[Path | None, Path | None]:
    reference_cache = (
        default_render_alignment_cache_path(reference_path)
        if reference_path and use_reference_cache
        else None
    )
    if transform_path:
        path = Path(transform_path)
        return (path if path.exists() else None), (None if path.exists() else path)
    if reference_cache is not None and reference_cache.exists():
        return reference_cache, None
    if save_transform_path:
        return None, Path(save_transform_path)
    return None, reference_cache


def render_up_axis_to_bool(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "y_up"


def clone_initial_pose_data(pose_data: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(pose_data)


def print_summary(run_log: dict[str, Any], log_path: Path) -> None:
    print(f"Tổng thời gian thực hiện: {run_log['total_seconds']}s")
    print(f"Run log: {log_path}")
    for item in run_log["sequences"]:
        print(f"- {item['sequence']}: {item['seconds']}s")
        if item["video"]:
            print(f"  Video: {item['video']}")
        if item["figures"]:
            print(f"  Biểu đồ: {len(item['figures'])} file")
        if item["benchmark"]:
            print(f"  Benchmark: {item['benchmark']}")


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
