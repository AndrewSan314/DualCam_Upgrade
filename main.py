from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from pose_pipeline.benchmark.evaluator import evaluate_benchmark
from pose_pipeline.config import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, SUPPORTED_SEQUENCES
from pose_pipeline.io_utils.input_loader import load_inputs
from pose_pipeline.pipelines.judgement import run_pose_judgement
from pose_pipeline.pipelines.learnable_smplify import run_learnable_smplify
from pose_pipeline.pipelines.refinement import run_pose_refinement
from pose_pipeline.visualization.video_composer import compose_output_video
from pose_pipeline.visualization.waveform import draw_waveform_analysis


PIPELINE_MAP = {
    "R": run_pose_refinement,
    "J": run_pose_judgement,
    "L": run_learnable_smplify,
}


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

    sequence_results = []
    for sequence in sequences:
        seq_start = time.perf_counter()
        print(f"\nBắt đầu sequence {sequence}", flush=True)
        current = clone_initial_pose_data(pose_data)
        current, pose_snapshots = run_pipeline_sequence(sequence, current, config)
        final_pose = select_final_pose(current)
        render_pose = select_render_pose(final_pose, pose_snapshots, args.render_stage)

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
                "figures": [str(path) for path in figure_paths],
                "benchmark": benchmark_result,
                "metadata": {
                    "left": current["left"].get("metadata", {}),
                    "right": current["right"].get("metadata", {}),
                    "fused": current["fused"].get("metadata", {}),
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
    parser.add_argument("--render-zoom", type=float, default=1.15)
    parser.add_argument("--render-stage", default="final", choices=("final", "input", "R", "J", "L"))
    parser.add_argument("--max-judgement-frames", type=int, default=None)
    parser.add_argument("--judgement-log-interval", type=int, default=10)
    parser.add_argument("--judgement-regularization", action="store_true")
    parser.add_argument("--judgement-regularization-lambda", type=float, default=1.0)
    parser.add_argument("--soft-tail-temperature", type=float, default=0.05)
    parser.add_argument("--soft-tail-weight", type=float, default=1.0)
    parser.add_argument("--learnable-smplify-src", default=None)
    parser.add_argument("--learnable-checkpoint", default=None)
    parser.add_argument("--smpl-model-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-refiner", default="smooth", choices=("smooth", "none"))
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--opencap-iterations", type=int, default=75)
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
    for child in ("videos", "figures", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def run_pipeline_sequence(
    sequence: str, pose_data: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshots = {"input": select_final_pose(pose_data).copy()}
    for symbol in sequence:
        print(f"  Đang chạy pipeline {symbol}...", flush=True)
        step_start = time.perf_counter()
        pose_data = PIPELINE_MAP[symbol](pose_data, config)
        snapshots[symbol] = select_final_pose(pose_data).copy()
        print(
            f"  Xong pipeline {symbol} sau {time.perf_counter() - step_start:.2f}s",
            flush=True,
        )
    return pose_data, snapshots


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
