#!/usr/bin/env python
"""Minimal child-video face preprocessing.

Pipeline:
1. Track people with YOLO.
2. Drop processed frames where no person is detected.
3. Merge raw tracker fragments into stable person ids.
4. Detect face boxes, render labels, and save face-only crops.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlretrieve


FACE_DETECTOR = "yolov8s_face_lindevs"
DEFAULT_PERSON_MODEL = "yolo26n.pt"
DEFAULT_PERSON_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
DEFAULT_FACE_MODEL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8s-face-lindevs.pt"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache" / "child-video-preprocess"
TRACK_MERGE_METHOD = "raw_track_bbox_appearance_v1"
TRACK_MERGE_SCORE_THRESHOLD = 0.34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop non-human frames, detect face boxes per person id, and render faces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", type=Path, help="Single input video path. Relative paths are resolved from the repo folder.")
    parser.add_argument("--video-dir", "--video_dir", dest="video_dir", type=Path, help="Folder of input videos to process.")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default=Path("outputs"), type=Path, help="Output root directory.")
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL, help="Ultralytics YOLO model name or local weight path.")
    parser.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort", help="Ultralytics tracker backend.")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=0,
        help="Optional cap for stable person objects. Use 0 for a robust automatic cap based on person counts over time.",
    )
    parser.add_argument("--render", action="store_true", help="Write an annotated video that draws detected face boxes.")
    parser.add_argument("--save-crops", action="store_true", help="Save face-only crops to face_crops/.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame. Use 1 for the smoothest render.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum YOLO person detection confidence.")
    parser.add_argument(
        "--face-model",
        default=DEFAULT_FACE_MODEL,
        help="Ultralytics-compatible YOLO face model path or URL.",
    )
    parser.add_argument("--face-conf", type=float, default=0.35, help="Minimum YOLO face detection confidence.")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, mps, or a CUDA device id such as 0.")
    parser.add_argument("--recursive", action="store_true", help="When --video-dir is used, include nested folders.")

    # Backward-compatible no-op flags from earlier versions.
    parser.add_argument("--target-mode", choices=["auto", "none"], default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--detect-faces", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save-face-crops", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--export-filtered-json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-non-human-crops", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--face-model-selection", type=int, choices=[0, 1], default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.video is None) == (args.video_dir is None):
        parser.error("provide exactly one of --video or --video-dir")
    return args


def main() -> int:
    args = parse_args()
    videos = collect_video_inputs(args.video, args.video_dir, args.recursive)
    output_root = resolve_repo_path(args.out_dir)
    written_by_video: dict[str, dict[str, Path]] = {}

    for index, video_path in enumerate(videos, start=1):
        if len(videos) > 1:
            print(f"\nProcessing video {index}/{len(videos)}: {video_path.name}")
        written_by_video[str(video_path)] = run_pipeline(
            video_path=video_path,
            out_dir=args.out_dir,
            person_model=args.person_model,
            tracker=args.tracker,
            max_objects=args.max_objects,
            conf=args.conf,
            face_model=args.face_model,
            face_conf=args.face_conf,
            device=args.device,
            frame_stride=args.frame_stride,
            render=args.render,
            save_crops=args.save_crops or args.save_face_crops,
        )

    print(f"Pipeline complete: {len(videos)} video(s).")
    print(f"Output root: {output_root}")
    if len(videos) == 1:
        for name, path in next(iter(written_by_video.values())).items():
            print(f"{name}: {path}")
    return 0


def collect_video_inputs(video: Path | None, video_dir: Path | None, recursive: bool) -> list[Path]:
    if (video is None) == (video_dir is None):
        raise ValueError("Provide exactly one of --video or --video-dir.")
    if video is not None:
        resolved = resolve_repo_path(video)
        if not resolved.exists():
            raise FileNotFoundError(f"Video not found: {resolved}")
        return [resolved]

    assert video_dir is not None
    resolved_dir = resolve_repo_path(video_dir)
    if not resolved_dir.is_dir():
        raise NotADirectoryError(f"Video folder not found: {resolved_dir}")
    pattern = "**/*" if recursive else "*"
    videos = sorted(path for path in resolved_dir.glob(pattern) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {resolved_dir}")
    return videos


def run_pipeline(
    video_path: Path,
    out_dir: Path,
    person_model: str = DEFAULT_PERSON_MODEL,
    tracker: str = "botsort",
    max_objects: int = 0,
    conf: float = 0.25,
    face_model: str = DEFAULT_FACE_MODEL,
    face_conf: float = 0.35,
    device: str = "auto",
    frame_stride: int = 1,
    render: bool = False,
    save_crops: bool = False,
) -> dict[str, Path]:
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if max_objects < 0:
        raise ValueError("--max-objects must be >= 0")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    if not 0.0 <= face_conf <= 1.0:
        raise ValueError("--face-conf must be between 0 and 1")

    video_path = resolve_repo_path(video_path)
    output_root = resolve_repo_path(out_dir)
    out_dir = make_video_output_dir(output_root, video_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cv2 = require_cv2()
    metadata = read_video_metadata(cv2, video_path)
    result = track_people_and_faces(
        cv2=cv2,
        video_path=video_path,
        metadata=metadata,
        out_dir=out_dir,
        person_model=person_model,
        tracker=tracker,
        max_objects=max_objects,
        conf=conf,
        face_model=face_model,
        face_conf=face_conf,
        device=device,
        frame_stride=frame_stride,
        save_crops=save_crops,
    )

    result_path = out_dir / "result.json"
    summary_path = out_dir / "summary.csv"
    batch_summary_path = output_root / "batch_summary.csv"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_summary_csv(result, summary_path)
    write_batch_summary_csv(result, batch_summary_path, out_dir)

    written = {"result_json": result_path, "summary_csv": summary_path, "batch_summary_csv": batch_summary_path}
    if render and result["human_frame_count"] > 0:
        written["render"] = render_face_video(cv2, video_path, result, out_dir / "render.mp4")
    elif render:
        print("Warning: no human frames were kept, so render.mp4 was not written.")
    report_path = out_dir / "report.txt"
    written["report"] = report_path
    report_text = build_report(result, out_dir, written)
    report_path.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)
    return written


def read_video_metadata(cv2: Any, video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    metadata = {
        "fps": fps if fps > 0 else 30.0,
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    return metadata


def track_people_and_faces(
    cv2: Any,
    video_path: Path,
    metadata: dict[str, Any],
    out_dir: Path,
    person_model: str,
    tracker: str,
    max_objects: int,
    conf: float,
    face_model: str,
    face_conf: float,
    device: str,
    frame_stride: int,
    save_crops: bool,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics to run person tracking.") from exc

    model = YOLO(resolve_person_model_spec(person_model))
    face_model_path = resolve_model_spec(face_model)
    face_detector = YOLO(face_model_path)
    crop_dir = out_dir / "face_crops"
    if save_crops:
        crop_dir.mkdir(parents=True, exist_ok=True)

    track_args: dict[str, Any] = {
        "source": str(video_path),
        "tracker": tracker_config(tracker),
        "classes": [0],
        "conf": conf,
        "stream": True,
        "verbose": False,
        "vid_stride": frame_stride,
    }
    if device and device.lower() != "auto":
        track_args["device"] = device

    frames: list[dict[str, Any]] = []
    dropped_non_human_frame_indices: list[int] = []
    processed_count = 0
    face_count = 0
    face_crop_count = 0
    person_detection_count = 0
    raw_tracker_ids: set[int] = set()
    progress_total = expected_processed_frames(metadata["num_frames"], frame_stride)
    tracked_results = model.track(**track_args)

    for result_idx, yolo_result in enumerate(
        progress_bar(tracked_results, total=progress_total, desc="Tracking people and detecting faces")
    ):
        processed_count += 1
        frame_idx = result_idx * frame_stride
        frame = yolo_result.orig_img
        persons = extract_persons(yolo_result, conf)

        if not persons:
            dropped_non_human_frame_indices.append(frame_idx)
            continue

        add_appearance_features(cv2, frame, persons)
        person_detection_count += len(persons)
        raw_tracker_ids.update(person["track_id"] for person in persons if person["track_id"] >= 0)
        faces = detect_faces_in_frame(
            face_model=face_detector,
            frame=frame,
            persons=persons,
            face_conf=face_conf,
            device=device,
        )
        face_count += len(faces)

        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp_sec": frame_idx / metadata["fps"],
                "persons": persons,
                "quality": {
                    "human_present": True,
                    "face_present": any(person["faces"] for person in persons),
                },
            }
        )

    stable_context = merge_raw_tracks_into_persons(frames, metadata, max_objects=max_objects)
    apply_stable_person_ids(frames, stable_context["raw_to_person"])
    objects = build_person_objects(frames)
    if save_crops and face_count:
        face_crop_count = export_face_crops(cv2, video_path, frames, crop_dir)

    return {
        "video_path": str(video_path),
        "fps": metadata["fps"],
        "num_frames": metadata["num_frames"],
        "frame_stride": frame_stride,
        "processed_frame_count": processed_count,
        "human_frame_count": len(frames),
        "dropped_non_human_frame_count": len(dropped_non_human_frame_indices),
        "dropped_non_human_frame_indices": dropped_non_human_frame_indices,
        "person_detection_count": person_detection_count,
        "raw_tracker_id_count": len(raw_tracker_ids),
        "raw_tracker_ids": sorted(raw_tracker_ids),
        "object_count": len(objects),
        "object_cap": stable_context["object_cap"],
        "auto_object_cap": stable_context["auto_object_cap"],
        "max_simultaneous_persons": stable_context["max_simultaneous_persons"],
        "track_merge_method": TRACK_MERGE_METHOD,
        "face_count": face_count,
        "face_crop_count": face_crop_count,
        "settings": {
            "person_model": person_model,
            "tracker": tracker,
            "conf": conf,
            "max_objects": max_objects,
            "face_detector": FACE_DETECTOR,
            "face_model": face_model,
            "face_conf": face_conf,
            "device": device,
        },
        "objects": objects,
        "frames": frames,
    }


def extract_persons(yolo_result: Any, conf_threshold: float) -> list[dict[str, Any]]:
    boxes = getattr(yolo_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = to_list(getattr(boxes, "xyxy", []))
    conf_values = to_list(getattr(boxes, "conf", []))
    cls_values = to_list(getattr(boxes, "cls", []))
    id_values = to_list(getattr(boxes, "id", None))

    persons: list[dict[str, Any]] = []
    for idx, bbox in enumerate(xyxy_values):
        cls_id = int(cls_values[idx]) if idx < len(cls_values) else 0
        confidence = float(conf_values[idx]) if idx < len(conf_values) else 0.0
        if cls_id != 0 or confidence < conf_threshold:
            continue
        persons.append(
            {
                "track_id": int(id_values[idx]) if idx < len(id_values) else -1,
                "bbox_xyxy": [float(value) for value in bbox],
                "confidence": confidence,
                "faces": [],
            }
        )
    return persons


def add_appearance_features(cv2: Any, frame: Any, persons: list[dict[str, Any]]) -> None:
    """Attach temporary color features used only for raw-track merging."""
    for person in persons:
        person["_appearance_feature"] = person_appearance_feature(cv2, frame, person["bbox_xyxy"])


def person_appearance_feature(cv2: Any, frame: Any, bbox: list[float]) -> list[float] | None:
    height, width = frame.shape[:2]
    clamped = clamp_bbox(bbox, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    values = [float(value) for value in hist.flatten()]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return None
    return [value / norm for value in values]


def merge_raw_tracks_into_persons(
    frames: list[dict[str, Any]],
    metadata: dict[str, Any],
    max_objects: int,
) -> dict[str, Any]:
    """Merge raw online tracker fragments into stable person ids.

    This is a lightweight heuristic layer. It uses temporal exclusivity, bbox
    continuity, and coarse color appearance. It is useful for cleaner research
    outputs, but it is not a formal person re-identification model.
    """
    tracklets = build_raw_tracklets(frames)
    max_simultaneous_persons = max((len(frame["persons"]) for frame in frames), default=0)
    auto_object_cap = robust_object_cap(frames)
    object_cap = max_objects if max_objects > 0 else auto_object_cap
    if object_cap <= 0:
        object_cap = len(tracklets)

    clusters: list[dict[str, Any]] = []
    raw_to_person: dict[int, int] = {}

    for tracklet in sorted(
        tracklets.values(),
        key=lambda item: (item["first_frame_idx"], -item["visible_frame_count"], item["raw_track_id"]),
    ):
        best_cluster_idx, best_score = best_cluster_for_tracklet(tracklet, clusters, metadata)
        can_create = len(clusters) < object_cap

        if best_cluster_idx is not None and (best_score >= TRACK_MERGE_SCORE_THRESHOLD or not can_create):
            cluster = clusters[best_cluster_idx]
        elif can_create:
            cluster = new_person_cluster(len(clusters) + 1)
            clusters.append(cluster)
        elif best_cluster_idx is not None:
            cluster = clusters[best_cluster_idx]
        else:
            fallback_idx, _ = best_cluster_for_tracklet(tracklet, clusters, metadata, allow_overlap=True)
            if fallback_idx is not None:
                cluster = clusters[fallback_idx]
            else:
                cluster = new_person_cluster(len(clusters) + 1)
                clusters.append(cluster)

        add_tracklet_to_cluster(cluster, tracklet)
        raw_to_person[tracklet["raw_track_id"]] = cluster["person_id"]

    return {
        "raw_to_person": raw_to_person,
        "object_cap": object_cap,
        "auto_object_cap": auto_object_cap,
        "max_simultaneous_persons": max_simultaneous_persons,
    }


def robust_object_cap(frames: list[dict[str, Any]]) -> int:
    """Estimate persistent object count while ignoring brief detector spikes."""
    counts = sorted(len(frame["persons"]) for frame in frames if frame["persons"])
    if not counts:
        return 0
    index = min(len(counts) - 1, max(0, math.ceil(0.95 * len(counts)) - 1))
    return max(1, int(counts[index]))


def build_raw_tracklets(frames: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    tracklets: dict[int, dict[str, Any]] = {}
    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        for person_idx, person in enumerate(frame["persons"]):
            raw_track_id = int(person["track_id"])
            if raw_track_id < 0:
                raw_track_id = -((frame_idx + 1) * 1000 + person_idx + 1)
                person["track_id"] = raw_track_id

            if raw_track_id not in tracklets:
                tracklets[raw_track_id] = {
                    "raw_track_id": raw_track_id,
                    "first_frame_idx": frame_idx,
                    "last_frame_idx": frame_idx,
                    "first_bbox": person["bbox_xyxy"],
                    "last_bbox": person["bbox_xyxy"],
                    "frame_indices": set(),
                    "visible_frame_count": 0,
                    "appearance_feature": None,
                    "appearance_count": 0,
                }

            tracklet = tracklets[raw_track_id]
            tracklet["first_frame_idx"] = min(tracklet["first_frame_idx"], frame_idx)
            tracklet["last_frame_idx"] = max(tracklet["last_frame_idx"], frame_idx)
            if frame_idx <= tracklet["first_frame_idx"]:
                tracklet["first_bbox"] = person["bbox_xyxy"]
            if frame_idx >= tracklet["last_frame_idx"]:
                tracklet["last_bbox"] = person["bbox_xyxy"]
            tracklet["frame_indices"].add(frame_idx)
            tracklet["visible_frame_count"] += 1

            feature = person.get("_appearance_feature")
            if feature:
                tracklet["appearance_feature"] = average_feature(
                    tracklet["appearance_feature"],
                    tracklet["appearance_count"],
                    feature,
                    1,
                )
                tracklet["appearance_count"] += 1
    return tracklets


def new_person_cluster(person_id: int) -> dict[str, Any]:
    return {
        "person_id": person_id,
        "raw_track_ids": [],
        "frame_indices": set(),
        "tracklets": [],
        "first_frame_idx": None,
        "last_frame_idx": None,
        "first_bbox": None,
        "last_bbox": None,
        "appearance_feature": None,
        "appearance_count": 0,
    }


def add_tracklet_to_cluster(cluster: dict[str, Any], tracklet: dict[str, Any]) -> None:
    cluster["raw_track_ids"].append(tracklet["raw_track_id"])
    cluster["frame_indices"].update(tracklet["frame_indices"])
    cluster["tracklets"].append(tracklet)

    if cluster["first_frame_idx"] is None or tracklet["first_frame_idx"] < cluster["first_frame_idx"]:
        cluster["first_frame_idx"] = tracklet["first_frame_idx"]
        cluster["first_bbox"] = tracklet["first_bbox"]
    if cluster["last_frame_idx"] is None or tracklet["last_frame_idx"] > cluster["last_frame_idx"]:
        cluster["last_frame_idx"] = tracklet["last_frame_idx"]
        cluster["last_bbox"] = tracklet["last_bbox"]

    feature = tracklet.get("appearance_feature")
    feature_count = int(tracklet.get("appearance_count") or 0)
    if feature and feature_count:
        cluster["appearance_feature"] = average_feature(
            cluster["appearance_feature"],
            cluster["appearance_count"],
            feature,
            feature_count,
        )
        cluster["appearance_count"] += feature_count


def best_cluster_for_tracklet(
    tracklet: dict[str, Any],
    clusters: list[dict[str, Any]],
    metadata: dict[str, Any],
    allow_overlap: bool = False,
) -> tuple[int | None, float]:
    best_idx: int | None = None
    best_score = -1.0
    for idx, cluster in enumerate(clusters):
        if not allow_overlap and not tracklet["frame_indices"].isdisjoint(cluster["frame_indices"]):
            continue
        score = tracklet_cluster_score(tracklet, cluster, metadata, allow_overlap=allow_overlap)
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def tracklet_cluster_score(
    tracklet: dict[str, Any],
    cluster: dict[str, Any],
    metadata: dict[str, Any],
    allow_overlap: bool = False,
) -> float:
    scores = []
    for existing_tracklet in cluster["tracklets"]:
        if tracklet["frame_indices"].isdisjoint(existing_tracklet["frame_indices"]):
            scores.append(tracklet_pair_score(tracklet, existing_tracklet, metadata))
        elif allow_overlap:
            scores.append(overlapping_tracklet_score(tracklet, existing_tracklet))
    return max(scores) if scores else -1.0


def overlapping_tracklet_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    appearance_score = feature_similarity(a.get("appearance_feature"), b.get("appearance_feature"))
    area_score = bbox_area_ratio(a["first_bbox"], b["first_bbox"])
    return 0.8 * appearance_score + 0.2 * area_score


def tracklet_pair_score(a: dict[str, Any], b: dict[str, Any], metadata: dict[str, Any]) -> float:
    if not a["frame_indices"].isdisjoint(b["frame_indices"]):
        return -1.0

    if a["last_frame_idx"] < b["first_frame_idx"]:
        earlier, later = a, b
        gap = b["first_frame_idx"] - a["last_frame_idx"]
    elif b["last_frame_idx"] < a["first_frame_idx"]:
        earlier, later = b, a
        gap = a["first_frame_idx"] - b["last_frame_idx"]
    else:
        earlier, later = a, b
        gap = 0

    width = float(metadata.get("width") or 1)
    height = float(metadata.get("height") or 1)
    diagonal = max(math.sqrt(width * width + height * height), 1.0)
    earlier_center = bbox_center(earlier["last_bbox"])
    later_center = bbox_center(later["first_bbox"])
    center_distance = math.sqrt(
        (earlier_center[0] - later_center[0]) ** 2 + (earlier_center[1] - later_center[1]) ** 2
    )
    spatial_score = max(0.0, 1.0 - 3.5 * center_distance / diagonal)
    area_score = bbox_area_ratio(earlier["last_bbox"], later["first_bbox"])
    appearance_score = feature_similarity(a.get("appearance_feature"), b.get("appearance_feature"))
    fps = float(metadata.get("fps") or 30.0)
    gap_score = math.exp(-float(max(gap, 0)) / max(fps * 6.0, 1.0))

    return (
        0.56 * appearance_score
        + 0.22 * spatial_score
        + 0.12 * area_score
        + 0.10 * gap_score
    )


def average_feature(
    existing: list[float] | None,
    existing_count: int,
    new: list[float],
    new_count: int,
) -> list[float]:
    if existing is None or existing_count <= 0:
        return list(new)
    total = existing_count + new_count
    merged = [
        (old_value * existing_count + new_value * new_count) / total
        for old_value, new_value in zip(existing, new)
    ]
    norm = math.sqrt(sum(value * value for value in merged))
    return [value / norm for value in merged] if norm > 0 else merged


def feature_similarity(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.5
    return max(0.0, min(1.0, sum(left * right for left, right in zip(a, b))))


def bbox_area_ratio(a: Any, b: Any) -> float:
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    larger = max(area_a, area_b)
    return min(area_a, area_b) / larger if larger else 0.0


def bbox_area(bbox: Any) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def apply_stable_person_ids(frames: list[dict[str, Any]], raw_to_person: dict[int, int]) -> None:
    next_person_id = max(raw_to_person.values(), default=0) + 1
    for frame in frames:
        for person in frame["persons"]:
            raw_track_id = int(person["track_id"])
            if raw_track_id not in raw_to_person:
                raw_to_person[raw_track_id] = next_person_id
                next_person_id += 1

            person_id = int(raw_to_person[raw_track_id])
            person["person_id"] = person_id
            person.pop("_appearance_feature", None)
            for face in person["faces"]:
                face["person_id"] = person_id
                face["person_track_id"] = raw_track_id
                face["raw_track_id"] = raw_track_id


def build_person_objects(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame_infos = [(int(frame["frame_idx"]), float(frame["timestamp_sec"])) for frame in frames]
    by_person: dict[int, dict[str, Any]] = {}

    for frame in frames:
        frame_idx = int(frame["frame_idx"])
        timestamp_sec = float(frame["timestamp_sec"])
        for person in frame["persons"]:
            person_id = int(person["person_id"])
            bucket = by_person.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "raw_track_ids": set(),
                    "visible_by_frame": {},
                },
            )
            bucket["raw_track_ids"].add(int(person["track_id"]))
            record = {
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "visible": True,
                "bbox_xyxy": person["bbox_xyxy"],
                "confidence": person["confidence"],
                "raw_track_id": int(person["track_id"]),
                "faces": person["faces"],
            }
            current = bucket["visible_by_frame"].get(frame_idx)
            if current is None or float(record["confidence"]) > float(current["confidence"]):
                bucket["visible_by_frame"][frame_idx] = record

    objects: list[dict[str, Any]] = []
    for person_id, bucket in sorted(by_person.items()):
        visible_by_frame = bucket["visible_by_frame"]
        object_frames = []
        for frame_idx, timestamp_sec in frame_infos:
            record = visible_by_frame.get(frame_idx)
            object_frames.append(
                record
                if record is not None
                else {
                    "frame_idx": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "visible": False,
                    "bbox_xyxy": None,
                    "confidence": None,
                    "raw_track_id": None,
                    "faces": [],
                }
            )

        visible_frame_indices = sorted(visible_by_frame)
        face_frame_count = sum(1 for record in visible_by_frame.values() if record["faces"])
        objects.append(
            {
                "person_id": person_id,
                "raw_track_ids": sorted(bucket["raw_track_ids"]),
                "visible_frame_count": len(visible_by_frame),
                "visible_frame_ratio_among_human": safe_ratio(len(visible_by_frame), len(frame_infos)),
                "face_frame_count": face_frame_count,
                "first_frame_idx": visible_frame_indices[0] if visible_frame_indices else None,
                "last_frame_idx": visible_frame_indices[-1] if visible_frame_indices else None,
                "frames": object_frames,
            }
        )
    return objects


def detect_faces_in_frame(
    face_model: Any,
    frame: Any,
    persons: list[dict[str, Any]],
    face_conf: float,
    device: str,
) -> list[dict[str, Any]]:
    predict_args: dict[str, Any] = {"source": frame, "conf": face_conf, "verbose": False}
    if device and device.lower() != "auto":
        predict_args["device"] = device

    face_results = face_model.predict(**predict_args)
    if not face_results:
        return []

    height, width = frame.shape[:2]
    boxes = getattr(face_results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = to_list(getattr(boxes, "xyxy", []))
    conf_values = to_list(getattr(boxes, "conf", []))
    assigned_faces: list[dict[str, Any]] = []

    for idx, bbox in enumerate(xyxy_values):
        confidence = float(conf_values[idx]) if idx < len(conf_values) else 0.0
        if confidence < face_conf:
            continue
        clamped = clamp_bbox([float(value) for value in bbox], width, height)
        if clamped is None:
            continue
        person = best_person_for_face(clamped, persons)
        if person is None:
            continue

        face = build_face_record(clamped, confidence, person)
        person["faces"].append(face)
        assigned_faces.append(face)

    return assigned_faces


def build_face_record(face_bbox: tuple[int, int, int, int], confidence: float, person: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = face_bbox
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    person_relative_bbox = relative_bbox(face_bbox, person["bbox_xyxy"])
    return {
        "person_track_id": person["track_id"],
        "raw_track_id": person["track_id"],
        "person_id": None,
        "bbox_xyxy": [float(value) for value in face_bbox],
        "confidence": confidence,
        "detector": FACE_DETECTOR,
        "features": {
            "center_xy": [center_x, center_y],
            "area_px": float((x2 - x1) * (y2 - y1)),
            "person_relative_bbox_xyxy": person_relative_bbox,
        },
    }


def best_person_for_face(face_bbox: tuple[int, int, int, int], persons: list[dict[str, Any]]) -> dict[str, Any] | None:
    face_center = bbox_center(face_bbox)
    best_person = None
    best_score = 0.0
    for person in persons:
        person_bbox = tuple(float(value) for value in person["bbox_xyxy"])
        overlap = iou(face_bbox, person_bbox)
        center_bonus = 1.0 if point_in_bbox(face_center, person_bbox) else 0.0
        score = center_bonus + overlap
        if score > best_score:
            best_score = score
            best_person = person
    return best_person if best_score > 0 else None


def relative_bbox(face_bbox: tuple[int, int, int, int], person_bbox: list[float]) -> list[float]:
    px1, py1, px2, py2 = [float(value) for value in person_bbox]
    width = max(px2 - px1, 1.0)
    height = max(py2 - py1, 1.0)
    x1, y1, x2, y2 = face_bbox
    return [
        float((x1 - px1) / width),
        float((y1 - py1) / height),
        float((x2 - px1) / width),
        float((y2 - py1) / height),
    ]


def render_face_video(cv2: Any, video_path: Path, result: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_by_idx = {frame["frame_idx"]: frame for frame in result["frames"]}
    if not frames_by_idx:
        raise ValueError("No human frames are available for rendering.")
    max_idx = max(frames_by_idx)
    wanted_indices = set(frames_by_idx)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video for rendering: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(result.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    render_fps = max(1.0, fps / max(int(result.get("frame_stride") or 1), 1))
    writer, actual_path = open_video_writer(cv2, output_path, render_fps, width, height)
    if actual_path != output_path:
        print(f"Warning: MP4 writer failed; wrote fallback render to {actual_path}")

    frame_idx = 0
    written = 0
    try:
        with progress_bar(total=max_idx + 1, desc="Rendering face video") as progress:
            while frame_idx <= max_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx in wanted_indices:
                    draw_faces(cv2, frame, frames_by_idx[frame_idx])
                    writer.write(frame)
                    written += 1
                frame_idx += 1
                progress.update(1)
    finally:
        cap.release()
        writer.release()

    if written == 0:
        raise ValueError("No frames were written to render.")
    return actual_path


def draw_faces(cv2: Any, frame: Any, frame_data: dict[str, Any]) -> None:
    for person in frame_data["persons"]:
        for face in person["faces"]:
            x1, y1, x2, y2 = [int(round(v)) for v in face["bbox_xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 240, 255), 2)
            center = face.get("features", {}).get("center_xy")
            if center:
                px, py = [int(round(v)) for v in center]
                cv2.circle(frame, (px, py), 3, (255, 120, 60), -1)
            cv2.putText(
                frame,
                f"person_{int(face['person_id']):04d}",
                (max(0, x1), max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 240, 255),
                2,
                cv2.LINE_AA,
            )


def write_summary_csv(result: dict[str, Any], path: Path) -> None:
    row = summary_row(result)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)


def write_batch_summary_csv(result: dict[str, Any], path: Path, out_dir: Path) -> None:
    row = {"video_id": safe_path_name(Path(result["video_path"]).stem), "output_dir": str(out_dir), **summary_row(result)}
    existing_rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))

    key = (row["video_path"], row["video_id"])
    filtered_rows = [
        existing
        for existing in existing_rows
        if (existing.get("video_path"), existing.get("video_id")) != key
    ]
    filtered_rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filtered_rows)


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    processed = result["processed_frame_count"]
    human = result["human_frame_count"]
    face_frames = sum(frame["quality"]["face_present"] for frame in result["frames"])
    return {
        "video_path": result["video_path"],
        "num_frames": result["num_frames"],
        "processed_frame_count": processed,
        "human_frame_count": human,
        "dropped_non_human_frame_count": result["dropped_non_human_frame_count"],
        "human_frame_ratio": safe_ratio(human, processed),
        "face_frame_count": face_frames,
        "face_frame_ratio_among_human": safe_ratio(face_frames, human),
        "face_frame_ratio_among_processed": safe_ratio(face_frames, processed),
        "person_detection_count": result["person_detection_count"],
        "object_count": result["object_count"],
        "object_cap": result["object_cap"],
        "auto_object_cap": result["auto_object_cap"],
        "max_simultaneous_persons": result["max_simultaneous_persons"],
        "raw_tracker_id_count": result["raw_tracker_id_count"],
        "face_count": result["face_count"],
        "face_crop_count": result["face_crop_count"],
    }


def build_report(result: dict[str, Any], out_dir: Path, written: dict[str, Path]) -> str:
    processed = result["processed_frame_count"]
    human = result["human_frame_count"]
    dropped = result["dropped_non_human_frame_count"]
    face_frames = sum(frame["quality"]["face_present"] for frame in result["frames"])

    lines = [
        "Run report",
        f"Video: {Path(result['video_path']).name}",
        f"Output directory: {out_dir}",
        f"Human frames: {human}/{processed} ({percent(human, processed)})",
        f"Dropped non-human frames: {dropped}/{processed} ({percent(dropped, processed)})",
        f"Frames with at least one face: {face_frames}/{human} human frames ({percent(face_frames, human)})",
        f"Frames with at least one face: {face_frames}/{processed} processed frames ({percent(face_frames, processed)})",
        f"Person detections: {result['person_detection_count']}",
        f"Stable person objects: {result['object_count']} (cap={result['object_cap']}, max simultaneous detections={result['max_simultaneous_persons']})",
        f"Raw tracker IDs: {result['raw_tracker_id_count']} (tracker fragments, not person count)",
        f"Face detections: {result['face_count']}",
        f"Face crops saved: {result['face_crop_count']}",
        "Files:",
    ]
    for name, path in written.items():
        lines.append(f"  {name}: {path}")
    return "\n".join(lines)


def export_face_crops(cv2: Any, video_path: Path, frames: list[dict[str, Any]], crop_dir: Path) -> int:
    frames_with_faces = {
        int(frame["frame_idx"]): frame
        for frame in frames
        if any(person["faces"] for person in frame["persons"])
    }
    if not frames_with_faces:
        return 0

    crop_dir.mkdir(parents=True, exist_ok=True)
    max_idx = max(frames_with_faces)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video for crop export: {video_path}")

    frame_idx = 0
    saved = 0
    try:
        with progress_bar(total=max_idx + 1, desc="Saving face crops") as progress:
            while frame_idx <= max_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_data = frames_with_faces.get(frame_idx)
                if frame_data is not None:
                    for person in frame_data["persons"]:
                        saved += save_face_crops(
                            cv2=cv2,
                            frame=frame,
                            faces=person["faces"],
                            crop_dir=crop_dir,
                            frame_idx=frame_idx,
                            person_id=person["person_id"],
                            raw_track_id=person["track_id"],
                        )
                frame_idx += 1
                progress.update(1)
    finally:
        cap.release()
    return saved


def save_face_crops(
    cv2: Any,
    frame: Any,
    faces: list[dict[str, Any]],
    crop_dir: Path,
    frame_idx: int,
    person_id: int,
    raw_track_id: int,
) -> int:
    height, width = frame.shape[:2]
    person_dir = crop_dir / person_folder_name(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for face_idx, face in enumerate(faces):
        bbox = clamp_bbox(face["bbox_xyxy"], width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_path = person_dir / f"frame_{frame_idx:06d}_raw_track_{raw_track_id:04d}_face_{face_idx}.jpg"
        if cv2.imwrite(str(crop_path), crop):
            saved += 1
    return saved


def person_folder_name(person_id: int) -> str:
    return f"person_{person_id:04d}" if person_id >= 0 else "person_unknown"


def resolve_person_model_spec(model_spec: str) -> str:
    if model_spec == DEFAULT_PERSON_MODEL:
        return resolve_model_spec(DEFAULT_PERSON_MODEL_URL)
    return resolve_model_spec(model_spec)


def resolve_model_spec(model_spec: str) -> str:
    if not is_url(model_spec):
        path = Path(model_spec).expanduser()
        return str(path.resolve()) if path.exists() else model_spec

    cache_path = CACHE_DIR / "models" / Path(urlparse(model_spec).path).name
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model to {cache_path}")
        urlretrieve(model_spec, cache_path)
    return str(cache_path)


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def open_video_writer(cv2: Any, output_path: Path, fps: float, width: int, height: int) -> tuple[Any, Path]:
    for path, codec in [
        (output_path, "mp4v"),
        (output_path.with_suffix(".avi"), "XVID"),
        (output_path.with_name(f"{output_path.stem}_mjpg.avi"), "MJPG"),
    ]:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            return writer, path
        writer.release()
    raise RuntimeError("Could not open video writer.")


def tracker_config(tracker: str) -> str:
    if tracker == "botsort":
        return "botsort.yaml"
    if tracker == "bytetrack":
        return "bytetrack.yaml"
    raise ValueError("tracker must be 'botsort' or 'bytetrack'")


def clamp_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1_i = max(0, min(width, int(round(x1))))
    y1_i = max(0, min(height, int(round(y1))))
    x2_i = max(0, min(width, int(round(x2))))
    y2_i = max(0, min(height, int(round(y2))))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def bbox_center(bbox: Any) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bbox(point: tuple[float, float], bbox: Any) -> bool:
    x, y = point
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in a]
    bx1, by1, bx2, by2 = [float(value) for value in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def percent(numerator: int | float, denominator: int | float) -> str:
    return f"{safe_ratio(numerator, denominator) * 100:.1f}%"


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def make_video_output_dir(output_root: Path, video_path: Path) -> Path:
    return output_root / safe_path_name(video_path.stem)


def safe_path_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return cleaned.strip("._") or "video"


def expected_processed_frames(num_frames: int, frame_stride: int) -> int | None:
    if num_frames <= 0:
        return None
    return max(1, num_frames // frame_stride)


def progress_bar(*args: Any, **kwargs: Any) -> Any:
    from tqdm.auto import tqdm

    kwargs.setdefault("unit", "frame")
    kwargs.setdefault("dynamic_ncols", True)
    return tqdm(*args, **kwargs)


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python or opencv-python-headless.") from exc
    return cv2


if __name__ == "__main__":
    raise SystemExit(main())
