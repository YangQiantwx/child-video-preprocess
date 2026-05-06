# child-video-preprocess

Minimal video preprocessing utility for child-centered behavioral videos.

The v1 pipeline does four things:

1. Tracks people with Ultralytics YOLO and BoT-SORT or ByteTrack.
2. Drops processed frames where no person is detected.
3. Merges raw tracker fragments into stable `person_id` objects.
4. Detects face boxes for each person, then optionally renders labels and saves face-only crops.

This is a practical research baseline, not a complete child-of-interest identification system and not a state-of-the-art person re-identification or face analysis system. Do not commit private videos, patient videos, identifiable child videos, or other sensitive data.

## Install

macOS or Linux CPU:

```bash
python -m pip install -r /path/to/child-video-preprocess/requirements.txt
```

Linux GPU: install the correct PyTorch/CUDA build first, then:

```bash
python -m pip install -r /path/to/child-video-preprocess/requirements.txt
```

Google Colab:

```python
!pip install -r /content/child-video-preprocess/requirements-colab.txt
```

Editable install:

```bash
python -m pip install -e /path/to/child-video-preprocess
```

## Run

After editable install:

```bash
cvp-preprocess --video test.mp4 --render --save-crops
```

Without editable install:

```bash
python /path/to/child-video-preprocess/run_pipeline.py --video test.mp4 --render --save-crops
```

Process a folder of videos:

```bash
cvp-preprocess --video-dir videos --out-dir outputs --render --save-crops
```

If you know the video should contain a fixed number of people, pass it explicitly:

```bash
cvp-preprocess --video test.mp4 --max-objects 3 --render --save-crops
```

Relative `--video`, `--video-dir`, and `--out-dir` paths are resolved from the folder containing `run_pipeline.py`, so commands do not depend on the current shell folder.

## Outputs

The output root is organized for future multi-video batch processing. Each video gets one subfolder named after the video file stem, plus one root-level batch summary.

```text
outputs/
  batch_summary.csv
  test/
    result.json
    summary.csv
    report.txt
    render.mp4        # only with --render
    face_crops/       # only with --save-crops
      person_0001/
        frame_000000_raw_track_0001_face_0.jpg
      person_0002/
      person_0003/
```

`result.json` keeps processed human frames only. Processed frames with no person are dropped and counted in:

```json
{
  "dropped_non_human_frame_count": 3,
  "dropped_non_human_frame_indices": [12, 13, 14]
}
```

Frames with a person but no detected face are not dropped. They remain in `result.json` with empty `faces` lists and are rendered without face boxes.

## Object JSON

The main output is object-centric. `raw_tracker_ids` are tracker fragments; `objects` are the stable person-level output.

```json
{
  "object_count": 3,
  "raw_tracker_id_count": 24,
  "track_merge_method": "raw_track_bbox_appearance_v1",
  "objects": [
    {
      "person_id": 1,
      "raw_track_ids": [1, 8, 15],
      "visible_frame_count": 1116,
      "face_frame_count": 856,
      "frames": [
        {
          "frame_idx": 0,
          "timestamp_sec": 0.0,
          "visible": true,
          "bbox_xyxy": [271.5, 635.0, 586.1, 1279.2],
          "confidence": 0.91,
          "raw_track_id": 1,
          "faces": [
            {
              "person_id": 1,
              "raw_track_id": 1,
              "bbox_xyxy": [365.0, 700.0, 430.0, 770.0],
              "confidence": 0.86,
              "detector": "yolov8s_face_lindevs"
            }
          ]
        },
        {
          "frame_idx": 1,
          "timestamp_sec": 0.033,
          "visible": false,
          "bbox_xyxy": null,
          "confidence": null,
          "raw_track_id": null,
          "faces": []
        }
      ]
    }
  ]
}
```

The frame-centric `frames` list is also kept for quick per-frame inspection. Each frame-level person entry includes both `person_id` and raw `track_id`.

## Arguments

`--video`: single input video path. Use exactly one of `--video` or `--video-dir`.

`--video-dir`: folder of videos to process. Supported extensions: `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`.

`--out-dir` / `--out_dir`: output root. The script creates a per-video subfolder inside this directory. Default: `outputs`.

`--person-model`: Ultralytics YOLO person model name or local weight path. Default: `yolo26n.pt`. The default weight is cached under `~/.cache/child-video-preprocess/models/`.

`--tracker`: Ultralytics tracker backend. Choices: `botsort`, `bytetrack`. Default: `botsort`.

`--max-objects`: optional cap for stable person objects. Use `0` for the automatic robust cap. If you know the video has three people, use `--max-objects 3`.

`--face-model`: Ultralytics-compatible YOLO face model path or URL. By default the script downloads public YOLOv8s-Face Lindevs weights into `~/.cache/child-video-preprocess/models/`.

`--render`: write a face-box render video. Human frames with no detected face are kept but have no overlay.

`--save-crops`: save face-only image crops under `face_crops/person_XXXX/` inside the per-video output directory.

`--frame-stride`: process every Nth frame. Default: `1`. Use `1` for the smoothest render and most complete JSON.

`--conf`: minimum YOLO person detection confidence. Default: `0.25`.

`--face-conf`: minimum YOLO face detection confidence. Default: `0.35`.

`--device`: inference device. Use `auto`, `cpu`, `mps`, or a CUDA device id such as `0`. Default: `auto`.

`--recursive`: with `--video-dir`, include nested folders.

Backward-compatible legacy flags are accepted but hidden from `--help`: `--target-mode`, `--detect-faces`, `--save-face-crops`, `--export-filtered-json`, `--skip-non-human-crops`, and `--face-model-selection`.

## Terminal Report

The terminal and `report.txt` both summarize the run:

```text
Run report
Video: test.mp4
Human frames: 1334/1334 (100.0%)
Dropped non-human frames: 0/1334 (0.0%)
Frames with at least one face: 1264/1334 human frames (94.8%)
Person detections: 3348
Stable person objects: 3 (cap=3, max simultaneous detections=5)
Raw tracker IDs: 24 (tracker fragments, not person count)
Face detections: 2450
Face crops saved: 2450
```

Raw tracker ids are not the same as the number of people in the room. A three-person video can still produce many raw ids because online trackers may start a new id after occlusion, missed detections, people moving close together, or a person leaving and re-entering the frame.

## Known Limitations

- Stable `person_id` assignment is a heuristic merge layer over raw online tracker ids. It uses temporal exclusivity, bbox continuity, and coarse color appearance.
- The automatic object cap ignores brief detector spikes using a robust person-count estimate. Use `--max-objects N` when the expected number of people is known.
- Non-human processed frames are dropped from `result.json`; their frame indices are recorded for audit.
- A face is assigned to a person when the face box overlaps or falls inside that tracked person box.
- No manual target selection is included in v1.
- No private data, example patient data, or real child videos are included.

## Citation / Acknowledgment

If this repository helps your work, cite the repository URL and acknowledge Ultralytics YOLO and the public face detection weights you choose to use.
