# child-video-preprocess

Minimal Python pipeline for preprocessing child-centered behavioral videos.

It does three practical things:

1. Detect and track people with Ultralytics YOLO.
2. Drop frames with no detected person.
3. Detect face boxes and export stable person-level JSON.

This is a research utility, not a complete child-identification or state-of-the-art tracking system.

## Install

```bash
python -m pip install -r requirements.txt
```

Editable install:

```bash
python -m pip install -e .
```

Google Colab:

```bash
pip install -r requirements-colab.txt
```

## Run

Single video:

```bash
run_pipeline.py --video test.mp4 --render --save-crops
```

If installed with `pip install -e .`:

```bash
cvp-preprocess --video test.mp4 --render --save-crops
```

Batch folder:

```bash
run_pipeline.py --video-dir videos --out-dir outputs --render --save-crops
```

If you know the expected number of people:

```bash
run_pipeline.py --video test.mp4 --max-objects 3 --render --save-crops
```

## Outputs

```text
outputs/
  batch_summary.csv
  video_name/
    result.json
    summary.csv
    report.txt
    render.mp4
    face_crops/
      person_0001/
      person_0002/
```

`result.json` contains:

- `objects`: stable person-level tracks.
- `objects[].frames`: each person's per-frame visibility, bbox, raw tracker id, and face boxes.
- `frames`: frame-level detections for quick inspection.
- `dropped_non_human_frame_indices`: processed frames removed because no person was detected.

Raw tracker ids are kept for audit, but the main output is `person_id`.

## Main Arguments

`--video`: process one video.

`--video-dir`: process all supported videos in a folder.

`--out-dir`: output root. Default: `outputs`.

`--tracker`: `botsort` or `bytetrack`. Default: `botsort`.

`--max-objects`: optional cap for stable people. Use `0` for automatic.

`--render`: write `render.mp4`.

`--save-crops`: save face-only crops.

`--frame-stride`: process every Nth frame. Default: `1`.

`--conf`: person detection confidence. Default: `0.25`.

`--face-conf`: face detection confidence. Default: `0.35`.

`--device`: `auto`, `cpu`, `mps`, or CUDA id such as `0`.

## Notes

- The default person model is `yolo26n.pt`.
- The default face detector is a public Ultralytics-compatible YOLO face model.
- Model weights are cached under `~/.cache/child-video-preprocess/models/`.
- Stable `person_id` merging is heuristic and may fail under heavy occlusion, similar clothing, or crowded scenes.
- Do not include private, patient, or sensitive child videos in this repository.

## Citation

If this helps your work, cite this repository and acknowledge Ultralytics YOLO and the face model weights you use.
