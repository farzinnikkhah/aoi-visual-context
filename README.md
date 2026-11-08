# Visual context for pedestrian crossings

Computer-vision code for the full-frame area-of-interest (AOI) pipeline used in the pedestrian day–night crossing study. It detects the 14 AOI classes in first-person video with [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), checks those boxes against independent human annotations, and separates moving vehicles from parked vehicles with a position-first rule plus a short video track.

The repository is code only. It does not include participant video, gaze tables, annotation images, or trained weights.

## What each part does

- `run_full_frame_groundingdino.py` runs the zero-shot detector on every usable frame and merges highly overlapping boxes of the same class (`box_merge.py`).
- `human_validation/` draws a stratified annotation sample, builds the train and test JSON, and fine-tunes Grounding DINO. The image backbone and text encoder stay frozen. The class list is `human_validation/finetune/data/classes.json`. The annotation rules are in `human_validation/ANNOTATION_PROTOCOL.md`.
- `run_full_frame_finetuned.py` runs the fine-tuned checkpoint on the corpus. Storefront uses a score cutoff of 0.20. Every other class uses 0.30.
- `vehicle_discrimination/` merges overlapping vehicle boxes, labels travel-lane versus curb from image position, then uses about ±15 frames of residual motion only to raise a moving score when the box is already lane-like. Lack of motion does not force a parked label.
- `merge_groundingdino_to_separate_excels.py` attaches the detection columns to the Tobii export tables.
- `align_frames_to_tobii.py` aligns extracted frames to the gaze timeline.
- `analysis_stats/object_in_attended_areas_stats.py` summarizes how often a detected class falls in the manually attended AOIs.

## Setup

Python 3.10 or newer. Install PyTorch for the machine you are using, then:

```bash
pip install -r requirements.txt
```

Point the scripts at the data directory. On the machine where the study files live, that directory is the existing `full_frame_aoi_pipeline` folder:

```bash
export AOI_DATA_ROOT=/path/to/full_frame_aoi_pipeline
export AOI_STUDY_ROOT=/path/to/AOI
```

`AOI_STUDY_ROOT` is only needed for the original Tobii tables under `seperate_excels/Separate_excels`.

The public starting checkpoint is `IDEA-Research/grounding-dino-base`. A fine-tuned run is loaded from:

```text
$AOI_DATA_ROOT/human_validation/finetune/runs/<run-id>/best
```

## Main commands

```bash
# Zero-shot full-frame detection
python run_full_frame_groundingdino.py --dry-run

# Build the annotation splits from LabelMe or AnyLabeling exports
python human_validation/finetune/build_dataset.py

# Fine-tune, then score the held-out frames
python human_validation/finetune/train_grounding_dino.py
python human_validation/finetune/score_run.py human_validation/finetune/runs/<run-id>

# Run the fine-tuned detector
python run_full_frame_finetuned.py --model-dir /path/to/checkpoint

# Moving versus parked, then write a new detection tree
cd vehicle_discrimination
python two_stage_vehicle.py --train-json /path/to/train.json
python apply_vehicle_video_to_corpus.py \
  --csv-root "$AOI_DATA_ROOT/GroundingDINO_finetuned_full_frame_output" \
  --out-root "$AOI_DATA_ROOT/GroundingDINO_finetuned_full_frame_output_vehicle_video"
```

Match rule used for the reported box scores: two boxes of the same class match when the intersection covers more than half of either box, or their intersection-over-union is above 0.5.
