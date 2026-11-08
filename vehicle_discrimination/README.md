# Moving vs parked vehicle discrimination

Paths below are relative to this folder. Detection CSVs and annotation JSON live under `AOI_DATA_ROOT` (see the repository README) and are not stored in git.

## What is in place

1. **Position-based prompts** in the finetuned runner/trainer  
   `vehicle in travel lane` / `parked vehicle at curb`

2. **Targeted fine-tune** `human_validation/finetune/runs/20261102_143436`

3. **Two-stage classifier** (`two_stage_vehicle.py`)  
   - Merge overlapping vehicle boxes (ignore moving/parked subtype)  
   - Classify travel-lane vs curb with road/curb geometry  

4. **Video tracks** (`video_track_motion.py`) — use video the way humans do  
   - Short tracks over ±0.5–1 s from consecutive corpus frames (~25 fps CSVs)  
   - Ego-motion from matched buildings/sidewalks; residual path/net boosts *moving* only  
   - Low motion never forces parked (queued in-lane cars)

5. **Hard annotation pack** (for the next label round)  
   - 48 clear frames (6/stratum), all with **both** model vehicle classes  
   - Zip: `human_validation/vehicle_hard_annotation_pack.zip`  
   - Manifest: `human_validation/batch_vehicle_hard/manifest.csv`

## Commands

```bash
# Hard pack (already generated)
cd ../human_validation && python export_vehicle_hard_batch.py

# Train / apply two-stage
python two_stage_vehicle.py --train-json ../human_validation/finetune/data/train.json
python two_stage_vehicle.py \
  --model-in two_stage_vehicle_classifier.json \
  --pred-json ../human_validation/finetune/runs/20261102_143436/finetuned_test.json \
  --gt-json ../human_validation/finetune/data/test.json \
  --out reports/two_stage_test.json

# Video-track A/B (geometry vs ±0.6s / ±1s tracks)
python run_video_track_eval.py

# Apply geometry + video corrections to corpus CSVs (writes a NEW root; originals kept)
python apply_vehicle_video_to_corpus.py \
  --csv-root ../GroundingDINO_finetuned_full_frame_output \
  --out-root ../GroundingDINO_finetuned_full_frame_output_vehicle_video

# Remake gaze-merged Tobii excels from the corrected DINO root
python ../merge_groundingdino_to_separate_excels.py \
  --dino-root ../GroundingDINO_finetuned_full_frame_output_vehicle_video \
  --output-root ../seperate_excels_with_finetuned_vehicle_video/Separate_excels \
  --summary-name merge_summary_finetuned_vehicle_video.csv
```

## Reports

- `reports/two_stage_ab_report.txt`
- `reports/manuscript_two_stage_split_tables.tex`
- `reports/video_track_ab_report.txt`
- `reports/manuscript_video_track_split_tables.tex`
