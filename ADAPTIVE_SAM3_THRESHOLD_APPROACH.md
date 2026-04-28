# Adaptive SAM3 Threshold Approach

When SAM3 is run with `SEGMENTATION_CONFIDENCE=-1`, the pipeline switches to adaptive scoring instead of a fixed cutoff.

## Behavior

1. Run SAM3 with a permissive detection threshold so low-confidence candidates are still collected.
2. Sort detections by score.
3. Compute an image-specific adaptive threshold from the score distribution.
4. Keep detections whose score is at or above that adaptive threshold.
5. If that would leave too few masks, fall back to keeping at least `CONCEPT_MASKS_PER_IMAGE` masks.

## Why this helps

This preserves recall when the target concept is present but the detector is uncertain, while still filtering obvious false positives. The minimum-keep floor prevents the adaptive cutoff from collapsing to zero masks on difficult images.

## Configuration

- `SEGMENTATION_CONFIDENCE > 0 and < 1`: use a fixed confidence threshold.
- `SEGMENTATION_CONFIDENCE = -1`: use adaptive thresholding.
- `CONCEPT_MASKS_PER_IMAGE`: used as the minimum number of masks to keep in adaptive mode.

## Where it is wired

- `src/sam3_utils.py` applies adaptive filtering and the minimum-keep floor.
- `preprocessing/crops_to_json.py` passes `CONCEPT_MASKS_PER_IMAGE` into the SAM3 model dict.
- `scripts/run_full_pipeline.py` reads `SEGMENTATION_CONFIDENCE` from the environment.
