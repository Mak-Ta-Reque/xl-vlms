"""Diagnostic: test SAM3 multi-tag non-concept detection on one image."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from PIL import Image

# Get an image path from crops.json
crops = json.load(open('outputs/bird/inference/crops.json'))
tag = 'bird'
entries = crops[tag]
rel_path = list(entries.keys())[0]
img_path = os.path.join('data/train/bird', rel_path)
print(f'Testing: {img_path}')

# Load and resize
img = Image.open(img_path).convert('RGB')
isw = 512
w, h = img.size
nw = isw
nh = round(h * isw / w)
img = img.resize((nw, nh), Image.LANCZOS)
print(f'Image size after resize: {img.size}')

# Load SAM3 model
from src.sam3_utils import load_sam3, predict_all_masks_sam3
device = 'cuda:0'
model = load_sam3(device=device)

# Get all tags from the mapping
concept_map = json.load(open('outputs/bird/inference/concepts_to_images.json'))
all_tags = list(concept_map.keys())
print(f'All tags in mapping ({len(all_tags)}): {all_tags}')

# Non-tag detection using multi-tag approach
nontag_pairs = predict_all_masks_sam3(
    model, [img], topn=10, min_mask_area=100,
    other_tags=all_tags, exclude_tag='bird',
)
print(f'\nNon-tag pairs (multi-tag approach): {len(nontag_pairs[0])} detections')
for i, (bbox, mask) in enumerate(nontag_pairs[0]):
    area = int(mask.sum()) if mask is not None else 0
    print(f'  nontag[{i}]: bbox={bbox}, mask_area={area}')

# Test full combined function
from preprocessing.crops_to_json import build_concept_and_nontag_detections_for_image
combined = build_concept_and_nontag_detections_for_image(
    img_pil=img, tag='bird', detector='sam3', model=model,
    topn_nontag=10, min_mask_area=100, all_tags=all_tags,
)
print(f'\nCombined result: {len(combined)} total pairs')
for i, (bbox, mask) in enumerate(combined):
    area = int(mask.sum()) if mask is not None else 0
    label = 'CONCEPT' if i == 0 else 'non-tag'
    print(f'  [{i}] {label}: bbox={bbox}, mask_area={area}')
