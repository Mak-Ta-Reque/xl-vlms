## Sample COCO Dataset Based on Token

This script allows you to sample a subset of the COCO dataset (in JSON format) by filtering images that contain a specific token in the "raw" field of their associated sentences.

### Usage

To use the script, run the following command:

```bash
python sample_coco_json.py <input_json_file> <n_train> <n_val> <token/ tokens> <output_directory>
```

```bash
python sample_data.py /mnt/abka03/mscoco2014/dataset_coco.json 20 20 cat dog horse human shoes /mnt/abka03/mscoco2014/xl-vlm

```

# Image Patch Generation Script

This script processes JSON annotation files, extracts image patches, and updates JSON filenames. The patches are organized into structured directories for training and validation datasets.

## Usage
```bash
python generate_patches.py <json_directory> <root_image_folder> <patch_size> <num_patches> <--technique>
```

```bash
python generate_patches.py /mnt/abka03/mscoco2014/xl-vlm  /mnt/abka03/mscoco2014 250 10 --technique vqa-seg
```

## Output Structure
```
/json_directory/
├── captions1.json
├── captions1_patch.json
├── captions2.json
├── captions2_patch.json
├── captions1/
│   ├── train2014/
│   │   ├── image1_patch_0.png
│   │   ├── image1_patch_1.png
│   ├── val2014/
├── captions2/
│   ├── train2014/
│   ├── val2014/
```
## Requirements for vqa-seq based patching
```bash
transformers==4.44.2
pip install torch==2.4.1 torchvision==0.19.1 --extra-index-url https://download.pytorch.org/whl/cu124
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

