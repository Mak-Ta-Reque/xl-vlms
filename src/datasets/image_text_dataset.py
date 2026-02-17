import json
import os
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import random
from datasets.constants import WORDS
from models.constants import TASK_PROMPTS

__all__ = ["COCODataset"]


class ImageTextDataset(Dataset):
    def __init__(
        self,
        annotation_file: str = "dataset_coco.json",
        data_dir: str = "/data/mshukor/data",
        split: str = "train",
        dataset_size: int = -1,
        seed: int = 0,
        dataset_name: str = "coco",
        mode: str = "train",
        questions_file: str = "",
        prompt_template: str = "llava",
        token_of_interest_num_samples: int = -1,
        **kwargs: Any,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.annotation_file = annotation_file
        self.questions_file = questions_file
        self.dataset_size = dataset_size
        self.seed = seed
        self.dataset_name = dataset_name
        self.mode = mode
        self.prompt_template = prompt_template
        self.token_of_interest_num_samples = token_of_interest_num_samples

        self.rng = np.random.default_rng(seed)

        self.split = split

        self.create_dataset()

    def create_dataset(
        self,
    ) -> None:
        raise NotImplementedError(
            f"create_dataset() is not defined from dataset: {self.dataset_name}"
        )

    def apply_prompt(self, item: Dict[str, Any], mode: str = "train") -> str:
        if "train" in mode:
            text = f"{item['instruction']}{item['response']}"
        else:
            text = item["instruction"]
        return text

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        item = self.data[idx]
        item["text"] = self.apply_prompt(item, mode=self.mode)

        return item

    def __len__(
        self,
    ) -> int:
        if isinstance(self.data, dict):
            return sum(len(v) for v in self.data.values())
        return len(self.data)

    def get_difference_variations_of_token_of_interest(
        self, token_of_interest: str
    ) -> List[str]:

        tokens_of_interest = set(
            [
                token_of_interest,
                token_of_interest.capitalize(),
                token_of_interest.lower(),
            ]
        )
        return list(tokens_of_interest)

    def token_of_interest_presence(
        self,
        idx: int = 0,
        token_of_interests: List[str] = ["dog"],
        token_of_interest_key: str = "response",
    ) -> bool:
        """
        Checking a single sample's caption for presence of TOI (token of interest)
        """
        if isinstance(token_of_interests, str):
            token_of_interests = [token_of_interests]
        return any(
            [k in self.data[idx][token_of_interest_key] for k in token_of_interests]
        )

    def token_of_interest_idx_extractor(
        self,
        token_of_interest: str = "dog",
        token_of_interest_key: str = "response",
        allow_different_variations: bool = False,
        token_of_interest_class: str = None,
        logger: Callable = None,
    ) -> List[int]:
        """
        To extract the indices of samples where the TOI (token of interest) appears
        in their caption.
        """

        if token_of_interest_class is not None:
            token_of_interests = list(WORDS[token_of_interest_class])
        elif isinstance(token_of_interest, str):
            token_of_interests = [token_of_interest]
        else:
            token_of_interests = token_of_interest

        if allow_different_variations:
            for tok in token_of_interests:
                token_of_interests.extend(
                    self.get_difference_variations_of_token_of_interest(tok)
                )
            token_of_interests = list(set(token_of_interests))

        if logger is not None:
            logger.info(f"Start selecting samples containing {token_of_interests}")

        token_of_interest_indices = []
        num_samples = 0
        for idx in range(len(self)):
            if self.token_of_interest_presence(
                idx, token_of_interests, token_of_interest_key=token_of_interest_key
            ):
                token_of_interest_indices.append(idx)
                num_samples += 1
                if (
                    self.token_of_interest_num_samples > 0
                    and num_samples >= self.token_of_interest_num_samples
                ):
                    break
        if logger is not None:
            logger.info(
                f"Selecting {len(token_of_interest_indices)} indices for token_of_interests: {token_of_interests}"
            )
        return token_of_interest_indices

    def idx_in_ids(
        self,
        idx: int = 0,
        img_ids: List[str] = None,
    ) -> bool:

        return True if self.data[idx]["img_id"] in img_ids else False

    def indices_from_ids_extractor(self, img_ids: List[int] = None) -> List[int]:

        filtered_indices = []

        for idx in range(len(self)):
            if self.idx_in_ids(idx, img_ids):
                filtered_indices.append(idx)

        return filtered_indices


class COCODataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        annotation_path = os.path.join(self.data_dir, self.annotation_file)
        with open(annotation_path) as f:
            karpathy_data = json.load(f)

        data = []
        for datum in karpathy_data["images"]:
            split_ = datum["split"]
            if split_ != self.split:
                continue

            img_id = datum["filename"].split(".")[0]

            if "train" in img_id:
                source = "train2014"
            elif "val" in img_id:
                source = "val2014"        
            else:
                raise NotImplementedError(
                    f"Please specify the image directory for the image: {img_id}"
                )

            image_path = os.path.join(self.data_dir, source, datum["filename"])
            instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                "ShortCaptioning", "An image of "
            )
            targets = [d["raw"].strip() for d in datum["sentences"]]
            response = " ".join(targets)#targets[0]  # take only the first caption

            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "targets": "$$".join(targets),
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data



class _ImageDataset(ImageTextDataset):
    def create_dataset(self) -> None:
        data = []

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg', '.dcm')):
                    img_id = os.path.splitext(file)[0]
                    image_path = os.path.join(root, file)

                    instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                        "ShortCaptioning", "An image of "
                    )
                    response = ""  # Assuming response generation is handled elsewhere

                    item = {
                        "img_id": img_id,
                        "instruction": instruction,
                        "response": response,
                        "image": image_path,
                        "targets": "",  # No targets since no JSON file
                    }
                    data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data


class JSONDataset(ImageTextDataset):
    def create_dataset(self) -> None:
        data = {}

        # Helper to resolve the annotation JSON path robustly
        def _resolve_json_path() -> str:
            # Prefer explicit file path in annotation_file
            if isinstance(self.annotation_file, str) and os.path.isfile(self.annotation_file):
                return self.annotation_file
            # Try relative to data_dir
            if isinstance(self.annotation_file, str):
                candidate = os.path.join(self.data_dir, self.annotation_file)
                if os.path.isfile(candidate):
                    return candidate
            # As a fallback, if data_dir itself is a json file, use it as annotation
            if isinstance(self.data_dir, str) and os.path.isfile(self.data_dir) and self.data_dir.endswith(".json"):
                return self.data_dir
            return ""

        # Try to load annotations if a JSON path is provided
        annotation_json_path = _resolve_json_path()
        if annotation_json_path.endswith(".json") and os.path.isfile(annotation_json_path) and self.prompt_template == "cgdl" :
            with open(annotation_json_path, "r") as f:
                annotations = json.load(f)

            # Case 1: annotations is a dict of dicts (e.g., {"key": {"relative/image.jpg": {...}}})
            if isinstance(annotations, dict) and all(
                isinstance(v, dict) for v in annotations.values()
            ):
                for top_key, file_map in annotations.items():
                    concept_data = []
                    if not isinstance(file_map, dict):
                        continue
                    for rel_path, meta in file_map.items():
                        # Filter by split if rel_path starts with split directory (e.g., "train/"
                        # Resolve image path
                        image_path = os.path.join(self.data_dir, rel_path) if self.data_dir else rel_path

                        # Derive img_id
                        if isinstance(meta, dict):
                            image_size = meta.get("meta", {}).get("image_size", None)
                            patch_size = meta.get("meta", {}).get("patch_size", None)
                            if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
                                try:
                                    w = int(image_size[0])
                                    h = int(image_size[1])
                                    image_size = [w, h]
                                except Exception:
                                    image_size = [0, 0]

                            instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                                "ShortCaptioning", "An image of "
                            )
                            response = ""

                            # --- Mask-centric path (only RLE masks) ---
                            masks_rle_list = meta.get("masks_rle", None)
                            if masks_rle_list and isinstance(masks_rle_list, list) and len(masks_rle_list) > 0:
                                # Use masks_rle as the primary data source
                                for i, mask_entry in enumerate(masks_rle_list):
                                    rle = mask_entry.get("rle", None) if isinstance(mask_entry, dict) else None
                                    is_concept = mask_entry.get("is_concept", False) if isinstance(mask_entry, dict) else False
                                    if rle is None:
                                        continue  # skip entries with no mask
                                    img_id = f"{os.path.splitext(os.path.basename(rel_path))[0]}_mask_{i}"
                                    item = {
                                        "img_id": img_id,
                                        "instruction": instruction,
                                        "response": response,
                                        "image": image_path,
                                        "targets": "",
                                        "image_size": image_size,
                                        "patch_size": patch_size,
                                        "concept": top_key,
                                        "seg_mask_rle": rle,
                                        "is_concept": is_concept,
                                    }
                                    concept_data.append(item)
                    if self.dataset_size > 0 and len(concept_data) > self.dataset_size:
                        random.shuffle(concept_data)
                        concept_data = self.rng.choice(concept_data, size=self.dataset_size, replace=False)
                        # Ensure list type if numpy array is returned
                        if hasattr(concept_data, "tolist"):
                            concept_data = concept_data.tolist()
                    data[top_key] = concept_data

        # Store grouped data by concept (top_key)
        self.data = data

        # Build flat index for unified indexing, useful if this dataset is used directly
        self._rebuild_flat_index()

    # ----- Helpers for grouped datasets -----
    def _rebuild_flat_index(self) -> None:
        """
        Build a flat index mapping global idx -> (concept_key, local_idx).
        This lets the dataset behave like a standard Dataset even when data is grouped by concept.
        """
        self._flat_index = []
        if isinstance(self.data, dict):
            for concept_key, items in self.data.items():
                for i in range(len(items)):
                    self._flat_index.append((concept_key, i))

    def __len__(self) -> int:  # type: ignore[override]
        if isinstance(self.data, dict):
            return sum(len(v) for v in self.data.values())
        return super().__len__()

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        # Support both grouped (dict) and flat (list) storage
        if isinstance(self.data, dict):
            # Rebuild index if missing or out-of-date
            if not hasattr(self, "_flat_index") or len(self._flat_index) != sum(
                len(v) for v in self.data.values()
            ):
                self._rebuild_flat_index()
            concept_key, local_idx = self._flat_index[idx]
            item = dict(self.data[concept_key][local_idx])  # shallow copy
            # Keep track of concept the sample came from
            item["concept"] = concept_key
            item["text"] = self.apply_prompt(item, mode=self.mode)
            return item
        # Fallback to base implementation for non-dict storage
        return super().__getitem__(idx)

    def per_concept_datasets(self) -> Dict[str, Dataset]:
        """
        Return a dict mapping concept_key -> torch Dataset containing only that concept's items.
        Each item is augmented with a `text` field using the dataset's prompt mode.
        """
        if not isinstance(self.data, dict):
            # If data isn't grouped, treat the whole thing as a single concept
            return {"all": _ListDictDataset(self.data, mode=self.mode, apply_prompt_fn=self.apply_prompt)}  # type: ignore[arg-type]

        per_ds: Dict[str, Dataset] = {}
        for concept_key, items in self.data.items():
            per_ds[concept_key] = _ListDictDataset(items, mode=self.mode, apply_prompt_fn=self.apply_prompt, concept_key=concept_key)
        return per_ds

    def build_dataloaders(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> Tuple[Dict[str, DataLoader], DataLoader]:
        """
        Build DataLoaders for each concept, plus a single combined DataLoader.

        Returns:
            - loaders_by_concept: Dict[concept_key, DataLoader]
            - combined_loader: DataLoader concatenating all concept datasets
        """
        per_ds = self.per_concept_datasets()
        loaders_by_concept: Dict[str, DataLoader] = {}
        for k, ds in per_ds.items():
            loaders_by_concept[k] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        if len(per_ds) == 0:
            empty_ds = _ListDictDataset([], mode=self.mode, apply_prompt_fn=self.apply_prompt)
            combined_loader = DataLoader(
                empty_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
        else:
            combined_loader = DataLoader(
                ConcatDataset(list(per_ds.values())),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        return loaders_by_concept, combined_loader

    def build_dataloaders_list(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
    ) -> Tuple[List[DataLoader], DataLoader, List[str]]:
        """
        Build and return (list_of_loaders, combined_loader, concept_keys).
        The order of loaders corresponds to the order of concept_keys.
        """
        per_ds = self.per_concept_datasets()
        concept_keys = list(per_ds.keys())
        loader_list: List[DataLoader] = []
        for k in concept_keys:
            ds = per_ds[k]
            loader_list.append(
                DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    collate_fn=collate_fn,
                )
            )

        if len(per_ds) == 0:
            empty_ds = _ListDictDataset([], mode=self.mode, apply_prompt_fn=self.apply_prompt)
            combined_loader = DataLoader(
                empty_ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
        else:
            combined_loader = DataLoader(
                ConcatDataset(list(per_ds.values())),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )

        return loader_list, combined_loader, concept_keys


class _ListDictDataset(Dataset):
    """Simple wrapper dataset over a list of item dicts.

    Adds a `text` field using the provided apply_prompt_fn and optionally attaches `concept`.
    """

    def __init__(
        self,
        items: List[Dict[str, Any]],
        mode: str,
        apply_prompt_fn: Callable[[Dict[str, Any], str], str],
    concept_key: Optional[str] = None,
    ) -> None:
        self.items = items
        self.mode = mode
        self.apply_prompt_fn = apply_prompt_fn
        self.concept_key = concept_key

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.items[idx])  # shallow copy
        if self.concept_key is not None:
            item["concept"] = self.concept_key
        item["text"] = self.apply_prompt_fn(item, mode=self.mode)
        return item

            


class ImageDataset(ImageTextDataset):
    def create_dataset(self) -> None:
        data = []

        if os.path.isfile(self.data_dir):
            # Single file path provided
            if self.data_dir.endswith(('.jpg', '.png', '.jpeg', '.dcm','.JPEG')):
                img_id = os.path.splitext(os.path.basename(self.data_dir))[0]
                image_path = self.data_dir

                instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                    "List of item", "An image of "
                )
                response = ""  # Assuming response generation is handled elsewhere

                item = {
                    "img_id": img_id,
                    "instruction": instruction,
                    "response": response,
                    "image": image_path,
                    "targets": "",  # No targets since no JSON file
                }
                self.data = [item]
                return
        elif "val" in self.data_dir:
            # Directory path provided
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.dcm', '.JPEG')):
                        img_id = os.path.splitext(file)[0]
                        image_path = os.path.join(root, file)

                        instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                            "ShortVQA", "An image of "
                        )
                        response = ""  # Assuming response generation is handled elsewhere

                        item = {
                            "img_id": img_id,
                            "instruction": instruction,
                            "response": response,
                            "image": image_path,
                            "targets": "",  # No targets since no JSON file
                        }
                        data.append(item)
        else:
            # Directory path provided
            for root, _, files in os.walk(self.data_dir):
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.dcm', '.JPEG')):
                        img_id = os.path.splitext(file)[0]
                        image_path = os.path.join(root, file)

                        instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                            "ShortCaptioning", "An image of "
                        )
                        response = ""  # Assuming response generation is handled elsewhere

                        item = {
                            "img_id": img_id,
                            "instruction": instruction,
                            "response": response,
                            "image": image_path,
                            "targets": "",  # No targets since no JSON file
                        }
                        data.append(item)

        if self.dataset_size > 0:
            random.shuffle(data)
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data



class TextDataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        # if self annotation_file is is a text file path then load the text file
        # and create a list of sentences
        # if self.annotation_file is a string then just put in to a list 
        
        lines = []
        if os.path.isfile(self.data_dir):
            annotation_path = self.data_dir
            with open(annotation_path) as f:
            # Load the text file and read all the lines and create a list of sentences
                lines = f.readlines()
        else:
            # If it's not a file, just put the string into a list
            lines = [self.data_dir]
        
        

        data = []
        for datum in lines:
            instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                "Repeat the text", "An image of "
            )
            if len(datum)< 2:
                print(f"Skipping empty line in ")
                continue
            item = {
                "instruction": f"{instruction}:' '{datum}",
                "response": "",
                "targets": f"{datum}",
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data



class VQAv2Dataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:

        annotations = json.load(
            open(os.path.join(self.data_dir, self.annotation_file))
        )["annotations"]
        questions = json.load(open(os.path.join(self.data_dir, self.questions_file)))[
            "questions"
        ]
        qid_2_question = {d["question_id"]: d["question"] for d in questions}

        assert self.split in [
            "val2014",
            "train2014",
        ], f"{self.split} split is not supported."

        data = []
        for datum in annotations:

            img_id = datum["image_id"]
            image_name = f"COCO_{self.split}_{img_id:012d}.jpg"
            image_path = os.path.join(self.data_dir, self.split, image_name)

            question_id = datum["question_id"]
            instruction = f"{qid_2_question[question_id].strip()}"
            instruction += TASK_PROMPTS.get(self.prompt_template, {}).get(
                "ShortVQA", " "
            )
            response = datum["multiple_choice_answer"].strip()
            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "answer_type": datum["answer_type"],
                "question_id": datum["question_id"],
                "targets": "$$".join([d["answer"] for d in datum["answers"]]),
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data

class XRAYdataset(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        annotation_path = os.path.join(self.data_dir, self.annotation_file)
        with open(annotation_path) as f:
            karpathy_data = json.load(f)

        data = []
        for datum in karpathy_data["images"]:
            split_ = datum["split"]
            if split_ != self.split:
                continue

            img_id = datum["filename"].split(".")[0]


            image_path = os.path.join(self.data_dir, datum["filepath"], datum["filename"])
            instruction = TASK_PROMPTS.get(self.prompt_template, {}).get(
                 "Findings", "Please provide a detailed finding of chest X-ray "
            )
            targets = [d["raw"].strip() for d in datum["sentences"]]
            response = targets[0]  # take only the first caption

            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "targets": "$$".join(targets),
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data


class XRAYdataset_view(ImageTextDataset):
    def create_dataset(
        self,
    ) -> None:
        annotation_path = os.path.join(self.data_dir, self.annotation_file)
        with open(annotation_path) as f:
            karpathy_data = json.load(f)

        data = []
        for datum in karpathy_data["View Classification"]:
            split_ = datum["split"]
            if split_ != self.split:
                continue

            img_id = datum["image_path"].split(".")[0]


            image_path = os.path.join(self.data_dir, datum["image_path"])
            instruction = f'Identify the view of this CXR. (A) {datum["option_0"]}, (B) {datum["option_1"]}, (B) {datum["option_3"]}'
            targets = datum["slice"]
            response = targets  # take only the first caption

            item = {
                "img_id": img_id,
                "instruction": instruction,
                "response": response,
                "image": image_path,
                "targets": targets,
            }
            data.append(item)

        if self.dataset_size > 0:
            data = self.rng.choice(data, size=self.dataset_size, replace=False)

        self.data = data