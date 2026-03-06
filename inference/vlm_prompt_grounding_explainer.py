#!/usr/bin/env python3
"""Vision-Language Model Grounding Explainer (prompt-based)

- Loads a supported VLM via repo's model loader
- Prompts it to detect and localize objects (names + bounding boxes)
- Parses a structured JSON response and saves a uniform output schema

Differences from concept-based explainer:
- No feature hooks or concept vectors
- One explanation per object (its bounding box) instead of ranked concepts
"""

from __future__ import annotations

import os
import json
import argparse
import logging
import random
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
from PIL import Image
import torch


def set_seed_all(seed: int, deterministic: bool = True) -> None:
    try:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        try:
            from transformers import set_seed as hf_set_seed  # type: ignore
            hf_set_seed(seed)
        except Exception:
            pass
    except Exception:
        pass


class VLMGroundingExplainer:
    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        trust_remote_code: bool = True,
        verbose: bool = False,
        bbox_format: str = "xyxy_norm",
        ) -> None:
                """Initialize the model and processor via project loader.

                bbox_format:
                    - "xyxy_norm": [x_min, y_min, x_max, y_max], each in [0,1]
                    - "xywh_norm": [x, y, w, h], each in [0,1]
                """
                self.model_name = model_name
                self.trust_remote_code = trust_remote_code
                self.verbose = verbose
                self.bbox_format = bbox_format

                self.model, self.processor, self._model_class = self._load_model(device)
                self.device = next(self.model.parameters()).device

    def _load_model(self, device_override: Optional[Union[str, torch.device]] = None):
        project_root = Path(__file__).resolve().parents[1]
        src_dir = project_root / "src"
        if str(src_dir) not in os.sys.path:
            os.sys.path.insert(0, str(src_dir))



        from device_utils import get_device_config  # type: ignore
        device_config = get_device_config(
            str(device_override) if device_override is not None else None
        )
        device = device_config.primary_device
        self._device_config = device_config

        from models import get_model_class  # type: ignore

        args = argparse.Namespace(
            local_files_only=False,
            cache_dir=os.environ.get("HF_HOME"),
        )
        model_class = get_model_class(
            model_name_or_path=self.model_name,
            processor_name=self.model_name,
            device=device,
            logger=None,
            args=args,
            device_config=device_config,
        )
        model = model_class.get_model()
        processor = model_class.get_processor()
        return model, processor, model_class

    def _build_grounding_instruction(self) -> str:
        # Request strict JSON to ease parsing; normalized coordinates avoid dependency on image size
        if self.bbox_format == "xywh_norm":
            box_spec = "bbox as [x, y, w, h] normalized to [0,1]"
        else:
            box_spec = "bbox as [x_min, y_min, x_max, y_max] normalized to [0,1]"

        return (
            "Detect and localize every salient object in the image. "
            f"Return ONLY a compact JSON object with an 'objects' array; each item has 'name' (string) and {box_spec}. "
            "Optionally include 'score' in [0,1]. No text outside JSON."
        )

    def _prepare_inputs_single(self, image: Union[str, Path, Image.Image]):
        instruction = self._build_grounding_instruction()
        if getattr(self, "_model_class", None) is not None:
            return self._model_class.preprocessor(
                instruction=instruction,
                image_file=image,
                response="",
                generation_mode=True,
            )
        # Fallback: let processor build chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    @staticmethod
    def _extract_json_blob(text: str) -> Optional[str]:
        # Common patterns: pure JSON, fenced code blocks, preambles
        text = text.strip()
        # Remove code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_\-]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        # If already looks like JSON
        if text.startswith("{") and text.endswith("}"):
            return text
        # Find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return m.group(0)
        return None

    @staticmethod
    def _safe_parse_objects(js: Dict[str, Any]) -> List[Dict[str, Any]]:
        objs = js.get("objects") if isinstance(js, dict) else None
        if not isinstance(objs, list):
            return []
        parsed = []
        for o in objs:
            if not isinstance(o, dict):
                continue
            name = o.get("name")
            bbox = o.get("bbox") or o.get("box")
            score = o.get("score")
            if isinstance(name, str) and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    bb = VLMGroundingExplainer._clip_bbox([float(x) for x in bbox])
                    item = {"name": name, "bbox": bb}
                    if isinstance(score, (int, float)):
                        item["score"] = float(score)
                    parsed.append(item)
                except Exception:
                    continue
        return parsed

    @staticmethod
    def _fallback_extract_objects_from_text(text: str) -> List[Dict[str, Any]]:
        """Lenient extractor for objects when JSON is truncated.

        Looks for patterns like: "name": "..." followed by "bbox": [a, b, c, d]
        Returns list of {name, bbox[, score]} when possible.
        """
        objs: List[Dict[str, Any]] = []
        try:
            # Normalize whitespace for regex
            t = text
            # Find all occurrences of name and bbox pairs
            name_iter = list(re.finditer(r'"name"\s*:\s*"([^"]+)"', t))
            bbox_iter = list(re.finditer(r'"bbox"\s*:\s*\[([^\]]+)\]', t))
            score_iter = list(re.finditer(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', t))

            # Build proximity-based associations: for each name, find the nearest bbox that follows
            bboxes = [(m.start(), m.group(1)) for m in bbox_iter]
            scores = [(m.start(), float(m.group(1))) for m in score_iter]
            for nm in name_iter:
                name = nm.group(1)
                pos = nm.end()
                # Find first bbox after name
                bbox_str = None
                bbox_pos = None
                for p, bstr in bboxes:
                    if p > pos:
                        bbox_pos = p
                        bbox_str = bstr
                        break
                if bbox_str is None:
                    continue
                # Parse up to 4 floats
                nums = re.findall(r'-?\d*\.?\d+', bbox_str)
                if len(nums) < 4:
                    continue
                try:
                    bb = [float(nums[i]) for i in range(4)]
                except Exception:
                    continue
                # Find nearest score after bbox (optional)
                sc = None
                if scores:
                    for sp, sv in scores:
                        if bbox_pos is not None and sp > bbox_pos:
                            sc = sv
                            break
                item = {"name": name, "bbox": VLMGroundingExplainer._clip_bbox(bb)}
                if sc is not None:
                    item["score"] = sc
                objs.append(item)
        except Exception:
            pass
        return objs

    @staticmethod
    def _clip_bbox(bb: List[float]) -> List[float]:
        return [max(0.0, min(1.0, float(v))) for v in bb[:4]]

    @staticmethod
    def _clean_token_text(s: str) -> str:
        # Trim quotes/whitespace; keep alnum and dashes/underscores
        s = re.sub(r'^[\s\"]+|[\s\"]+$', '', s)
        return s.strip()

    @torch.inference_mode()
    def explain_with_grounding(
        self,
        images: List[Union[str, Path, Image.Image]],
        ground_truth_labels: Optional[List[Optional[str]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        batch_size: int = 1,
    ) -> List[Dict[str, Any]]:
        if ground_truth_labels and len(ground_truth_labels) != len(images):
            raise ValueError("ground_truth_labels length must match images length")

        # Prepare inputs (paths preferred for model preprocessor)
        prepped_inputs: List[Union[Image.Image, str, Path]] = []
        abs_image_paths: List[Optional[str]] = []
        for img_in in images:
            if isinstance(img_in, (str, Path)):
                prepped_inputs.append(str(img_in))
                try:
                    abs_image_paths.append(os.path.abspath(str(img_in)))
                except Exception:
                    abs_image_paths.append(None)
            elif isinstance(img_in, Image.Image):
                prepped_inputs.append(img_in.convert('RGB'))
                abs_image_paths.append(None)
            else:
                raise TypeError(f"Unsupported image type: {type(img_in)}")

        results: List[Dict[str, Any]] = []

        def chunks(seq, n):
            for i in range(0, len(seq), n):
                yield i, seq[i:i+n]

        for base_idx, img_chunk in chunks(prepped_inputs, batch_size):
            inputs = self._prepare_inputs_single(img_chunk[0])
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

            tokenizer = None
            if getattr(self, "_model_class", None) is not None:
                tokenizer = getattr(self._model_class, "get_tokenizer", lambda: None)()
            if tokenizer is None and hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                pad_token_id=(tokenizer.eos_token_id if tokenizer is not None else None),
                use_cache=True,
                early_stopping=True,
            )
            gen_config = getattr(self.model, 'generation_config', None)
            if gen_config is not None:
                try:
                    gen_config = gen_config.clone()
                except Exception:
                    from copy import deepcopy
                    gen_config = deepcopy(gen_config)
                if temperature > 0:
                    try:
                        gen_config.temperature = float(temperature)
                    except Exception:
                        pass

            if gen_config is not None:
                out_tokens = self.model.generate(**inputs, generation_config=gen_config, **gen_kwargs)
            else:
                out_tokens = self.model.generate(**inputs, **gen_kwargs)

            input_len = inputs['input_ids'].shape[-1]
            batch_texts: List[str] = []
            gen_token_ids: List[List[int]] = []
            for row in out_tokens:
                new = row[input_len:]
                if tokenizer is not None:
                    decoded = tokenizer.decode(new, skip_special_tokens=True)
                else:
                    decoded = self.processor.decode(new, skip_special_tokens=True)
                batch_texts.append(decoded.strip())
                gen_token_ids.append(new.tolist())

            for j, text in enumerate(batch_texts):
                js_blob = self._extract_json_blob(text)
                objs: List[Dict[str, Any]] = []
                if js_blob is not None:
                    try:
                        parsed = json.loads(js_blob)
                        objs = self._safe_parse_objects(parsed)
                    except Exception:
                        objs = []
                if not objs:
                    # Fallback to lenient extractor for truncated JSON
                    objs = self._fallback_extract_objects_from_text(text)
                if self.verbose:
                    logging.debug(f"Raw output: {text}")
                    logging.debug(f"Parsed objects: {objs}")

                # Decode per-token strings for alignment (not stored in output)
                if tokenizer is not None:
                    token_texts = [
                        (tokenizer.decode([tid], skip_special_tokens=True) if isinstance(tid, int) else "")
                        for tid in gen_token_ids[j]
                    ]
                else:
                    token_texts = [""] * len(gen_token_ids[j])
                token_texts_clean = [self._clean_token_text(t) for t in token_texts]

                # Map object names to token id subsequences and align to generated ids
                per_token_explantion: List[Dict[str, Any]] = []
                explanations_by_idx: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(gen_token_ids[j]))}
                if tokenizer is not None and objs:
                    try:
                        # Pre-tokenize each object name
                        name_to_ids: List[tuple[List[int], Dict[str, Any]]] = []
                        for o in objs:
                            name = o.get('name') or ""
                            enc = tokenizer(name, add_special_tokens=False)
                            ids = enc['input_ids'] if isinstance(enc, dict) else enc
                            if isinstance(ids, list) and len(ids) > 0:
                                name_to_ids.append((ids, o))

                        hay = gen_token_ids[j]
                        # Simple subsequence search for each name
                        for ids, o in name_to_ids:
                            n = len(ids)
                            for start in range(0, max(0, len(hay) - n + 1)):
                                if hay[start:start + n] == ids:
                                    for k in range(n):
                                        explanations_by_idx[start + k].append({
                                            'name': o.get('name'),
                                            'bbox': o.get('bbox'),
                                            'score': o.get('score') if isinstance(o.get('score'), (int, float)) else None,
                                        })
                        # Fallback: match by decoded token text equality (trim quotes/space)
                        for o in objs:
                            nm = (o.get('name') or '').strip()
                            if not nm:
                                continue
                            nm_lower = nm.lower()
                            # Skip if already matched via ids
                            if any(any((e.get('name') or '').lower() == nm_lower for e in lst) for lst in explanations_by_idx.values()):
                                continue
                            for idx, tt in enumerate(token_texts_clean):
                                if tt.lower() == nm_lower:
                                    explanations_by_idx[idx].append({
                                        'name': o.get('name'),
                                        'bbox': o.get('bbox'),
                                        'score': o.get('score') if isinstance(o.get('score'), (int, float)) else None,
                                    })
                                    break
                    except Exception:
                        # If tokenization/search fails, leave explanations empty
                        pass

                for idx, tok_id in enumerate(gen_token_ids[j]):
                    exp_list = explanations_by_idx.get(idx, [])
                    # Only store entries for object tokens that have explanations
                    if not exp_list:
                        continue
                    # Choose a single bbox to store (prefer highest score if available)
                    chosen = None
                    best_score = -1.0
                    for e in exp_list:
                        sc = e.get('score')
                        if isinstance(sc, (int, float)) and sc > best_score:
                            best_score = float(sc)
                            chosen = e
                        if sc is None and chosen is None:
                            chosen = e
                    bbox = chosen.get('bbox') if isinstance(chosen, dict) else None
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        continue
                    per_token_explantion.append({
                        'token_id': tok_id,
                        'explantion': VLMGroundingExplainer._clip_bbox(bbox),
                    })

                # Aggregate overall bboxes and names over the whole sequence
                names_all: List[str] = []
                boxes_all: List[List[float]] = []
                token_ids_all: List[List[int]] = []
                seen = set()
                for o in objs:
                    nm = o.get('name')
                    bb = o.get('bbox')
                    if not isinstance(nm, str) or not (isinstance(bb, list) and len(bb) == 4):
                        continue
                    key = (nm, tuple(bb))
                    if key in seen:
                        continue
                    seen.add(key)
                    names_all.append(nm)
                    boxes_all.append(VLMGroundingExplainer._clip_bbox(bb))
                    # Collect all token ids matching this name AND bbox
                    ids_for_entry: List[int] = []
                    nm_lower = nm.lower()
                    for idx, lst in explanations_by_idx.items():
                        for e in lst:
                            if (e.get('name') or '').lower() == nm_lower and isinstance(e.get('bbox'), list) and e.get('bbox') == bb:
                                ids_for_entry.append(int(gen_token_ids[j][idx]))
                                break
                    # dedupe while preserving order
                    seen_ids = set()
                    ids_unique = []
                    for tid in ids_for_entry:
                        if tid not in seen_ids:
                            seen_ids.add(tid)
                            ids_unique.append(tid)
                    token_ids_all.append(ids_unique)
                grounding_over_sequence = {
                    'overal': boxes_all,
                    'name': names_all,
                    'token_id': token_ids_all,
                }

                results.append({
                    'image_path': abs_image_paths[base_idx + j],
                    'ground_truth': ground_truth_labels[base_idx + j] if ground_truth_labels else None,
                    'model_output': text,
                    # 'generated_token_ids' omitted as requested
                    # Keep concept-related fields for schema compatibility but empty
                    'per_token_concepts': [],
                    'top_concepts_over_sequence': [],
                    # New grounding-based explanations
                    'grounded_objects': objs,
                    'per_token_explantion': per_token_explantion,
                    'grounding_over_sequence': grounding_over_sequence,
                    'hook_layer': None,
                    'layer_activation_shape': None,
                    'model_name': self.model_name,
                    'bbox_format': self.bbox_format,
                })

        return results

    def close(self):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prompt-based grounding explainer for VLMs")
    ap.add_argument('--model_name', default=os.environ.get('VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct'))
    ap.add_argument('--image', action='append', help='Image path; can be repeated')
    ap.add_argument('--image_root', default=None, help='Root dir to recursively collect images')
    ap.add_argument('--label', action='append', help='Ground-truth label per image (optional)')
    ap.add_argument('--batch_size', type=int, default=int(os.environ.get('BATCH_SIZE', '10')))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_new_tokens', type=int, default=128)
    ap.add_argument('--bbox_format', choices=['xyxy_norm', 'xywh_norm'], default='xyxy_norm')
    ap.add_argument('--out_json', default=os.path.join(os.environ.get('OUTPUT_DIR', '.'), 'vlm_groundings.json'))
    ap.add_argument('--data_root', default=None, help='Root path of dataset for metadata')
    args = ap.parse_args()

    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format='%(levelname)s: %(message)s')

    # Build image list from --image_root if provided
    if getattr(args, 'image_root', None):
        root = os.path.abspath(args.image_root)
        if not os.path.isdir(root):
            raise ValueError(f"--image_root must be a directory: {root}")
        valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
        collected = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in valid_exts:
                    collected.append(os.path.join(dirpath, fn))
        collected.sort()
        if args.image is None:
            args.image = []
        args.image.extend(collected)
        logging.info(f"Discovered {len(collected)} images under {root}")

    if not args.image:
        raise ValueError("No images provided. Use --image multiple times, or provide --image_root.")

    if args.batch_size > 1:
        raise ValueError("Batch size greater than 1 is not supported.")

    set_seed_all(args.seed, deterministic=args.deterministic)

    explainer = VLMGroundingExplainer(
        model_name=args.model_name,
        verbose=args.verbose,
        bbox_format=args.bbox_format,
    )

    res = explainer.explain_with_grounding(
        args.image,
        ground_truth_labels=args.label,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
    )

    for r in res:
        logging.info(f"Image {r.get('image_path')} -> gt={r.get('ground_truth')}\nObjects: {len(r.get('grounded_objects', []))}")
        if args.verbose:
            for o in r.get('grounded_objects', []):
                logging.info(f"  - {o['name']}: {o['bbox']} score={o.get('score')}")

    try:
        img_abs_list = [os.path.abspath(str(p)) for p in (args.image or [])]
        if args.data_root is not None:
            data_root = os.path.abspath(args.data_root)
        else:
            data_root = os.path.commonpath(img_abs_list) if img_abs_list else None
    except Exception:
        data_root = None

    out_payload: Dict[str, Any] = {
        'model_card': args.model_name,
        'layer_path': None,
        'data_root': data_root,
        'results': res,
    }
    try:
        out_path = args.out_json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved JSON to {out_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON: {e}")

    explainer.close()
