import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpers.layer_selection import (
    build_region_image,
    discover_decoder_layers,
    get_visual_token_ids,
    resolve_sweep_layers,
    sample_tag_regions,
)
from helpers.logit_lens import score_hidden_state_with_logit_lens


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {
            "cat": [3],
            " cat": [3],
            "Cat": [3],
            "CAT": [3],
            "dog": [2],
            " dog": [2],
        }
        return mapping.get(text, [1])

    def decode(self, token_ids, skip_special_tokens=True):
        reverse = {1: "unk", 2: "dog", 3: "cat"}
        return " ".join(reverse.get(int(t), f"tok{int(t)}") for t in token_ids)

    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 9}.get(token, None)


class DummyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Identity()


class DummyVLM(torch.nn.Module):
    def __init__(self, num_text_layers=4, num_visual_layers=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(num_text_layers)]
        )
        self.visual = torch.nn.Module()
        self.visual.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(num_visual_layers)]
        )


# ---------------------------------------------------------------------------
# Relative probability (p(tag) / p(top-1)) in the logit-lens scorer
# ---------------------------------------------------------------------------

def _make_lm_head():
    lm_head = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        lm_head.weight.zero_()
        lm_head.weight[3, 0] = 10.0  # "cat" direction
        lm_head.weight[2, 1] = 10.0  # "dog" direction
    return lm_head


def test_relative_probability_is_one_when_concept_is_top_token():
    result = score_hidden_state_with_logit_lens(
        hidden_state=torch.tensor([[1.0, 0.0]]),  # aligned with "cat"
        lm_head=_make_lm_head(),
        tokenizer=DummyTokenizer(),
        concept_text="cat",
        language_model=DummyLanguageModel(),
    )
    assert result["concept_token_id"] == 3
    assert abs(result["relative_probability"] - 1.0) < 1e-6
    assert 0.0 < result["top1_token_probability"] <= 1.0


def test_relative_probability_below_one_when_other_token_dominates():
    result = score_hidden_state_with_logit_lens(
        hidden_state=torch.tensor([[0.0, 1.0]]),  # aligned with "dog"
        lm_head=_make_lm_head(),
        tokenizer=DummyTokenizer(),
        concept_text="cat",
        language_model=DummyLanguageModel(),
    )
    assert result["relative_probability"] < 1.0
    assert result["relative_probability"] > 0.0
    # relative = p(cat) / p(dog at top)
    expected = result["concept_token_probability"] / result["top1_token_probability"]
    assert abs(result["relative_probability"] - expected) < 1e-6


def test_relative_probability_ranks_layers():
    # Layer A: concept dominant; layer B: other token dominant. The relative
    # score must rank A above B, as select_layer_for_tag relies on it.
    lm_head = _make_lm_head()
    tokenizer = DummyTokenizer()
    language_model = DummyLanguageModel()
    score_a = score_hidden_state_with_logit_lens(
        torch.tensor([[0.9, 0.1]]), lm_head, tokenizer, "cat", language_model
    )["relative_probability"]
    score_b = score_hidden_state_with_logit_lens(
        torch.tensor([[0.1, 0.9]]), lm_head, tokenizer, "cat", language_model
    )["relative_probability"]
    assert score_a > score_b


# ---------------------------------------------------------------------------
# Region sampling from the tag's crops.json entries
# ---------------------------------------------------------------------------

class ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return dict(self.items[idx])


def test_sample_tag_regions_uniform_over_all_items():
    # No is_concept filtering: sliding-window crops carry no such flag, so
    # the sweep samples uniformly from every region of the tag.
    items = (
        [{"image": f"neg{i}.jpg", "text": "probe [concept]", "is_concept": False} for i in range(5)]
        + [{"image": f"pos{i}.jpg", "text": "probe [concept]", "is_concept": True} for i in range(3)]
    )
    regions, prompt = sample_tag_regions(ListDataset(items), 8, random.Random(0))
    assert prompt == "probe [concept]"
    assert len(regions) == 8  # whole pool, no filtering


def test_sample_tag_regions_caps_and_is_deterministic():
    items = [
        {"image": f"pos{i}.jpg", "text": "t", "is_concept": True} for i in range(20)
    ]
    first, _ = sample_tag_regions(ListDataset(items), 4, random.Random(7))
    second, _ = sample_tag_regions(ListDataset(items), 4, random.Random(7))
    assert len(first) == 4
    assert [r["image"] for r in first] == [r["image"] for r in second]


def test_sample_tag_regions_falls_back_to_all_items():
    items = [{"image": "a.jpg", "text": "t", "is_concept": False}]
    regions, _ = sample_tag_regions(ListDataset(items), 4, random.Random(0))
    assert len(regions) == 1


# ---------------------------------------------------------------------------
# Region image construction (bbox path; mask path needs cv2/scipy stack)
# ---------------------------------------------------------------------------

def _write_test_image(tmp_name, size=(64, 48)):
    from PIL import Image

    path = os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"layer_selection_test_{tmp_name}.png"
    )
    Image.new("RGB", size, (10, 200, 30)).save(path)
    return path


def test_build_region_image_bbox_crop_and_patch_resize():
    path = _write_test_image("bbox")
    item = {
        "image": path,
        "bbox": [8, 4, 40, 36],
        "image_size": [64, 48],
        "patch_size": 16,
        "is_concept": True,
    }
    built = build_region_image(item)
    assert built is not None
    region, bbox = built
    assert bbox == [8, 4, 40, 36]
    assert region.size == (16, 16)  # PATCH resize with padding


def test_build_region_image_full_image_fallback():
    os.environ.pop("PATCH_SIZE", None)  # no patch resize in this test
    path = _write_test_image("full")
    built = build_region_image({"image": path, "is_concept": False})
    assert built is not None
    region, bbox = built
    assert bbox == [0, 0, 64, 48]
    assert region.size == (64, 48)


def test_build_region_image_missing_file_returns_none():
    assert build_region_image({"image": "/nonexistent/file.jpg"}) is None


# ---------------------------------------------------------------------------
# Sweep layer resolution
# ---------------------------------------------------------------------------

def test_discover_decoder_layers_prefers_language_model_and_skips_visual():
    model = DummyVLM(num_text_layers=4, num_visual_layers=2)
    layers = discover_decoder_layers(model)
    assert layers == [f"model.language_model.layers.{i}" for i in range(4)]


def test_resolve_sweep_layers_auto_and_specs():
    model = DummyVLM(num_text_layers=4)
    assert resolve_sweep_layers(model, "auto") == [
        f"model.language_model.layers.{i}" for i in range(4)
    ]
    assert resolve_sweep_layers(model, "1-2") == [
        "model.language_model.layers.1",
        "model.language_model.layers.2",
    ]
    assert resolve_sweep_layers(model, "0,3") == [
        "model.language_model.layers.0",
        "model.language_model.layers.3",
    ]
    assert resolve_sweep_layers(model, "model.language_model.layers.2") == [
        "model.language_model.layers.2"
    ]


def test_resolve_sweep_layers_rejects_unknown_module():
    model = DummyVLM()
    try:
        resolve_sweep_layers(model, "model.language_model.layers.99")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown module path")


# ---------------------------------------------------------------------------
# Visual token id discovery
# ---------------------------------------------------------------------------

def test_get_visual_token_ids_from_config():
    model = DummyVLM()
    model.config = type("Config", (), {"image_token_id": 151655})()
    assert get_visual_token_ids(model, None) == [151655]


def test_get_visual_token_ids_tokenizer_fallback():
    model = DummyVLM()  # no config attribute
    assert get_visual_token_ids(model, DummyTokenizer()) == [9]
