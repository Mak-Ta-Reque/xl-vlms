import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from helpers.logit_lens import expand_layer_specs, resolve_layer_modules, score_hidden_state_with_logit_lens


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
        reverse = {
            1: "unk",
            2: "dog",
            3: "cat",
        }
        return " ".join(reverse.get(int(token_id), f"tok{int(token_id)}") for token_id in token_ids)


class DummyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Identity()


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
        self.model.layers[5].dummy = torch.nn.Identity()


def test_expand_layer_specs_range_and_list():
    assert expand_layer_specs("[5-7]") == ["5", "6", "7"]
    assert expand_layer_specs("[5,7,9]") == ["5", "7", "9"]
    assert expand_layer_specs("model.layers.[5-6]") == ["model.layers.5", "model.layers.6"]


def test_resolve_layer_modules_numeric_index():
    model = DummyModel()
    resolved = resolve_layer_modules(model, "[5-6]")
    assert resolved == ["model.layers.5", "model.layers.6"]


def test_score_hidden_state_with_logit_lens_prefers_concept_token():
    lm_head = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        lm_head.weight.zero_()
        lm_head.weight[3, 0] = 10.0
        lm_head.weight[2, 1] = 5.0

    hidden_state = torch.tensor([[1.0, 0.0]])
    tokenizer = DummyTokenizer()
    language_model = DummyLanguageModel()

    result = score_hidden_state_with_logit_lens(
        hidden_state=hidden_state,
        lm_head=lm_head,
        tokenizer=tokenizer,
        concept_text="cat",
        language_model=language_model,
        top_k=3,
    )

    assert result["concept_token_id"] == 3
    assert result["concept_token"] == "cat"
    assert result["top_tokens"][0]["token_id"] == 3
    assert result["concept_token_probability"] > 0.0
