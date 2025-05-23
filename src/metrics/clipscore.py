"""
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
"""

import warnings
import os
import clip
import numpy as np
import sklearn
import torch
import tqdm
from packaging import version
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize,
                                    ToTensor)
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut





class CLIPCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix="A photo is described by the words "):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        from helpers.utils import load_image_as_rgb
        #image = Image.open(c_data, out_type="PIL")
        #image = Image.open(c_data, out_type="PIL")
        image = load_image_as_rgb(c_data,  out_type="PIL")
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


def extract_text_features(captions, model, device, batch_size=256):
    data = torch.utils.data.DataLoader(
        CLIPCaptionDataset(captions), batch_size=batch_size, shuffle=False
    )
    all_text_features = []
    with torch.no_grad():
        for idx, b in enumerate(data):
            b = b["caption"].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_image_features(images, model, device, batch_size=64):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images), batch_size=batch_size, shuffle=False
    )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["image"].to(device)
            if device == "cuda":
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)

    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    """
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    """
    if isinstance(images, list):
        # Extract image CLIP features
        images = extract_image_features(images, model, device)

    candidates = extract_text_features(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse("1.21"):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            "due to a numerical instability, new numpy normalization is slightly different than paper results. "
            "to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3."
        )
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def find_two_longest_words(words):
    if not words:
        return []

    # Sort the list by word length in descending order
    sorted_words = sorted(words, key=len, reverse=True)

    # Return the top two longest words (or fewer if list has < 2 items)
    return sorted_words[:2]
def remove_substrings(tokens):
    tokens = list(tokens)  # Convert to list to allow indexing
    result = set()

    for i, token in enumerate(tokens):
        if not any(token != other and token in other for other in tokens):
            result.add(token)

    return result

def img_clipscore(model, img_feat, activ, grounding_words, device, top_k=3):
    # Assume activ is of shape (n_comp,)
    top_comp = activ.argsort()[-top_k:]
    candidates = []
    for comp_idx in top_comp:
        comp_grounding_words = grounding_words[comp_idx]
        #print(f"Top {comp_idx} words: {comp_grounding_words}")
        comp_grounding_words= [item.lower() for item in comp_grounding_words]
        comp_grounding_words = list(set(comp_grounding_words))
        comp_grounding_words = [word for word in comp_grounding_words if len(word)> 2]
        comp_grounding_words = remove_substrings(comp_grounding_words) 
        cand = ""
        for word in comp_grounding_words:
            cand = cand + word + ", "
        cand = cand[: len(cand) - 2] + "."
        candidates.append(cand)

    image_feat = np.array([img_feat] * top_k)
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feat, candidates, device
    )
    score = np.array(per_instance_image_text)
    return score

def alterantive_img_clipscore(model, img_feat, activ, grounding_words, device, top_k=3):
    # Assume activ is of shape (n_comp,)
    top_comp = activ.argsort()[-top_k:]
    candidates = []
    for comp_idx in top_comp:
        comp_grounding_words = grounding_words[comp_idx]
        comp_grounding_words= [item.lower() for item in comp_grounding_words]
        comp_grounding_words = list(set(comp_grounding_words))
        if len(comp_grounding_words) > 2:
            longest_grounded_words = find_two_longest_words(comp_grounding_words)
        else:
            longest_grounded_words = comp_grounding_words
        if len(longest_grounded_words) == 1:
            cand = f"An image contains a {longest_grounded_words[0]}"
        else:
            cand = "An image contains " + ", ".join(f"a {w}" for w in longest_grounded_words[:-1]) + \
           " and a " + longest_grounded_words[-1]

        #for word in comp_grounding_words:
        #    cand = cand + word + ", "
        #cand = cand[: len(cand) - 2] + "."
        candidates.append(cand)

    image_feat = np.array([img_feat] * top_k)
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feat, candidates, device
    )
    score = np.array(per_instance_image_text)
    return score
