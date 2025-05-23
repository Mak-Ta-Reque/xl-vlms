import argparse
from typing import Any, Callable, Dict, List
from bert_score import score as bert_score
import clip
import numpy as np
import torch
from nltk.corpus import words
import copy
import analysis.feature_decomposition as analysis_decomposition
from metrics.clipscore import extract_image_features, img_clipscore
from metrics.utils import get_stopwords, valid_word

__all__ = [
    "get_clip_score",
    "get_random_words",
    "compute_grounding_words_overlap",
    "compute_test_clipscore",
]


def get_clip_score(
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    concepts_dict: Dict[str, Any] = {},
    model_class: Callable = None,
    device: torch.device = torch.device("cpu"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:
    features = copy.deepcopy(features)
    metadata = copy.deepcopy(metadata)
    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    analysis_model = concepts_dict["analysis_model"]
    grounding_words =  copy.deepcopy(concepts_dict["text_grounding"])
    projections = analysis_decomposition.project_test_sample_using_matix(
        sample=features,
        activations=concepts_dict,
        decomposition_type=concepts_dict["decomposition_method"],
    )

    
    if args.use_random_grounding_words:
        lm_head = model_class.get_lm_head().float()
        tokenizer = model_class.get_tokenizer()
        grounding_words = get_random_words(
            lm_head=lm_head,
            tokenizer=tokenizer,
            grounding_words=grounding_words,
        )
        logger.info(f"Random words usage is True. Only for CLIPScore evaluation")
    

    clipscore_dict = compute_test_clipscore(
        projections=projections,
        grounding_words=grounding_words,
        device=device,
        metadata=metadata,
    )
    logger.info(
        f"top-1 test CLIPScore (mean, std) {clipscore_dict['top_1_mean']: .3f} +/- {clipscore_dict['top_1_std']: .3f}"
    )
    return clipscore_dict


def get_bert_score(
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    concepts_dict: Dict[str, Any] = {},
    model_class: Callable = None,
    device: torch.device = torch.device("cpu"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:
    features = copy.deepcopy(features)
    metadata = copy.deepcopy(metadata)  
    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    analysis_model = concepts_dict["analysis_model"]
    grounding_words = copy.deepcopy(concepts_dict["text_grounding"])
    projections = analysis_decomposition.project_test_sample_using_matix(
        sample=features,
        activations=concepts_dict,
        decomposition_type=concepts_dict["decomposition_method"],
    )

    
    if args.use_random_grounding_words:
        lm_head = model_class.get_lm_head().float()
        tokenizer = model_class.get_tokenizer()
        grounding_words = get_random_words(
            lm_head=lm_head,
            tokenizer=tokenizer,
            grounding_words=grounding_words,
        )
        logger.info(f"Random words usage is True. Only for CLIPScore evaluation")
    

    bertscore_dict = compute_test_bertscore(
        projections=projections,
        grounding_words=grounding_words,
        device=device,
        metadata=metadata,
    )
    logger.info(
        f"top-1 test BERTScore (mean, std) {bertscore_dict['top_1_f1_mean']: .3f} +/- {bertscore_dict['top_1_f1_std']: .3f}"
    )
    return bertscore_dict

def get_jakard_score(
    features: Dict[str, torch.Tensor] = None,
    metadata: Dict[str, Any] = {},
    concepts_dict: Dict[str, Any] = {},
    model_class: Callable = None,
    device: torch.device = torch.device("cpu"),
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:
    features = copy.deepcopy(features)
    metadata = copy.deepcopy(metadata)
    features = list(features.values())[0]
    metadata = list(metadata.values())[0]
    analysis_model = concepts_dict["analysis_model"]
    grounding_words = copy.deepcopy(concepts_dict["text_grounding"])
    projections = analysis_decomposition.project_test_sample_using_matix(
        sample=features,
        activations=concepts_dict,
        decomposition_type=concepts_dict["decomposition_method"],
    )

    
    if args.use_random_grounding_words:
        lm_head = model_class.get_lm_head().float()
        tokenizer = model_class.get_tokenizer()
        grounding_words = get_random_words(
            lm_head=lm_head,
            tokenizer=tokenizer,
            grounding_words=grounding_words,
        )
        logger.info(f"Random words usage is True. Only for CLIPScore evaluation")
    

    jaccard_dict = compute_test_jaccard_score(
        projections=projections,
        grounding_words=grounding_words,
        metadata=metadata,
    )
    logger.info(
        f"top-1 test JACCARDcore (mean, std) {jaccard_dict['top_1_mean']: .3f} +/- {jaccard_dict['top_1_std']: .3f}" 
    )
    return jaccard_dict


def get_random_words(
    lm_head: Callable, tokenizer: Callable, grounding_words: List[List[str]] = []
) -> List[List[str]]:
    """
    This function replaces grounding words of each concept by a set of random words, possibly of same length
    Random words obtained by:
    (i) Sampling a random direction to decode with lm_head
    (ii) Decode top tokens which satisfy same valid word filters as grounding words
    """
    eng_corpus = words.words()
    stopwords = get_stopwords()
    all_random_words = []
    for k, concept_words in enumerate(grounding_words):
        # k is concept idx, words is grounded words for concept k
        desired_length = len(concept_words)
        num_top_tokens = min(
            10 * desired_length, lm_head.out_features
        )  # Should be more than enough
        random_direction = torch.rand(1, lm_head.in_features).float()
        token_logits = lm_head(random_direction)
        top_token_idx = token_logits.argsort(dim=-1, descending=True)[
            :, :num_top_tokens
        ]
        candidate_words = tokenizer.batch_decode(
            top_token_idx[0], skip_special_tokens=True
        )
        candidate_words = [
            word.lower().strip()
            for word in candidate_words
            if valid_word(word, eng_corpus=eng_corpus, stopwords=stopwords)
        ]
        if len(candidate_words) > desired_length:
            candidate_words = candidate_words[:desired_length]
        all_random_words.append(candidate_words)
    return all_random_words


def compute_grounding_words_overlap(
    grounding_words, logger: Callable = None
) -> Dict[str, Any]:
    """
    Function to compute overlap metric given the grounded words of a concept dictionary
    Input: List of grounded words for concepts: List[List]
    """
    grounding_words = copy.deepcopy(grounding_words)
    num_concepts = len(grounding_words)
    overlap_matrix = np.zeros([num_concepts, num_concepts])
    for i in range(num_concepts):
        words_i = grounding_words[i]
        if len(words_i) == 0:
            continue
        for j in range(num_concepts):
            words_j = grounding_words[j]
            overlap_ij = len([w for w in words_i if w in words_j])
            overlap_matrix[i, j] = overlap_ij * 1.0 / len(words_i)

    overlap_metric = overlap_matrix.sum() - np.diag(overlap_matrix).sum()
    overlap_metric = overlap_metric / (num_concepts * (num_concepts - 1))

    if logger is not None:
        logger.info(f"Overlap metric (lower is better): {overlap_metric: .3f}")

    scores = {}
    scores["grounding_words_overlap_metric"] = overlap_metric
    scores["grounding_words_overlap_matrix"] = overlap_matrix
    return scores


def compute_test_clipscore(
    projections: np.ndarray,
    grounding_words: List[List[str]],
    metadata: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
    top_k: int = 5,
) -> Dict[str, Any]:
    scores = []
    image_paths = []
    num_samples = projections.shape[0]
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()

    image_paths = metadata.get("image_paths", [])
    image_paths = metadata.get("image", [])
    token_of_interest_mask = metadata.get("token_of_interest_mask", None)
    if token_of_interest_mask is not None:
        image_paths = [
            image_paths[i][0]
            for i in range(len(image_paths))
            if token_of_interest_mask[i]
        ]
    image_features = extract_image_features(
        image_paths, clip_model, device, batch_size=8
    )  # image_features of shape (num_images, dim)

    for idx in range(num_samples):
        

        image_paths[idx] = image_paths[idx][0]
        #print("Predicted concept:", image_paths[idx].split("@")[0])
        if "No" in image_paths[idx].split("@")[0]:
            continue # MIss classified images are ignored 
        img_activations = projections[idx] #get 15 activations for
        img_feat = image_features[idx]
        img_score = img_clipscore(
            clip_model, img_feat, img_activations, grounding_words, device, top_k=top_k
        )
        scores.append(img_score)
    scores = np.array(scores)
    # Return dictionary containing all test sample scores, their mean, std
    scores_dict = {}
    for k in [1, 3]:
        key = f"top_{k}_all"
        key_mean = f"top_{k}_mean"
        key_std = f"top_{k}_std"
        all_test_scores = scores[:, -k:].mean(axis=1)
        mean_topk_score, std_topk = all_test_scores.mean(), all_test_scores.std()
        scores_dict[key] = all_test_scores
        scores_dict[key_mean] = mean_topk_score
        scores_dict[key_std] = std_topk

    return scores_dict


def compute_test_bertscore(
    projections: np.ndarray,
    grounding_words: List[List[str]],
    metadata: Dict[str, Any],
    device: torch.device = torch.device("cpu"),
    top_k: int = 5,
) -> Dict[str, Any]:
    all_P, all_R, all_F1 = [], [], []

    model_preds = metadata.get("model_predictions", [])
    predictions = [item[0].split("@")[1] for item in model_preds]
    image_paths = metadata.get("image", [])
    num_samples = projections.shape[0]

    for idx in range(num_samples):
        if idx >= len(image_paths):
            continue
        if "No" in image_paths[idx][0].split("@")[0]:
            continue 

        projection_scores = projections[idx]
        prediction = [predictions[idx]] * top_k

        top_comp = projection_scores.argsort()[-top_k:]
        top_grounding_words = []
        for comp_idx in top_comp:
            comp_grounding_words = grounding_words[comp_idx]
            comp_grounding_words = [w.lower() for w in comp_grounding_words if len(w)>2]
            comp_grounding_words = list(set(comp_grounding_words))
            grounding_tokens = remove_substrings(comp_grounding_words)
            cand = " ".join(grounding_tokens)
            top_grounding_words.append(cand)

        P, R, F1 = bert_score(top_grounding_words, prediction, lang="en", verbose=True)
        all_P.append(P)
        all_R.append(R)
        all_F1.append(F1)

    all_P = np.array([p.numpy() for p in all_P])
    all_R = np.array([r.numpy() for r in all_R])
    all_F1 = np.array([f1.numpy() for f1 in all_F1])

    scores_dict = {}

    for k in [1, 3]:
        for metric_name, values in zip(["precision", "recall", "f1"], [all_P, all_R, all_F1]):
            key_all = f"top_{k}_{metric_name}_all"
            key_mean = f"top_{k}_{metric_name}_mean"
            key_std = f"top_{k}_{metric_name}_std"
            topk_vals = values[:, -k:].mean(axis=1)
            scores_dict[key_all] = topk_vals
            scores_dict[key_mean] = topk_vals.mean()
            scores_dict[key_std] = topk_vals.std()

    return scores_dict



def jaccard_index(set1: set, set2: set) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union) if union else 0.0

def remove_substrings(tokens):
    tokens = list(tokens)  # Convert to list to allow indexing
    result = set()

    for i, token in enumerate(tokens):
        if not any(token != other and token in other for other in tokens):
            result.add(token)

    return result

def compute_test_jaccard_score(
    projections: np.ndarray,
    grounding_words: List[List[str]],
    metadata: Dict[str, Any],
    top_k: int = 5,
) -> Dict[str, Any]:
    scores = []

    model_preds = metadata.get("model_predictions", [])
    predictions = [item[0].split("@")[1] for item in model_preds]
    image_paths = metadata.get("image", [])
    num_samples = projections.shape[0]

    for idx in range(num_samples):
        if idx >= len(image_paths):
            continue
        if "No" in image_paths[idx][0].split("@")[0]:
            continue 

        projection_scores = projections[idx]
        prediction = predictions[idx].lower()
        pred_tokens = set(prediction.split())

        top_comp = projection_scores.argsort()[-top_k:]
        top_jaccard_scores = []

        for comp_idx in top_comp:
            comp_grounding_words = grounding_words[comp_idx]
            grounding_tokens = set(w.lower() for w in comp_grounding_words if len(w) > 2)
            grounding_tokens = remove_substrings(grounding_tokens)
            j_score = jaccard_index(pred_tokens, grounding_tokens)
            top_jaccard_scores.append(j_score)

        scores.append(top_jaccard_scores)

    scores = np.array(scores)

    # Calculate mean/std for top 1 and top 3
    scores_dict = {}
    for k in [1, 3]:
        key = f"top_{k}_all"
        key_mean = f"top_{k}_mean"
        key_std = f"top_{k}_std"
        all_test_scores = scores[:, -k:].mean(axis=1)
        scores_dict[key] = all_test_scores
        scores_dict[key_mean] = all_test_scores.mean()
        scores_dict[key_std] = all_test_scores.std()
    #print(scores_dict)
    return scores_dict
