import argparse
import pickle
from typing import Any, Callable, Tuple, List, Dict

from torch.utils.data import DataLoader, Subset

from datasets.image_text_dataset import COCODataset, VQAv2Dataset, XRAYdataset, XRAYdataset_view, TextDataset,ImageDataset, JSONDataset

__all__ = ["get_dataset_loader"]


def safe_collate_dicts(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of dicts into a dict of lists, sanitizing None values.

    - For strings: replace None with ""
    - For lists: replace None with []
    - For numbers/tensors: replace None with 0 / empty tensor is not handled here (not needed now)
    - For other types: replace None with ""
    """
    if not batch:
        return {}
    out: Dict[str, Any] = {}
    keys = set()
    for d in batch:
        keys.update(d.keys())
    for k in keys:
        vals = [d.get(k, None) for d in batch]
        # Infer a simple target type
        sample = next((v for v in vals if v is not None), None)
        if isinstance(sample, str):
            out[k] = [v if v is not None else "" for v in vals]
        elif isinstance(sample, list):
            out[k] = [v if v is not None else [] for v in vals]
        else:
            out[k] = [v if v is not None else "" for v in vals]
    return out


def get_dataset_loader(
    dataset_name: str,
    args: argparse.Namespace = None,
    logger: Callable = None,
    **kwargs: Any,
) -> Tuple[Callable, list]:
    """
    General function to return a DataLoader for a given dataset.

    This function also has the functionality of extracting of a subset of the dataset based on

    Args:
        dataset_name (str): The name of the dataset.
        args (argparse.Namespace): Arguments including paths, flags, and dataset options.

    Returns:
        loader (DataLoader): The DataLoader object for the (subset of) dataset.
        indices (list): The indices of the subset (if applicable) or an empty list.
    """

    batch_size = getattr(args, "batch_size", 1)

    if dataset_name == "coco":
        dataset_cls = COCODataset
    elif dataset_name == "vqav2":
        dataset_cls = VQAv2Dataset
    elif dataset_name == "text":
        dataset_cls = TextDataset
    elif dataset_name == "image": # load all trhe image from a dir
        dataset_cls = ImageDataset
    elif dataset_name == "xray":
        dataset_cls = XRAYdataset
    elif dataset_name == "xray_view":
        dataset_cls = XRAYdataset_view
    elif dataset_name == "json_crop_map":
        dataset_cls = JSONDataset
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented.")

    dataset = dataset_cls(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        questions_file=args.questions_file,
        split=args.split,
        dataset_size=args.dataset_size,
        seed=args.seed,
        dataset_name=args.dataset_name,
        mode=("val" if args.generation_mode else "train"),
        prompt_template=args.prompt_template,
        token_of_interest_num_samples=args.token_of_interest_num_samples,
    )

    loader = None
    token_of_interest_indices = []

    if logger is not None:
        logger.info(f"Successfully loaded {dataset_name}")

    if  dataset_name == "json_crop_map":
        # Build per-concept datasets and dataloaders using JSONDataset helpers
        assert isinstance(dataset, JSONDataset)
        per_ds = dataset.per_concept_datasets()
        dataloaders = {}
        for key, ds in per_ds.items():
            if logger:
                logger.info(f"Concept: {key}, samples: {len(ds)}")
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=safe_collate_dicts,
            )
            dataloaders[key] = loader
        # Return only the dict of per-concept loaders as requested
        return dataloaders

    elif args.select_token_of_interest_samples:
        token_of_interest_indices = dataset.token_of_interest_idx_extractor(
            token_of_interest=args.token_of_interest,
            token_of_interest_key=args.token_of_interest_key,
            allow_different_variations=args.allow_different_variations_of_token_of_interest,
            token_of_interest_class=args.token_of_interest_class,
            logger=logger,
        )
        dataset_subset = Subset(dataset, token_of_interest_indices)

        loader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    elif args.select_samples_from_ids:

        assert (
            args.path_to_samples_ids is not None
        ), 'A path to ids should be given when passing the flag "select_samples_from_ids"'
        with open(args.path_to_samples_ids, "rb") as handle:
            img_ids = pickle.load(handle)

        filtered_indices = dataset.indices_from_ids_extractor(img_ids=img_ids)
        dataset_subset = Subset(dataset, filtered_indices)

        loader = DataLoader(
            dataset_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    if logger is not None:
        logger.info(f"Reading dataset: {dataset_name} of size: {len(loader)}")
    assert len(loader) > 0, f"loader is empty. Check if the --token_of_interest is properly assigned"

    return loader
