from __future__ import annotations

import hashlib
import json
import logging
import os
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

MOLFORMER_DEFAULT_MODEL = "ibm-research/MoLFormer-XL-both-10pct"


def resolve_molformer_model_path(model_name: str) -> str:
    """Resolve a local MoLFormer model path, tolerating a parent hf_models dir."""

    path = Path(str(model_name)).expanduser()
    if not path.exists() or not path.is_dir():
        return model_name
    if (path / "config.json").exists():
        return str(path)

    children = [child for child in path.iterdir() if child.is_dir() and (child / "config.json").exists()]
    if len(children) == 1:
        return str(children[0])
    return str(path)


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for MoLFormer embedding: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _molformer_cache_path(
    cache_dir: PathLike | None,
    canonical_smiles: str,
    model_name: str,
    max_length: int,
    pooling: str,
) -> Path | None:
    if cache_dir is None:
        return None

    payload = {
        "version": 1,
        "canonical_smiles": canonical_smiles,
        "model_name": model_name,
        "max_length": int(max_length),
        "pooling": pooling,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir).expanduser() / "molformer" / f"{digest}.pt"


def _load_cached_embedding(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        import torch

        embedding = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        embedding = np.asarray(embedding, dtype=np.float32)
        return embedding if embedding.ndim == 1 else None
    except Exception as exc:
        logger.warning("Failed to load MoLFormer embedding cache %s: %s. Recomputing.", path, exc)
        return None


def _save_cached_embedding(path: Path | None, embedding: np.ndarray) -> None:
    if path is None:
        return
    try:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(torch.as_tensor(embedding, dtype=torch.float32).cpu(), tmp_path)
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to save MoLFormer embedding cache %s: %s", path, exc)


def _load_molformer(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "MoLFormer 1D embeddings require the optional `transformers` dependency. "
            "Install it with `pip install -e .[molformer]` or `pip install transformers`."
        ) from exc

    model_name = resolve_molformer_model_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        deterministic_eval=True,
        trust_remote_code=True,
    )
    model.to(torch.device(device))
    model.eval()
    return tokenizer, model


def _load_molformer_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "MoLFormer token features require the optional `transformers` dependency. "
            "Install it with `pip install -e .[molformer]` or `pip install transformers`."
        ) from exc

    return AutoTokenizer.from_pretrained(resolve_molformer_model_path(model_name), trust_remote_code=True)


def _pool_molformer_output(outputs, attention_mask, pooling: str):
    if pooling == "pooler":
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        pooling = "cls"

    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return hidden[:, 0]
    if pooling == "mean":
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unknown MoLFormer pooling mode: {pooling}")


def molformer_1d_embeddings(
    smiles: Sequence[str],
    model_name: str = MOLFORMER_DEFAULT_MODEL,
    cache_dir: PathLike | None = None,
    max_length: int = 256,
    pooling: str = "pooler",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Return frozen MoLFormer embeddings for SMILES strings.

    Embeddings are cached per canonical SMILES. The cache stores only the final pooled vector.
    """

    if pooling not in {"pooler", "cls", "mean"}:
        raise ValueError("MoLFormer pooling must be one of: pooler, cls, mean.")
    if max_length < 1:
        raise ValueError("MoLFormer max_length must be at least 1.")
    if batch_size < 1:
        raise ValueError("MoLFormer batch_size must be at least 1.")

    model_name = resolve_molformer_model_path(model_name)
    canonical = [_canonical_smiles(smi) for smi in smiles]
    cache_paths = [
        _molformer_cache_path(cache_dir, smi, model_name, max_length, pooling) for smi in canonical
    ]

    embeddings: list[np.ndarray | None] = [_load_cached_embedding(path) for path in cache_paths]
    missing_indices = [idx for idx, embedding in enumerate(embeddings) if embedding is None]
    if missing_indices:
        import torch

        tokenizer, model = _load_molformer(model_name, device)
        model_device = next(model.parameters()).device
        for start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[start : start + batch_size]
            batch_smiles = [canonical[idx] for idx in batch_indices]
            inputs = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                pooled = _pool_molformer_output(outputs, inputs["attention_mask"], pooling)
            batch_embeddings = pooled.detach().cpu().numpy().astype(np.float32)
            for idx, embedding in zip(batch_indices, batch_embeddings):
                embeddings[idx] = embedding
                _save_cached_embedding(cache_paths[idx], embedding)

    return np.vstack([embedding for embedding in embeddings if embedding is not None]).astype(
        np.float32
    )


def molformer_token_features(
    smiles: Sequence[str],
    model_name: str = MOLFORMER_DEFAULT_MODEL,
    max_length: int = 256,
) -> np.ndarray:
    """Return token ids and attention masks for trainable MoLFormer fine-tuning.

    The returned array has shape ``[n_molecules, 2 * max_length]`` and stores
    ``[input_ids ; attention_mask]``. It intentionally stays in ``X_d`` so the
    existing Chemprop batching path can carry it into the model, where the
    trainable MoLFormer encoder reconstructs integer tensors.
    """

    if max_length < 1:
        raise ValueError("MoLFormer max_length must be at least 1.")

    canonical = [_canonical_smiles(smi) for smi in smiles]
    model_name = resolve_molformer_model_path(model_name)
    tokenizer = _load_molformer_tokenizer(model_name)
    encoded = tokenizer(
        canonical,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.float32)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.float32)
    return np.hstack([input_ids, attention_mask]).astype(np.float32)
