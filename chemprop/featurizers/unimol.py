from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

UNIMOL_DEFAULT_MODEL = "unimolv1"


def _canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES for Uni-Mol embedding: {smiles!r}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _cache_path(
    cache_dir: PathLike | None,
    canonical_smiles: str,
    model_name: str,
    model_path: str | None,
    remove_hs: bool,
) -> Path | None:
    if cache_dir is None:
        return None

    resolved_model_path = None
    if model_path is not None:
        resolved_model_path = str(Path(model_path).expanduser().resolve())

    payload = {
        "version": 1,
        "canonical_smiles": canonical_smiles,
        "model_name": model_name,
        "model_path": resolved_model_path,
        "remove_hs": bool(remove_hs),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return Path(cache_dir).expanduser() / "unimol" / f"{digest}.npy"


def _load_cached_embedding(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        embedding = np.load(path)
        embedding = np.asarray(embedding, dtype=np.float32)
        return embedding if embedding.ndim == 1 else None
    except Exception as exc:
        logger.warning("Failed to load Uni-Mol embedding cache %s: %s. Recomputing.", path, exc)
        return None


def _save_cached_embedding(path: Path | None, embedding: np.ndarray) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        with open(tmp_path, "wb") as fh:
            np.save(fh, np.asarray(embedding, dtype=np.float32))
        tmp_path.replace(path)
    except Exception as exc:
        logger.warning("Failed to save Uni-Mol embedding cache %s: %s", path, exc)


def _accepts_kwargs(callable_obj) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    if _accepts_kwargs(callable_obj):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _build_unimol_repr(
    model_name: str,
    model_path: str | None,
    device: str,
    batch_size: int,
    remove_hs: bool,
):
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError(
            "Uni-Mol 3D embeddings require the optional `unimol_tools` dependency. "
            "Install it with `pip install unimol_tools` or `pip install -e .[unimol]`."
        ) from exc

    kwargs = {
        "data_type": "molecule",
        "remove_hs": remove_hs,
        "model_name": model_name,
        "model_path": model_path,
        "device": device,
        "batch_size": batch_size,
        "use_gpu": str(device).lower() not in {"cpu", "none", "-1"},
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    kwargs = _filter_supported_kwargs(UniMolRepr, kwargs)
    try:
        return UniMolRepr(**kwargs)
    except TypeError:
        # UniMolRepr has changed argument names across releases. Fall back to the
        # stable core constructor and keep the optional settings in get_repr where
        # supported.
        core_kwargs = _filter_supported_kwargs(
            UniMolRepr, {"data_type": "molecule", "remove_hs": remove_hs}
        )
        return UniMolRepr(**core_kwargs)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _coerce_repr_array(value) -> np.ndarray:
    """Coerce Uni-Mol representation outputs across unimol_tools versions."""

    if isinstance(value, dict):
        for key in ("cls_repr", "cls_reprs", "molecular_repr", "mol_repr", "repr"):
            if key in value:
                return _coerce_repr_array(value[key])
        raise KeyError(
            "Uni-Mol representation output did not contain `cls_repr` or a known molecule-level key."
        )

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return np.empty((0, 0), dtype=np.float32)
        arrays = [_coerce_repr_array(item) for item in value]
        if len(arrays) == 1:
            return arrays[0]
        if all(array.ndim == 1 for array in arrays):
            return np.stack(arrays, axis=0).astype(np.float32)
        if all(array.ndim == 2 and array.shape[0] == 1 for array in arrays):
            return np.concatenate(arrays, axis=0).astype(np.float32)
        try:
            return np.concatenate(arrays, axis=0).astype(np.float32)
        except ValueError:
            return np.stack(arrays, axis=0).astype(np.float32)

    array = _to_numpy(value)
    if array.dtype == object:
        return _coerce_repr_array(array.tolist())
    return array.astype(np.float32)


def _extract_cls_repr(repr_output) -> np.ndarray:
    if isinstance(repr_output, dict):
        for key in ("cls_repr", "cls_reprs", "molecular_repr", "mol_repr", "repr"):
            if key in repr_output:
                return _coerce_repr_array(repr_output[key])
        raise KeyError(
            "Uni-Mol representation output did not contain `cls_repr` or a known molecule-level key."
        )

    if isinstance(repr_output, (tuple, list)) and repr_output:
        return _coerce_repr_array(repr_output[0])

    return _coerce_repr_array(repr_output)


def _normalize_embeddings(embeddings: np.ndarray, expected_n: int) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    elif embeddings.ndim == 3:
        if embeddings.shape[0] == 1 and embeddings.shape[1] == expected_n:
            embeddings = embeddings[0]
        elif embeddings.shape[0] == expected_n and embeddings.shape[1] == 1:
            embeddings = embeddings[:, 0, :]
        else:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    if embeddings.ndim != 2:
        raise ValueError(f"Expected Uni-Mol cls_repr to be 2D, got shape {embeddings.shape}.")
    return np.nan_to_num(embeddings.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _get_repr(model, smiles: Sequence[str], batch_size: int) -> np.ndarray:
    kwargs = {"return_atomic_reprs": False, "batch_size": batch_size}
    kwargs = _filter_supported_kwargs(model.get_repr, kwargs)
    try:
        output = model.get_repr(list(smiles), **kwargs)
    except TypeError:
        try:
            output = model.get_repr(data=list(smiles), **kwargs)
        except TypeError:
            output = model.get_repr(list(smiles), return_atomic_reprs=False)

    embeddings = _normalize_embeddings(_extract_cls_repr(output), expected_n=len(smiles))
    if embeddings.shape[0] != len(smiles) and len(smiles) > 1:
        logger.warning(
            "Uni-Mol returned %s embeddings for a batch of %s molecules; falling back to "
            "one-molecule calls for this batch.",
            embeddings.shape[0],
            len(smiles),
        )
        return np.vstack([_get_repr(model, [smi], batch_size=1) for smi in smiles]).astype(
            np.float32
        )
    return embeddings


def unimol_3d_embeddings(
    smiles: Sequence[str],
    model_name: str = UNIMOL_DEFAULT_MODEL,
    model_path: str | None = None,
    cache_dir: PathLike | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    remove_hs: bool = False,
) -> np.ndarray:
    """Return frozen molecule-level Uni-Mol CLS embeddings for SMILES strings.

    The embeddings are intended to be used as descriptor-style 3D features, i.e.
    they are prepended to ``X_d`` and consumed through ``--x-d-3d-dim``.
    """

    if batch_size < 1:
        raise ValueError("Uni-Mol batch_size must be at least 1.")

    canonical = [_canonical_smiles(smi) for smi in smiles]
    cache_paths = [
        _cache_path(cache_dir, smi, model_name, model_path, remove_hs) for smi in canonical
    ]
    embeddings: list[np.ndarray | None] = [_load_cached_embedding(path) for path in cache_paths]
    missing_indices = [idx for idx, embedding in enumerate(embeddings) if embedding is None]

    if missing_indices:
        model = _build_unimol_repr(
            model_name=model_name,
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            remove_hs=remove_hs,
        )
        for start in range(0, len(missing_indices), batch_size):
            batch_indices = missing_indices[start : start + batch_size]
            batch_smiles = [canonical[idx] for idx in batch_indices]
            batch_embeddings = _get_repr(model, batch_smiles, batch_size=batch_size)
            if batch_embeddings.shape[0] != len(batch_indices):
                raise ValueError(
                    "Uni-Mol returned a different number of embeddings than requested: "
                    f"{batch_embeddings.shape[0]} vs {len(batch_indices)}."
                )
            for idx, embedding in zip(batch_indices, batch_embeddings):
                embeddings[idx] = embedding.astype(np.float32)
                _save_cached_embedding(cache_paths[idx], embedding)

    return np.vstack([embedding for embedding in embeddings if embedding is not None]).astype(
        np.float32
    )
