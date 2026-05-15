from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CACHE_DIR = Path("training/cache")
DATASET_PIPELINE_VERSION = "v1"


def _normalize_feature_columns(feature_columns: list[str]) -> list[str]:
    """
    Нормализует список признаков для стабильного построения cache key.
    """
    return [str(col).strip() for col in feature_columns]


def build_cache_signature(
    max_files: int | None,
    window_size_minutes: int,
    history_length: int,
    feature_columns: list[str],
    pipeline_version: str = DATASET_PIPELINE_VERSION,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Формирует словарь параметров, определяющих подготовленный датасет.
    """
    signature = {
        "pipeline_version": pipeline_version,
        "max_files": max_files,
        "window_size_minutes": window_size_minutes,
        "history_length": history_length,
        "feature_columns": _normalize_feature_columns(feature_columns),
    }

    if extra_params:
        signature["extra_params"] = extra_params

    return signature


def build_cache_key(
    max_files: int | None,
    window_size_minutes: int,
    history_length: int,
    feature_columns: list[str],
    pipeline_version: str = DATASET_PIPELINE_VERSION,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """
    Строит короткий cache key на основе параметров датасета.
    """
    signature = build_cache_signature(
        max_files=max_files,
        window_size_minutes=window_size_minutes,
        history_length=history_length,
        feature_columns=feature_columns,
        pipeline_version=pipeline_version,
        extra_params=extra_params,
    )

    serialized = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()[:12]

    feature_count = len(feature_columns)
    max_files_label = "all" if max_files is None else str(max_files)

    return (
        f"cesnet_mf{max_files_label}_"
        f"ws{window_size_minutes}_"
        f"hl{history_length}_"
        f"feat{feature_count}_"
        f"{digest}"
    )


def ensure_cache_dir(cache_dir: str | Path = CACHE_DIR) -> Path:
    """
    Создаёт директорию кэша, если она отсутствует.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_paths(
    cache_key: str,
    cache_dir: str | Path = CACHE_DIR,
) -> dict[str, Path]:
    """
    Возвращает пути ко всем артефактам кэша.
    """
    cache_dir = ensure_cache_dir(cache_dir)

    return {
        "signature": cache_dir / f"{cache_key}__signature.json",
        "windows_df": cache_dir / f"{cache_key}__windows_df.parquet",
        "X": cache_dir / f"{cache_key}__X.npy",
        "y": cache_dir / f"{cache_key}__y.npy",
        "metadata_df": cache_dir / f"{cache_key}__metadata_df.parquet",
    }


def cache_exists(
    cache_key: str,
    cache_dir: str | Path = CACHE_DIR,
) -> bool:
    """
    Проверяет, существует ли полный набор кэшированных файлов.
    """
    paths = get_cache_paths(cache_key, cache_dir=cache_dir)

    return all(path.exists() for path in paths.values())


def save_prepared_dataset(
    cache_key: str,
    windows_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    signature: dict[str, Any],
    cache_dir: str | Path = CACHE_DIR,
) -> dict[str, Path]:
    """
    Сохраняет подготовленный датасет в кэш.
    """
    paths = get_cache_paths(cache_key, cache_dir=cache_dir)

    with open(paths["signature"], "w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2)

    windows_df.to_parquet(paths["windows_df"], index=False)
    np.save(paths["X"], X)
    np.save(paths["y"], y)
    metadata_df.to_parquet(paths["metadata_df"], index=False)

    return paths


def load_prepared_dataset(
    cache_key: str,
    cache_dir: str | Path = CACHE_DIR,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """
    Загружает подготовленный датасет из кэша.

    Возвращает:
    - windows_df
    - X
    - y
    - metadata_df
    - signature
    """
    paths = get_cache_paths(cache_key, cache_dir=cache_dir)

    if not cache_exists(cache_key, cache_dir=cache_dir):
        raise FileNotFoundError(
            f"Кэш для ключа '{cache_key}' не найден или неполон."
        )

    with open(paths["signature"], "r", encoding="utf-8") as f:
        signature = json.load(f)

    windows_df = pd.read_parquet(paths["windows_df"])
    X = np.load(paths["X"])
    y = np.load(paths["y"])
    metadata_df = pd.read_parquet(paths["metadata_df"])

    return windows_df, X, y, metadata_df, signature


def print_cache_summary(
    cache_key: str,
    signature: dict[str, Any],
) -> None:
    """
    Печатает краткую сводку по кэшу.
    """
    print("\n=== DATASET CACHE ===")
    print(f"Cache key: {cache_key}")
    print(f"Pipeline version: {signature.get('pipeline_version')}")
    print(f"Max files: {signature.get('max_files')}")
    print(f"Window size minutes: {signature.get('window_size_minutes')}")
    print(f"History length: {signature.get('history_length')}")
    print(f"Feature columns: {signature.get('feature_columns')}")