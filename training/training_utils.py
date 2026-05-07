from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from src.models.lstm_predictor import (
    LSTMNextWindowPredictor,
    build_lstm_next_window_predictor,
)
from src.models.model_io import save_predictor_model


DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_RANDOM_SEED = 42


class WindowSequenceDataset(Dataset):
    """
    Dataset для LSTM-предиктора.

    X:
        np.ndarray формы [num_samples, sequence_length, feature_count]

    y:
        np.ndarray формы [num_samples, output_size]
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError(
                f"Размеры X и y не совпадают: len(X)={len(X)}, len(y)={len(y)}"
            )

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class TrainConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    num_epochs: int = DEFAULT_NUM_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    val_ratio: float = DEFAULT_VAL_RATIO
    test_ratio: float = DEFAULT_TEST_RATIO
    random_seed: int = DEFAULT_RANDOM_SEED
    device: str = "cpu"


def split_lstm_dataset(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """
    Делит датасет на train / val / test.
    """
    if len(X) == 0:
        raise ValueError("Пустой датасет: невозможно выполнить split.")

    if len(X) != len(metadata_df):
        raise ValueError(
            "Число sample в X не совпадает с числом строк metadata_df: "
            f"{len(X)} != {len(metadata_df)}"
        )

    full_dataset = WindowSequenceDataset(X, y)

    total_size = len(full_dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        raise ValueError(
            "После разбиения не осталось train-части. "
            "Уменьши val_ratio/test_ratio или увеличь объём данных."
        )

    generator = torch.Generator().manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_metadata_df": metadata_df.iloc[train_indices].reset_index(drop=True),
        "val_metadata_df": metadata_df.iloc[val_indices].reset_index(drop=True),
        "test_metadata_df": metadata_df.iloc[test_indices].reset_index(drop=True),
    }


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, DataLoader]:
    """
    Создаёт DataLoader для train / val / test.
    """
    return {
        "train_loader": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test_loader": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """
    Один проход обучения.
    """
    model.train()

    total_loss = 0.0
    batch_count = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        return 0.0

    return total_loss / batch_count


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """
    Один проход валидации/оценки.
    """
    model.eval()

    total_loss = 0.0
    batch_count = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        return 0.0

    return total_loss / batch_count


def train_lstm_predictor(
    X: np.ndarray,
    y: np.ndarray,
    metadata_df: pd.DataFrame,
    config: TrainConfig | None = None,
    model_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Главная функция обучения LSTM-предиктора.
    """
    if config is None:
        config = TrainConfig()

    split_data = split_lstm_dataset(
        X=X,
        y=y,
        metadata_df=metadata_df,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )

    dataloaders = create_dataloaders(
        train_dataset=split_data["train_dataset"],
        val_dataset=split_data["val_dataset"],
        test_dataset=split_data["test_dataset"],
        batch_size=config.batch_size,
    )

    model = build_lstm_next_window_predictor()
    model.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": None,
    }

    best_state_dict = None
    best_val_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train_loader"],
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
        )

        val_loss = validate_one_epoch(
            model=model,
            dataloader=dataloaders["val_loader"],
            criterion=criterion,
            device=config.device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_val_loss"] = best_val_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{config.num_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss = validate_one_epoch(
        model=model,
        dataloader=dataloaders["test_loader"],
        criterion=criterion,
        device=config.device,
    )

    history["test_loss"] = test_loss

    print(f"Final test_loss={test_loss:.6f}")

    if model_output_path is not None:
        save_predictor_model(
            model=model,
            output_path=model_output_path,
            extra_metadata={
                "history": history,
                "train_config": config.__dict__,
            },
        )

    return {
        "model": model,
        "history": history,
        "split_data": split_data,
        "dataloaders": dataloaders,
    }