from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_FEATURE_COUNT = 13
DEFAULT_HIDDEN_SIZE = 32
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_OUTPUT_SIZE = 13


class LSTMNextWindowPredictor(nn.Module):
    """
    LSTM-предиктор следующего агрегированного окна интерфейса.
    """

    def __init__(
        self,
        input_size: int = DEFAULT_FEATURE_COUNT,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        output_size: int = DEFAULT_OUTPUT_SIZE,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Ожидается вход формы [batch, seq_len, input_size], получено {tuple(x.shape)}"
            )

        if x.shape[-1] != self.input_size:
            raise ValueError(
                "Число признаков во входе не совпадает с input_size модели: "
                f"{x.shape[-1]} != {self.input_size}"
            )

        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        y_pred = self.head(last_hidden)
        return y_pred

    def get_model_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "output_size": self.output_size,
        }


def build_lstm_next_window_predictor(
    input_size: int = DEFAULT_FEATURE_COUNT,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dropout: float = DEFAULT_DROPOUT,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> LSTMNextWindowPredictor:
    return LSTMNextWindowPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size,
    )