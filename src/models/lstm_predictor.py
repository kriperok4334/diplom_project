from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_SEQUENCE_LENGTH = 5
DEFAULT_FEATURE_COUNT = 13
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_LAYERS = 1
DEFAULT_DROPOUT = 0.0
DEFAULT_OUTPUT_SIZE = 13


class LSTMNextWindowPredictor(nn.Module):
    """
    LSTM-предиктор следующего агрегированного окна интерфейса.

    Вход:
        x.shape = [batch_size, sequence_length, feature_count]

    Где:
        sequence_length = 5
        feature_count = 13

    Выход:
        y_pred.shape = [batch_size, output_size]

    Где:
        output_size = 13

    Модель предсказывает вектор признаков следующего окна
    на основе истории последних окон.
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

        # В PyTorch dropout внутри LSTM применяется только если num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.

        Параметры:
            x: torch.Tensor формы [batch_size, sequence_length, feature_count]

        Возвращает:
            torch.Tensor формы [batch_size, output_size]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Ожидается входной тензор размерности 3, получено x.ndim={x.ndim}"
            )

        batch_size, sequence_length, feature_count = x.shape

        if feature_count != self.input_size:
            raise ValueError(
                "Число признаков во входе не совпадает с input_size модели: "
                f"{feature_count} != {self.input_size}"
            )

        # lstm_output.shape = [batch_size, sequence_length, hidden_size]
        lstm_output, (hidden_n, cell_n) = self.lstm(x)

        # Берём выход LSTM на последнем временном шаге.
        # Это и есть компактное представление всей последовательности.
        last_output = lstm_output[:, -1, :]

        y_pred = self.head(last_output)
        return y_pred

    def get_model_config(self) -> dict[str, int | float]:
        """
        Возвращает конфигурацию модели в словарном виде.
        Удобно для логирования и сохранения метаданных.
        """
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
    """
    Удобная фабрика для создания модели с заданными параметрами.
    """
    return LSTMNextWindowPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=output_size,
    )
