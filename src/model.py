"""
model.py
========
The LSTM neural network that classifies a sequence of MFCC frames into a spoken
digit (0-9).

PIPELINE STAGE
--------------
MFCC sequence (features.py)  ->  [THIS FILE: LSTM]  ->  logits -> confidence (predict.py)

WHY AN LSTM?
------------
Speech is a *time sequence*: the digit "seven" is a pattern that unfolds over
time. MFCC features are therefore a sequence of frames. An LSTM (Long Short-Term
Memory network) is a recurrent network designed to read sequences and remember
context across time steps, which makes it a natural fit. (A CNN on mel
spectrograms is our planned backup/comparison model.)

INPUT / OUTPUT SHAPES
---------------------
Input  : x of shape (batch, time_steps, n_mfcc)   -- batch_first=True
Output : logits of shape (batch, NUM_CLASSES)
         (raw scores; apply softmax to get probabilities/confidence)

The model is intentionally small (2 LSTM layers, hidden size 128) so it trains
on a normal laptop CPU.
"""

import torch
import torch.nn as nn

from config import (
    DROPOUT,
    LSTM_BIDIRECTIONAL,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    N_MFCC,
    NUM_CLASSES,
)


class DigitLSTM(nn.Module):
    """LSTM classifier for spoken digits.

    Architecture:
        MFCC sequence
          -> LSTM (stacked, optionally bidirectional, with dropout between layers)
          -> take the final time step's hidden state (a summary of the whole clip)
          -> dropout
          -> fully connected layer -> one logit per class
    """

    def __init__(
        self,
        input_size: int = N_MFCC,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        bidirectional: bool = LSTM_BIDIRECTIONAL,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,                       # input is (batch, time, features)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # If bidirectional, the final feature vector is twice as wide.
        fc_in = hidden_size * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fc_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, time_steps, n_mfcc) -> logits: (batch, num_classes)."""
        # outputs: (batch, time_steps, hidden * num_directions)
        outputs, _ = self.lstm(x)

        # Use the last time step as a summary of the whole sequence.
        last_step = outputs[:, -1, :]               # (batch, hidden * num_directions)

        dropped = self.dropout(last_step)
        logits = self.fc(dropped)                    # (batch, num_classes)
        return logits


class DigitCNN(nn.Module):
    """CNN classifier for spoken digits (backup / comparison model).

    Treats the mel spectrogram like a small grayscale image: frequency on one
    axis, time on the other. Convolution layers learn local time-frequency
    patterns (how energy in certain bands rises/falls), which is a different and
    complementary view to the LSTM's sequential reading of MFCCs.

    Input : x of shape (batch, n_mels, time_steps)   -- the channel dim is added here
    Output: logits of shape (batch, num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),      # halves H and W
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                        # fixed size regardless of input
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mels, time) -> add channel -> (batch, 1, n_mels, time)
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def build_model(model_type: str = "lstm") -> nn.Module:
    """Factory used by train/evaluate/predict so they all build the same model.

    model_type="lstm" -> DigitLSTM (MFCC)   |   "cnn" -> DigitCNN (mel spectrogram)
    """
    if model_type == "lstm":
        return DigitLSTM()
    if model_type == "cnn":
        return DigitCNN()
    raise ValueError(f"Unknown model_type: {model_type!r} (expected 'lstm' or 'cnn')")


if __name__ == "__main__":
    # Self-test: push a fake batch through to confirm the shapes line up.
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Trainable parameters: {n_params:,}")

    fake_batch = torch.randn(4, 50, N_MFCC)  # (batch=4, time=50, n_mfcc)
    out = model(fake_batch)
    print(f"Input  shape: {tuple(fake_batch.shape)}  (batch, time, n_mfcc)")
    print(f"Output shape: {tuple(out.shape)}  (batch, num_classes={NUM_CLASSES})")
