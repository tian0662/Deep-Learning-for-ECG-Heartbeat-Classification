"""CNN architectures for ECG heartbeat classification.

This module mirrors the structure of ``utils.rnn_models`` by providing a
collection of ready-to-use 1D convolutional neural network (CNN) models.
The goal is to offer a simple template that can be imported inside notebooks
or scripts in the same fashion as the existing RNN utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class ConvBlockConfig:
    """Configuration for a single convolutional block.

    Attributes
    ----------
    out_channels:
        Number of channels produced by the convolution layer.
    kernel_size:
        Size of the temporal kernel. Padding is automatically computed as
        ``kernel_size // 2`` to preserve sequence length.
    stride:
        Stride applied in the convolution layer.
    dilation:
        Dilation factor used in the convolution.
    pool_kernel_size:
        Size of the max pooling kernel that follows the convolution.
    dropout:
        Dropout probability applied after the activation function.
    """

    out_channels: int
    kernel_size: int = 5
    stride: int = 1
    dilation: int = 1
    pool_kernel_size: int = 2
    dropout: float = 0.1


class ConvBlock(nn.Module):
    """Reusable 1D convolutional block used by the CNN classifier."""

    def __init__(self, in_channels: int, config: ConvBlockConfig) -> None:
        super().__init__()
        padding = ((config.kernel_size - 1) // 2) * config.dilation
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=config.out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=padding,
                dilation=config.dilation,
                bias=False,
            ),
            nn.BatchNorm1d(config.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.MaxPool1d(kernel_size=config.pool_kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Run the convolutional block on the input tensor."""

        return self.layers(x)


class ECG_CNN_Classifier(nn.Module):
    """Baseline 1D CNN classifier for ECG sequences.

    The architecture mirrors the ergonomic API offered by the RNN models:

    * The constructor exposes high-level knobs (number of channels, dropout,
      fully-connected size, etc.).
    * Inputs are expected as ``(batch, seq_len)`` tensors to stay compatible
      with the ``ECG_Dataset`` class. They are automatically reshaped to the
      ``(batch, channels, seq_len)`` format required by convolution layers.
    * The module returns logits that can be passed directly to
      ``nn.CrossEntropyLoss``.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        conv_channels: Sequence[int] | None = None,
        kernel_sizes: Sequence[int] | None = None,
        pool_kernel_sizes: Sequence[int] | None = None,
        dropout: float = 0.2,
        fc_hidden_dim: int = 128,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = (32, 64, 128)
        if kernel_sizes is None:
            kernel_sizes = (7, 5, 3)
        if pool_kernel_sizes is None:
            pool_kernel_sizes = (2,) * len(conv_channels)

        if not (
            len(conv_channels)
            == len(kernel_sizes)
            == len(pool_kernel_sizes)
        ):
            raise ValueError(
                "`conv_channels`, `kernel_sizes`, and `pool_kernel_sizes` "
                "must have the same length."
            )

        conv_blocks = []
        current_in_channels = in_channels
        for out_channels, kernel_size, pool_kernel in zip(
            conv_channels, kernel_sizes, pool_kernel_sizes
        ):
            block_config = ConvBlockConfig(
                out_channels=out_channels,
                kernel_size=kernel_size,
                pool_kernel_size=pool_kernel,
                dropout=dropout,
            )
            conv_blocks.append(ConvBlock(current_in_channels, block_config))
            current_in_channels = out_channels

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        classifier: list[nn.Module] = [nn.Flatten()]
        if fc_hidden_dim is not None:
            classifier.append(nn.Linear(current_in_channels, fc_hidden_dim))
            if use_batch_norm:
                classifier.append(nn.BatchNorm1d(fc_hidden_dim))
            classifier.extend(
                [
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(fc_hidden_dim, num_classes),
                ]
            )
        else:
            classifier.append(nn.Linear(current_in_channels, num_classes))

        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Compute class logits for the provided ECG batch."""

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            # [batch, seq_len] -> [batch, channels=1, seq_len]
            x = x.unsqueeze(1)

        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits


__all__ = ["ConvBlockConfig", "ConvBlock", "ECG_CNN_Classifier"]
