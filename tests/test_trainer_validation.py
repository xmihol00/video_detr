from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cnnSearch.search_space import ArchitectureConfig
from cnnSearch.trainer import recalibrateBatchNormStatistics


class _ToyValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv = nn.Conv2d(3, 2, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs: torch.Tensor, architectureConfig: ArchitectureConfig | None = None):
        outputs = self.bn(inputs)
        outputs = self.conv(outputs)
        outputs = self.pool(outputs)
        logits = outputs.flatten(1)
        return logits, architectureConfig, []


def testRecalibrateBatchNormStatisticsUpdatesRunningStats() -> None:
    model = _ToyValidationModel()
    device = torch.device("cpu")

    images = torch.full((6, 3, 16, 16), fill_value=2.5)
    labels = torch.zeros(6, dtype=torch.long)
    calibrationLoader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    architecture = ArchitectureConfig(
        inputResolution=16,
        outputStride=16,
        stageDepths=[1, 1, 1, 1],
        stageWidthMultipliers=[1.0, 1.0, 1.0, 1.0],
        stemChannels=32,
        enableAuxiliaryHeads=False,
    )

    assert model.bn.running_mean is not None
    assert model.bn.running_var is not None
    initialMean = model.bn.running_mean.detach().clone()
    initialVar = model.bn.running_var.detach().clone()

    steps = recalibrateBatchNormStatistics(
        model=model,
        calibrationLoader=calibrationLoader,
        device=device,
        architectureConfig=architecture,
        ampEnabled=False,
        maxCalibrationSteps=2,
    )

    assert steps == 2
    assert model.bn.running_mean is not None
    assert model.bn.running_var is not None
    assert not torch.allclose(model.bn.running_mean, initialMean)
    assert not torch.allclose(model.bn.running_var, initialVar)
