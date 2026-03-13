from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import cast

from cnnSearch.search_space import ArchitectureConfig, SearchSpaceConfig, DEFAULT_SEARCH_SPACE, decodeStageChannels


class SlimConv2d(nn.Module):
    def __init__(self, maxInChannels: int, maxOutChannels: int, kernelSize: int, stride: int = 1, padding: int = 0) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxOutChannels = maxOutChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(maxOutChannels, maxInChannels, kernelSize, kernelSize))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, inputs: Tensor, outChannels: int, strideOverride: Optional[int] = None) -> Tensor:
        inChannels = inputs.shape[1]
        weight = self.weight[:outChannels, :inChannels, :, :]
        stride = self.stride if strideOverride is None else strideOverride
        return F.conv2d(inputs, weight, bias=None, stride=stride, padding=self.padding)


class SlimBatchNorm2d(nn.Module):
    def __init__(self, maxChannels: int) -> None:
        super().__init__()
        self.maxChannels = maxChannels
        self.weight = nn.Parameter(torch.ones(maxChannels))
        self.bias = nn.Parameter(torch.zeros(maxChannels))
        self.register_buffer("runningMean", torch.zeros(maxChannels))
        self.register_buffer("runningVar", torch.ones(maxChannels))
        self.momentum = 0.1
        self.eps = 1e-5

    def forward(self, inputs: Tensor, activeChannels: int) -> Tensor:
        runningMean = cast(Tensor, self.runningMean)
        runningVar = cast(Tensor, self.runningVar)
        return F.batch_norm(
            inputs,
            runningMean[:activeChannels],
            runningVar[:activeChannels],
            self.weight[:activeChannels],
            self.bias[:activeChannels],
            self.training,
            self.momentum,
            self.eps,
        )


class SlimBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, maxInChannels: int, maxOutChannels: int, stride: int = 1) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxOutChannels = maxOutChannels
        self.stride = stride

        self.conv1 = SlimConv2d(maxInChannels, maxOutChannels, kernelSize=3, stride=stride, padding=1)
        self.bn1 = SlimBatchNorm2d(maxOutChannels)
        self.conv2 = SlimConv2d(maxOutChannels, maxOutChannels, kernelSize=3, stride=1, padding=1)
        self.bn2 = SlimBatchNorm2d(maxOutChannels)

        self.downsampleConv = SlimConv2d(maxInChannels, maxOutChannels, kernelSize=1, stride=stride, padding=0)
        self.downsampleBn = SlimBatchNorm2d(maxOutChannels)

    def forward(self, inputs: Tensor, outChannels: int, strideOverride: Optional[int] = None) -> Tensor:
        identity = inputs
        effectiveStride = self.stride if strideOverride is None else strideOverride

        outputs = self.conv1(inputs, outChannels=outChannels, strideOverride=effectiveStride)
        outputs = self.bn1(outputs, activeChannels=outChannels)
        outputs = F.relu(outputs, inplace=True)

        outputs = self.conv2(outputs, outChannels=outChannels)
        outputs = self.bn2(outputs, activeChannels=outChannels)

        if effectiveStride != 1 or inputs.shape[1] != outChannels:
            identity = self.downsampleConv(inputs, outChannels=outChannels, strideOverride=effectiveStride)
            identity = self.downsampleBn(identity, activeChannels=outChannels)

        outputs = outputs + identity
        outputs = F.relu(outputs, inplace=True)
        return outputs


class SlimLinear(nn.Module):
    def __init__(self, maxInFeatures: int, outFeatures: int) -> None:
        super().__init__()
        self.maxInFeatures = maxInFeatures
        self.outFeatures = outFeatures
        self.weight = nn.Parameter(torch.empty(outFeatures, maxInFeatures))
        self.bias = nn.Parameter(torch.zeros(outFeatures))
        nn.init.normal_(self.weight, 0, 0.01)

    def forward(self, inputs: Tensor) -> Tensor:
        inFeatures = inputs.shape[1]
        weight = self.weight[:, :inFeatures]
        return F.linear(inputs, weight, self.bias)


class ResNetSuperNet(nn.Module):
    def __init__(self, searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> None:
        super().__init__()
        self.searchSpace = searchSpace

        maxStemChannels = max(searchSpace.stemChannels)
        self.stemConv = SlimConv2d(3, maxStemChannels, kernelSize=7, stride=2, padding=3)
        self.stemBn = SlimBatchNorm2d(maxStemChannels)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.maxStageChannels = [
            int(base * max(multOptions)) for base, multOptions in zip(
                searchSpace.baseChannelsPerStage,
                searchSpace.widthMultipliersPerStage,
            )
        ]
        self.maxStageDepths = [max(options) for options in searchSpace.depthOptionsPerStage]

        self.stage1 = self._makeStage(maxStemChannels, self.maxStageChannels[0], self.maxStageDepths[0], firstStride=1)
        self.stage2 = self._makeStage(self.maxStageChannels[0], self.maxStageChannels[1], self.maxStageDepths[1], firstStride=2)
        self.stage3 = self._makeStage(self.maxStageChannels[1], self.maxStageChannels[2], self.maxStageDepths[2], firstStride=2)
        self.stage4 = self._makeStage(self.maxStageChannels[2], self.maxStageChannels[3], self.maxStageDepths[3], firstStride=2)

        self.classifier = SlimLinear(maxInFeatures=self.maxStageChannels[3], outFeatures=searchSpace.numClasses)

    def _makeStage(self, maxInChannels: int, maxOutChannels: int, maxDepth: int, firstStride: int) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for blockIndex in range(maxDepth):
            stride = firstStride if blockIndex == 0 else 1
            inputChannels = maxInChannels if blockIndex == 0 else maxOutChannels
            blocks.append(SlimBasicBlock(inputChannels, maxOutChannels, stride=stride))
        return blocks

    def _resolveStageStrides(self, outputStride: int) -> List[int]:
        if outputStride == 8:
            return [1, 2, 1, 1]
        if outputStride == 16:
            return [1, 2, 2, 1]
        return [1, 2, 2, 2]

    def _forwardStage(
        self,
        inputs: Tensor,
        stageBlocks: nn.ModuleList,
        activeDepth: int,
        activeOutChannels: int,
        strideOverride: int,
    ) -> Tensor:
        outputs = inputs
        for blockIndex in range(activeDepth):
            block = stageBlocks[blockIndex]
            if blockIndex == 0:
                outputs = block(outputs, outChannels=activeOutChannels, strideOverride=strideOverride)
            else:
                outputs = block(outputs, outChannels=activeOutChannels)
        return outputs

    def forwardFeatures(self, inputs: Tensor, architectureConfig: ArchitectureConfig) -> Tensor:
        stageChannels = decodeStageChannels(architectureConfig, self.searchSpace)
        stageDepths = architectureConfig.stageDepths
        stageStrides = self._resolveStageStrides(architectureConfig.outputStride)

        outputs = self.stemConv(inputs, outChannels=architectureConfig.stemChannels)
        outputs = self.stemBn(outputs, activeChannels=architectureConfig.stemChannels)
        outputs = F.relu(outputs, inplace=True)
        outputs = self.maxPool(outputs)

        outputs = self._forwardStage(outputs, self.stage1, stageDepths[0], stageChannels[0], strideOverride=stageStrides[0])
        outputs = self._forwardStage(outputs, self.stage2, stageDepths[1], stageChannels[1], strideOverride=stageStrides[1])
        outputs = self._forwardStage(outputs, self.stage3, stageDepths[2], stageChannels[2], strideOverride=stageStrides[2])
        outputs = self._forwardStage(outputs, self.stage4, stageDepths[3], stageChannels[3], strideOverride=stageStrides[3])

        return outputs

    def forward(self, inputs: Tensor, architectureConfig: Optional[ArchitectureConfig] = None) -> Tuple[Tensor, ArchitectureConfig]:
        if architectureConfig is None:
            architectureConfig = ArchitectureConfig(
                inputResolution=max(self.searchSpace.inputResolutions),
                outputStride=max(self.searchSpace.outputStrides),
                stageDepths=[max(options) for options in self.searchSpace.depthOptionsPerStage],
                stageWidthMultipliers=[max(options) for options in self.searchSpace.widthMultipliersPerStage],
                stemChannels=max(self.searchSpace.stemChannels),
            )

        features = self.forwardFeatures(inputs, architectureConfig)
        pooled = F.adaptive_avg_pool2d(features, output_size=1)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)

        return logits, architectureConfig
