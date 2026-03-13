from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import torch
from torch import Tensor, nn

from cnnSearch.search_space import ArchitectureConfig, SearchSpaceConfig, DEFAULT_SEARCH_SPACE, decodeStageChannels
from cnnSearch.models.supernet import SlimBasicBlock, SlimBatchNorm2d, SlimConv2d, SlimLinear, ResNetSuperNet


class BasicBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)

        if stride != 1 or inChannels != outChannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outChannels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        identity = self.downsample(inputs)

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu(outputs)

        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)

        outputs = outputs + identity
        outputs = self.relu(outputs)
        return outputs


class ResNetSubnet(nn.Module):
    def __init__(
        self,
        stageDepths: List[int],
        stageChannels: List[int],
        outputStride: int,
        stemChannels: int,
        numClasses: int,
    ) -> None:
        super().__init__()
        self.stageDepths = stageDepths
        self.stageChannels = stageChannels
        self.outputStride = outputStride

        self.stem = nn.Sequential(
            nn.Conv2d(3, stemChannels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stemChannels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        stageStrides = self._resolveStageStrides(outputStride)
        self.stage1 = self._makeStage(stemChannels, stageChannels[0], stageDepths[0], firstStride=stageStrides[0])
        self.stage2 = self._makeStage(stageChannels[0], stageChannels[1], stageDepths[1], firstStride=stageStrides[1])
        self.stage3 = self._makeStage(stageChannels[1], stageChannels[2], stageDepths[2], firstStride=stageStrides[2])
        self.stage4 = self._makeStage(stageChannels[2], stageChannels[3], stageDepths[3], firstStride=stageStrides[3])

        self.classifier = nn.Linear(stageChannels[3], numClasses)

    def _resolveStageStrides(self, outputStride: int) -> List[int]:
        if outputStride == 8:
            return [1, 2, 1, 1]
        if outputStride == 16:
            return [1, 2, 2, 1]
        return [1, 2, 2, 2]

    def _makeStage(self, inChannels: int, outChannels: int, depth: int, firstStride: int) -> nn.Sequential:
        blocks: List[nn.Module] = []
        for blockIndex in range(depth):
            stride = firstStride if blockIndex == 0 else 1
            blockIn = inChannels if blockIndex == 0 else outChannels
            blocks.append(BasicBlock(blockIn, outChannels, stride=stride))
        return nn.Sequential(*blocks)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.stage3(outputs)
        outputs = self.stage4(outputs)

        outputs = torch.nn.functional.adaptive_avg_pool2d(outputs, output_size=1)
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)
        return outputs


@dataclass(frozen=True)
class ExtractedSubnet:
    model: ResNetSubnet
    architectureConfig: ArchitectureConfig


def _copySlimConvWeights(slimConv: SlimConv2d, targetConv: nn.Conv2d, inChannels: int, outChannels: int) -> None:
    with torch.no_grad():
        targetConv.weight.copy_(slimConv.weight[:outChannels, :inChannels, :, :])


def _copySlimBatchNormWeights(slimBn: SlimBatchNorm2d, targetBn: nn.BatchNorm2d, channels: int) -> None:
    with torch.no_grad():
        runningMean = cast(torch.Tensor, slimBn.runningMean)
        runningVar = cast(torch.Tensor, slimBn.runningVar)
        targetBn.weight.copy_(slimBn.weight[:channels])
        targetBn.bias.copy_(slimBn.bias[:channels])
        if targetBn.running_mean is not None:
            targetBn.running_mean.copy_(runningMean[:channels])
        if targetBn.running_var is not None:
            targetBn.running_var.copy_(runningVar[:channels])


def extractSubnetFromSupernet(
    supernetModel: ResNetSuperNet,
    architectureConfig: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
) -> ExtractedSubnet:
    stageChannels = decodeStageChannels(architectureConfig, searchSpace)

    subnetModel = ResNetSubnet(
        stageDepths=architectureConfig.stageDepths,
        stageChannels=stageChannels,
        outputStride=architectureConfig.outputStride,
        stemChannels=architectureConfig.stemChannels,
        numClasses=searchSpace.numClasses,
    )

    stemConv = cast(nn.Conv2d, subnetModel.stem[0])
    stemBn = cast(nn.BatchNorm2d, subnetModel.stem[1])
    _copySlimConvWeights(supernetModel.stemConv, stemConv, inChannels=3, outChannels=architectureConfig.stemChannels)
    _copySlimBatchNormWeights(supernetModel.stemBn, stemBn, channels=architectureConfig.stemChannels)

    superStages = [supernetModel.stage1, supernetModel.stage2, supernetModel.stage3, supernetModel.stage4]
    subStages = [subnetModel.stage1, subnetModel.stage2, subnetModel.stage3, subnetModel.stage4]
    inputChannels = [architectureConfig.stemChannels] + stageChannels[:-1]

    for stageIndex, (superStageAny, subStageAny) in enumerate(zip(superStages, subStages)):
        superStage = cast(nn.ModuleList, superStageAny)
        subStage = cast(nn.Sequential, subStageAny)
        stageOutChannels = stageChannels[stageIndex]
        for blockIndex, subBlock in enumerate(subStage):
            superBlock = cast(SlimBasicBlock, superStage[blockIndex])
            subBlock = cast(BasicBlock, subBlock)
            blockInChannels = inputChannels[stageIndex] if blockIndex == 0 else stageOutChannels

            _copySlimConvWeights(superBlock.conv1, subBlock.conv1, inChannels=blockInChannels, outChannels=stageOutChannels)
            _copySlimBatchNormWeights(superBlock.bn1, subBlock.bn1, channels=stageOutChannels)

            _copySlimConvWeights(superBlock.conv2, subBlock.conv2, inChannels=stageOutChannels, outChannels=stageOutChannels)
            _copySlimBatchNormWeights(superBlock.bn2, subBlock.bn2, channels=stageOutChannels)

            if isinstance(subBlock.downsample, nn.Sequential):
                downsampleConv = cast(nn.Conv2d, subBlock.downsample[0])
                downsampleBn = cast(nn.BatchNorm2d, subBlock.downsample[1])
                _copySlimConvWeights(
                    superBlock.downsampleConv,
                    downsampleConv,
                    inChannels=blockInChannels,
                    outChannels=stageOutChannels,
                )
                _copySlimBatchNormWeights(superBlock.downsampleBn, downsampleBn, channels=stageOutChannels)

    with torch.no_grad():
        classifierHead = cast(SlimLinear, supernetModel.classifier)
        classifierWeight = classifierHead.weight[:, :stageChannels[-1]]
        subnetModel.classifier.weight.copy_(classifierWeight)
        subnetModel.classifier.bias.copy_(classifierHead.bias)

    return ExtractedSubnet(model=subnetModel, architectureConfig=architectureConfig)
