from __future__ import annotations

from dataclasses import dataclass
from typing import List, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from cnnSearch.models.supernet import (
    ResNetSuperNet,
    SlimBasicBlock,
    SlimBatchNorm2d,
    SlimConv2d,
    SlimLinear,
    SlimStemPath,
    SlimStemSelector,
    SlimStagePath,
    SlimStageSelector,
    _centerCropKernel,
)
from cnnSearch.search_space import (
    ArchitectureConfig,
    SearchSpaceConfig,
    DEFAULT_SEARCH_SPACE,
    decodeStageChannels,
    decodeStagePathChannels,
    resolveStagePathDepth,
)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inChannels: int,
        outChannels: int,
        kernelSize: int,
        stride: int = 1,
        dilation: int = 1,
        useSE: bool = False,
    ) -> None:
        super().__init__()
        padding = (kernelSize // 2) * dilation
        self.useSE = useSE
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outChannels, outChannels, kernel_size=kernelSize, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(outChannels)
        self.seReduce = nn.Conv2d(outChannels, max(8, outChannels // 4), kernel_size=1, stride=1, padding=0, bias=False)
        self.seExpand = nn.Conv2d(max(8, outChannels // 4), outChannels, kernel_size=1, stride=1, padding=0, bias=False)

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

        if self.useSE:
            squeeze = F.adaptive_avg_pool2d(outputs, output_size=1)
            squeeze = self.seReduce(squeeze)
            squeeze = self.relu(squeeze)
            squeeze = self.seExpand(squeeze)
            squeeze = torch.sigmoid(squeeze)
            outputs = outputs * squeeze

        outputs = outputs + identity
        outputs = self.relu(outputs)
        return outputs


class ResNetSubnet(nn.Module):
    def __init__(
        self,
        architectureConfig: ArchitectureConfig,
        searchSpace: SearchSpaceConfig,
    ) -> None:
        super().__init__()
        self.architectureConfig = architectureConfig
        self.searchSpace = searchSpace

        stageChannels = decodeStageChannels(architectureConfig, searchSpace)
        stagePathChannels = decodeStagePathChannels(stageChannels, architectureConfig, searchSpace)
        stageStrides = self._resolveStageStrides(architectureConfig.outputStride)
        self.stemPathIndex = architectureConfig.stemPathIndex
        self.stem = self._makeStem(architectureConfig.stemChannels, architectureConfig.stemPathIndex)

        stageInputs = [architectureConfig.stemChannels] + stageChannels[:-1]
        self.stage1 = self._makeStage(
            inChannels=stageInputs[0],
            pathOutChannels=stagePathChannels[0],
            canonicalOutChannels=stageChannels[0],
            baseDepth=architectureConfig.stageDepths[0],
            pathIndex=architectureConfig.stagePathIndices[0],
            kernelSize=architectureConfig.stageKernelSizes[0],
            firstStride=stageStrides[0] * architectureConfig.stageExtraStrides[0],
        )
        self.stage2 = self._makeStage(
            inChannels=stageInputs[1],
            pathOutChannels=stagePathChannels[1],
            canonicalOutChannels=stageChannels[1],
            baseDepth=architectureConfig.stageDepths[1],
            pathIndex=architectureConfig.stagePathIndices[1],
            kernelSize=architectureConfig.stageKernelSizes[1],
            firstStride=stageStrides[1] * architectureConfig.stageExtraStrides[1],
        )
        self.stage3 = self._makeStage(
            inChannels=stageInputs[2],
            pathOutChannels=stagePathChannels[2],
            canonicalOutChannels=stageChannels[2],
            baseDepth=architectureConfig.stageDepths[2],
            pathIndex=architectureConfig.stagePathIndices[2],
            kernelSize=architectureConfig.stageKernelSizes[2],
            firstStride=stageStrides[2] * architectureConfig.stageExtraStrides[2],
        )
        self.stage4 = self._makeStage(
            inChannels=stageInputs[3],
            pathOutChannels=stagePathChannels[3],
            canonicalOutChannels=stageChannels[3],
            baseDepth=architectureConfig.stageDepths[3],
            pathIndex=architectureConfig.stagePathIndices[3],
            kernelSize=architectureConfig.stageKernelSizes[3],
            firstStride=stageStrides[3] * architectureConfig.stageExtraStrides[3],
        )

        self.classifier = nn.Linear(stageChannels[3], searchSpace.numClasses)

    def _resolveStageStrides(self, outputStride: int) -> List[int]:
        if outputStride == 8:
            return [1, 2, 1, 1]
        if outputStride == 16:
            return [1, 2, 2, 1]
        return [1, 2, 2, 2]

    def _makeStem(self, stemChannels: int, stemPathIndex: int) -> nn.Sequential:
        if stemPathIndex == 0:
            return nn.Sequential(
                nn.Conv2d(3, stemChannels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stemChannels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        if stemPathIndex == 1:
            return nn.Sequential(
                nn.Conv2d(3, stemChannels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stemChannels),
                nn.ReLU(inplace=True),
                nn.Conv2d(stemChannels, stemChannels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stemChannels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        return nn.Sequential(
            nn.Conv2d(3, stemChannels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(stemChannels),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(stemChannels, stemChannels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(stemChannels),
            nn.ReLU(inplace=True),
        )

    def _makeStage(
        self,
        inChannels: int,
        pathOutChannels: int,
        canonicalOutChannels: int,
        baseDepth: int,
        pathIndex: int,
        kernelSize: int,
        firstStride: int,
    ) -> nn.Module:
        safePathIndex = min(max(pathIndex, 0), len(self.searchSpace.pathDepthMultipliers) - 1)
        effectiveKernelSize = max(kernelSize, self.searchSpace.pathMinKernelSizes[safePathIndex])
        pathDilation = self.searchSpace.pathDilations[safePathIndex]
        pathUseSE = self.searchSpace.pathUseSE[safePathIndex]
        depth = resolveStagePathDepth(baseDepth, pathIndex, self.searchSpace)
        blocks: List[nn.Module] = []
        for blockIndex in range(depth):
            stride = firstStride if blockIndex == 0 else 1
            blockIn = inChannels if blockIndex == 0 else pathOutChannels
            blocks.append(
                BasicBlock(
                    blockIn,
                    pathOutChannels,
                    kernelSize=effectiveKernelSize,
                    stride=stride,
                    dilation=pathDilation,
                    useSE=pathUseSE,
                )
            )

        return nn.ModuleDict(
            {
                "blocks": nn.Sequential(*blocks),
                "projection": nn.Sequential(
                    nn.Conv2d(pathOutChannels, canonicalOutChannels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(canonicalOutChannels),
                    nn.ReLU(inplace=True),
                ),
            }
        )

    def _forwardStage(self, stageModule: nn.Module, inputs: Tensor) -> Tensor:
        stageModule = cast(nn.ModuleDict, stageModule)
        blocks = cast(nn.Sequential, stageModule["blocks"])
        projection = cast(nn.Sequential, stageModule["projection"])
        outputs = blocks(inputs)
        outputs = projection(outputs)
        return outputs

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stem(inputs)
        outputs = self._forwardStage(self.stage1, outputs)
        outputs = self._forwardStage(self.stage2, outputs)
        outputs = self._forwardStage(self.stage3, outputs)
        outputs = self._forwardStage(self.stage4, outputs)

        outputs = F.adaptive_avg_pool2d(outputs, output_size=1)
        outputs = torch.flatten(outputs, 1)
        outputs = self.classifier(outputs)
        return outputs


@dataclass(frozen=True)
class ExtractedSubnet:
    model: ResNetSubnet
    architectureConfig: ArchitectureConfig


def _copySlimConvWeights(
    slimConv: SlimConv2d,
    targetConv: nn.Conv2d,
    inChannels: int,
    outChannels: int,
    kernelSize: int,
) -> None:
    with torch.no_grad():
        weight = slimConv.weight[:outChannels, :inChannels, :, :]
        weight = _centerCropKernel(weight, kernelSize=kernelSize)
        targetConv.weight.copy_(weight)


def _copySlimPointwiseConvWeights(slimConv: SlimConv2d, targetConv: nn.Conv2d, inChannels: int, outChannels: int) -> None:
    with torch.no_grad():
        weight = slimConv.weight[:outChannels, :inChannels, :1, :1]
        targetConv.weight.copy_(weight)


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
    stagePathChannels = decodeStagePathChannels(stageChannels, architectureConfig, searchSpace)

    subnetModel = ResNetSubnet(
        architectureConfig=architectureConfig,
        searchSpace=searchSpace,
    )

    superStem = cast(SlimStemSelector, supernetModel.stem)
    stemPathOptions = searchSpace.stemPathOptions
    selectedStemPath = architectureConfig.stemPathIndex
    if selectedStemPath in stemPathOptions:
        selectedStemSlot = stemPathOptions.index(selectedStemPath)
    else:
        selectedStemSlot = min(max(selectedStemPath, 0), len(superStem.paths) - 1)

    superStemPath = cast(SlimStemPath, superStem.paths[selectedStemSlot])
    if selectedStemPath == 0:
        stemConv = cast(nn.Conv2d, subnetModel.stem[0])
        stemBn = cast(nn.BatchNorm2d, subnetModel.stem[1])
        _copySlimConvWeights(
            superStemPath.conv7,
            stemConv,
            inChannels=3,
            outChannels=architectureConfig.stemChannels,
            kernelSize=7,
        )
        _copySlimBatchNormWeights(superStemPath.bn7, stemBn, channels=architectureConfig.stemChannels)
    elif selectedStemPath == 1:
        stemConvA = cast(nn.Conv2d, subnetModel.stem[0])
        stemBnA = cast(nn.BatchNorm2d, subnetModel.stem[1])
        stemConvB = cast(nn.Conv2d, subnetModel.stem[3])
        stemBnB = cast(nn.BatchNorm2d, subnetModel.stem[4])
        _copySlimConvWeights(
            superStemPath.conv3a,
            stemConvA,
            inChannels=3,
            outChannels=architectureConfig.stemChannels,
            kernelSize=3,
        )
        _copySlimBatchNormWeights(superStemPath.bn3a, stemBnA, channels=architectureConfig.stemChannels)
        _copySlimConvWeights(
            superStemPath.conv3b,
            stemConvB,
            inChannels=architectureConfig.stemChannels,
            outChannels=architectureConfig.stemChannels,
            kernelSize=3,
        )
        _copySlimBatchNormWeights(superStemPath.bn3b, stemBnB, channels=architectureConfig.stemChannels)
    else:
        stemConv5 = cast(nn.Conv2d, subnetModel.stem[0])
        stemBn5 = cast(nn.BatchNorm2d, subnetModel.stem[1])
        stemConv1 = cast(nn.Conv2d, subnetModel.stem[4])
        stemBn1 = cast(nn.BatchNorm2d, subnetModel.stem[5])
        _copySlimConvWeights(
            superStemPath.conv5,
            stemConv5,
            inChannels=3,
            outChannels=architectureConfig.stemChannels,
            kernelSize=5,
        )
        _copySlimBatchNormWeights(superStemPath.bn5, stemBn5, channels=architectureConfig.stemChannels)
        _copySlimPointwiseConvWeights(
            superStemPath.conv1,
            stemConv1,
            inChannels=architectureConfig.stemChannels,
            outChannels=architectureConfig.stemChannels,
        )
        _copySlimBatchNormWeights(superStemPath.bn1, stemBn1, channels=architectureConfig.stemChannels)

    superStages = [supernetModel.stage1, supernetModel.stage2, supernetModel.stage3, supernetModel.stage4]
    subStages = [subnetModel.stage1, subnetModel.stage2, subnetModel.stage3, subnetModel.stage4]
    inputChannels = [architectureConfig.stemChannels] + stageChannels[:-1]

    for stageIndex, (superStageAny, subStageAny) in enumerate(zip(superStages, subStages)):
        stagePathIndex = architectureConfig.stagePathIndices[stageIndex]
        safePathIndex = min(max(stagePathIndex, 0), len(searchSpace.pathDepthMultipliers) - 1)
        stageKernelSize = max(architectureConfig.stageKernelSizes[stageIndex], searchSpace.pathMinKernelSizes[safePathIndex])
        stagePathOutChannels = stagePathChannels[stageIndex]
        stageCanonicalChannels = stageChannels[stageIndex]

        superStage = cast(SlimStageSelector, superStageAny)
        selectedSuperPath = cast(SlimStagePath, superStage.paths[stagePathIndex])

        subStage = cast(nn.ModuleDict, subStageAny)
        subBlocks = cast(nn.Sequential, subStage["blocks"])
        subProjection = cast(nn.Sequential, subStage["projection"])

        for blockIndex, subBlockAny in enumerate(subBlocks):
            superBlock = cast(SlimBasicBlock, selectedSuperPath.blocks[blockIndex])
            subBlock = cast(BasicBlock, subBlockAny)
            blockInChannels = inputChannels[stageIndex] if blockIndex == 0 else stagePathOutChannels

            _copySlimConvWeights(
                superBlock.conv1,
                subBlock.conv1,
                inChannels=blockInChannels,
                outChannels=stagePathOutChannels,
                kernelSize=stageKernelSize,
            )
            _copySlimBatchNormWeights(superBlock.bn1, subBlock.bn1, channels=stagePathOutChannels)

            _copySlimConvWeights(
                superBlock.conv2,
                subBlock.conv2,
                inChannels=stagePathOutChannels,
                outChannels=stagePathOutChannels,
                kernelSize=stageKernelSize,
            )
            _copySlimBatchNormWeights(superBlock.bn2, subBlock.bn2, channels=stagePathOutChannels)

            if isinstance(subBlock.downsample, nn.Sequential):
                downsampleConv = cast(nn.Conv2d, subBlock.downsample[0])
                downsampleBn = cast(nn.BatchNorm2d, subBlock.downsample[1])
                _copySlimConvWeights(
                    superBlock.downsampleConv,
                    downsampleConv,
                    inChannels=blockInChannels,
                    outChannels=stagePathOutChannels,
                    kernelSize=1,
                )
                _copySlimBatchNormWeights(superBlock.downsampleBn, downsampleBn, channels=stagePathOutChannels)

            if subBlock.useSE:
                seHiddenChannels = max(8, stagePathOutChannels // 4)
                _copySlimPointwiseConvWeights(
                    superBlock.seReduce,
                    subBlock.seReduce,
                    inChannels=stagePathOutChannels,
                    outChannels=seHiddenChannels,
                )
                _copySlimPointwiseConvWeights(
                    superBlock.seExpand,
                    subBlock.seExpand,
                    inChannels=seHiddenChannels,
                    outChannels=stagePathOutChannels,
                )

        projectionConv = cast(nn.Conv2d, subProjection[0])
        projectionBn = cast(nn.BatchNorm2d, subProjection[1])
        _copySlimConvWeights(
            selectedSuperPath.projectionConv,
            projectionConv,
            inChannels=stagePathOutChannels,
            outChannels=stageCanonicalChannels,
            kernelSize=1,
        )
        _copySlimBatchNormWeights(selectedSuperPath.projectionBn, projectionBn, channels=stageCanonicalChannels)

    with torch.no_grad():
        classifierHead = cast(SlimLinear, supernetModel.classifier)
        classifierWeight = classifierHead.weight[:, :stageChannels[-1]]
        subnetModel.classifier.weight.copy_(classifierWeight)
        subnetModel.classifier.bias.copy_(classifierHead.bias)

    return ExtractedSubnet(model=subnetModel, architectureConfig=architectureConfig)
