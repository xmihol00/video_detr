from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import cast

from cnnSearch.search_space import (
    ArchitectureConfig,
    SearchSpaceConfig,
    DEFAULT_SEARCH_SPACE,
    alignChannels,
    decodeStageChannels,
)


def _centerCropKernel(weight: Tensor, kernelSize: int) -> Tensor:
    fullKernel = weight.shape[-1]
    if kernelSize == fullKernel:
        return weight
    offset = (fullKernel - kernelSize) // 2
    return weight[:, :, offset:offset + kernelSize, offset:offset + kernelSize]


class SlimConv2d(nn.Module):
    def __init__(self, maxInChannels: int, maxOutChannels: int, kernelSize: int, stride: int = 1, padding: Optional[int] = None) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxOutChannels = maxOutChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = kernelSize // 2 if padding is None else padding
        self.weight = nn.Parameter(torch.empty(maxOutChannels, maxInChannels, kernelSize, kernelSize))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(
        self,
        inputs: Tensor,
        outChannels: int,
        strideOverride: Optional[int] = None,
        kernelSizeOverride: Optional[int] = None,
        paddingOverride: Optional[int] = None,
        dilationOverride: Optional[int] = None,
    ) -> Tensor:
        inChannels = inputs.shape[1]
        kernelSize = self.kernelSize if kernelSizeOverride is None else kernelSizeOverride
        weight = self.weight[:outChannels, :inChannels, :, :]
        weight = _centerCropKernel(weight, kernelSize=kernelSize)
        stride = self.stride if strideOverride is None else strideOverride
        padding = self.padding if paddingOverride is None else paddingOverride
        dilation = 1 if dilationOverride is None else dilationOverride
        return F.conv2d(inputs, weight, bias=None, stride=stride, padding=padding, dilation=dilation)


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

    def __init__(
        self,
        maxInChannels: int,
        maxOutChannels: int,
        stride: int = 1,
        maxKernelSize: int = 7,
        dilation: int = 1,
        useSE: bool = False,
    ) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxOutChannels = maxOutChannels
        self.stride = stride
        self.dilation = dilation
        self.useSE = useSE

        self.maxKernelSize = maxKernelSize
        self.conv1 = SlimConv2d(maxInChannels, maxOutChannels, kernelSize=maxKernelSize, stride=stride)
        self.bn1 = SlimBatchNorm2d(maxOutChannels)
        self.conv2 = SlimConv2d(maxOutChannels, maxOutChannels, kernelSize=maxKernelSize, stride=1)
        self.bn2 = SlimBatchNorm2d(maxOutChannels)

        self.downsampleConv = SlimConv2d(maxInChannels, maxOutChannels, kernelSize=1, stride=stride, padding=0)
        self.downsampleBn = SlimBatchNorm2d(maxOutChannels)
        self.seReduce = SlimConv2d(maxOutChannels, max(8, maxOutChannels // 4), kernelSize=1, stride=1, padding=0)
        self.seExpand = SlimConv2d(max(8, maxOutChannels // 4), maxOutChannels, kernelSize=1, stride=1, padding=0)

    def forward(
        self,
        inputs: Tensor,
        outChannels: int,
        strideOverride: Optional[int] = None,
        kernelSize: int = 3,
        dilation: Optional[int] = None,
    ) -> Tensor:
        identity = inputs
        effectiveStride = self.stride if strideOverride is None else strideOverride
        effectiveDilation = self.dilation if dilation is None else dilation
        effectivePadding = (kernelSize // 2) * effectiveDilation

        outputs = self.conv1(
            inputs,
            outChannels=outChannels,
            strideOverride=effectiveStride,
            kernelSizeOverride=kernelSize,
            paddingOverride=effectivePadding,
            dilationOverride=effectiveDilation,
        )
        outputs = self.bn1(outputs, activeChannels=outChannels)
        outputs = F.relu(outputs, inplace=True)

        outputs = self.conv2(
            outputs,
            outChannels=outChannels,
            kernelSizeOverride=kernelSize,
            paddingOverride=effectivePadding,
            dilationOverride=effectiveDilation,
        )
        outputs = self.bn2(outputs, activeChannels=outChannels)

        if self.useSE:
            squeeze = F.adaptive_avg_pool2d(outputs, output_size=1)
            seHiddenChannels = max(8, outChannels // 4)
            squeeze = self.seReduce(squeeze, outChannels=seHiddenChannels)
            squeeze = F.relu(squeeze, inplace=True)
            squeeze = self.seExpand(squeeze, outChannels=outChannels)
            squeeze = torch.sigmoid(squeeze)
            outputs = outputs * squeeze

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


class SlimStemPath(nn.Module):
    def __init__(self, maxStemChannels: int, pathType: int) -> None:
        super().__init__()
        self.pathType = pathType

        self.conv7 = SlimConv2d(3, maxStemChannels, kernelSize=7, stride=2, padding=3)
        self.bn7 = SlimBatchNorm2d(maxStemChannels)

        self.conv3a = SlimConv2d(3, maxStemChannels, kernelSize=3, stride=2, padding=1)
        self.bn3a = SlimBatchNorm2d(maxStemChannels)
        self.conv3b = SlimConv2d(maxStemChannels, maxStemChannels, kernelSize=3, stride=1, padding=1)
        self.bn3b = SlimBatchNorm2d(maxStemChannels)

        self.conv5 = SlimConv2d(3, maxStemChannels, kernelSize=5, stride=2, padding=2)
        self.bn5 = SlimBatchNorm2d(maxStemChannels)
        self.conv1 = SlimConv2d(maxStemChannels, maxStemChannels, kernelSize=1, stride=1, padding=0)
        self.bn1 = SlimBatchNorm2d(maxStemChannels)

        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgPool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs: Tensor, stemChannels: int) -> Tensor:
        if self.pathType == 0:
            outputs = self.conv7(inputs, outChannels=stemChannels)
            outputs = self.bn7(outputs, activeChannels=stemChannels)
            outputs = F.relu(outputs, inplace=True)
            outputs = self.maxPool(outputs)
            return outputs

        if self.pathType == 1:
            outputs = self.conv3a(inputs, outChannels=stemChannels)
            outputs = self.bn3a(outputs, activeChannels=stemChannels)
            outputs = F.relu(outputs, inplace=True)
            outputs = self.conv3b(outputs, outChannels=stemChannels)
            outputs = self.bn3b(outputs, activeChannels=stemChannels)
            outputs = F.relu(outputs, inplace=True)
            outputs = self.maxPool(outputs)
            return outputs

        outputs = self.conv5(inputs, outChannels=stemChannels)
        outputs = self.bn5(outputs, activeChannels=stemChannels)
        outputs = F.relu(outputs, inplace=True)
        outputs = self.avgPool(outputs)
        outputs = self.conv1(outputs, outChannels=stemChannels)
        outputs = self.bn1(outputs, activeChannels=stemChannels)
        outputs = F.relu(outputs, inplace=True)
        return outputs


class SlimStemSelector(nn.Module):
    def __init__(self, maxStemChannels: int, stemPathOptions: List[int]) -> None:
        super().__init__()
        self.stemPathOptions = stemPathOptions
        self.paths = nn.ModuleList([SlimStemPath(maxStemChannels=maxStemChannels, pathType=pathIndex) for pathIndex in stemPathOptions])

    def forward(self, inputs: Tensor, stemChannels: int) -> Tensor:
        pathOutputs = [path(inputs, stemChannels=stemChannels) for path in self.paths]
        combinedOutput = torch.zeros_like(pathOutputs[0])
        equalWeight = 1.0 / len(pathOutputs)
        for pathOutput in pathOutputs:
            combinedOutput = combinedOutput + equalWeight * pathOutput
        return combinedOutput


class SlimStagePath(nn.Module):
    def __init__(
        self,
        maxInChannels: int,
        maxPathOutChannels: int,
        maxCanonicalOutChannels: int,
        maxDepth: int,
        maxKernelSize: int,
        firstStride: int,
        pathDilation: int,
        pathUseSE: bool,
        pathMinKernelSize: int,
        pathName: str,
    ) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxPathOutChannels = maxPathOutChannels
        self.maxCanonicalOutChannels = maxCanonicalOutChannels
        self.maxDepth = maxDepth
        self.firstStride = firstStride
        self.pathDilation = pathDilation
        self.pathUseSE = pathUseSE
        self.pathMinKernelSize = pathMinKernelSize
        self.pathName = pathName

        blocks = nn.ModuleList()
        for blockIndex in range(maxDepth):
            blockStride = firstStride if blockIndex == 0 else 1
            blockInChannels = maxInChannels if blockIndex == 0 else maxPathOutChannels
            blocks.append(
                SlimBasicBlock(
                    blockInChannels,
                    maxPathOutChannels,
                    stride=blockStride,
                    maxKernelSize=maxKernelSize,
                    dilation=pathDilation,
                    useSE=pathUseSE,
                )
            )
        self.blocks = blocks

        self.projectionConv = SlimConv2d(maxPathOutChannels, maxCanonicalOutChannels, kernelSize=1, stride=1, padding=0)
        self.projectionBn = SlimBatchNorm2d(maxCanonicalOutChannels)

    def forward(
        self,
        inputs: Tensor,
        activeDepth: int,
        activePathOutChannels: int,
        activeCanonicalOutChannels: int,
        kernelSize: int,
        strideOverride: int,
    ) -> Tensor:
        effectiveKernelSize = max(kernelSize, self.pathMinKernelSize)
        outputs = inputs
        for blockIndex in range(activeDepth):
            block = self.blocks[blockIndex]
            if blockIndex == 0:
                outputs = block(
                    outputs,
                    outChannels=activePathOutChannels,
                    strideOverride=strideOverride,
                    kernelSize=effectiveKernelSize,
                )
            else:
                outputs = block(outputs, outChannels=activePathOutChannels, kernelSize=effectiveKernelSize)

        outputs = self.projectionConv(outputs, outChannels=activeCanonicalOutChannels)
        outputs = self.projectionBn(outputs, activeChannels=activeCanonicalOutChannels)
        outputs = F.relu(outputs, inplace=True)
        return outputs


class SlimStageSelector(nn.Module):
    def __init__(
        self,
        maxInChannels: int,
        maxCanonicalOutChannels: int,
        maxBaseDepth: int,
        maxKernelSize: int,
        firstStride: int,
        pathDepthMultipliers: List[float],
        pathWidthMultipliers: List[float],
        pathDilations: List[int],
        pathUseSE: List[bool],
        pathMinKernelSizes: List[int],
        pathNames: List[str],
    ) -> None:
        super().__init__()
        self.maxInChannels = maxInChannels
        self.maxCanonicalOutChannels = maxCanonicalOutChannels
        self.maxBaseDepth = maxBaseDepth
        self.maxKernelSize = maxKernelSize
        self.firstStride = firstStride
        self.pathDepthMultipliers = pathDepthMultipliers
        self.pathWidthMultipliers = pathWidthMultipliers
        self.pathDilations = pathDilations
        self.pathUseSE = pathUseSE
        self.pathMinKernelSizes = pathMinKernelSizes
        self.pathNames = pathNames

        paths = nn.ModuleList()
        self.pathMaxDepths: List[int] = []
        self.pathMaxOutChannels: List[int] = []

        for depthMultiplier, widthMultiplier, pathDilation, pathSE, pathMinKernelSize, pathName in zip(
            pathDepthMultipliers,
            pathWidthMultipliers,
            pathDilations,
            pathUseSE,
            pathMinKernelSizes,
            pathNames,
        ):
            pathMaxDepth = max(1, int(math.ceil(maxBaseDepth * depthMultiplier)))
            pathMaxOutChannels = alignChannels(int(round(maxCanonicalOutChannels * widthMultiplier)))
            self.pathMaxDepths.append(pathMaxDepth)
            self.pathMaxOutChannels.append(pathMaxOutChannels)

            paths.append(
                SlimStagePath(
                    maxInChannels=maxInChannels,
                    maxPathOutChannels=pathMaxOutChannels,
                    maxCanonicalOutChannels=maxCanonicalOutChannels,
                    maxDepth=pathMaxDepth,
                    maxKernelSize=maxKernelSize,
                    firstStride=firstStride,
                    pathDilation=pathDilation,
                    pathUseSE=pathSE,
                    pathMinKernelSize=pathMinKernelSize,
                    pathName=pathName,
                )
            )

        self.paths = paths

    def forward(
        self,
        inputs: Tensor,
        canonicalOutChannels: int,
        baseDepth: int,
        pathPreferenceIndex: int,
        kernelSize: int,
        stageStride: int,
        extraStride: int,
    ) -> Tensor:
        _ = pathPreferenceIndex
        effectiveStride = max(1, stageStride * extraStride)

        pathOutputs: List[Tensor] = []
        equalWeight = 1.0 / len(self.paths)

        for pathIndex, path in enumerate(self.paths):
            pathDepthMultiplier = self.pathDepthMultipliers[pathIndex]
            activeDepth = max(1, int(round(baseDepth * pathDepthMultiplier)))
            activeDepth = min(activeDepth, self.pathMaxDepths[pathIndex])

            pathWidthMultiplier = self.pathWidthMultipliers[pathIndex]
            activePathOutChannels = alignChannels(int(round(canonicalOutChannels * pathWidthMultiplier)))
            activePathOutChannels = min(activePathOutChannels, self.pathMaxOutChannels[pathIndex])

            pathOutputs.append(
                path(
                    inputs,
                    activeDepth=activeDepth,
                    activePathOutChannels=activePathOutChannels,
                    activeCanonicalOutChannels=canonicalOutChannels,
                    kernelSize=kernelSize,
                    strideOverride=effectiveStride,
                )
            )

        combinedOutput = torch.zeros_like(pathOutputs[0])
        for pathOutput in pathOutputs:
            combinedOutput = combinedOutput + equalWeight * pathOutput

        return combinedOutput


class ResNetSuperNet(nn.Module):
    def __init__(self, searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> None:
        super().__init__()
        self.searchSpace = searchSpace

        maxStemChannels = max(searchSpace.stemChannels)
        self.stem = SlimStemSelector(maxStemChannels=maxStemChannels, stemPathOptions=searchSpace.stemPathOptions)

        self.maxStageChannels = [
            int(base * max(multOptions)) for base, multOptions in zip(
                searchSpace.baseChannelsPerStage,
                searchSpace.widthMultipliersPerStage,
            )
        ]
        self.maxStageDepths = [max(options) for options in searchSpace.depthOptionsPerStage]
        self.maxKernelSize = max(max(options) for options in searchSpace.stageKernelSizeOptionsPerStage)

        self.stage1 = SlimStageSelector(
            maxInChannels=maxStemChannels,
            maxCanonicalOutChannels=self.maxStageChannels[0],
            maxBaseDepth=self.maxStageDepths[0],
            maxKernelSize=self.maxKernelSize,
            firstStride=1,
            pathDepthMultipliers=searchSpace.pathDepthMultipliers,
            pathWidthMultipliers=searchSpace.pathWidthMultipliers,
            pathDilations=searchSpace.pathDilations,
            pathUseSE=searchSpace.pathUseSE,
            pathMinKernelSizes=searchSpace.pathMinKernelSizes,
            pathNames=searchSpace.pathNames,
        )
        self.stage2 = SlimStageSelector(
            maxInChannels=self.maxStageChannels[0],
            maxCanonicalOutChannels=self.maxStageChannels[1],
            maxBaseDepth=self.maxStageDepths[1],
            maxKernelSize=self.maxKernelSize,
            firstStride=2,
            pathDepthMultipliers=searchSpace.pathDepthMultipliers,
            pathWidthMultipliers=searchSpace.pathWidthMultipliers,
            pathDilations=searchSpace.pathDilations,
            pathUseSE=searchSpace.pathUseSE,
            pathMinKernelSizes=searchSpace.pathMinKernelSizes,
            pathNames=searchSpace.pathNames,
        )
        self.stage3 = SlimStageSelector(
            maxInChannels=self.maxStageChannels[1],
            maxCanonicalOutChannels=self.maxStageChannels[2],
            maxBaseDepth=self.maxStageDepths[2],
            maxKernelSize=self.maxKernelSize,
            firstStride=2,
            pathDepthMultipliers=searchSpace.pathDepthMultipliers,
            pathWidthMultipliers=searchSpace.pathWidthMultipliers,
            pathDilations=searchSpace.pathDilations,
            pathUseSE=searchSpace.pathUseSE,
            pathMinKernelSizes=searchSpace.pathMinKernelSizes,
            pathNames=searchSpace.pathNames,
        )
        self.stage4 = SlimStageSelector(
            maxInChannels=self.maxStageChannels[2],
            maxCanonicalOutChannels=self.maxStageChannels[3],
            maxBaseDepth=self.maxStageDepths[3],
            maxKernelSize=self.maxKernelSize,
            firstStride=2,
            pathDepthMultipliers=searchSpace.pathDepthMultipliers,
            pathWidthMultipliers=searchSpace.pathWidthMultipliers,
            pathDilations=searchSpace.pathDilations,
            pathUseSE=searchSpace.pathUseSE,
            pathMinKernelSizes=searchSpace.pathMinKernelSizes,
            pathNames=searchSpace.pathNames,
        )

        self.auxiliaryHeads = nn.ModuleDict(
            {
                f"stage{stageNumber}": SlimLinear(
                    maxInFeatures=self.maxStageChannels[stageNumber - 1],
                    outFeatures=searchSpace.numClasses,
                )
                for stageNumber in searchSpace.auxiliaryHeadStages
            }
        )

        self.classifier = SlimLinear(maxInFeatures=self.maxStageChannels[3], outFeatures=searchSpace.numClasses)

    def _resolveStageStrides(self, outputStride: int) -> List[int]:
        if outputStride == 8:
            return [1, 2, 1, 1]
        if outputStride == 16:
            return [1, 2, 2, 1]
        return [1, 2, 2, 2]

    def _forwardAuxiliaryHeads(self, stageFeatures: Dict[int, Tensor]) -> List[Tensor]:
        auxiliaryLogits: List[Tensor] = []
        for stageNumber in self.searchSpace.auxiliaryHeadStages:
            if stageNumber not in stageFeatures:
                continue
            pooled = F.adaptive_avg_pool2d(stageFeatures[stageNumber], output_size=1)
            pooled = torch.flatten(pooled, 1)
            head = self.auxiliaryHeads[f"stage{stageNumber}"]
            logits = head(pooled)
            auxiliaryLogits.append(logits)

        return auxiliaryLogits

    def forwardFeatures(self, inputs: Tensor, architectureConfig: ArchitectureConfig) -> Tuple[Tensor, Dict[int, Tensor]]:
        stageChannels = decodeStageChannels(architectureConfig, self.searchSpace)
        stageDepths = architectureConfig.stageDepths
        stageStrides = self._resolveStageStrides(architectureConfig.outputStride)
        stagePathIndices = architectureConfig.stagePathIndices
        stageKernelSizes = architectureConfig.stageKernelSizes
        stageExtraStrides = architectureConfig.stageExtraStrides
        stageFeatures: Dict[int, Tensor] = {}

        outputs = self.stem(inputs, stemChannels=architectureConfig.stemChannels)

        outputs = self.stage1(
            outputs,
            canonicalOutChannels=stageChannels[0],
            baseDepth=stageDepths[0],
            pathPreferenceIndex=stagePathIndices[0],
            kernelSize=stageKernelSizes[0],
            stageStride=stageStrides[0],
            extraStride=stageExtraStrides[0],
        )
        stageFeatures[1] = outputs

        outputs = self.stage2(
            outputs,
            canonicalOutChannels=stageChannels[1],
            baseDepth=stageDepths[1],
            pathPreferenceIndex=stagePathIndices[1],
            kernelSize=stageKernelSizes[1],
            stageStride=stageStrides[1],
            extraStride=stageExtraStrides[1],
        )
        stageFeatures[2] = outputs

        outputs = self.stage3(
            outputs,
            canonicalOutChannels=stageChannels[2],
            baseDepth=stageDepths[2],
            pathPreferenceIndex=stagePathIndices[2],
            kernelSize=stageKernelSizes[2],
            stageStride=stageStrides[2],
            extraStride=stageExtraStrides[2],
        )
        stageFeatures[3] = outputs

        outputs = self.stage4(
            outputs,
            canonicalOutChannels=stageChannels[3],
            baseDepth=stageDepths[3],
            pathPreferenceIndex=stagePathIndices[3],
            kernelSize=stageKernelSizes[3],
            stageStride=stageStrides[3],
            extraStride=stageExtraStrides[3],
        )
        stageFeatures[4] = outputs

        return outputs, stageFeatures

    def forward(self, inputs: Tensor, architectureConfig: Optional[ArchitectureConfig] = None) -> Tuple[Tensor, ArchitectureConfig, List[Tensor]]:
        if architectureConfig is None:
            architectureConfig = ArchitectureConfig(
                inputResolution=max(self.searchSpace.inputResolutions),
                outputStride=max(self.searchSpace.outputStrides),
                stageDepths=[max(options) for options in self.searchSpace.depthOptionsPerStage],
                stageWidthMultipliers=[max(options) for options in self.searchSpace.widthMultipliersPerStage],
                stemChannels=max(self.searchSpace.stemChannels),
                stemPathIndex=1,
                stagePathIndices=[1 for _ in self.searchSpace.stagePathOptionsPerStage],
                stageKernelSizes=[3 for _ in self.searchSpace.stageKernelSizeOptionsPerStage],
                stageExtraStrides=[min(max(options), 1) for options in self.searchSpace.stageExtraStrideOptionsPerStage],
                enableAuxiliaryHeads=True,
            )

        features, stageFeatures = self.forwardFeatures(inputs, architectureConfig)
        pooled = F.adaptive_avg_pool2d(features, output_size=1)
        pooled = torch.flatten(pooled, 1)
        logits = self.classifier(pooled)
        auxiliaryLogits = self._forwardAuxiliaryHeads(stageFeatures) if architectureConfig.enableAuxiliaryHeads else []

        return logits, architectureConfig, auxiliaryLogits
