from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import random
from typing import Dict, List


@dataclass(frozen=True)
class SearchSpaceConfig:
    inputResolutions: List[int]
    outputStrides: List[int]
    depthOptionsPerStage: List[List[int]]
    widthMultipliersPerStage: List[List[float]]
    baseChannelsPerStage: List[int]
    stemChannels: List[int]
    stemPathOptions: List[int]
    stagePathOptionsPerStage: List[List[int]]
    stageKernelSizeOptionsPerStage: List[List[int]]
    stageExtraStrideOptionsPerStage: List[List[int]]
    pathDepthMultipliers: List[float]
    pathWidthMultipliers: List[float]
    pathDilations: List[int]
    pathUseSE: List[bool]
    pathMinKernelSizes: List[int]
    pathNames: List[str]
    auxiliaryHeadStages: List[int]
    numClasses: int = 1000


@dataclass(frozen=True)
class ArchitectureConfig:
    inputResolution: int
    outputStride: int
    stageDepths: List[int]
    stageWidthMultipliers: List[float]
    stemChannels: int
    stemPathIndex: int = 1
    stagePathIndices: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    stageKernelSizes: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    stageExtraStrides: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    enableAuxiliaryHeads: bool = True

    def toDict(self) -> Dict[str, object]:
        return {
            "inputResolution": self.inputResolution,
            "outputStride": self.outputStride,
            "stageDepths": list(self.stageDepths),
            "stageWidthMultipliers": list(self.stageWidthMultipliers),
            "stemChannels": self.stemChannels,
            "stemPathIndex": self.stemPathIndex,
            "stagePathIndices": list(self.stagePathIndices),
            "stageKernelSizes": list(self.stageKernelSizes),
            "stageExtraStrides": list(self.stageExtraStrides),
            "enableAuxiliaryHeads": self.enableAuxiliaryHeads,
        }


DEFAULT_SEARCH_SPACE = SearchSpaceConfig(
    inputResolutions=[192, 224, 256, 288, 320],
    outputStrides=[8, 16, 32],
    depthOptionsPerStage=[
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3],
    ],
    widthMultipliersPerStage=[
        [0.5, 0.75, 1.0],
        [0.5, 0.75, 1.0],
        [0.5, 0.75, 1.0],
        [0.5, 0.75, 1.0],
    ],
    baseChannelsPerStage=[64, 128, 256, 512],
    stemChannels=[32, 48, 64],
    stemPathOptions=[0, 1, 2],
    stagePathOptionsPerStage=[
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ],
    stageKernelSizeOptionsPerStage=[
        [3, 5, 7],
        [3, 5, 7],
        [3, 5, 7],
        [3, 5, 7],
    ],
    stageExtraStrideOptionsPerStage=[
        [1],
        [1, 2],
        [1, 2],
        [1, 2],
    ],
    pathDepthMultipliers=[0.75, 1.0, 1.25, 1.0, 1.35],
    pathWidthMultipliers=[1.25, 1.0, 0.75, 1.0, 0.85],
    pathDilations=[1, 1, 1, 1, 2],
    pathUseSE=[False, False, False, True, True],
    pathMinKernelSizes=[3, 3, 3, 5, 3],
    pathNames=["shortWide", "balanced", "deepNarrow", "largeKernelSE", "dilatedSE"],
    auxiliaryHeadStages=[1, 2, 3, 4],
)


def sampleRandomArchitecture(searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> ArchitectureConfig:
    stageDepths = [random.choice(options) for options in searchSpace.depthOptionsPerStage]
    stageWidthMultipliers = [random.choice(options) for options in searchSpace.widthMultipliersPerStage]
    stagePathIndices = [random.choice(options) for options in searchSpace.stagePathOptionsPerStage]
    stageKernelSizes = [random.choice(options) for options in searchSpace.stageKernelSizeOptionsPerStage]
    stageExtraStrides = [random.choice(options) for options in searchSpace.stageExtraStrideOptionsPerStage]

    return ArchitectureConfig(
        inputResolution=random.choice(searchSpace.inputResolutions),
        outputStride=random.choice(searchSpace.outputStrides),
        stageDepths=stageDepths,
        stageWidthMultipliers=stageWidthMultipliers,
        stemChannels=random.choice(searchSpace.stemChannels),
        stemPathIndex=random.choice(searchSpace.stemPathOptions),
        stagePathIndices=stagePathIndices,
        stageKernelSizes=stageKernelSizes,
        stageExtraStrides=stageExtraStrides,
        enableAuxiliaryHeads=True,
    )


def decodeStageChannels(
    architectureConfig: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
) -> List[int]:
    channels: List[int] = []
    for baseChannels, widthMultiplier in zip(searchSpace.baseChannelsPerStage, architectureConfig.stageWidthMultipliers):
        channelCount = int(round(baseChannels * widthMultiplier))
        channelCount = max(8, (channelCount // 8) * 8)
        channels.append(channelCount)

    return channels


def alignChannels(channelCount: int, alignment: int = 8, minChannels: int = 8) -> int:
    aligned = max(minChannels, (channelCount // alignment) * alignment)
    return aligned


def resolveStagePathDepth(baseDepth: int, pathIndex: int, searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> int:
    depthMultiplier = searchSpace.pathDepthMultipliers[pathIndex]
    return max(1, int(round(baseDepth * depthMultiplier)))


def decodeStagePathChannels(
    canonicalStageChannels: List[int],
    architectureConfig: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
) -> List[int]:
    pathChannels: List[int] = []
    for stageIndex, stageChannels in enumerate(canonicalStageChannels):
        pathIndex = architectureConfig.stagePathIndices[stageIndex]
        widthMultiplier = searchSpace.pathWidthMultipliers[pathIndex]
        stagePathChannels = alignChannels(int(round(stageChannels * widthMultiplier)))
        pathChannels.append(stagePathChannels)

    return pathChannels


def countCombinations(searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> int:
    numResolutions = len(searchSpace.inputResolutions)
    numStemOptions = len(searchSpace.stemChannels) * len(searchSpace.stemPathOptions)
    numOutputStrides = len(searchSpace.outputStrides)

    stageCombinations = 1
    for i in range(len(searchSpace.baseChannelsPerStage)):
        stageOpts = (
            len(searchSpace.depthOptionsPerStage[i])
            * len(searchSpace.widthMultipliersPerStage[i])
            * len(searchSpace.stagePathOptionsPerStage[i])
            * len(searchSpace.stageKernelSizeOptionsPerStage[i])
            * len(searchSpace.stageExtraStrideOptionsPerStage[i])
        )
        stageCombinations *= stageOpts

    return numResolutions * numStemOptions * numOutputStrides * stageCombinations


def calculateSearchSpaceSize(searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> int:
    numGlobal = (
        len(searchSpace.inputResolutions) * 
        len(searchSpace.outputStrides) * 
        len(searchSpace.stemChannels) * 
        len(searchSpace.stemPathOptions)
    )
    
    stageVariations = 1
    for i in range(len(searchSpace.baseChannelsPerStage)):
        stageOpts = (
            len(searchSpace.depthOptionsPerStage[i]) * 
            len(searchSpace.widthMultipliersPerStage[i]) * 
            len(searchSpace.stagePathOptionsPerStage[i]) * 
            len(searchSpace.stageKernelSizeOptionsPerStage[i]) * 
            len(searchSpace.stageExtraStrideOptionsPerStage[i])
        )
        stageVariations *= stageOpts
        
    return numGlobal * stageVariations


def iterateAllArchitectures(searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE):
    # Order: Resolution -> OutputStride -> Stem -> Stages
    
    # 1. Global options
    global_options = list(itertools.product(
        searchSpace.inputResolutions,
        searchSpace.outputStrides,
        searchSpace.stemChannels,
        searchSpace.stemPathOptions
    ))
    
    # 2. Per-stage options pre-calculation
    stage_sequences = []
    for i in range(len(searchSpace.baseChannelsPerStage)):
        stage_opts = list(itertools.product(
            searchSpace.depthOptionsPerStage[i],
            searchSpace.widthMultipliersPerStage[i],
            searchSpace.stagePathOptionsPerStage[i],
            searchSpace.stageKernelSizeOptionsPerStage[i],
            searchSpace.stageExtraStrideOptionsPerStage[i]
        ))
        stage_sequences.append(stage_opts)
        
    # 3. Cartesian product of stages
    # WARNING: This might still be too large to materialize fully if search space is huge.
    # We iterate stages dynamically.
    
    for (res, stride, stemCh, stemPath) in global_options:
        for stage_configs in itertools.product(*stage_sequences):
            # stage_configs contains one tuple per stage: ((d1,w1...), (d2,w2...), ...)
            
            stageDepths = [s[0] for s in stage_configs]
            stageWidths = [s[1] for s in stage_configs]
            stagePaths = [s[2] for s in stage_configs]
            stageKernels = [s[3] for s in stage_configs]
            stageExtraStrides = [s[4] for s in stage_configs]
            
            yield ArchitectureConfig(
                inputResolution=res,
                outputStride=stride,
                stageDepths=stageDepths,
                stageWidthMultipliers=stageWidths,
                stemChannels=stemCh,
                stemPathIndex=stemPath,
                stagePathIndices=stagePaths,
                stageKernelSizes=stageKernels,
                stageExtraStrides=stageExtraStrides,
                enableAuxiliaryHeads=True,
            )
