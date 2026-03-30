from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
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
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
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
    pathDepthMultipliers=[0.75, 1.0, 1.25],
    pathWidthMultipliers=[1.25, 1.0, 0.75],
    pathDilations=[1, 1, 1],
    pathUseSE=[False, False, False],
    pathMinKernelSizes=[3, 3, 3],
    pathNames=["shortWide", "balanced", "deepNarrow"],
    auxiliaryHeadStages=[1, 2, 3, 4],
)

COMPLEX_SEARCH_SPACE = SearchSpaceConfig(
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

def getSearchSpace(useComplexPaths: bool = False) -> SearchSpaceConfig:
    return COMPLEX_SEARCH_SPACE if useComplexPaths else DEFAULT_SEARCH_SPACE


def _selectNearestOption(candidateValue: float, availableOptions: List[float]) -> float:
    return min(availableOptions, key=lambda option: abs(option - candidateValue))


def normalizeArchitectureForSearchSpace(
    architectureConfig: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
    enableAuxiliaryHeads: bool = False,
) -> ArchitectureConfig:
    """Convert a potentially stale architecture config into a static valid config for this search space."""
    stageCount = len(searchSpace.baseChannelsPerStage)

    stageDepths = [
        int(_selectNearestOption(float(architectureConfig.stageDepths[i]), [float(v) for v in searchSpace.depthOptionsPerStage[i]]))
        for i in range(stageCount)
    ]
    stageWidthMultipliers = [
        float(_selectNearestOption(float(architectureConfig.stageWidthMultipliers[i]), searchSpace.widthMultipliersPerStage[i]))
        for i in range(stageCount)
    ]

    stagePathIndices = []
    stageKernelSizes = []
    stageExtraStrides = []
    for stageIndex in range(stageCount):
        pathOptions = searchSpace.stagePathOptionsPerStage[stageIndex]
        kernelOptions = searchSpace.stageKernelSizeOptionsPerStage[stageIndex]
        strideOptions = searchSpace.stageExtraStrideOptionsPerStage[stageIndex]

        stagePathIndices.append(
            int(_selectNearestOption(float(architectureConfig.stagePathIndices[stageIndex]), [float(v) for v in pathOptions]))
        )
        stageKernelSizes.append(
            int(_selectNearestOption(float(architectureConfig.stageKernelSizes[stageIndex]), [float(v) for v in kernelOptions]))
        )
        stageExtraStrides.append(
            int(_selectNearestOption(float(architectureConfig.stageExtraStrides[stageIndex]), [float(v) for v in strideOptions]))
        )

    inputResolution = int(_selectNearestOption(float(architectureConfig.inputResolution), [float(v) for v in searchSpace.inputResolutions]))
    outputStride = int(_selectNearestOption(float(architectureConfig.outputStride), [float(v) for v in searchSpace.outputStrides]))
    stemChannels = int(_selectNearestOption(float(architectureConfig.stemChannels), [float(v) for v in searchSpace.stemChannels]))
    stemPathIndex = int(_selectNearestOption(float(architectureConfig.stemPathIndex), [float(v) for v in searchSpace.stemPathOptions]))

    return ArchitectureConfig(
        inputResolution=inputResolution,
        outputStride=outputStride,
        stageDepths=stageDepths,
        stageWidthMultipliers=stageWidthMultipliers,
        stemChannels=stemChannels,
        stemPathIndex=stemPathIndex,
        stagePathIndices=stagePathIndices,
        stageKernelSizes=stageKernelSizes,
        stageExtraStrides=stageExtraStrides,
        enableAuxiliaryHeads=enableAuxiliaryHeads,
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


def architectureDistance(
    firstArchitecture: ArchitectureConfig,
    secondArchitecture: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
) -> float:
    """Compute a normalized architecture distance in [0, 1] using per-field option ranges."""
    firstConfig = normalizeArchitectureForSearchSpace(firstArchitecture, searchSpace=searchSpace, enableAuxiliaryHeads=False)
    secondConfig = normalizeArchitectureForSearchSpace(secondArchitecture, searchSpace=searchSpace, enableAuxiliaryHeads=False)

    optionRanges: List[int] = []
    absoluteDiffs: List[float] = []

    optionRanges.extend([
        max(1, len(searchSpace.inputResolutions) - 1),
        max(1, len(searchSpace.outputStrides) - 1),
        max(1, len(searchSpace.stemChannels) - 1),
        max(1, len(searchSpace.stemPathOptions) - 1),
    ])
    absoluteDiffs.extend([
        abs(searchSpace.inputResolutions.index(firstConfig.inputResolution) - searchSpace.inputResolutions.index(secondConfig.inputResolution)),
        abs(searchSpace.outputStrides.index(firstConfig.outputStride) - searchSpace.outputStrides.index(secondConfig.outputStride)),
        abs(searchSpace.stemChannels.index(firstConfig.stemChannels) - searchSpace.stemChannels.index(secondConfig.stemChannels)),
        abs(searchSpace.stemPathOptions.index(firstConfig.stemPathIndex) - searchSpace.stemPathOptions.index(secondConfig.stemPathIndex)),
    ])

    for stageIndex in range(len(searchSpace.baseChannelsPerStage)):
        depthOptions = searchSpace.depthOptionsPerStage[stageIndex]
        widthOptions = searchSpace.widthMultipliersPerStage[stageIndex]
        pathOptions = searchSpace.stagePathOptionsPerStage[stageIndex]
        kernelOptions = searchSpace.stageKernelSizeOptionsPerStage[stageIndex]
        strideOptions = searchSpace.stageExtraStrideOptionsPerStage[stageIndex]

        optionRanges.extend([
            max(1, len(depthOptions) - 1),
            max(1, len(widthOptions) - 1),
            max(1, len(pathOptions) - 1),
            max(1, len(kernelOptions) - 1),
            max(1, len(strideOptions) - 1),
        ])
        absoluteDiffs.extend([
            abs(depthOptions.index(firstConfig.stageDepths[stageIndex]) - depthOptions.index(secondConfig.stageDepths[stageIndex])),
            abs(widthOptions.index(firstConfig.stageWidthMultipliers[stageIndex]) - widthOptions.index(secondConfig.stageWidthMultipliers[stageIndex])),
            abs(pathOptions.index(firstConfig.stagePathIndices[stageIndex]) - pathOptions.index(secondConfig.stagePathIndices[stageIndex])),
            abs(kernelOptions.index(firstConfig.stageKernelSizes[stageIndex]) - kernelOptions.index(secondConfig.stageKernelSizes[stageIndex])),
            abs(strideOptions.index(firstConfig.stageExtraStrides[stageIndex]) - strideOptions.index(secondConfig.stageExtraStrides[stageIndex])),
        ])

    normalizedDistance = sum(diff / optionRange for diff, optionRange in zip(absoluteDiffs, optionRanges)) / float(len(optionRanges))
    return max(0.0, min(1.0, normalizedDistance))


def architectureSimilarityScore(
    firstArchitecture: ArchitectureConfig,
    secondArchitecture: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
) -> float:
    """Compute a normalized architecture similarity in [0, 1]."""
    return 1.0 - architectureDistance(firstArchitecture, secondArchitecture, searchSpace=searchSpace)


def generateSimilarArchitectures(
    seedArchitecture: ArchitectureConfig,
    searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE,
    maxCandidates: int = 16,
    maxMutations: int = 2,
) -> List[ArchitectureConfig]:
    """Generate nearby architecture variants by mutating a few categorical choices around a seed."""
    normalizedSeed = normalizeArchitectureForSearchSpace(seedArchitecture, searchSpace=searchSpace, enableAuxiliaryHeads=False)
    generatedArchitectures: List[ArchitectureConfig] = []
    seenConfigs = {json.dumps(normalizedSeed.toDict(), sort_keys=True)}

    mutationFields = [
        "inputResolution",
        "outputStride",
        "stemChannels",
        "stemPathIndex",
        "stageDepths",
        "stageWidthMultipliers",
        "stagePathIndices",
        "stageKernelSizes",
        "stageExtraStrides",
    ]

    maxAttempts = maxCandidates * 12
    attemptIndex = 0
    while len(generatedArchitectures) < maxCandidates and attemptIndex < maxAttempts:
        attemptIndex += 1
        mutatedConfig = ArchitectureConfig(
            inputResolution=normalizedSeed.inputResolution,
            outputStride=normalizedSeed.outputStride,
            stageDepths=list(normalizedSeed.stageDepths),
            stageWidthMultipliers=list(normalizedSeed.stageWidthMultipliers),
            stemChannels=normalizedSeed.stemChannels,
            stemPathIndex=normalizedSeed.stemPathIndex,
            stagePathIndices=list(normalizedSeed.stagePathIndices),
            stageKernelSizes=list(normalizedSeed.stageKernelSizes),
            stageExtraStrides=list(normalizedSeed.stageExtraStrides),
            enableAuxiliaryHeads=False,
        )

        numMutations = random.randint(1, max(1, maxMutations))
        for _ in range(numMutations):
            fieldName = random.choice(mutationFields)
            if fieldName == "inputResolution":
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "inputResolution": random.choice(searchSpace.inputResolutions),
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "outputStride":
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "outputStride": random.choice(searchSpace.outputStrides),
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stemChannels":
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stemChannels": random.choice(searchSpace.stemChannels),
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stemPathIndex":
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stemPathIndex": random.choice(searchSpace.stemPathOptions),
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue

            stageIndex = random.randint(0, len(searchSpace.baseChannelsPerStage) - 1)
            if fieldName == "stageDepths":
                nextValues = list(mutatedConfig.stageDepths)
                nextValues[stageIndex] = random.choice(searchSpace.depthOptionsPerStage[stageIndex])
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stageDepths": nextValues,
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stageWidthMultipliers":
                nextValues = list(mutatedConfig.stageWidthMultipliers)
                nextValues[stageIndex] = random.choice(searchSpace.widthMultipliersPerStage[stageIndex])
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stageWidthMultipliers": nextValues,
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stagePathIndices":
                nextValues = list(mutatedConfig.stagePathIndices)
                nextValues[stageIndex] = random.choice(searchSpace.stagePathOptionsPerStage[stageIndex])
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stagePathIndices": nextValues,
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stageKernelSizes":
                nextValues = list(mutatedConfig.stageKernelSizes)
                nextValues[stageIndex] = random.choice(searchSpace.stageKernelSizeOptionsPerStage[stageIndex])
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stageKernelSizes": nextValues,
                        "enableAuxiliaryHeads": False,
                    }
                )
                continue
            if fieldName == "stageExtraStrides":
                nextValues = list(mutatedConfig.stageExtraStrides)
                nextValues[stageIndex] = random.choice(searchSpace.stageExtraStrideOptionsPerStage[stageIndex])
                mutatedConfig = ArchitectureConfig(
                    **{
                        **mutatedConfig.toDict(),
                        "stageExtraStrides": nextValues,
                        "enableAuxiliaryHeads": False,
                    }
                )

        normalizedMutated = normalizeArchitectureForSearchSpace(mutatedConfig, searchSpace=searchSpace, enableAuxiliaryHeads=False)
        configKey = json.dumps(normalizedMutated.toDict(), sort_keys=True)
        if configKey in seenConfigs:
            continue

        seenConfigs.add(configKey)
        generatedArchitectures.append(normalizedMutated)

    return generatedArchitectures
