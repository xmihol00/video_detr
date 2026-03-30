from __future__ import annotations

import random

from cnnSearch.search_space import (
    ArchitectureConfig,
    DEFAULT_SEARCH_SPACE,
    architectureDistance,
    architectureSimilarityScore,
    generateSimilarArchitectures,
    normalizeArchitectureForSearchSpace,
)


def _buildSeedArchitecture() -> ArchitectureConfig:
    return ArchitectureConfig(
        inputResolution=224,
        outputStride=16,
        stageDepths=[2, 3, 4, 2],
        stageWidthMultipliers=[0.75, 1.0, 0.75, 0.5],
        stemChannels=48,
        stemPathIndex=1,
        stagePathIndices=[0, 1, 2, 1],
        stageKernelSizes=[3, 5, 5, 3],
        stageExtraStrides=[1, 1, 1, 1],
        enableAuxiliaryHeads=False,
    )


def testArchitectureDistanceAndSimilarityIdentity() -> None:
    architecture = _buildSeedArchitecture()

    assert architectureDistance(architecture, architecture, searchSpace=DEFAULT_SEARCH_SPACE) == 0.0
    assert architectureSimilarityScore(architecture, architecture, searchSpace=DEFAULT_SEARCH_SPACE) == 1.0


def testArchitectureDistanceSymmetry() -> None:
    firstArchitecture = _buildSeedArchitecture()
    secondArchitecture = ArchitectureConfig(
        inputResolution=320,
        outputStride=32,
        stageDepths=[3, 4, 6, 3],
        stageWidthMultipliers=[1.0, 1.0, 1.0, 1.0],
        stemChannels=64,
        stemPathIndex=2,
        stagePathIndices=[2, 2, 2, 2],
        stageKernelSizes=[7, 7, 7, 7],
        stageExtraStrides=[1, 2, 2, 2],
        enableAuxiliaryHeads=False,
    )

    firstToSecond = architectureDistance(firstArchitecture, secondArchitecture, searchSpace=DEFAULT_SEARCH_SPACE)
    secondToFirst = architectureDistance(secondArchitecture, firstArchitecture, searchSpace=DEFAULT_SEARCH_SPACE)

    assert firstToSecond == secondToFirst
    assert 0.0 <= firstToSecond <= 1.0


def testGenerateSimilarArchitecturesProducesValidNearbyConfigs() -> None:
    random.seed(123)
    seedArchitecture = _buildSeedArchitecture()

    similarArchitectures = generateSimilarArchitectures(
        seedArchitecture,
        searchSpace=DEFAULT_SEARCH_SPACE,
        maxCandidates=12,
        maxMutations=2,
    )

    assert 1 <= len(similarArchitectures) <= 12

    normalizedSeed = normalizeArchitectureForSearchSpace(
        seedArchitecture,
        searchSpace=DEFAULT_SEARCH_SPACE,
        enableAuxiliaryHeads=False,
    )
    normalizedSeedDict = normalizedSeed.toDict()

    for candidateArchitecture in similarArchitectures:
        normalizedCandidate = normalizeArchitectureForSearchSpace(
            candidateArchitecture,
            searchSpace=DEFAULT_SEARCH_SPACE,
            enableAuxiliaryHeads=False,
        )

        assert normalizedCandidate.toDict() != normalizedSeedDict
        assert normalizedCandidate.inputResolution in DEFAULT_SEARCH_SPACE.inputResolutions
        assert normalizedCandidate.outputStride in DEFAULT_SEARCH_SPACE.outputStrides
        assert normalizedCandidate.stemChannels in DEFAULT_SEARCH_SPACE.stemChannels
        assert normalizedCandidate.stemPathIndex in DEFAULT_SEARCH_SPACE.stemPathOptions

        for stageIndex in range(4):
            assert normalizedCandidate.stageDepths[stageIndex] in DEFAULT_SEARCH_SPACE.depthOptionsPerStage[stageIndex]
            assert normalizedCandidate.stageWidthMultipliers[stageIndex] in DEFAULT_SEARCH_SPACE.widthMultipliersPerStage[stageIndex]
            assert normalizedCandidate.stagePathIndices[stageIndex] in DEFAULT_SEARCH_SPACE.stagePathOptionsPerStage[stageIndex]
            assert normalizedCandidate.stageKernelSizes[stageIndex] in DEFAULT_SEARCH_SPACE.stageKernelSizeOptionsPerStage[stageIndex]
            assert normalizedCandidate.stageExtraStrides[stageIndex] in DEFAULT_SEARCH_SPACE.stageExtraStrideOptionsPerStage[stageIndex]
