from __future__ import annotations

from cnnSearch.search_space import (
    ArchitectureConfig,
    DEFAULT_SEARCH_SPACE,
    normalizeArchitectureForSearchSpace,
)


def testNormalizeArchitectureForSearchSpaceProducesValidStaticConfig() -> None:
    architecture = ArchitectureConfig(
        inputResolution=300,
        outputStride=20,
        stageDepths=[9, 0, 8, 5],
        stageWidthMultipliers=[1.1, 0.8, 0.6, 0.4],
        stemChannels=60,
        stemPathIndex=4,
        stagePathIndices=[5, -1, 2, 9],
        stageKernelSizes=[9, 2, 5, 11],
        stageExtraStrides=[3, 7, 0, 2],
        enableAuxiliaryHeads=True,
    )

    normalized = normalizeArchitectureForSearchSpace(
        architecture,
        searchSpace=DEFAULT_SEARCH_SPACE,
        enableAuxiliaryHeads=False,
    )

    assert normalized.inputResolution in DEFAULT_SEARCH_SPACE.inputResolutions
    assert normalized.outputStride in DEFAULT_SEARCH_SPACE.outputStrides
    assert normalized.stemChannels in DEFAULT_SEARCH_SPACE.stemChannels
    assert normalized.stemPathIndex in DEFAULT_SEARCH_SPACE.stemPathOptions

    for stageIndex in range(4):
        assert normalized.stageDepths[stageIndex] in DEFAULT_SEARCH_SPACE.depthOptionsPerStage[stageIndex]
        assert normalized.stageWidthMultipliers[stageIndex] in DEFAULT_SEARCH_SPACE.widthMultipliersPerStage[stageIndex]
        assert normalized.stagePathIndices[stageIndex] in DEFAULT_SEARCH_SPACE.stagePathOptionsPerStage[stageIndex]
        assert normalized.stageKernelSizes[stageIndex] in DEFAULT_SEARCH_SPACE.stageKernelSizeOptionsPerStage[stageIndex]
        assert normalized.stageExtraStrides[stageIndex] in DEFAULT_SEARCH_SPACE.stageExtraStrideOptionsPerStage[stageIndex]

    assert normalized.enableAuxiliaryHeads is False
