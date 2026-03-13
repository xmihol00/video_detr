from __future__ import annotations

from dataclasses import dataclass
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
    numClasses: int = 1000


@dataclass(frozen=True)
class ArchitectureConfig:
    inputResolution: int
    outputStride: int
    stageDepths: List[int]
    stageWidthMultipliers: List[float]
    stemChannels: int

    def toDict(self) -> Dict[str, object]:
        return {
            "inputResolution": self.inputResolution,
            "outputStride": self.outputStride,
            "stageDepths": list(self.stageDepths),
            "stageWidthMultipliers": list(self.stageWidthMultipliers),
            "stemChannels": self.stemChannels,
        }


DEFAULT_SEARCH_SPACE = SearchSpaceConfig(
    inputResolutions=[128, 160, 192, 224],
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
)


def sampleRandomArchitecture(searchSpace: SearchSpaceConfig = DEFAULT_SEARCH_SPACE) -> ArchitectureConfig:
    stageDepths = [random.choice(options) for options in searchSpace.depthOptionsPerStage]
    stageWidthMultipliers = [random.choice(options) for options in searchSpace.widthMultipliersPerStage]

    return ArchitectureConfig(
        inputResolution=random.choice(searchSpace.inputResolutions),
        outputStride=random.choice(searchSpace.outputStrides),
        stageDepths=stageDepths,
        stageWidthMultipliers=stageWidthMultipliers,
        stemChannels=random.choice(searchSpace.stemChannels),
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
