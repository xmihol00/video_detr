from __future__ import annotations

import json
from pathlib import Path

from cnnSearch.search_space import ArchitectureConfig


def saveArchitectureConfig(architectureConfig: ArchitectureConfig, filePath: str) -> None:
    targetPath = Path(filePath)
    targetPath.parent.mkdir(parents=True, exist_ok=True)
    with targetPath.open("w", encoding="utf-8") as handle:
        json.dump(architectureConfig.toDict(), handle, indent=2)


def loadArchitectureConfig(filePath: str) -> ArchitectureConfig:
    with Path(filePath).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return ArchitectureConfig(
        inputResolution=int(payload["inputResolution"]),
        outputStride=int(payload["outputStride"]),
        stageDepths=[int(value) for value in payload["stageDepths"]],
        stageWidthMultipliers=[float(value) for value in payload["stageWidthMultipliers"]],
        stemChannels=int(payload["stemChannels"]),
    )
