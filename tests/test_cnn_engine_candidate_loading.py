from __future__ import annotations

import json

from cnnSearch.engine import loadCandidateArchitecturesFromCompilationJson


def _config(inputResolution: int) -> dict:
    return {
        "inputResolution": inputResolution,
        "outputStride": 16,
        "stageDepths": [1, 2, 3, 1],
        "stageWidthMultipliers": [0.5, 0.75, 1.0, 0.5],
        "stemChannels": 32,
        "stemPathIndex": 1,
        "stagePathIndices": [0, 1, 2, 1],
        "stageKernelSizes": [3, 3, 5, 3],
        "stageExtraStrides": [1, 1, 2, 1],
        "enableAuxiliaryHeads": False,
    }


def testLoadCandidateArchitecturesFromCompilationJsonDeduplicatesByConfig(tmp_path) -> None:
    payload = {
        "verified_compilable_architectures": [
            {"id": 10, "source": "SAMPLED", "config": _config(224)},
            {"id": 11, "source": "SAMPLED", "config": _config(256)},
        ],
        "likely_compilable_candidates": [
            {"id": 1001, "source": "SIMILARITY", "config": _config(256)},
            {"id": 1002, "source": "SIMILARITY", "config": _config(288)},
        ],
    }

    jsonPath = tmp_path / "candidates.json"
    jsonPath.write_text(json.dumps(payload), encoding="utf-8")

    candidates = loadCandidateArchitecturesFromCompilationJson(str(jsonPath))

    assert len(candidates) == 3
    assert [int(item["id"]) for item in candidates] == [10, 11, 1002]
    assert {int(item["config"]["inputResolution"]) for item in candidates} == {224, 256, 288}
