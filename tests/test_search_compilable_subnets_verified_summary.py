from __future__ import annotations

import json

import cnnSearch.search_compilable_subnets as searchSubnets


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


def testSaveVerifiedCandidatesDbIncludesEvaluationMetrics(tmp_path) -> None:
    dbPath = tmp_path / "compilation_search_test.json"
    searchSubnets.configureDatabasePaths(str(dbPath))

    experiments = [
        {
            "id": 1,
            "config": _config(224),
            "param_count": 123456,
            "status": "SUCCESS",
            "source": "SAMPLED",
            "evaluation": {
                "loss": 1.23,
                "top1": 76.5,
                "top5": 92.0,
                "num_samples": 64,
                "num_classes": 10,
            },
        },
        {
            "id": 2,
            "config": _config(256),
            "param_count": 223456,
            "status": "PENDING",
            "source": "SIMILARITY",
            "predicted_similarity": 0.9,
            "predicted_likely_score": 0.8,
        },
    ]

    envelope = searchSubnets.getCompilableEnvelope(experiments)
    searchSubnets.saveVerifiedCandidatesDb(experiments, envelope)

    with open(searchSubnets.VERIFIED_DB_PATH, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["compilable_envelope"] is not None
    assert len(payload["verified_compilable_architectures"]) == 1
    assert payload["verified_compilable_architectures"][0]["evaluation"]["top1"] == 76.5
    assert len(payload["likely_compilable_candidates"]) == 1
    assert payload["likely_compilable_candidates"][0]["predicted_likely_score"] == 0.8
