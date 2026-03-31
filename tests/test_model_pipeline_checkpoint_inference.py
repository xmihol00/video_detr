from __future__ import annotations

import torch

import cnnSearch.model_pipeline as modelPipeline


def testBuildSearchSpaceForCheckpointDisablesAuxHeadsWhenMissing(monkeypatch) -> None:
    fakeStateDict = {
        "classifier.weight": torch.zeros((7, 64)),
        "classifier.bias": torch.zeros(7),
        "stem.path0.conv7.weight": torch.zeros((64, 3, 7, 7)),
    }

    monkeypatch.setattr(
        modelPipeline,
        "loadModelStateDictFromCheckpoint",
        lambda checkpointPath: fakeStateDict,
    )

    inferredSearchSpace = modelPipeline.buildSearchSpaceForCheckpoint(
        checkpointPath="dummy_checkpoint.pth",
        useComplexPaths=False,
    )

    assert inferredSearchSpace.numClasses == 7
    assert inferredSearchSpace.auxiliaryHeadStages == []


def testBuildSearchSpaceForCheckpointKeepsDetectedAuxHeads(monkeypatch) -> None:
    fakeStateDict = {
        "classifier.weight": torch.zeros((5, 64)),
        "classifier.bias": torch.zeros(5),
        "auxiliaryHeads.stage1.weight": torch.zeros((5, 64)),
        "auxiliaryHeads.stage1.bias": torch.zeros(5),
        "auxiliaryHeads.stage3.weight": torch.zeros((5, 256)),
        "auxiliaryHeads.stage3.bias": torch.zeros(5),
    }

    monkeypatch.setattr(
        modelPipeline,
        "loadModelStateDictFromCheckpoint",
        lambda checkpointPath: fakeStateDict,
    )

    inferredSearchSpace = modelPipeline.buildSearchSpaceForCheckpoint(
        checkpointPath="dummy_checkpoint.pth",
        useComplexPaths=False,
    )

    assert inferredSearchSpace.numClasses == 5
    assert inferredSearchSpace.auxiliaryHeadStages == [1, 3]
