from __future__ import annotations

import pytest

from cnnSearch.trainer import shouldEvaluateOnEpoch, shouldSaveCheckpointOnEpoch


def testShouldEvaluateOnEpochUsesOneBasedCadenceAndFinalEpoch() -> None:
    # With eval every 10 epochs over 12 epochs, evaluate at epochs 9 and 11 (0-based).
    evaluatedEpochs = [
        epochIndex
        for epochIndex in range(12)
        if shouldEvaluateOnEpoch(epochIndex=epochIndex, totalEpochs=12, evalEveryEpochs=10)
    ]
    assert evaluatedEpochs == [9, 11]


def testShouldEvaluateOnEpochEveryEpochWhenIntervalOne() -> None:
    evaluatedEpochs = [
        epochIndex
        for epochIndex in range(5)
        if shouldEvaluateOnEpoch(epochIndex=epochIndex, totalEpochs=5, evalEveryEpochs=1)
    ]
    assert evaluatedEpochs == [0, 1, 2, 3, 4]


def testShouldEvaluateOnEpochRejectsNonPositiveInterval() -> None:
    with pytest.raises(ValueError):
        shouldEvaluateOnEpoch(epochIndex=0, totalEpochs=10, evalEveryEpochs=0)


def testShouldSaveCheckpointOnEpochRespectsOneBasedCadence() -> None:
    savedEpochs = [
        epochIndex
        for epochIndex in range(10)
        if shouldSaveCheckpointOnEpoch(epochIndex=epochIndex, saveEveryEpoch=3)
    ]
    assert savedEpochs == [2, 5, 8]


def testShouldSaveCheckpointOnEpochDisablesWhenZero() -> None:
    savedEpochs = [
        epochIndex
        for epochIndex in range(5)
        if shouldSaveCheckpointOnEpoch(epochIndex=epochIndex, saveEveryEpoch=0)
    ]
    assert savedEpochs == []


def testShouldSaveCheckpointOnEpochRejectsNegativeInterval() -> None:
    with pytest.raises(ValueError):
        shouldSaveCheckpointOnEpoch(epochIndex=0, saveEveryEpoch=-1)
