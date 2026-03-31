from __future__ import annotations

from cnnSearch.model_pipeline import selectOnnxExecutionProviders


def testSelectOnnxExecutionProvidersPrefersCudaWhenAvailable() -> None:
    selected = selectOnnxExecutionProviders(
        availableProviders=["CPUExecutionProvider", "CUDAExecutionProvider"],
        requestedProviders=None,
        preferCuda=True,
    )
    assert selected == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def testSelectOnnxExecutionProvidersFallsBackToCpu() -> None:
    selected = selectOnnxExecutionProviders(
        availableProviders=["CPUExecutionProvider"],
        requestedProviders=None,
        preferCuda=True,
    )
    assert selected == ["CPUExecutionProvider"]


def testSelectOnnxExecutionProvidersRespectsRequestedOrder() -> None:
    selected = selectOnnxExecutionProviders(
        availableProviders=["CPUExecutionProvider", "CUDAExecutionProvider"],
        requestedProviders=["CPUExecutionProvider"],
        preferCuda=True,
    )
    assert selected == ["CPUExecutionProvider"]
