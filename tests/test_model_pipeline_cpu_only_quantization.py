from __future__ import annotations

import cnnSearch.model_pipeline as modelPipeline


class _DummyExporter:
    def __init__(self) -> None:
        self.seenCudaIsAvailable = None
        self.seenCudaDeviceCount = None

    def quantize(self, model, representativeDataGenerator, outputOnnxPath) -> None:
        self.seenCudaIsAvailable = modelPipeline.torch.cuda.is_available()
        self.seenCudaDeviceCount = modelPipeline.torch.cuda.device_count()


def testRunQuantizationCpuOnlyTemporarilyDisablesCuda(monkeypatch) -> None:
    originalIsAvailable = modelPipeline.torch.cuda.is_available
    originalDeviceCount = modelPipeline.torch.cuda.device_count

    dummyExporter = _DummyExporter()

    modelPipeline._runQuantizationCpuOnly(
        exporter=dummyExporter,
        model=object(),
        representativeDataGenerator=object(),
        outputOnnxPath="dummy.onnx",
    )

    assert dummyExporter.seenCudaIsAvailable is False
    assert dummyExporter.seenCudaDeviceCount == 0
    assert modelPipeline.torch.cuda.is_available is originalIsAvailable
    assert modelPipeline.torch.cuda.device_count is originalDeviceCount
