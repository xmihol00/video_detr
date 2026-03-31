from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from cnnSearch.model_pipeline import OnnxClassificationEvaluator


def _writeDummyImage(path: Path) -> None:
    image = Image.new("RGB", (16, 16), color=(127, 127, 127))
    image.save(path)


def testOnnxEvaluatorDataLoaderRespectsMaxImages(tmp_path) -> None:
    pytest.importorskip("onnxruntime")

    class0Dir = tmp_path / "class0"
    class1Dir = tmp_path / "class1"
    class0Dir.mkdir(parents=True)
    class1Dir.mkdir(parents=True)

    for imageIndex in range(4):
        _writeDummyImage(class0Dir / f"img_{imageIndex}.png")
    for imageIndex in range(3):
        _writeDummyImage(class1Dir / f"img_{imageIndex}.png")

    evaluator = OnnxClassificationEvaluator(
        datasetPath=str(tmp_path),
        imageSize=16,
        batchSize=2,
        numWorkers=0,
        maxImages=5,
        enableProgressLogging=False,
    )

    dataLoader, numClasses, totalImages = evaluator._buildDataLoader()

    assert numClasses == 2
    assert totalImages == 5
    assert len(dataLoader.dataset) == 5
