from __future__ import annotations

from pathlib import Path
import tempfile

from PIL import Image
import torch

from cnnSearch.data import buildImageFolderLoaders
from cnnSearch.models.subnet import extractSubnetFromSupernet
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.search_space import ArchitectureConfig, DEFAULT_SEARCH_SPACE


def _createTinyImageFolder(rootPath: Path) -> None:
    for classIndex in range(2):
        classDir = rootPath / f"class_{classIndex}"
        classDir.mkdir(parents=True, exist_ok=True)
        for imageIndex in range(4):
            image = Image.new("RGB", (64, 64), color=(classIndex * 100, imageIndex * 20, 50))
            image.save(classDir / f"image_{imageIndex}.jpg")


def _createSingleImageClass(rootPath: Path, className: str, colorValue: int) -> None:
    classDir = rootPath / className
    classDir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=(colorValue, 10, 10))
    image.save(classDir / "sample.jpg")


def testSupernetForwardAndSubnetExtraction() -> None:
    searchSpace = DEFAULT_SEARCH_SPACE
    supernet = ResNetSuperNet(searchSpace=searchSpace)

    architecture = ArchitectureConfig(
        inputResolution=128,
        outputStride=16,
        stageDepths=[1, 2, 2, 1],
        stageWidthMultipliers=[0.5, 0.5, 0.5, 0.5],
        stemChannels=32,
    )

    inputs = torch.randn(2, 3, 128, 128)
    logits, returnedArchitecture, auxiliaryLogits = supernet(inputs, architecture)

    assert logits.shape == (2, searchSpace.numClasses)
    assert returnedArchitecture.outputStride == architecture.outputStride
    assert len(auxiliaryLogits) == len(searchSpace.auxiliaryHeadStages)

    extracted = extractSubnetFromSupernet(supernet, architectureConfig=architecture, searchSpace=searchSpace)
    subnetLogits = extracted.model(inputs)
    assert subnetLogits.shape == (2, searchSpace.numClasses)


def testImageFolderLoaderWithAutoSplit() -> None:
    with tempfile.TemporaryDirectory() as tempDir:
        rootPath = Path(tempDir)
        _createTinyImageFolder(rootPath)

        dataBundle = buildImageFolderLoaders(
            trainDir=str(rootPath),
            valDir=None,
            imageSize=128,
            batchSize=2,
            numWorkers=0,
            distributed=False,
            valSplitRatio=0.25,
            splitSeed=123,
        )

        trainBatchImages, trainBatchTargets = next(iter(dataBundle.trainLoader))
        valBatchImages, valBatchTargets = next(iter(dataBundle.valLoader))

        assert trainBatchImages.shape[0] == 2
        assert trainBatchImages.shape[1] == 3
        assert trainBatchTargets.ndim == 1

        assert valBatchImages.shape[0] > 0
        assert valBatchTargets.ndim == 1
        assert dataBundle.numClasses == 2


def testImageFolderLoaderRemapsExplicitValClassIndicesToTrainSpace() -> None:
    with tempfile.TemporaryDirectory() as tempDir:
        tempRoot = Path(tempDir)
        trainRoot = tempRoot / "train"
        valRoot = tempRoot / "val"

        _createSingleImageClass(trainRoot, "class_a", 40)
        _createSingleImageClass(trainRoot, "class_b", 90)
        _createSingleImageClass(trainRoot, "class_c", 140)

        # Validation intentionally omits class_a. Without remapping this would become targets {0, 1}.
        _createSingleImageClass(valRoot, "class_b", 80)
        _createSingleImageClass(valRoot, "class_c", 130)

        dataBundle = buildImageFolderLoaders(
            trainDir=str(trainRoot),
            valDir=str(valRoot),
            imageSize=64,
            batchSize=4,
            numWorkers=0,
            distributed=False,
        )

        valTargets = []
        for _, targets in dataBundle.valLoader:
            valTargets.extend(targets.tolist())

        assert dataBundle.numClasses == 3
        assert set(valTargets) == {1, 2}
