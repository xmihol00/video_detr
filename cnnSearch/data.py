from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset, random_split
from torchvision.datasets import ImageFolder

from cnnSearch.augmentations import buildEvalTransform, buildTrainTransform
from cnnSearch.logging_utils import EventLogger, getEventLogger


LOGGER = getEventLogger(__name__)


@dataclass(frozen=True)
class DataLoaderBundle:
    trainLoader: DataLoader
    valLoader: DataLoader
    trainSampler: Optional[DistributedSampler]
    valSampler: Optional[DistributedSampler]
    numClasses: int


def _buildAutoValidationSplit(
    trainDataset: ImageFolder,
    valSplitRatio: float,
    splitSeed: int,
) -> Tuple[Subset, Subset]:
    datasetSize = len(trainDataset)
    valSize = int(datasetSize * valSplitRatio)
    trainSize = datasetSize - valSize

    generator = torch.Generator()
    generator.manual_seed(splitSeed)

    trainSubset, valSubset = random_split(trainDataset, [trainSize, valSize], generator=generator)
    LOGGER.info(
        "Built automatic validation split",
        datasetSize=datasetSize,
        trainSize=trainSize,
        valSize=valSize,
        valSplitRatio=valSplitRatio,
        splitSeed=splitSeed,
    )
    return trainSubset, valSubset


def buildImageFolderLoaders(
    trainDir: str,
    valDir: Optional[str],
    imageSize: int,
    batchSize: int,
    numWorkers: int,
    distributed: bool,
    valSplitRatio: float = 0.1,
    splitSeed: int = 42,
    eventLogger: Optional[EventLogger] = None,
) -> DataLoaderBundle:
    activeLogger = eventLogger if eventLogger is not None else LOGGER
    trainPath = Path(trainDir)
    if not trainPath.exists():
        raise FileNotFoundError(f"Training directory does not exist: {trainDir}")

    activeLogger.logOnce(
        "data.loaders.start",
        "Starting ImageFolder loader construction",
        trainDir=str(trainPath),
        valDir=valDir,
        imageSize=imageSize,
        batchSize=batchSize,
        numWorkers=numWorkers,
        distributed=distributed,
    )

    trainDatasetFull = ImageFolder(root=str(trainPath), transform=buildTrainTransform(imageSize))
    activeLogger.info(
        "Loaded training dataset metadata",
        trainSamples=len(trainDatasetFull),
        numClasses=len(trainDatasetFull.classes),
    )

    if valDir is not None:
        valPath = Path(valDir)
        if not valPath.exists():
            raise FileNotFoundError(f"Validation directory does not exist: {valDir}")

        valDataset = ImageFolder(root=str(valPath), transform=buildEvalTransform(imageSize))
        trainDataset = trainDatasetFull
        activeLogger.info(
            "Using explicit validation directory",
            valDir=str(valPath),
            valSamples=len(valDataset),
        )
    else:
        trainSubset, valSubset = _buildAutoValidationSplit(trainDatasetFull, valSplitRatio=valSplitRatio, splitSeed=splitSeed)
        trainDataset = trainSubset

        valDatasetFull = ImageFolder(root=str(trainPath), transform=buildEvalTransform(imageSize))
        valDataset = Subset(valDatasetFull, valSubset.indices)
        activeLogger.info(
            "Using auto-generated validation split",
            trainSamples=len(trainSubset),
            valSamples=len(valSubset),
        )

    trainSampler: Optional[DistributedSampler]
    valSampler: Optional[DistributedSampler]
    if distributed:
        trainSampler = DistributedSampler(trainDataset, shuffle=True, drop_last=True)
        valSampler = DistributedSampler(valDataset, shuffle=False, drop_last=False)
        activeLogger.info("Enabled distributed samplers")
    else:
        trainSampler = None
        valSampler = None
        activeLogger.info("Using non-distributed samplers")

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=trainSampler is None,
        sampler=trainSampler,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=numWorkers > 0,
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        sampler=valSampler,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=numWorkers > 0,
    )

    return DataLoaderBundle(
        trainLoader=trainLoader,
        valLoader=valLoader,
        trainSampler=trainSampler,
        valSampler=valSampler,
        numClasses=len(trainDatasetFull.classes),
    )
