from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from cnnSearch.data import buildImageFolderLoaders
from cnnSearch.distributed import cleanupDistributed, getRank, isMainProcess, setupDistributed
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.search_space import ArchitectureConfig, DEFAULT_SEARCH_SPACE
from cnnSearch.trainer import TrainConfig, appendJsonLog, evaluate, saveCheckpoint, trainOneEpoch


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ResNet SuperNet for NAS")

    parser.add_argument("--trainDir", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/train", help="Path to ImageFolder training root")
    parser.add_argument("--valDir", type=str, default=None, help="Optional ImageFolder validation root")

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--imageSize", type=int, default=224)

    parser.add_argument("--learningRate", type=float, default=0.1)
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradientClipNorm", type=float, default=0.0)

    parser.add_argument("--saveDir", type=str, default="cnnSearch/outputs/supernet")
    parser.add_argument("--evalEveryEpochs", type=int, default=1)
    parser.add_argument("--valSplitRatio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def _buildEvaluationArchitecture(imageSize: int) -> ArchitectureConfig:
    return ArchitectureConfig(
        inputResolution=imageSize,
        outputStride=16,
        stageDepths=[max(options) for options in DEFAULT_SEARCH_SPACE.depthOptionsPerStage],
        stageWidthMultipliers=[max(options) for options in DEFAULT_SEARCH_SPACE.widthMultipliersPerStage],
        stemChannels=max(DEFAULT_SEARCH_SPACE.stemChannels),
    )


def _setSeeds(seed: int, rank: int) -> None:
    effectiveSeed = seed + rank
    torch.manual_seed(effectiveSeed)
    torch.cuda.manual_seed_all(effectiveSeed)


def main() -> None:
    args = parseArguments()

    isDistributed, device, localRank = setupDistributed()
    rank = getRank()
    _setSeeds(args.seed, rank)

    dataBundle = buildImageFolderLoaders(
        trainDir=args.trainDir,
        valDir=args.valDir,
        imageSize=args.imageSize,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
        distributed=isDistributed,
        valSplitRatio=args.valSplitRatio,
        splitSeed=args.seed,
    )

    searchSpace = DEFAULT_SEARCH_SPACE
    searchSpace = type(searchSpace)(
        inputResolutions=searchSpace.inputResolutions,
        outputStrides=searchSpace.outputStrides,
        depthOptionsPerStage=searchSpace.depthOptionsPerStage,
        widthMultipliersPerStage=searchSpace.widthMultipliersPerStage,
        baseChannelsPerStage=searchSpace.baseChannelsPerStage,
        stemChannels=searchSpace.stemChannels,
        numClasses=dataBundle.numClasses,
    )

    model = ResNetSuperNet(searchSpace=searchSpace).to(device)

    if isDistributed:
        model = DDP(model, device_ids=[localRank] if device.type == "cuda" else None)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learningRate,
        momentum=args.momentum,
        weight_decay=args.weightDecay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    startEpoch = 0
    bestTop1 = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        startEpoch = int(checkpoint["epoch"]) + 1
        bestTop1 = float(checkpoint.get("bestMetric", 0.0))

    trainConfig = TrainConfig(
        epochs=args.epochs,
        learningRate=args.learningRate,
        weightDecay=args.weightDecay,
        ampEnabled=args.amp,
        gradientClipNorm=args.gradientClipNorm,
        saveDir=args.saveDir,
        evalEveryEpochs=args.evalEveryEpochs,
    )

    evalArchitecture = _buildEvaluationArchitecture(args.imageSize)

    for epochIndex in range(startEpoch, trainConfig.epochs):
        if dataBundle.trainSampler is not None:
            dataBundle.trainSampler.set_epoch(epochIndex)

        trainMetrics = trainOneEpoch(
            model=model,
            optimizer=optimizer,
            trainLoader=dataBundle.trainLoader,
            device=device,
            scaler=scaler,
            epochIndex=epochIndex,
            ampEnabled=trainConfig.ampEnabled,
            gradientClipNorm=trainConfig.gradientClipNorm,
            referenceResolution=args.imageSize,
        )

        scheduler.step()

        metricsForLog = {
            "epoch": epochIndex,
            "trainLoss": trainMetrics["loss"],
            "trainTop1": trainMetrics["top1"],
            "trainTop5": trainMetrics["top5"],
            "learningRate": scheduler.get_last_lr()[0],
        }

        if epochIndex % trainConfig.evalEveryEpochs == 0:
            evalMetrics = evaluate(
                model=model,
                valLoader=dataBundle.valLoader,
                device=device,
                architectureConfig=evalArchitecture,
                ampEnabled=trainConfig.ampEnabled,
            )
            metricsForLog.update(
                {
                    "valLoss": evalMetrics.loss,
                    "valTop1": evalMetrics.top1,
                    "valTop5": evalMetrics.top5,
                }
            )

            if evalMetrics.top1 > bestTop1:
                bestTop1 = evalMetrics.top1
                if isMainProcess():
                    bestPath = Path(trainConfig.saveDir) / "best_model.pth"
                    torch.save({"model": model.state_dict(), "epoch": epochIndex, "bestTop1": bestTop1}, bestPath)

        if isMainProcess():
            appendJsonLog(trainConfig.saveDir, metricsForLog)
            saveCheckpoint(
                saveDir=trainConfig.saveDir,
                epochIndex=epochIndex,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                bestMetric=bestTop1,
                extraState={"rank": rank},
            )

    cleanupDistributed()


if __name__ == "__main__":
    main()
