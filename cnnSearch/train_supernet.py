from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from cnnSearch.data import buildImageFolderLoaders
from cnnSearch.distributed import cleanupDistributed, getRank, isMainProcess, setupDistributed
from cnnSearch.logging_utils import LoggingConfig, configureLogging, getEventLogger
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.search_space import ArchitectureConfig, DEFAULT_SEARCH_SPACE
from cnnSearch.trainer import TrainConfig, appendJsonLog, evaluate, saveCheckpoint, trainOneEpoch

NUM_GPUS = 1

import safe_gpu
while True:
    try:
        safe_gpu.claim_gpus(NUM_GPUS)
        break
    except:
        print("Waiting for free GPU")
        time.sleep(5)
        pass

LOGGER = getEventLogger(__name__)

def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ResNet SuperNet for NAS")

    parser.add_argument("--trainDir", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/train", help="Path to ImageFolder training root")
    parser.add_argument("--valDir", type=str, default=None, help="Optional ImageFolder validation root")

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batchSize", type=int, default=72)
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--imageSize", type=int, default=320)

    parser.add_argument("--learningRate", type=float, default=0.1)
    parser.add_argument("--weightDecay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradientClipNorm", type=float, default=0.0)
    parser.add_argument("--auxiliaryLossWeight", type=float, default=0.3)
    parser.add_argument("--aux-heads", action="store_true", help="Enable auxiliary classification heads during training")
    
    # New argument to enable SE paths (index 3 and 4) which are now disabled by default
    parser.add_argument("--enable-complex-paths", action="store_true", help="Enable complex SE and dilated paths (paths 3 and 4) which are disabled by default")

    parser.add_argument("--saveDir", type=str, default="./checkpoints")
    parser.add_argument("--evalEveryEpochs", type=int, default=10)
    parser.add_argument("--valSplitRatio", type=float, default=0.15)
    parser.add_argument("--disableCheckpointing", action="store_true", help="Skip writing checkpoints and best model files")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--logLevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--logFormat", type=str, default="text", choices=["text", "json"])
    parser.add_argument("--logFile", type=str, default=None)
    parser.add_argument("--logIntervalSteps", type=int, default=50)

    return parser.parse_args()


def _buildEvaluationArchitecture(imageSize: int) -> ArchitectureConfig:
    return ArchitectureConfig(
        inputResolution=imageSize,
        outputStride=16,
        stageDepths=[max(options) for options in DEFAULT_SEARCH_SPACE.depthOptionsPerStage],
        stageWidthMultipliers=[max(options) for options in DEFAULT_SEARCH_SPACE.widthMultipliersPerStage],
        stemChannels=max(DEFAULT_SEARCH_SPACE.stemChannels),
        stemPathIndex=1,
        stagePathIndices=[1, 1, 1, 1],
        stageKernelSizes=[3, 3, 3, 3],
        stageExtraStrides=[1, 1, 1, 1],
        enableAuxiliaryHeads=False,
    )


def _setSeeds(seed: int, rank: int) -> None:
    effectiveSeed = seed + rank
    torch.manual_seed(effectiveSeed)
    torch.cuda.manual_seed_all(effectiveSeed)


def main() -> None:
    args = parseArguments()

    configureLogging(
        LoggingConfig(
            logLevel=args.logLevel,
            logFormat=args.logFormat,
            logFilePath=args.logFile,
            enableConsole=True,
        )
    )

    LOGGER.info(
        "Starting supernet training entrypoint",
        trainDir=args.trainDir,
        valDir=args.valDir,
        epochs=args.epochs,
        batchSize=args.batchSize,
        imageSize=args.imageSize,
        amp=args.amp,
        logLevel=args.logLevel,
        logFormat=args.logFormat,
        logFile=args.logFile,
    )

    Path(args.saveDir).mkdir(parents=True, exist_ok=True)

    isDistributed, device, localRank = setupDistributed()
    rank = getRank()
    _setSeeds(args.seed, rank)
    LOGGER.info("Initialized runtime and seeds", rank=rank, localRank=localRank, distributed=isDistributed, device=str(device), seed=args.seed)

    dataBundle = buildImageFolderLoaders(
        trainDir=args.trainDir,
        valDir=args.valDir,
        imageSize=args.imageSize,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
        distributed=isDistributed,
        valSplitRatio=args.valSplitRatio,
        splitSeed=args.seed,
        eventLogger=LOGGER,
    )
    LOGGER.info(
        "Constructed dataloaders",
        trainBatches=len(dataBundle.trainLoader),
        valBatches=len(dataBundle.valLoader),
        numClasses=dataBundle.numClasses,
    )

    searchSpace = DEFAULT_SEARCH_SPACE
    if args.enable_complex_paths:
        from cnnSearch.search_space import COMPLEX_SEARCH_SPACE
        searchSpace = COMPLEX_SEARCH_SPACE
        LOGGER.info("Enabling complex SE and dilated paths (paths 3 and 4)")
    else:
        LOGGER.info("Using simplified search space (paths 0, 1, 2 only)")

    # Update auxiliary heads config based on args
    searchSpace = type(searchSpace)(
        inputResolutions=searchSpace.inputResolutions,
        outputStrides=searchSpace.outputStrides,
        depthOptionsPerStage=searchSpace.depthOptionsPerStage,
        widthMultipliersPerStage=searchSpace.widthMultipliersPerStage,
        baseChannelsPerStage=searchSpace.baseChannelsPerStage,
        stemChannels=searchSpace.stemChannels,
        stemPathOptions=searchSpace.stemPathOptions,
        stagePathOptionsPerStage=searchSpace.stagePathOptionsPerStage,
        stageKernelSizeOptionsPerStage=searchSpace.stageKernelSizeOptionsPerStage,
        stageExtraStrideOptionsPerStage=searchSpace.stageExtraStrideOptionsPerStage,
        pathDepthMultipliers=searchSpace.pathDepthMultipliers,
        pathWidthMultipliers=searchSpace.pathWidthMultipliers,
        pathDilations=searchSpace.pathDilations,
        pathUseSE=searchSpace.pathUseSE,
        pathMinKernelSizes=searchSpace.pathMinKernelSizes,
        pathNames=searchSpace.pathNames,
        auxiliaryHeadStages=searchSpace.auxiliaryHeadStages if args.aux_heads else [],
        numClasses=dataBundle.numClasses,
    )

    model = ResNetSuperNet(searchSpace=searchSpace).to(device)

    if isDistributed:
        model = DDP(model, device_ids=[localRank] if device.type == "cuda" else None)
    LOGGER.info("Model instantiated", distributedWrapped=isDistributed)

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
        LOGGER.info("Loading checkpoint for resume", checkpointPath=args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        startEpoch = int(checkpoint["epoch"]) + 1
        bestTop1 = float(checkpoint.get("bestMetric", 0.0))
        LOGGER.info("Checkpoint loaded", startEpoch=startEpoch, bestTop1=bestTop1)

    trainConfig = TrainConfig(
        epochs=args.epochs,
        learningRate=args.learningRate,
        weightDecay=args.weightDecay,
        ampEnabled=args.amp,
        gradientClipNorm=args.gradientClipNorm,
        auxiliaryLossWeight=args.auxiliaryLossWeight,
        saveDir=args.saveDir,
        evalEveryEpochs=args.evalEveryEpochs,
    )

    evalArchitecture = _buildEvaluationArchitecture(args.imageSize)

    for epochIndex in range(startEpoch, trainConfig.epochs):
        LOGGER.info("Epoch started", epoch=epochIndex)
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
            auxiliaryLossWeight=trainConfig.auxiliaryLossWeight,
            referenceResolution=args.imageSize,
            eventLogger=LOGGER,
            logIntervalSteps=args.logIntervalSteps,
        )

        scheduler.step()

        metricsForLog = {
            "epoch": epochIndex,
            "trainLoss": trainMetrics["loss"],
            "trainTop1": trainMetrics["top1"],
            "trainTop5": trainMetrics["top5"],
            "learningRate": scheduler.get_last_lr()[0],
        }

        # Log epoch summary clearly
        if isMainProcess():
            print("\n" + "="*60)
            print(f"EPOCH {epochIndex} COMPLETED")
            print(f"TRAIN Average Loss: {trainMetrics['loss']:.4f}")
            print(f"TRAIN Top-1 Acc:    {trainMetrics['top1']:.2f}%")
            print(f"TRAIN Top-5 Acc:    {trainMetrics['top5']:.2f}%")

        if epochIndex % trainConfig.evalEveryEpochs == 0:
            evalMetrics = evaluate(
                model=model,
                valLoader=dataBundle.valLoader,
                device=device,
                architectureConfig=evalArchitecture,
                ampEnabled=trainConfig.ampEnabled,
                eventLogger=LOGGER,
                logIntervalSteps=max(1, args.logIntervalSteps * 2),
                epochIndex=epochIndex,
            )
            metricsForLog.update(
                {
                    "valLoss": evalMetrics.loss,
                    "valTop1": evalMetrics.top1,
                    "valTop5": evalMetrics.top5,
                }
            )
            
            if isMainProcess():
                print(f"VAL   Average Loss: {evalMetrics.loss:.4f}")
                print(f"VAL   Top-1 Acc:    {evalMetrics.top1:.2f}%")
                print(f"VAL   Top-5 Acc:    {evalMetrics.top5:.2f}%")

            if evalMetrics.top1 > bestTop1:
                bestTop1 = evalMetrics.top1
                if isMainProcess() and not args.disableCheckpointing:
                    bestPath = Path(trainConfig.saveDir) / "best_model.pth"
                    try:
                        torch.save(
                            {"model": model.state_dict(), "epoch": epochIndex, "bestTop1": bestTop1},
                            bestPath,
                            _use_new_zipfile_serialization=False,
                        )
                        LOGGER.info("Saved new best model", epoch=epochIndex, bestTop1=bestTop1, bestPath=str(bestPath))
                    except Exception as e:
                        LOGGER.error("Failed to save best model", error=str(e))
        
        if isMainProcess():
            print("="*60 + "\n")

        if isMainProcess():
            appendJsonLog(trainConfig.saveDir, metricsForLog)
            if not args.disableCheckpointing:
                saveCheckpoint(
                    saveDir=trainConfig.saveDir,
                    epochIndex=epochIndex,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    bestMetric=bestTop1,
                    extraState={"rank": rank},
                )
            
            # Print End of Epoch Summary
            print("\n" + "="*50)
            print(f"End of Epoch {epochIndex} Summary:")
            print(f"  Train Loss: {trainMetrics['loss']:.4f}")
            print(f"  Train Top1: {trainMetrics['top1']:.2f}%")
            print(f"  Train Top5: {trainMetrics['top5']:.2f}%")
            if "valLoss" in metricsForLog:
                print(f"  Val   Loss: {metricsForLog['valLoss']:.4f}")
                print(f"  Val   Top1: {metricsForLog['valTop1']:.2f}%")
                print(f"  Val   Top5: {metricsForLog['valTop5']:.2f}%")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Best Val Top1: {bestTop1:.2f}%")
            print("="*50 + "\n")

        LOGGER.info(
            "Epoch completed",
            epoch=epochIndex,
            trainLoss=trainMetrics["loss"],
            trainTop1=trainMetrics["top1"],
            trainTop5=trainMetrics["top5"],
            learningRate=scheduler.get_last_lr()[0],
            bestTop1=bestTop1,
        )

    cleanupDistributed()
    LOGGER.info("Training run completed", bestTop1=bestTop1)


if __name__ == "__main__":
    main()
