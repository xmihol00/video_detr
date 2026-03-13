from __future__ import annotations

import argparse
import json

import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

from cnnSearch.architecture_io import loadArchitectureConfig
from cnnSearch.data import buildImageFolderLoaders
from cnnSearch.models.subnet import extractSubnetFromSupernet
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.profiling import collectModelResourceMetrics
from cnnSearch.search_space import DEFAULT_SEARCH_SPACE


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate extracted subnet candidate")
    parser.add_argument("--supernetCheckpoint", type=str, required=True)
    parser.add_argument("--architectureJson", type=str, required=True)
    parser.add_argument("--valDir", type=str, required=True)
    parser.add_argument("--batchSize", type=int, default=128)
    parser.add_argument("--numWorkers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--outputJson", type=str, default=None)
    return parser.parse_args()


def evaluateTopK(
    model: torch.nn.Module,
    valLoader: torch.utils.data.DataLoader,
    device: torch.device,
    ampEnabled: bool,
) -> dict:
    model.eval()

    runningLoss = 0.0
    runningTop1 = 0.0
    runningTop5 = 0.0
    steps = 0

    with torch.no_grad():
        for images, targets in valLoader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=ampEnabled and device.type == "cuda"):
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

            _, top5 = logits.topk(5, dim=1)
            correctTop1 = (logits.argmax(dim=1) == targets).float().mean() * 100.0
            correctTop5 = top5.eq(targets.view(-1, 1)).any(dim=1).float().mean() * 100.0

            runningLoss += float(loss.item())
            runningTop1 += float(correctTop1.item())
            runningTop5 += float(correctTop5.item())
            steps += 1

    return {
        "loss": runningLoss / max(steps, 1),
        "top1": runningTop1 / max(steps, 1),
        "top5": runningTop5 / max(steps, 1),
    }


def main() -> None:
    args = parseArguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    architectureConfig = loadArchitectureConfig(args.architectureJson)

    valBundle = buildImageFolderLoaders(
        trainDir=args.valDir,
        valDir=args.valDir,
        imageSize=architectureConfig.inputResolution,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
        distributed=False,
    )

    searchSpace = type(DEFAULT_SEARCH_SPACE)(
        inputResolutions=DEFAULT_SEARCH_SPACE.inputResolutions,
        outputStrides=DEFAULT_SEARCH_SPACE.outputStrides,
        depthOptionsPerStage=DEFAULT_SEARCH_SPACE.depthOptionsPerStage,
        widthMultipliersPerStage=DEFAULT_SEARCH_SPACE.widthMultipliersPerStage,
        baseChannelsPerStage=DEFAULT_SEARCH_SPACE.baseChannelsPerStage,
        stemChannels=DEFAULT_SEARCH_SPACE.stemChannels,
        numClasses=valBundle.numClasses,
    )

    supernet = ResNetSuperNet(searchSpace=searchSpace)
    checkpoint = torch.load(args.supernetCheckpoint, map_location="cpu")
    if "model" in checkpoint:
        modelState = checkpoint["model"]
    else:
        modelState = checkpoint

    if any(key.startswith("module.") for key in modelState.keys()):
        modelState = {key.replace("module.", "", 1): value for key, value in modelState.items()}

    supernet.load_state_dict(modelState)

    extracted = extractSubnetFromSupernet(supernet, architectureConfig=architectureConfig, searchSpace=searchSpace)
    subnet = extracted.model.to(device)

    accuracyMetrics = evaluateTopK(
        model=subnet,
        valLoader=valBundle.valLoader,
        device=device,
        ampEnabled=args.amp,
    )

    resourceMetrics = collectModelResourceMetrics(
        model=subnet,
        inputResolution=architectureConfig.inputResolution,
        device=device,
    )

    result = {
        "architecture": architectureConfig.toDict(),
        "accuracy": accuracyMetrics,
        "resources": resourceMetrics,
    }

    print(json.dumps(result, indent=2))
    if args.outputJson is not None:
        with open(args.outputJson, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
