from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from cnnSearch.augmentations import buildEvalTransform
from cnnSearch.export_utils import Imx500Exporter, RepresentativeDataGenerator
from cnnSearch.models.subnet import extractSubnetFromSupernet
from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.search_space import (
    ArchitectureConfig,
    SearchSpaceConfig,
    getSearchSpace,
    normalizeArchitectureForSearchSpace,
)


@dataclass(frozen=True)
class QuantizationResult:
    architectureConfig: ArchitectureConfig
    paramCount: int
    onnxPath: str


@dataclass(frozen=True)
class CompilationResult:
    success: bool
    returnCode: int
    errorMessage: Optional[str]
    stdout: str
    stderr: str
    outputDirectory: str


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    top1: float
    top5: float
    numSamples: int
    numClasses: int


@dataclass(frozen=True)
class CandidatePipelineResult:
    architectureConfig: ArchitectureConfig
    paramCount: int
    quantizedOnnxPath: str
    compilation: CompilationResult
    evaluation: Optional[EvaluationResult]


def withSearchSpaceNumClasses(searchSpace: SearchSpaceConfig, numClasses: int) -> SearchSpaceConfig:
    return SearchSpaceConfig(
        inputResolutions=list(searchSpace.inputResolutions),
        outputStrides=list(searchSpace.outputStrides),
        depthOptionsPerStage=[list(options) for options in searchSpace.depthOptionsPerStage],
        widthMultipliersPerStage=[list(options) for options in searchSpace.widthMultipliersPerStage],
        baseChannelsPerStage=list(searchSpace.baseChannelsPerStage),
        stemChannels=list(searchSpace.stemChannels),
        stemPathOptions=list(searchSpace.stemPathOptions),
        stagePathOptionsPerStage=[list(options) for options in searchSpace.stagePathOptionsPerStage],
        stageKernelSizeOptionsPerStage=[list(options) for options in searchSpace.stageKernelSizeOptionsPerStage],
        stageExtraStrideOptionsPerStage=[list(options) for options in searchSpace.stageExtraStrideOptionsPerStage],
        pathDepthMultipliers=list(searchSpace.pathDepthMultipliers),
        pathWidthMultipliers=list(searchSpace.pathWidthMultipliers),
        pathDilations=list(searchSpace.pathDilations),
        pathUseSE=list(searchSpace.pathUseSE),
        pathMinKernelSizes=list(searchSpace.pathMinKernelSizes),
        pathNames=list(searchSpace.pathNames),
        auxiliaryHeadStages=list(searchSpace.auxiliaryHeadStages),
        numClasses=int(numClasses),
    )


def loadModelStateDictFromCheckpoint(checkpointPath: str) -> Dict[str, Any]:
    checkpointData = torch.load(checkpointPath, map_location="cpu")
    modelStateDict = checkpointData["model"] if isinstance(checkpointData, dict) and "model" in checkpointData else checkpointData
    if any(str(key).startswith("module.") for key in modelStateDict.keys()):
        modelStateDict = {
            str(key).replace("module.", "", 1): value
            for key, value in modelStateDict.items()
        }
    return modelStateDict


def inferNumClassesFromStateDict(modelStateDict: Dict[str, Any], defaultNumClasses: int = 1000) -> int:
    classifierWeight = modelStateDict.get("classifier.weight")
    if isinstance(classifierWeight, torch.Tensor) and classifierWeight.ndim == 2:
        return int(classifierWeight.shape[0])
    return int(defaultNumClasses)


def inferAuxiliaryHeadStagesFromStateDict(
    modelStateDict: Dict[str, Any],
    defaultAuxiliaryHeadStages: Sequence[int],
) -> List[int]:
    auxiliaryHeadPattern = re.compile(r"^auxiliaryHeads\.stage(\d+)\.(weight|bias)$")
    detectedStages = set()
    for parameterName in modelStateDict.keys():
        patternMatch = auxiliaryHeadPattern.match(str(parameterName))
        if patternMatch is None:
            continue
        detectedStages.add(int(patternMatch.group(1)))

    if not detectedStages:
        return []

    if defaultAuxiliaryHeadStages:
        allowedStages = set(int(stage) for stage in defaultAuxiliaryHeadStages)
        filteredStages = sorted(stage for stage in detectedStages if stage in allowedStages)
        if filteredStages:
            return filteredStages

    return sorted(detectedStages)


def buildSearchSpaceForCheckpoint(checkpointPath: str, useComplexPaths: bool) -> SearchSpaceConfig:
    baseSearchSpace = getSearchSpace(useComplexPaths=useComplexPaths)
    modelStateDict = loadModelStateDictFromCheckpoint(checkpointPath)
    numClasses = inferNumClassesFromStateDict(modelStateDict, defaultNumClasses=baseSearchSpace.numClasses)
    auxiliaryHeadStages = inferAuxiliaryHeadStagesFromStateDict(
        modelStateDict,
        defaultAuxiliaryHeadStages=baseSearchSpace.auxiliaryHeadStages,
    )
    return SearchSpaceConfig(
        inputResolutions=list(baseSearchSpace.inputResolutions),
        outputStrides=list(baseSearchSpace.outputStrides),
        depthOptionsPerStage=[list(options) for options in baseSearchSpace.depthOptionsPerStage],
        widthMultipliersPerStage=[list(options) for options in baseSearchSpace.widthMultipliersPerStage],
        baseChannelsPerStage=list(baseSearchSpace.baseChannelsPerStage),
        stemChannels=list(baseSearchSpace.stemChannels),
        stemPathOptions=list(baseSearchSpace.stemPathOptions),
        stagePathOptionsPerStage=[list(options) for options in baseSearchSpace.stagePathOptionsPerStage],
        stageKernelSizeOptionsPerStage=[list(options) for options in baseSearchSpace.stageKernelSizeOptionsPerStage],
        stageExtraStrideOptionsPerStage=[list(options) for options in baseSearchSpace.stageExtraStrideOptionsPerStage],
        pathDepthMultipliers=list(baseSearchSpace.pathDepthMultipliers),
        pathWidthMultipliers=list(baseSearchSpace.pathWidthMultipliers),
        pathDilations=list(baseSearchSpace.pathDilations),
        pathUseSE=list(baseSearchSpace.pathUseSE),
        pathMinKernelSizes=list(baseSearchSpace.pathMinKernelSizes),
        pathNames=list(baseSearchSpace.pathNames),
        auxiliaryHeadStages=list(auxiliaryHeadStages),
        numClasses=int(numClasses),
    )


def loadSupernetFromCheckpoint(
    checkpointPath: str,
    searchSpace: SearchSpaceConfig,
) -> ResNetSuperNet:
    supernetModel = ResNetSuperNet(searchSpace=searchSpace)
    modelStateDict = loadModelStateDictFromCheckpoint(checkpointPath)
    supernetModel.load_state_dict(modelStateDict)
    supernetModel.eval()
    return supernetModel


def _runQuantizationCpuOnly(
    exporter: Imx500Exporter,
    model: torch.nn.Module,
    representativeDataGenerator: RepresentativeDataGenerator,
    outputOnnxPath: str,
) -> None:
    """Run MCT quantization while forcing CPU path even if CUDA is visible but incompatible."""
    originalCudaIsAvailable = torch.cuda.is_available
    originalCudaDeviceCount = torch.cuda.device_count

    try:
        torch.cuda.is_available = lambda: False  # type: ignore[assignment]
        torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
        exporter.quantize(model, representativeDataGenerator, outputOnnxPath)
    finally:
        torch.cuda.is_available = originalCudaIsAvailable  # type: ignore[assignment]
        torch.cuda.device_count = originalCudaDeviceCount  # type: ignore[assignment]


class OnnxClassificationEvaluator:
    """Evaluate an ONNX classification model on an ImageFolder dataset."""

    def __init__(
        self,
        datasetPath: str,
        imageSize: int,
        batchSize: int = 32,
        numWorkers: int = 4,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as importError:
            raise ImportError("onnxruntime is required for ONNX model evaluation") from importError

        self._ort = ort
        self.datasetPath = datasetPath
        self.imageSize = int(imageSize)
        self.batchSize = int(batchSize)
        self.numWorkers = int(numWorkers)

        availableProviders = set(self._ort.get_available_providers())
        if providers is not None:
            self.providers = [provider for provider in providers if provider in availableProviders]
        else:
            providerPreference = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.providers = [provider for provider in providerPreference if provider in availableProviders]

        if not self.providers:
            self.providers = ["CPUExecutionProvider"]

    def _buildDataLoader(self) -> tuple[DataLoader, int]:
        evalDataset = ImageFolder(self.datasetPath, transform=buildEvalTransform(self.imageSize))
        dataLoader = DataLoader(
            evalDataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=self.numWorkers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self.numWorkers > 0,
        )
        return dataLoader, len(evalDataset.classes)

    def evaluateModel(self, onnxModelPath: str) -> EvaluationResult:
        dataLoader, numClasses = self._buildDataLoader()

        session = self._ort.InferenceSession(onnxModelPath, providers=list(self.providers))
        modelInputName = session.get_inputs()[0].name

        totalLoss = 0.0
        totalTop1Correct = 0
        totalTop5Correct = 0
        totalSamples = 0

        for images, labels in dataLoader:
            logitsOutput = session.run(None, {modelInputName: images.detach().cpu().numpy()})[0]
            logitsTensor = torch.from_numpy(logitsOutput)
            labelsTensor = labels.detach().cpu().long()

            batchLoss = F.cross_entropy(logitsTensor, labelsTensor, reduction="sum")
            totalLoss += float(batchLoss.item())

            top1Pred = logitsTensor.argmax(dim=1)
            totalTop1Correct += int((top1Pred == labelsTensor).sum().item())

            topK = min(5, logitsTensor.shape[1])
            topKPred = logitsTensor.topk(topK, dim=1).indices
            top5Matches = topKPred.eq(labelsTensor.view(-1, 1)).any(dim=1)
            totalTop5Correct += int(top5Matches.sum().item())
            totalSamples += int(labelsTensor.shape[0])

        safeDenominator = max(1, totalSamples)
        return EvaluationResult(
            loss=float(totalLoss / safeDenominator),
            top1=float(100.0 * totalTop1Correct / safeDenominator),
            top5=float(100.0 * totalTop5Correct / safeDenominator),
            numSamples=int(totalSamples),
            numClasses=int(numClasses),
        )


class SubnetCompilationPipeline:
    """Shared subnet quantize/compile/evaluate pipeline used by search and standalone CLI."""

    def __init__(
        self,
        searchSpace: SearchSpaceConfig,
        calibrationImagesDir: str,
        exporterDevice: str = "cpu",
        compilerBinary: str = "imxconv-pt",
        compilerExtraArgs: Optional[List[str]] = None,
    ) -> None:
        self.searchSpace = searchSpace
        self.calibrationImagesDir = calibrationImagesDir
        self.exporterDevice = exporterDevice
        self.compilerBinary = compilerBinary
        self.compilerExtraArgs = list(compilerExtraArgs) if compilerExtraArgs is not None else [
            "--no-input-persistency",
            "--overwrite",
        ]

    @staticmethod
    def _countParameters(model: torch.nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in model.parameters()))

    def quantizeSubnet(
        self,
        supernetModel: ResNetSuperNet,
        architectureConfig: ArchitectureConfig,
        outputOnnxPath: str,
        numCalibrationImages: int = 1,
    ) -> QuantizationResult:
        normalizedConfig = normalizeArchitectureForSearchSpace(
            architectureConfig,
            searchSpace=self.searchSpace,
            enableAuxiliaryHeads=False,
        )

        extractedSubnet = extractSubnetFromSupernet(
            supernetModel,
            normalizedConfig,
            searchSpace=self.searchSpace,
        )
        subnetModel = extractedSubnet.model.to(device="cpu").eval()
        paramCount = self._countParameters(subnetModel)

        representativeDataGenerator = RepresentativeDataGenerator(
            self.calibrationImagesDir,
            input_shape=(3, normalizedConfig.inputResolution, normalizedConfig.inputResolution),
            batch_size=1,
            num_images=numCalibrationImages,
            device="cpu",
        )
        exporter = Imx500Exporter(device=self.exporterDevice)
        _runQuantizationCpuOnly(
            exporter=exporter,
            model=subnetModel,
            representativeDataGenerator=representativeDataGenerator,
            outputOnnxPath=outputOnnxPath,
        )

        return QuantizationResult(
            architectureConfig=normalizedConfig,
            paramCount=paramCount,
            onnxPath=outputOnnxPath,
        )

    def compileQuantizedOnnx(
        self,
        onnxPath: str,
        outputDirectory: str,
        compilerEnvironment: Optional[Dict[str, str]] = None,
    ) -> CompilationResult:
        command = [
            self.compilerBinary,
            "-i",
            onnxPath,
            "-o",
            outputDirectory,
            *self.compilerExtraArgs,
        ]

        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=compilerEnvironment,
            )
        except FileNotFoundError:
            return CompilationResult(
                success=False,
                returnCode=127,
                errorMessage=(
                    f"Compiler binary '{self.compilerBinary}' was not found in PATH. "
                    "Install it or run with skip-compilation mode in standalone engine."
                ),
                stdout="",
                stderr="",
                outputDirectory=outputDirectory,
            )

        success = completed.returncode == 0
        errorMessage = None
        if not success:
            details = completed.stderr if completed.stderr else completed.stdout
            errorMessage = f"IMX500 compilation failed (code {completed.returncode}): {details}"

        return CompilationResult(
            success=success,
            returnCode=int(completed.returncode),
            errorMessage=errorMessage,
            stdout=str(completed.stdout),
            stderr=str(completed.stderr),
            outputDirectory=outputDirectory,
        )

    def quantizeCompileEvaluateArchitecture(
        self,
        supernetModel: ResNetSuperNet,
        architectureConfig: ArchitectureConfig,
        experimentLabel: str,
        compilerEnvironment: Optional[Dict[str, str]] = None,
        evaluationDatasetPath: Optional[str] = None,
        evaluationBatchSize: int = 32,
        evaluationNumWorkers: int = 4,
        skipCompilation: bool = False,
        keepArtifacts: bool = False,
        artifactsRootDir: Optional[str] = None,
    ) -> CandidatePipelineResult:
        if artifactsRootDir is None:
            artifactsDirectory = tempfile.mkdtemp(prefix=f"subnet_eval_{experimentLabel}_")
        else:
            artifactsDirectory = os.path.join(artifactsRootDir, f"subnet_eval_{experimentLabel}")
            os.makedirs(artifactsDirectory, exist_ok=True)

        onnxPath = os.path.join(artifactsDirectory, "quantized_model.onnx")
        compileOutputDirectory = os.path.join(artifactsDirectory, "compiled")

        quantizationResult: Optional[QuantizationResult] = None
        compilationResult: Optional[CompilationResult] = None
        evaluationResult: Optional[EvaluationResult] = None
        try:
            quantizationResult = self.quantizeSubnet(
                supernetModel=supernetModel,
                architectureConfig=architectureConfig,
                outputOnnxPath=onnxPath,
            )

            if skipCompilation:
                compilationResult = CompilationResult(
                    success=True,
                    returnCode=0,
                    errorMessage=None,
                    stdout="Compilation skipped by configuration",
                    stderr="",
                    outputDirectory=compileOutputDirectory,
                )
            else:
                compilationResult = self.compileQuantizedOnnx(
                    onnxPath=quantizationResult.onnxPath,
                    outputDirectory=compileOutputDirectory,
                    compilerEnvironment=compilerEnvironment,
                )

            if compilationResult.success and evaluationDatasetPath is not None and evaluationDatasetPath.strip() != "":
                evaluator = OnnxClassificationEvaluator(
                    datasetPath=evaluationDatasetPath,
                    imageSize=quantizationResult.architectureConfig.inputResolution,
                    batchSize=evaluationBatchSize,
                    numWorkers=evaluationNumWorkers,
                )
                evaluationResult = evaluator.evaluateModel(quantizationResult.onnxPath)

            return CandidatePipelineResult(
                architectureConfig=quantizationResult.architectureConfig,
                paramCount=quantizationResult.paramCount,
                quantizedOnnxPath=quantizationResult.onnxPath,
                compilation=compilationResult,
                evaluation=evaluationResult,
            )
        finally:
            if not keepArtifacts:
                if os.path.isdir(artifactsDirectory):
                    shutil.rmtree(artifactsDirectory, ignore_errors=True)


def parseArchitectureConfig(inputData: Any) -> ArchitectureConfig:
    if isinstance(inputData, ArchitectureConfig):
        return inputData
    if isinstance(inputData, str):
        rawDict = json.loads(inputData)
    else:
        rawDict = dict(inputData)
    return ArchitectureConfig(**rawDict)
