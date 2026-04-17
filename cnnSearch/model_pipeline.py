from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
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
    preQuantizationEvaluation: Optional[EvaluationResult]
    evaluationDelta: Optional[Dict[str, float]]


def selectOnnxExecutionProviders(
    availableProviders: Sequence[str],
    requestedProviders: Optional[Sequence[str]] = None,
    preferCuda: bool = True,
) -> List[str]:
    availableSet = set(str(provider) for provider in availableProviders)

    if requestedProviders is not None:
        return [str(provider) for provider in requestedProviders if str(provider) in availableSet]

    providerPreference: List[str] = []
    if preferCuda:
        providerPreference.append("CUDAExecutionProvider")
    providerPreference.append("CPUExecutionProvider")

    selectedProviders = [provider for provider in providerPreference if provider in availableSet]
    if selectedProviders:
        return selectedProviders

    if availableProviders:
        return [str(provider) for provider in availableProviders]
    return ["CPUExecutionProvider"]


def getOrtSessionOptionsFromMctQuantizers() -> Any:
    """Return ORT session options registering MCT custom ops when available."""
    try:
        from mct_quantizers import get_ort_session_options as getOrtSessionOptions  # type: ignore
    except Exception:
        return None

    try:
        return getOrtSessionOptions()
    except Exception:
        return None


def _logPipeline(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[pipeline {timestamp}] {message}")


def _buildEvaluationDelta(
    preQuantizationEvaluation: Optional[EvaluationResult],
    quantizedEvaluation: Optional[EvaluationResult],
) -> Optional[Dict[str, float]]:
    if preQuantizationEvaluation is None or quantizedEvaluation is None:
        return None
    return {
        "top1_drop": float(preQuantizationEvaluation.top1 - quantizedEvaluation.top1),
        "top5_drop": float(preQuantizationEvaluation.top5 - quantizedEvaluation.top5),
        "loss_delta": float(quantizedEvaluation.loss - preQuantizationEvaluation.loss),
    }


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
    supernetModel.load_state_dict(modelStateDict, strict=False)
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
        preferCuda: bool = True,
        requireCuda: bool = False,
        maxImages: Optional[int] = None,
        logEveryBatches: int = 20,
        progressPrefix: str = "",
        enableProgressLogging: bool = True,
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as importError:
            raise ImportError("onnxruntime is required for ONNX model evaluation") from importError

        self._ort = ort
        self._sessionOptions = getOrtSessionOptionsFromMctQuantizers()
        self.datasetPath = datasetPath
        self.imageSize = int(imageSize)
        self.batchSize = int(batchSize)
        self.numWorkers = int(numWorkers)
        self.preferCuda = bool(preferCuda)
        self.requireCuda = bool(requireCuda)
        self.maxImages = int(maxImages) if maxImages is not None and int(maxImages) > 0 else None
        self.logEveryBatches = max(1, int(logEveryBatches))
        self.progressPrefix = progressPrefix.strip()
        self.enableProgressLogging = bool(enableProgressLogging)

        self.availableProviders = [str(provider) for provider in self._ort.get_available_providers()]
        self.providers = selectOnnxExecutionProviders(
            availableProviders=self.availableProviders,
            requestedProviders=providers,
            preferCuda=self.preferCuda,
        )

        if self.requireCuda and "CUDAExecutionProvider" not in self.providers:
            raise RuntimeError(
                "CUDA execution provider is required but unavailable. "
                f"Available ORT providers: {self.availableProviders}. "
                "Install GPU-enabled ONNX Runtime (e.g. onnxruntime-gpu) or disable --require-cuda."
            )

    def _buildDataLoader(self) -> tuple[DataLoader, int, int]:
        fullDataset = ImageFolder(self.datasetPath, transform=buildEvalTransform(self.imageSize))
        numClasses = len(fullDataset.classes)
        evalDataset = fullDataset
        if self.maxImages is not None and self.maxImages < len(evalDataset):
            evalDataset = Subset(evalDataset, list(range(self.maxImages)))

        dataLoader = DataLoader(
            evalDataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=self.numWorkers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=self.numWorkers > 0,
        )
        return dataLoader, numClasses, len(evalDataset)

    def _progressTag(self) -> str:
        return f"[{self.progressPrefix}] " if self.progressPrefix != "" else ""

    def evaluateModel(self, onnxModelPath: str) -> EvaluationResult:
        dataLoader, numClasses, totalImages = self._buildDataLoader()
        totalBatches = len(dataLoader)

        try:
            session = self._ort.InferenceSession(
                onnxModelPath,
                sess_options=self._sessionOptions,
                providers=list(self.providers),
            )
        except Exception as sessionError:
            errorText = str(sessionError)
            usesMctCustomOps = (
                "ActivationPOTQuantizer" in errorText
                or "mct_quantizers" in errorText
                or "is not a registered function/op" in errorText
            )
            if usesMctCustomOps:
                raise RuntimeError(
                    "Failed to load quantized ONNX in ONNXRuntime due to missing MCT custom ops registration. "
                    "Install/import `mct_quantizers` in this environment and ensure "
                    "`mct_quantizers.get_ort_session_options()` is available."
                ) from sessionError
            raise

        modelInputName = session.get_inputs()[0].name

        if self.enableProgressLogging:
            _logPipeline(
                f"{self._progressTag()}ONNX evaluation start: {totalImages} images, "
                f"{totalBatches} batches, batchSize={self.batchSize}, requestedProviders={self.providers}, "
                f"availableProviders={self.availableProviders}"
            )

        totalLoss = 0.0
        totalTop1Correct = 0
        totalTop5Correct = 0
        totalSamples = 0
        import time
        wallStart = time.time()

        for batchIndex, (images, labels) in enumerate(dataLoader, start=1):
            try:
                logitsOutput = session.run(None, {modelInputName: images.detach().cpu().numpy()})[0]
            except KeyboardInterrupt as interruptError:
                raise KeyboardInterrupt(
                    "Interrupted during ONNXRuntime execution. "
                    "For faster diagnostics, rerun with smaller `--max-eval-images` or `--max-candidates`."
                ) from interruptError
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

            if self.enableProgressLogging and (batchIndex % self.logEveryBatches == 0 or batchIndex == totalBatches):
                elapsed = max(1e-6, time.time() - wallStart)
                imagesPerSecond = float(totalSamples) / elapsed
                _logPipeline(
                    f"{self._progressTag()}ONNX evaluation progress: batch {batchIndex}/{totalBatches}, "
                    f"processed={totalSamples}/{totalImages} images, throughput={imagesPerSecond:.2f} img/s"
                )

        safeDenominator = max(1, totalSamples)
        result = EvaluationResult(
            loss=float(totalLoss / safeDenominator),
            top1=float(100.0 * totalTop1Correct / safeDenominator),
            top5=float(100.0 * totalTop5Correct / safeDenominator),
            numSamples=int(totalSamples),
            numClasses=int(numClasses),
        )
        if self.enableProgressLogging:
            elapsed = max(1e-6, time.time() - wallStart)
            activeProviders = session.get_providers()
            _logPipeline(
                f"{self._progressTag()}ONNX evaluation done: top1={result.top1:.3f}, "
                f"top5={result.top5:.3f}, loss={result.loss:.5f}, elapsed={elapsed:.2f}s, "
                f"activeProviders={activeProviders}"
            )
        return result


class PytorchClassificationEvaluator:
    """Evaluate a PyTorch classification model on an ImageFolder dataset."""

    def __init__(
        self,
        datasetPath: str,
        imageSize: int,
        batchSize: int = 32,
        numWorkers: int = 4,
        maxImages: Optional[int] = None,
        logEveryBatches: int = 20,
        progressPrefix: str = "",
        enableProgressLogging: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        self.datasetPath = datasetPath
        self.imageSize = int(imageSize)
        self.batchSize = int(batchSize)
        self.numWorkers = int(numWorkers)
        self.maxImages = int(maxImages) if maxImages is not None and int(maxImages) > 0 else None
        self.logEveryBatches = max(1, int(logEveryBatches))
        self.progressPrefix = progressPrefix.strip()
        self.enableProgressLogging = bool(enableProgressLogging)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _progressTag(self) -> str:
        return f"[{self.progressPrefix}] " if self.progressPrefix != "" else ""

    def _buildDataLoader(self) -> tuple[DataLoader, int, int]:
        fullDataset = ImageFolder(self.datasetPath, transform=buildEvalTransform(self.imageSize))
        numClasses = len(fullDataset.classes)
        evalDataset = fullDataset
        if self.maxImages is not None and self.maxImages < len(evalDataset):
            evalDataset = Subset(evalDataset, list(range(self.maxImages)))

        dataLoader = DataLoader(
            evalDataset,
            batch_size=self.batchSize,
            shuffle=False,
            num_workers=self.numWorkers,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            persistent_workers=self.numWorkers > 0,
        )
        return dataLoader, numClasses, len(evalDataset)

    def evaluateModel(self, model: torch.nn.Module) -> EvaluationResult:
        import time

        model = model.to(self.device)
        model.eval()
        dataLoader, numClasses, totalImages = self._buildDataLoader()
        totalBatches = len(dataLoader)

        if self.enableProgressLogging:
            _logPipeline(
                f"{self._progressTag()}Float evaluation start: {totalImages} images, "
                f"{totalBatches} batches, batchSize={self.batchSize}, device={self.device}"
            )

        totalLoss = 0.0
        totalTop1Correct = 0
        totalTop5Correct = 0
        totalSamples = 0
        wallStart = time.time()

        with torch.no_grad():
            for batchIndex, (images, labels) in enumerate(dataLoader, start=1):
                images = images.to(self.device, non_blocking=self.device.type == "cuda")
                labels = labels.to(self.device, non_blocking=self.device.type == "cuda")

                logitsTensor = model(images)

                batchLoss = F.cross_entropy(logitsTensor, labels, reduction="sum")
                totalLoss += float(batchLoss.item())

                top1Pred = logitsTensor.argmax(dim=1)
                totalTop1Correct += int((top1Pred == labels).sum().item())

                topK = min(5, logitsTensor.shape[1])
                topKPred = logitsTensor.topk(topK, dim=1).indices
                top5Matches = topKPred.eq(labels.view(-1, 1)).any(dim=1)
                totalTop5Correct += int(top5Matches.sum().item())
                totalSamples += int(labels.shape[0])

                if self.enableProgressLogging and (batchIndex % self.logEveryBatches == 0 or batchIndex == totalBatches):
                    elapsed = max(1e-6, time.time() - wallStart)
                    imagesPerSecond = float(totalSamples) / elapsed
                    _logPipeline(
                        f"{self._progressTag()}Float evaluation progress: batch {batchIndex}/{totalBatches}, "
                        f"processed={totalSamples}/{totalImages} images, throughput={imagesPerSecond:.2f} img/s"
                    )

        safeDenominator = max(1, totalSamples)
        result = EvaluationResult(
            loss=float(totalLoss / safeDenominator),
            top1=float(100.0 * totalTop1Correct / safeDenominator),
            top5=float(100.0 * totalTop5Correct / safeDenominator),
            numSamples=int(totalSamples),
            numClasses=int(numClasses),
        )
        if self.enableProgressLogging:
            elapsed = max(1e-6, time.time() - wallStart)
            _logPipeline(
                f"{self._progressTag()}Float evaluation done: top1={result.top1:.3f}, "
                f"top5={result.top5:.3f}, loss={result.loss:.5f}, elapsed={elapsed:.2f}s"
            )
        return result


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
        evaluationRequireCuda: bool = False,
        evaluationMaxImages: Optional[int] = None,
        evaluationLogEveryBatches: int = 20,
        enableProgressLogging: bool = True,
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
        preQuantizationEvaluation: Optional[EvaluationResult] = None
        evaluationResult: Optional[EvaluationResult] = None
        try:
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

            if evaluationDatasetPath is not None and evaluationDatasetPath.strip() != "":
                floatEvaluator = PytorchClassificationEvaluator(
                    datasetPath=evaluationDatasetPath,
                    imageSize=normalizedConfig.inputResolution,
                    batchSize=evaluationBatchSize,
                    numWorkers=evaluationNumWorkers,
                    maxImages=evaluationMaxImages,
                    logEveryBatches=evaluationLogEveryBatches,
                    progressPrefix=experimentLabel,
                    enableProgressLogging=enableProgressLogging,
                )
                preQuantizationEvaluation = floatEvaluator.evaluateModel(subnetModel)

            if enableProgressLogging:
                _logPipeline(f"[{experimentLabel}] Quantization started")
            representativeDataGenerator = RepresentativeDataGenerator(
                self.calibrationImagesDir,
                input_shape=(3, normalizedConfig.inputResolution, normalizedConfig.inputResolution),
                batch_size=1,
                num_images=1,
                device="cpu",
            )
            exporter = Imx500Exporter(device=self.exporterDevice)
            _runQuantizationCpuOnly(
                exporter=exporter,
                model=subnetModel,
                representativeDataGenerator=representativeDataGenerator,
                outputOnnxPath=onnxPath,
            )
            quantizationResult = QuantizationResult(
                architectureConfig=normalizedConfig,
                paramCount=paramCount,
                onnxPath=onnxPath,
            )
            if enableProgressLogging:
                _logPipeline(
                    f"[{experimentLabel}] Quantization finished: params={quantizationResult.paramCount}, "
                    f"input={quantizationResult.architectureConfig.inputResolution}"
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
                if enableProgressLogging:
                    _logPipeline(f"[{experimentLabel}] Compilation skipped by configuration")
            else:
                if enableProgressLogging:
                    _logPipeline(f"[{experimentLabel}] Compilation started")
                compilationResult = self.compileQuantizedOnnx(
                    onnxPath=quantizationResult.onnxPath,
                    outputDirectory=compileOutputDirectory,
                    compilerEnvironment=compilerEnvironment,
                )
                if enableProgressLogging:
                    compileStatus = "SUCCESS" if compilationResult.success else "FAILED"
                    _logPipeline(f"[{experimentLabel}] Compilation {compileStatus}")

            if compilationResult.success and evaluationDatasetPath is not None and evaluationDatasetPath.strip() != "":
                evaluator = OnnxClassificationEvaluator(
                    datasetPath=evaluationDatasetPath,
                    imageSize=quantizationResult.architectureConfig.inputResolution,
                    batchSize=evaluationBatchSize,
                    numWorkers=evaluationNumWorkers,
                    preferCuda=True,
                    requireCuda=evaluationRequireCuda,
                    maxImages=evaluationMaxImages,
                    logEveryBatches=evaluationLogEveryBatches,
                    progressPrefix=experimentLabel,
                    enableProgressLogging=enableProgressLogging,
                )
                evaluationResult = evaluator.evaluateModel(quantizationResult.onnxPath)

            evaluationDelta = _buildEvaluationDelta(preQuantizationEvaluation, evaluationResult)
            if enableProgressLogging and evaluationDelta is not None:
                assert preQuantizationEvaluation is not None
                assert evaluationResult is not None
                _logPipeline(
                    f"[{experimentLabel}] Accuracy summary: "
                    f"preQ_top1={preQuantizationEvaluation.top1:.3f}, "
                    f"postQ_top1={evaluationResult.top1:.3f}, "
                    f"top1_drop={evaluationDelta['top1_drop']:.3f}"
                )

            return CandidatePipelineResult(
                architectureConfig=quantizationResult.architectureConfig,
                paramCount=quantizationResult.paramCount,
                quantizedOnnxPath=quantizationResult.onnxPath,
                compilation=compilationResult,
                evaluation=evaluationResult,
                preQuantizationEvaluation=preQuantizationEvaluation,
                evaluationDelta=evaluationDelta,
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
