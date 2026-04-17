from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from cnnSearch.model_pipeline import (
    CandidatePipelineResult,
    OnnxClassificationEvaluator,
    SubnetCompilationPipeline,
    buildSearchSpaceForCheckpoint,
    loadSupernetFromCheckpoint,
    parseArchitectureConfig,
)
from cnnSearch.search_space import ArchitectureConfig, getSearchSpace


CALIBRATION_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "calibration_images")

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

def _formatFloat(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _configHash(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


def loadCandidateArchitecturesFromCompilationJson(compilationJsonPath: str) -> List[Dict[str, Any]]:
    with open(compilationJsonPath, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    candidateSources = []
    if isinstance(payload, dict):
        candidateSources.extend(payload.get("verified_compilable_architectures", []))
        candidateSources.extend(payload.get("likely_compilable_candidates", []))

    deduplicated: List[Dict[str, Any]] = []
    seenHashes = set()
    for candidate in candidateSources:
        configDict = dict(candidate["config"])
        configKey = _configHash(configDict)
        if configKey in seenHashes:
            continue

        deduplicated.append(
            {
                "id": candidate.get("id"),
                "source": candidate.get("source", "UNKNOWN"),
                "config": configDict,
            }
        )
        seenHashes.add(configKey)

    return deduplicated


def _printResultSummary(results: Sequence[Dict[str, Any]]) -> None:
    successCount = sum(1 for result in results if bool(result.get("compile_success")))
    evaluatedCount = sum(1 for result in results if result.get("evaluation") is not None)

    print("\n" + "=" * 96)
    print("Evaluation summary")
    print("=" * 96)
    print(f"Candidates: {len(results)} | Compiled: {successCount} | Evaluated: {evaluatedCount}")
    print("-" * 96)
    print(f"{'ID':>8} | {'Source':>10} | {'Compiled':>8} | {'PreTop1':>8} | {'PostTop1':>9} | {'Drop':>7}")
    print("-" * 96)

    for result in results:
        evaluation = result.get("evaluation")
        preQuantizationEvaluation = result.get("pre_quantization_evaluation")
        evaluationDelta = result.get("evaluation_delta")
        preTop1 = _formatFloat(preQuantizationEvaluation.get("top1") if preQuantizationEvaluation is not None else None)
        postTop1 = _formatFloat(evaluation.get("top1") if evaluation is not None else None)
        top1Drop = _formatFloat(evaluationDelta.get("top1_drop") if evaluationDelta is not None else None)
        candidateId = str(result.get("id", "-"))
        source = str(result.get("source", "-") or "-")
        compiledFlag = "yes" if bool(result.get("compile_success")) else "no"

        print(f"{candidateId:>8} | {source:>10} | {compiledFlag:>8} | {preTop1:>8} | {postTop1:>9} | {top1Drop:>7}")

    print("=" * 96)


def _logEngine(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[engine {timestamp}] {message}")


def _candidateResultToRecord(
    candidateId: Optional[int],
    source: str,
    pipelineResult: CandidatePipelineResult,
) -> Dict[str, Any]:
    evaluationPayload = None
    preQuantizationEvaluationPayload = None
    evaluationDeltaPayload = None
    if pipelineResult.evaluation is not None:
        evaluationPayload = {
            "loss": float(pipelineResult.evaluation.loss),
            "top1": float(pipelineResult.evaluation.top1),
            "top5": float(pipelineResult.evaluation.top5),
            "num_samples": int(pipelineResult.evaluation.numSamples),
            "num_classes": int(pipelineResult.evaluation.numClasses),
        }
    if pipelineResult.preQuantizationEvaluation is not None:
        preQuantizationEvaluationPayload = {
            "loss": float(pipelineResult.preQuantizationEvaluation.loss),
            "top1": float(pipelineResult.preQuantizationEvaluation.top1),
            "top5": float(pipelineResult.preQuantizationEvaluation.top5),
            "num_samples": int(pipelineResult.preQuantizationEvaluation.numSamples),
            "num_classes": int(pipelineResult.preQuantizationEvaluation.numClasses),
        }
    if pipelineResult.evaluationDelta is not None:
        evaluationDeltaPayload = dict(pipelineResult.evaluationDelta)

    return {
        "id": candidateId,
        "source": source,
        "config": pipelineResult.architectureConfig.toDict(),
        "param_count": int(pipelineResult.paramCount),
        "compile_success": bool(pipelineResult.compilation.success),
        "compile_return_code": int(pipelineResult.compilation.returnCode),
        "compile_error": pipelineResult.compilation.errorMessage,
        "pre_quantization_evaluation": preQuantizationEvaluationPayload,
        "evaluation": evaluationPayload,
        "evaluation_delta": evaluationDeltaPayload,
    }


def _resolveSearchSpaceMode(
    cliEnableComplexPaths: bool,
    explicitArchitectures: Sequence[ArchitectureConfig],
) -> bool:
    if cliEnableComplexPaths:
        return True

    for architectureConfig in explicitArchitectures:
        if any(int(pathIndex) > 2 for pathIndex in architectureConfig.stagePathIndices):
            return True
    return False


def parseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser("cnnSearch model evaluation engine")

    parser.add_argument(
        "--quantized-onnx-path",
        type=str,
        default="quantized_model.onnx",
        help="Path to pre-quantized ONNX model for direct evaluation",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="ImageFolder dataset root with one level of class subdirectories",
    )
    parser.add_argument(
        "--supernet-path",
        type=str,
        default="",
        help="Supernet checkpoint path; overrides --quantized-onnx-path workflows",
    )
    parser.add_argument(
        "--compilation-json",
        type=str,
        default="",
        help="Compilation-search JSON with architecture candidates to evaluate",
    )
    parser.add_argument(
        "--architecture-json",
        type=str,
        default="",
        help="Single architecture JSON path when using --supernet-path without --compilation-json",
    )
    parser.add_argument(
        "--enable-complex-paths",
        action="store_true",
        help="Use complex search space (path indices 0..4)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--artifacts-root", type=str, default="")
    parser.add_argument("--max-candidates", type=int, default=0, help="Evaluate only first N candidates (0 = all)")
    parser.add_argument("--max-eval-images", type=int, default=0, help="Evaluate only first N dataset images per candidate (0 = all)")
    parser.add_argument("--log-every-batches", type=int, default=20, help="Print ONNX evaluation progress every N batches")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDAExecutionProvider is unavailable in ONNX Runtime",
    )
    parser.add_argument(
        "--skip-compilation",
        action="store_true",
        help="Skip IMX compilation step and evaluate quantized ONNX directly",
    )
    return parser.parse_args()


def _evaluateDirectOnnx(args: argparse.Namespace) -> Dict[str, Any]:
    _logEngine("Starting direct ONNX evaluation mode")
    evaluator = OnnxClassificationEvaluator(
        datasetPath=args.dataset_path,
        imageSize=224,
        batchSize=args.batch_size,
        numWorkers=args.num_workers,
        preferCuda=True,
        requireCuda=bool(args.require_cuda),
        maxImages=(args.max_eval_images if int(args.max_eval_images) > 0 else None),
        logEveryBatches=int(args.log_every_batches),
        progressPrefix="onnx_only",
        enableProgressLogging=True,
    )
    evaluationResult = evaluator.evaluateModel(args.quantized_onnx_path)
    return {
        "mode": "onnx_only",
        "quantized_onnx_path": args.quantized_onnx_path,
        "evaluation": {
            "loss": float(evaluationResult.loss),
            "top1": float(evaluationResult.top1),
            "top5": float(evaluationResult.top5),
            "num_samples": int(evaluationResult.numSamples),
            "num_classes": int(evaluationResult.numClasses),
        },
    }


def _loadSingleArchitecture(path: str) -> ArchitectureConfig:
    with open(path, "r", encoding="utf-8") as handle:
        configData = json.load(handle)
    return parseArchitectureConfig(configData)


def _evaluateSupernetArchitectures(args: argparse.Namespace) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    architectureConfigsForMode: List[ArchitectureConfig] = []

    if args.compilation_json.strip() != "":
        candidates = loadCandidateArchitecturesFromCompilationJson(args.compilation_json)
        architectureConfigsForMode = [parseArchitectureConfig(candidate["config"]) for candidate in candidates]
    elif args.architecture_json.strip() != "":
        architectureConfig = _loadSingleArchitecture(args.architecture_json)
        architectureConfigsForMode = [architectureConfig]
        candidates = [{"id": 1, "source": "SINGLE", "config": architectureConfig.toDict()}]
    else:
        raise ValueError("When --supernet-path is provided, also specify --compilation-json or --architecture-json")

    originalCandidateCount = len(candidates)
    if int(args.max_candidates) > 0:
        candidates = candidates[: int(args.max_candidates)]
    _logEngine(
        f"Prepared {len(candidates)} candidates (source had {originalCandidateCount}). "
        f"skip_compilation={bool(args.skip_compilation)}, max_eval_images={int(args.max_eval_images)}, "
        f"require_cuda={bool(args.require_cuda)}"
    )

    useComplexPaths = _resolveSearchSpaceMode(
        cliEnableComplexPaths=bool(args.enable_complex_paths),
        explicitArchitectures=architectureConfigsForMode,
    )
    searchSpace = buildSearchSpaceForCheckpoint(args.supernet_path, useComplexPaths=useComplexPaths)
    supernetModel = loadSupernetFromCheckpoint(args.supernet_path, searchSpace=searchSpace)

    artifactsRoot = args.artifacts_root.strip() or None
    pipeline = SubnetCompilationPipeline(
        searchSpace=searchSpace,
        calibrationImagesDir=CALIBRATION_IMAGES_DIR,
    )

    perCandidateResults: List[Dict[str, Any]] = []
    interrupted = False
    for candidateIndex, candidate in enumerate(candidates, start=1):
        architectureConfig = parseArchitectureConfig(candidate["config"])
        candidateId = int(candidate.get("id", 0)) if candidate.get("id") is not None else None
        source = str(candidate.get("source", "UNKNOWN"))
        candidateTag = f"{candidateIndex}/{len(candidates)} id={candidateId if candidateId is not None else '-'}"

        _logEngine(f"Candidate {candidateTag} started (source={source})")
        candidateStart = time.time()

        try:
            pipelineResult = pipeline.quantizeCompileEvaluateArchitecture(
                supernetModel=supernetModel,
                architectureConfig=architectureConfig,
                experimentLabel=f"cand_{candidateId if candidateId is not None else len(perCandidateResults)+1}",
                evaluationDatasetPath=args.dataset_path,
                evaluationBatchSize=args.batch_size,
                evaluationNumWorkers=args.num_workers,
                evaluationRequireCuda=bool(args.require_cuda),
                evaluationMaxImages=(args.max_eval_images if int(args.max_eval_images) > 0 else None),
                evaluationLogEveryBatches=int(args.log_every_batches),
                enableProgressLogging=True,
                skipCompilation=bool(args.skip_compilation),
                keepArtifacts=bool(args.keep_artifacts),
                artifactsRootDir=artifactsRoot,
            )
        except KeyboardInterrupt:
            interrupted = True
            _logEngine(f"Interrupted during candidate {candidateTag}. Returning partial results")
            break

        perCandidateResults.append(
            _candidateResultToRecord(
                candidateId=candidateId,
                source=source,
                pipelineResult=pipelineResult,
            )
        )
        elapsed = max(1e-6, time.time() - candidateStart)
        _logEngine(f"Candidate {candidateTag} finished in {elapsed:.2f}s")

    _printResultSummary(perCandidateResults)

    return {
        "mode": "supernet_quantize_compile_evaluate",
        "supernet_path": args.supernet_path,
        "dataset_path": args.dataset_path,
        "use_complex_paths": bool(useComplexPaths),
        "skip_compilation": bool(args.skip_compilation),
        "require_cuda": bool(args.require_cuda),
        "interrupted": bool(interrupted),
        "results": perCandidateResults,
    }


def main() -> None:
    args = parseArguments()

    if args.supernet_path.strip() == "":
        payload = _evaluateDirectOnnx(args)
    else:
        payload = _evaluateSupernetArchitectures(args)

    payload["generated_at"] = datetime.now().isoformat()

    print(json.dumps(payload, indent=2))

    if args.output_json.strip() != "":
        outputDirectory = os.path.dirname(args.output_json)
        if outputDirectory:
            os.makedirs(outputDirectory, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
