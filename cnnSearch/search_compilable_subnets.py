import argparse
from datetime import datetime
import json
import os
import torch
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import traceback
import subprocess
import shutil

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cnnSearch.models.supernet import ResNetSuperNet
from cnnSearch.models.subnet import extractSubnetFromSupernet
from cnnSearch.search_space import (
    sampleRandomArchitecture,
    DEFAULT_SEARCH_SPACE,
    ArchitectureConfig,
    calculateSearchSpaceSize,
    iterateAllArchitectures,
    getSearchSpace,
    normalizeArchitectureForSearchSpace,
    architectureSimilarityScore,
    generateSimilarArchitectures,
)
from cnnSearch.export_utils import Imx500Exporter, RepresentativeDataGenerator

DB_PATH = "compilation_search.json"
VERIFIED_DB_PATH = "compilation_search_verified_candidates.json"
CALIBRATION_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "calibration_images")
# Will be initialized in main based on args
SEARCH_SPACE = DEFAULT_SEARCH_SPACE
SIMILARITY_CANDIDATE_BUDGET = 120
SIMILARITY_COMPILE_BUDGET = 20
SIMILARITY_PER_SEED = 24
SIMILARITY_MAX_MUTATIONS = 2
THRESHOLD_BAND_RATIO = 0.15
PARAM_BYTES_FP32 = 4


def logEvent(eventType, message):
    """Print concise, emoji-like logs for easy progress scanning."""
    icons = {
        "START": "🚀",
        "INFO": "ℹ️",
        "CHECK": "🔍",
        "SUCCESS": "✅",
        "FAILED": "❌",
        "PROGRESS": "📈",
        "SAVE": "💾",
        "WARN": "⚠️",
        "DONE": "🏁",
        "DENSE": "🧪",
    }
    icon = icons.get(eventType, "📌")
    print(f"{icon} [{eventType}] {message}")

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())


def get_param_memory_bytes(paramCount: int) -> int:
    return int(paramCount * PARAM_BYTES_FP32)


def get_param_memory_mib(paramCount: int) -> float:
    return float(get_param_memory_bytes(paramCount)) / float(1024 * 1024)

def get_config_hash(config_dict):
    """Create a unique hash/string for a config dictionary to check for duplicates."""
    return json.dumps(config_dict, sort_keys=True)


def buildTimestampedDbPath() -> str:
    timestampText = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"compilation_search_{timestampText}.json"


def deriveVerifiedDbPath(databasePath: str) -> str:
    if databasePath.lower().endswith(".json"):
        return databasePath[:-5] + "_verified_candidates.json"
    return databasePath + "_verified_candidates.json"


def configureDatabasePaths(databasePathArg: str) -> None:
    global DB_PATH
    global VERIFIED_DB_PATH

    dbPathValue = databasePathArg.strip()
    if dbPathValue == "":
        DB_PATH = buildTimestampedDbPath()
    else:
        DB_PATH = dbPathValue

    VERIFIED_DB_PATH = deriveVerifiedDbPath(DB_PATH)

    dbDirectory = os.path.dirname(DB_PATH)
    if dbDirectory:
        os.makedirs(dbDirectory, exist_ok=True)

    verifiedDirectory = os.path.dirname(VERIFIED_DB_PATH)
    if verifiedDirectory:
        os.makedirs(verifiedDirectory, exist_ok=True)

def load_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_db(data):
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def init_db():
    if not os.path.exists(DB_PATH):
        save_db([])


def getNextExperimentId(experiments: Sequence[Dict[str, Any]]) -> int:
    if not experiments:
        return 1
    return max(int(experiment['id']) for experiment in experiments) + 1


def getSupernetForSearchSpace() -> ResNetSuperNet:
    supernet = ResNetSuperNet(SEARCH_SPACE)
    supernet.eval()
    return supernet


def getNormalizedArchitectureConfig(configData: Any) -> ArchitectureConfig:
    if isinstance(configData, str):
        configDict = json.loads(configData)
    else:
        configDict = dict(configData)

    parsedConfig = ArchitectureConfig(**configDict)
    staticConfig = normalizeArchitectureForSearchSpace(parsedConfig, SEARCH_SPACE, enableAuxiliaryHeads=False)
    return staticConfig


def estimateParamCountForConfig(supernet: ResNetSuperNet, architectureConfig: ArchitectureConfig) -> int:
    subnetData = extractSubnetFromSupernet(supernet, architectureConfig, searchSpace=SEARCH_SPACE)
    subnetData.model.eval()
    return int(get_param_count(subnetData.model))


def ensureParamCountPresent(experiment: Dict[str, Any], supernet: ResNetSuperNet) -> None:
    if experiment.get('param_count') is not None:
        return
    config = getNormalizedArchitectureConfig(experiment['config'])
    experiment['param_count'] = estimateParamCountForConfig(supernet, config)

def populate_candidates(target_count=None):
    experiments = load_db()
    existing_hashes = {get_config_hash(e['config']) for e in experiments}

    supernet = getSupernetForSearchSpace()

    next_id = getNextExperimentId(experiments)

    is_exhaustive = target_count is None

    if is_exhaustive:
        logEvent("START", "Mode EXHAUSTIVE: generating all possible architectures")
        generator = iterateAllArchitectures(SEARCH_SPACE)
    else:
        current_count = len(experiments)
        if current_count >= target_count:
            logEvent("INFO", f"DB has {current_count} candidates, target is {target_count}. Skipping population")
            return
        to_generate = target_count - current_count
        logEvent("START", f"Mode SAMPLING: generating {to_generate} random architectures")
        generator = (sampleRandomArchitecture(SEARCH_SPACE) for _ in range(to_generate * 4))
    
    added_count = 0
    sampled_attempts = 0
    maxSamplingAttempts = 0 if is_exhaustive else max(2000, (target_count or 0) * 20)
    try:
        for config in generator:
            if not is_exhaustive and len(experiments) >= target_count:
                break
            if not is_exhaustive and sampled_attempts >= maxSamplingAttempts:
                logEvent("WARN", f"Sampling attempts reached {maxSamplingAttempts}. Stopping early")
                break

            sampled_attempts += 1

            staticConfig = normalizeArchitectureForSearchSpace(config, SEARCH_SPACE, enableAuxiliaryHeads=False)
            config_dict = staticConfig.toDict()
            config_hash = get_config_hash(config_dict)
            
            if config_hash in existing_hashes:
                continue

            # Extract subnet to count parameters accurately
            subnet_data = extractSubnetFromSupernet(supernet, staticConfig, searchSpace=SEARCH_SPACE)
            subnet_data.model.eval()
            param_count = get_param_count(subnet_data.model)
            
            experiments.append({
                "id": next_id,
                "config": config_dict,
                "param_count": int(param_count),
                "status": "PENDING",
                "error_msg": None,
                "source": "SAMPLED"
            })
            existing_hashes.add(config_hash)
            next_id += 1
            added_count += 1
            
            # Periodically save to allow resume
            if added_count % 100 == 0:
                logEvent("PROGRESS", f"Generated {added_count} new candidates so far")
                save_db(experiments)
                
    except KeyboardInterrupt:
        logEvent("WARN", "Population interrupted by user. Saving current progress")
        save_db(experiments)
        return

    if added_count > 0:
        logEvent("DONE", f"Population finished. Added {added_count} new candidates")
        save_db(experiments)
    else:
        logEvent("INFO", "No new candidates added")

def attempt_compilation(config_data, experiment_id):
    output_onnx = f"temp_quant_{experiment_id}.onnx"
    output_compile_dir = f"temp_compile_{experiment_id}"

    try:
        if isinstance(config_data, str):
            config_dict = json.loads(config_data)
        else:
            config_dict = config_data

        parsedConfig = ArchitectureConfig(**config_dict)
        config = normalizeArchitectureForSearchSpace(parsedConfig, SEARCH_SPACE, enableAuxiliaryHeads=False)
        
        supernet = getSupernetForSearchSpace()
        subnet_data = extractSubnetFromSupernet(supernet, config, searchSpace=SEARCH_SPACE)
        model = subnet_data.model
        model.eval()
        
        # Prepare for export
        calib_gen = RepresentativeDataGenerator(
            CALIBRATION_IMAGES_DIR,
            input_shape=(3, config.inputResolution, config.inputResolution),
            batch_size=1,
            num_images=1, # 1 image as requested
            device='cpu' 
        )
        
        exporter = Imx500Exporter(device='cpu')
        exporter.quantize(model, calib_gen, output_onnx)
        
        # Run compilation using imxconv-pt
        # Command: imxconv-pt -i {onnx_quant_path} -o ./imx500_output --no-input-persistency --overwrite
        cmd = [
            "imxconv-pt",
            "-i", output_onnx,
            "-o", output_compile_dir,
            "--no-input-persistency",
            "--overwrite"
        ]
        
        # Run command and capture output
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        
        if result.returncode != 0:
            error_details = result.stderr if result.stderr else result.stdout
            raise RuntimeError(f"IMX500 Compilation failed (code {result.returncode}): {error_details}")
        
        # Cleanup on success
        if os.path.exists(output_onnx):
            os.remove(output_onnx)
        if os.path.exists(output_compile_dir):
            shutil.rmtree(output_compile_dir)
            
        logEvent("SUCCESS", f"Compilation succeeded for candidate {experiment_id}")
        return "SUCCESS", None

    except Exception as e:
        # Cleanup on failure
        if os.path.exists(output_onnx):
            try:
                os.remove(output_onnx)
            except OSError:
                pass
        if os.path.exists(output_compile_dir):
            try:
                shutil.rmtree(output_compile_dir)
            except OSError:
                pass

        logEvent("FAILED", f"Compilation failed for candidate {experiment_id}: {e}")
        traceback.print_exc()
        return "FAILED", str(e)


def compileExperimentAtIndex(experiments: List[Dict[str, Any]], candidateIndex: int) -> str:
    experiment = experiments[candidateIndex]
    status = str(experiment.get('status', 'PENDING'))
    if status == "PENDING":
        logEvent("CHECK", f"Checking candidate {experiment['id']} (Params: {experiment['param_count']})")
        compiledStatus, errorMessage = attempt_compilation(experiment['config'], experiment['id'])
        experiment['status'] = compiledStatus
        experiment['error_msg'] = errorMessage
        save_db(experiments)
        status = compiledStatus
    return status


def runLargestCompilableBinarySearch(experiments: List[Dict[str, Any]]) -> int:
    low = 0
    high = len(experiments) - 1
    bestSuccessIndex = -1

    logEvent("START", f"Starting largest-compilable binary search over {len(experiments)} candidates")
    while low <= high:
        mid = (low + high) // 2
        status = compileExperimentAtIndex(experiments, mid)
        if status == "SUCCESS":
            bestSuccessIndex = mid
            low = mid + 1
            logEvent("SUCCESS", f"Candidate {experiments[mid]['id']} compilable. Moving higher -> [{low}, {high}]")
        else:
            high = mid - 1
            logEvent("FAILED", f"Candidate {experiments[mid]['id']} not compilable. Moving lower -> [{low}, {high}]")

    return bestSuccessIndex


def runSmallestCompilableBinarySearch(experiments: List[Dict[str, Any]]) -> int:
    low = 0
    high = len(experiments) - 1
    bestSuccessIndex = -1

    logEvent("START", f"Starting smallest-compilable binary search over {len(experiments)} candidates")
    while low <= high:
        mid = (low + high) // 2
        status = compileExperimentAtIndex(experiments, mid)
        if status == "SUCCESS":
            bestSuccessIndex = mid
            high = mid - 1
            logEvent("SUCCESS", f"Candidate {experiments[mid]['id']} compilable. Moving lower -> [{low}, {high}]")
        else:
            low = mid + 1
            logEvent("FAILED", f"Candidate {experiments[mid]['id']} not compilable. Moving higher -> [{low}, {high}]")

    return bestSuccessIndex


def runDenseBoundaryChecks(
    experiments: List[Dict[str, Any]],
    centerIndices: Sequence[int],
    denseWidth: int,
) -> None:
    ranges: List[Tuple[int, int]] = []
    for centerIndex in centerIndices:
        if centerIndex < 0:
            continue
        startIndex = max(0, centerIndex - denseWidth)
        endIndex = min(len(experiments), centerIndex + denseWidth + 1)
        ranges.append((startIndex, endIndex))

    if not ranges:
        logEvent("WARN", "Dense boundary checks skipped (no boundary index found)")
        return

    # Merge overlapping ranges to avoid duplicate checks.
    ranges.sort(key=lambda item: item[0])
    mergedRanges: List[Tuple[int, int]] = []
    for startIndex, endIndex in ranges:
        if not mergedRanges or startIndex > mergedRanges[-1][1]:
            mergedRanges.append((startIndex, endIndex))
            continue
        previousStart, previousEnd = mergedRanges[-1]
        mergedRanges[-1] = (previousStart, max(previousEnd, endIndex))

    for startIndex, endIndex in mergedRanges:
        logEvent("DENSE", f"Checking dense neighborhood [{startIndex}, {endIndex})")
        for candidateIndex in range(startIndex, endIndex):
            if str(experiments[candidateIndex].get('status', 'PENDING')) == "PENDING":
                compileExperimentAtIndex(experiments, candidateIndex)


def getCompilableEnvelope(experiments: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    compilableExperiments = [experiment for experiment in experiments if str(experiment.get('status')) == "SUCCESS"]
    if not compilableExperiments:
        return None

    smallestCompilable = min(compilableExperiments, key=lambda experiment: int(experiment['param_count']))
    largestCompilable = max(compilableExperiments, key=lambda experiment: int(experiment['param_count']))
    return {
        "smallestCompilable": {
            "id": int(smallestCompilable['id']),
            "param_count": int(smallestCompilable['param_count']),
            "param_memory_bytes": get_param_memory_bytes(int(smallestCompilable['param_count'])),
            "param_memory_mib": get_param_memory_mib(int(smallestCompilable['param_count'])),
            "config": smallestCompilable['config'],
        },
        "largestCompilable": {
            "id": int(largestCompilable['id']),
            "param_count": int(largestCompilable['param_count']),
            "param_memory_bytes": get_param_memory_bytes(int(largestCompilable['param_count'])),
            "param_memory_mib": get_param_memory_mib(int(largestCompilable['param_count'])),
            "config": largestCompilable['config'],
        },
        "numCompilable": len(compilableExperiments),
    }


def getSuccessExperiments(experiments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [experiment for experiment in experiments if str(experiment.get('status')) == "SUCCESS"]


def buildLikelyCompilableCandidates(
    experiments: Sequence[Dict[str, Any]],
    envelope: Optional[Dict[str, Any]],
    targetCount: int,
) -> List[Dict[str, Any]]:
    if envelope is None:
        return []

    successExperiments = sorted(getSuccessExperiments(experiments), key=lambda item: int(item['param_count']), reverse=True)
    if not successExperiments:
        return []

    maxCompilableParams = int(envelope['largestCompilable']['param_count'])
    minCompilableParams = int(envelope['smallestCompilable']['param_count'])
    paramRange = max(1, maxCompilableParams - minCompilableParams)

    seedExperiments = successExperiments[: min(8, len(successExperiments))]
    existingConfigHashes = {get_config_hash(dict(experiment['config'])) for experiment in experiments}

    supernet = getSupernetForSearchSpace()
    candidatePool: List[Dict[str, Any]] = []
    localHashes = set()

    successArchitectures = [
        getNormalizedArchitectureConfig(successExperiment['config'])
        for successExperiment in seedExperiments
    ]

    for seedArchitecture in successArchitectures:
        similarArchitectures = generateSimilarArchitectures(
            seedArchitecture,
            searchSpace=SEARCH_SPACE,
            maxCandidates=SIMILARITY_PER_SEED,
            maxMutations=SIMILARITY_MAX_MUTATIONS,
        )

        for similarArchitecture in similarArchitectures:
            candidateConfigDict = similarArchitecture.toDict()
            candidateHash = get_config_hash(candidateConfigDict)
            if candidateHash in existingConfigHashes or candidateHash in localHashes:
                continue

            paramCount = estimateParamCountForConfig(supernet, similarArchitecture)
            if paramCount < minCompilableParams or paramCount > maxCompilableParams:
                continue

            similarityToEnvelope = max(
                architectureSimilarityScore(similarArchitecture, verifiedArchitecture, searchSpace=SEARCH_SPACE)
                for verifiedArchitecture in successArchitectures
            )
            if similarityToEnvelope < 0.65:
                continue

            memoryProximity = 1.0 - (abs(paramCount - maxCompilableParams) / float(paramRange))
            memoryProximity = max(0.0, min(1.0, memoryProximity))
            likelyScore = 0.7 * similarityToEnvelope + 0.3 * memoryProximity

            candidatePool.append({
                "config": candidateConfigDict,
                "param_count": int(paramCount),
                "param_memory_bytes": get_param_memory_bytes(int(paramCount)),
                "param_memory_mib": get_param_memory_mib(int(paramCount)),
                "predicted_similarity": float(similarityToEnvelope),
                "predicted_likely_score": float(likelyScore),
                "status": "PREDICTED_LIKELY",
                "source": "SIMILARITY",
            })
            localHashes.add(candidateHash)

            if len(candidatePool) >= targetCount:
                break
        if len(candidatePool) >= targetCount:
            break

    candidatePool.sort(
        key=lambda candidate: (float(candidate['predicted_likely_score']), int(candidate['param_count'])),
        reverse=True,
    )
    return candidatePool[:targetCount]


def addLikelyCandidatesToDb(experiments: List[Dict[str, Any]], likelyCandidates: Sequence[Dict[str, Any]]) -> List[int]:
    if not likelyCandidates:
        return []

    existingHashes = {get_config_hash(dict(experiment['config'])) for experiment in experiments}
    nextId = getNextExperimentId(experiments)
    addedIds: List[int] = []

    for likelyCandidate in likelyCandidates:
        configDict = dict(likelyCandidate['config'])
        configHash = get_config_hash(configDict)
        if configHash in existingHashes:
            continue

        candidateRecord = {
            "id": nextId,
            "config": configDict,
            "param_count": int(likelyCandidate['param_count']),
            "status": "PENDING",
            "error_msg": None,
            "source": "SIMILARITY",
            "predicted_similarity": float(likelyCandidate['predicted_similarity']),
            "predicted_likely_score": float(likelyCandidate['predicted_likely_score']),
        }
        experiments.append(candidateRecord)
        existingHashes.add(configHash)
        addedIds.append(nextId)
        nextId += 1

    if addedIds:
        save_db(experiments)
    return addedIds


def runSimilarityThresholdChecks(
    experiments: List[Dict[str, Any]],
    envelope: Optional[Dict[str, Any]],
    compileBudget: int,
) -> None:
    if envelope is None or compileBudget <= 0:
        return

    maxCompilableParams = int(envelope['largestCompilable']['param_count'])
    minCompilableParams = int(envelope['smallestCompilable']['param_count'])
    thresholdMin = max(minCompilableParams, int(maxCompilableParams * (1.0 - THRESHOLD_BAND_RATIO)))

    pendingSimilarityCandidates: List[Tuple[int, Dict[str, Any]]] = []
    for candidateIndex, experiment in enumerate(experiments):
        if str(experiment.get('source')) != "SIMILARITY":
            continue
        if str(experiment.get('status', 'PENDING')) != "PENDING":
            continue

        paramCount = int(experiment['param_count'])
        if thresholdMin <= paramCount <= maxCompilableParams:
            pendingSimilarityCandidates.append((candidateIndex, experiment))

    pendingSimilarityCandidates.sort(
        key=lambda item: (float(item[1].get('predicted_likely_score', 0.0)), int(item[1]['param_count'])),
        reverse=True,
    )

    for candidateIndex, experiment in pendingSimilarityCandidates[:compileBudget]:
        logEvent(
            "DENSE",
            f"Threshold check for likely candidate {experiment['id']} ({experiment['param_count']} params)",
        )
        compileExperimentAtIndex(experiments, candidateIndex)


def saveVerifiedCandidatesDb(
    experiments: Sequence[Dict[str, Any]],
    envelope: Optional[Dict[str, Any]],
) -> None:
    verifiedCompilable = []
    likelyCompilable = []

    for experiment in experiments:
        paramCount = int(experiment['param_count']) if experiment.get('param_count') is not None else 0
        record = {
            "id": int(experiment['id']),
            "config": experiment['config'],
            "param_count": paramCount,
            "param_memory_bytes": get_param_memory_bytes(paramCount),
            "param_memory_mib": get_param_memory_mib(paramCount),
            "status": str(experiment.get('status', 'PENDING')),
            "source": str(experiment.get('source', 'SAMPLED')),
        }

        if str(experiment.get('status')) == "SUCCESS":
            verifiedCompilable.append(record)

        if str(experiment.get('source')) == "SIMILARITY":
            record["predicted_similarity"] = float(experiment.get('predicted_similarity', 0.0))
            record["predicted_likely_score"] = float(experiment.get('predicted_likely_score', 0.0))
            likelyCompilable.append(record)

    verifiedCompilable.sort(key=lambda item: item['param_count'])
    likelyCompilable.sort(key=lambda item: item.get('predicted_likely_score', 0.0), reverse=True)

    payload = {
        "source_db": DB_PATH,
        "generated_at": datetime.now().isoformat(),
        "search_space_mode": "complex" if len(SEARCH_SPACE.stagePathOptionsPerStage[0]) > 3 else "simplified",
        "compilable_envelope": envelope,
        "verified_compilable_architectures": verifiedCompilable,
        "likely_compilable_candidates": likelyCompilable,
    }

    with open(VERIFIED_DB_PATH, 'w') as verifiedFile:
        json.dump(payload, verifiedFile, indent=2)

    logEvent("SAVE", f"Saved verified/likely summary to {VERIFIED_DB_PATH}")

def search_loop(args):
    experiments = load_db()

    if not experiments:
        logEvent("WARN", "No experiments found")
        return

    supernet = getSupernetForSearchSpace()
    for experiment in experiments:
        ensureParamCountPresent(experiment, supernet)
    save_db(experiments)

    # Sort by param_count
    experiments.sort(key=lambda x: x['param_count'])

    largestCompilableIndex = runLargestCompilableBinarySearch(experiments)
    smallestCompilableIndex = runSmallestCompilableBinarySearch(experiments)

    runDenseBoundaryChecks(
        experiments,
        centerIndices=[largestCompilableIndex, smallestCompilableIndex],
        denseWidth=50,
    )

    envelope = getCompilableEnvelope(experiments)
    if envelope is not None:
        logEvent(
            "INFO",
            "Compilable envelope: "
            f"smallest={envelope['smallestCompilable']['param_count']} params, "
            f"largest={envelope['largestCompilable']['param_count']} params",
        )
    else:
        logEvent("WARN", "No compilable architecture found yet")

    likelyCandidates = buildLikelyCompilableCandidates(
        experiments,
        envelope=envelope,
        targetCount=SIMILARITY_CANDIDATE_BUDGET,
    )
    if likelyCandidates:
        addedIds = addLikelyCandidatesToDb(experiments, likelyCandidates)
        logEvent("INFO", f"Added {len(addedIds)} similarity-guided candidates to DB")
    else:
        logEvent("INFO", "No similarity-guided candidate generated for current envelope")

    # Refresh ordering after adding similarity candidates.
    experiments.sort(key=lambda item: int(item['param_count']))
    runSimilarityThresholdChecks(
        experiments,
        envelope=envelope,
        compileBudget=SIMILARITY_COMPILE_BUDGET,
    )

    envelope = getCompilableEnvelope(experiments)
    save_db(experiments)
    saveVerifiedCandidatesDb(experiments, envelope)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples to generate. If not set, exhaustive search is performed.")
    parser.add_argument("--enable-complex-paths", action="store_true", help="Enable complex SE and dilated paths (paths 3 and 4) which are disabled by default")
    parser.add_argument("--dv", type=str, default="", help="Database file path to continue from. Leave empty to create compilation_search_<datetime>.json")
    args = parser.parse_args()
    
    global SEARCH_SPACE
    from cnnSearch.search_space import getSearchSpace
    SEARCH_SPACE = getSearchSpace(useComplexPaths=args.enable_complex_paths)
    
    if args.enable_complex_paths:
        logEvent("INFO", "Enabling complex SE and dilated paths (paths 3 and 4)")
    else:
        logEvent("INFO", "Using simplified search space (paths 0, 1, 2 only)")

    configureDatabasePaths(args.dv)
    logEvent("INFO", f"Using DB file: {DB_PATH}")
    logEvent("INFO", f"Verified summary file: {VERIFIED_DB_PATH}")

    init_db()

    # Logging combinatorics
    total_combinations = calculateSearchSpaceSize(SEARCH_SPACE)
    print("\n" + "=" * 60)
    logEvent("INFO", f"Total possible architectures in search space: {total_combinations:,}")
    
    current_db = load_db()
    pending_count = sum(1 for e in current_db if e.get('status') == 'PENDING')
    completed_count = sum(1 for e in current_db if e.get('status') in ['SUCCESS', 'FAILED'])
    logEvent("INFO", f"Existing candidates in DB: {len(current_db):,}")
    logEvent("INFO", f"Pending: {pending_count:,}")
    logEvent("INFO", f"Completed: {completed_count:,}")
    
    if args.num_samples is None:
        logEvent("START", "Mode EXHAUSTIVE SEARCH (checking ALL combinations)")
    else:
        logEvent("START", f"Mode RANDOM SAMPLING (target: {args.num_samples} candidates)")
    print("=" * 60 + "\n")

    try:
        populate_candidates(args.num_samples)
    except Exception as e:
        logEvent("FAILED", f"Error during population: {e}")
        pass

    search_loop(args)

if __name__ == "__main__":
    main()
