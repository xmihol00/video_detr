import argparse
import json
import os
import torch
import sys
from pathlib import Path
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
    getSearchSpace
)
from cnnSearch.export_utils import Imx500Exporter, RepresentativeDataGenerator

DB_PATH = "compilation_search.json"
CALIBRATION_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "calibration_images")
# Will be initialized in main based on args
SEARCH_SPACE = DEFAULT_SEARCH_SPACE

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())

def get_config_hash(config_dict):
    """Create a unique hash/string for a config dictionary to check for duplicates."""
    return json.dumps(config_dict, sort_keys=True)

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

def populate_candidates(target_count=None):
    experiments = load_db()
    existing_hashes = {get_config_hash(e['config']) for e in experiments}
    
    supernet = ResNetSuperNet(SEARCH_SPACE)
    
    # Determine next ID
    next_id = 1
    if experiments:
        next_id = max(e['id'] for e in experiments) + 1

    configs_source = []
    is_exhaustive = target_count is None

    if is_exhaustive:
        print("Mode: EXHAUSTIVE. generating all possible architectures...")
        generator = iterateAllArchitectures(SEARCH_SPACE)
    else:
        current_count = len(experiments)
        if current_count >= target_count:
            print(f"DB has {current_count} candidates, target is {target_count}. Skipping population.")
            return
        to_generate = target_count - current_count
        print(f"Mode: SAMPLING. Generating {to_generate} random architectures...")
        generator = (sampleRandomArchitecture(SEARCH_SPACE) for _ in range(to_generate))

    # Iterate through the generator (either exhaustive or random range)
    # Note: For random sampling, we might hit duplicates, so we might need a while loop logic, 
    # but for simplicity in this structure we just iterate.
    # If we need strictly 'to_generate' NEW items, the generator logic for random needs to be different.
    # But let's stick to a robust loop that adds unique items.
    
    added_count = 0
    try:
        for config in generator:
            # If we are in sampling mode and reached target, stop.
            if not is_exhaustive and len(experiments) >= target_count:
                break

            config_dict = config.toDict()
            config_hash = get_config_hash(config_dict)
            
            if config_hash in existing_hashes:
                continue

            # Extract subnet to count parameters accurately
            subnet_data = extractSubnetFromSupernet(supernet, config)
            param_count = get_param_count(subnet_data.model)
            
            experiments.append({
                "id": next_id,
                "config": config_dict,
                "param_count": int(param_count),
                "status": "PENDING",
                "error_msg": None
            })
            existing_hashes.add(config_hash)
            next_id += 1
            added_count += 1
            
            # Periodically save to allow resume
            if added_count % 100 == 0:
                print(f"Generated {added_count} candidates so far...")
                save_db(experiments)
                
    except KeyboardInterrupt:
        print("Population interrupted by user. Saving current progress...")
        save_db(experiments)
        return

    if added_count > 0:
        print(f"Population finished. Added {added_count} new candidates.")
        save_db(experiments)
    else:
        print("No new candidates added.")

def attempt_compilation(config_data, experiment_id):
    output_onnx = f"temp_quant_{experiment_id}.onnx"
    output_compile_dir = f"temp_compile_{experiment_id}"

    try:
        if isinstance(config_data, str):
            config_dict = json.loads(config_data)
        else:
            config_dict = config_data

        config = ArchitectureConfig(**config_dict)
        
        supernet = ResNetSuperNet(SEARCH_SPACE)
        subnet_data = extractSubnetFromSupernet(supernet, config)
        model = subnet_data.model
        
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

        traceback.print_exc() # Uncomment for debugging
        return "FAILED", str(e)

def search_loop(args):
    experiments = load_db()
    
    if not experiments:
        print("No experiments found.")
        return

    # Sort by param_count
    experiments.sort(key=lambda x: x['param_count'])
    
    # Binary Search Logic on the sorted list
    low = 0
    high = len(experiments) - 1
    
    print(f"Starting binary search on {len(experiments)} ordered candidates...")
    
    mid = 0
    while low <= high:
        mid = (low + high) // 2
        exp = experiments[mid]
        
        status = exp.get('status', 'PENDING')
        
        if status == "PENDING":
             print(f"Checking candidate {exp['id']} (Params: {exp['param_count']})...")
             res_status, error_msg = attempt_compilation(exp['config'], exp['id'])
             
             # Update in-memory and save
             exp['status'] = res_status
             exp['error_msg'] = error_msg
             # Update status for loop logic
             status = res_status
             save_db(experiments)
        
        if status == "SUCCESS":
            # If success, we can try larger models (higher index)
            # The boundary is to the right
            low = mid + 1
            print(f"Candidate {exp['id']} SUCCESS. Moving search higher -> range [{low}, {high}]")
        else:
            # If failed, we need simpler models (lower index)
            # The boundary is to the left
            high = mid - 1
            print(f"Candidate {exp['id']} FAILED. Moving search lower -> range [{low}, {high}]")

    print(f"Binary search converged at index {mid} (approximate boundary).")
    
    # Dense search around boundary
    # Check +/- 50 candidates around the found boundary
    boundary_idx = max(low, 0)
    
    dense_range_width = 50
    start_dense = max(0, boundary_idx - dense_range_width)
    end_dense = min(len(experiments), boundary_idx + dense_range_width)
    
    print(f"Starting dense search in range [{start_dense}, {end_dense}]...")
    
    for idx in range(start_dense, end_dense):
        exp = experiments[idx]
        
        if exp.get('status', 'PENDING') == "PENDING":
             print(f"Dense Check: Candidate {exp['id']} (Params: {exp['param_count']})...")
             res_status, error_msg = attempt_compilation(exp['config'], exp['id'])
             
             exp['status'] = res_status
             exp['error_msg'] = error_msg
             save_db(experiments)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=None, help="Number of random samples to generate. If not set, exhaustive search is performed.")
    parser.add_argument("--enable-complex-paths", action="store_true", help="Enable complex SE and dilated paths (paths 3 and 4) which are disabled by default")
    args = parser.parse_args()
    
    global SEARCH_SPACE
    from cnnSearch.search_space import getSearchSpace
    SEARCH_SPACE = getSearchSpace(useComplexPaths=args.enable_complex_paths)
    
    if args.enable_complex_paths:
        print("Enabling complex SE and dilated paths (paths 3 and 4)")
    else:
        print("Using simplified search space (paths 0, 1, 2 only)")
    
    init_db()

    # Logging combinatorics
    total_combinations = calculateSearchSpaceSize(SEARCH_SPACE)
    print("=" * 60)
    print(f"Total possible architectures in search space: {total_combinations:,}")
    
    current_db = load_db()
    pending_count = sum(1 for e in current_db if e.get('status') == 'PENDING')
    completed_count = sum(1 for e in current_db if e.get('status') in ['SUCCESS', 'FAILED'])
    print(f"Existing candidates in DB: {len(current_db):,}")
    print(f"  - Pending: {pending_count:,}")
    print(f"  - Completed: {completed_count:,}")
    
    if args.num_samples is None:
        print("Mode: EXHAUSTIVE SEARCH (checking ALL combinations)")
    else:
        print(f"Mode: RANDOM SAMPLING (target: {args.num_samples} candidates)")
    print("=" * 60)

    try:
        populate_candidates(args.num_samples)
    except Exception as e:
        print(f"Error during population: {e}")
        # Continue to search loop even if population failed/stopped
        pass

    search_loop(args)

if __name__ == "__main__":
    main()
