import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import sys
import multiprocessing as mp
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try relative import first (when used as module), fallback to absolute import
try:
    from .utils import (
        forward_kinematics,
        generate_dh_parameters,
        save_json,
        load_config,
        set_random_seed
    )
except ImportError:
    # Fallback for direct script execution
    from Scripts.utils import (
        forward_kinematics,
        generate_dh_parameters,
        save_json,
        load_config,
        set_random_seed
    )

def compute_sample_batch(args: Tuple[np.ndarray, List[Dict[str, float]], int]) -> List[Dict[str, Any]]:
    joint_angles_batch, dh_params, start_id = args
    
    batch_results = []
    for i, joint_angles in enumerate(joint_angles_batch):
        sample_id = start_id + i
        
        # Compute forward kinematics using DH parameters
        fk_result = forward_kinematics(joint_angles, dh_params)
        
        # Create pose vector [x, y, z, roll, pitch, yaw]
        pose = np.concatenate([
            fk_result['position'],
            fk_result['euler_angles']
        ])
        
        batch_results.append({
            'id': sample_id,
            'pose': pose.tolist(),
            'joint_angles': joint_angles.tolist()
        })
    
    return batch_results

def generate_dataset(
    dof: int, 
    samples: int, config: Dict[str, Any], seed: int = None, multiprocessing: bool = True) -> Dict[str, Any]:
    if seed is not None:
        np.random.seed(seed)
        set_random_seed(seed)
    
    # Generate DH parameters for this DOF configuration
    dh_params = generate_dh_parameters(dof, config['dh_parameters'])
    
    # Get joint angle limits
    angle_limits = config['angle_limits']
    
    # Generate random joint configurations
    joint_angles = np.random.uniform(
        angle_limits[0],
        angle_limits[1],
        size=(samples, dof)
    )
    
    print(f"Generating {samples} samples for DOF={dof}...")
    print(f"DH Parameters: {len(dh_params)} joints with link length {config['dh_parameters']['link_length']}")
    
    # Get multiprocessing settings
    mp_config = config.get('multiprocessing', {'enabled': True, 'max': 2, 'batch_size': 50})
    batch_size = mp_config.get('batch_size', 50)
    
    # Process samples
    if multiprocessing and mp_config.get('enabled', True) and samples > batch_size:
        # Use multiprocessing for larger datasets
        poses_data, solutions_data = _generate_with_multiprocessing(
            joint_angles, dh_params, batch_size, mp_config.get('max', 2)
        )
    else:
        # Single-threaded processing
        poses_data, solutions_data = _generate_single_threaded(
            joint_angles, dh_params
        )
    
    # Create comprehensive metadata
    metadata = {
        'dof': dof,
        'samples': samples,
        'dh_parameters': dh_params,
        'joint_angle_limits': list(angle_limits),
        'link_length': config['dh_parameters']['link_length'],
        'units': {
            'angles': 'radians',
            'position': 'meters',
            'orientation': 'radians (XYZ Euler - Roll, Pitch, Yaw)'
        },
        'description': f'{dof}-DOF serial manipulator with unit link lengths and alternating twist angles',
        'data_format': {
            'pose_structure': '[x, y, z, roll, pitch, yaw]',
            'joint_angles_structure': f'[theta_1, theta_2, ..., theta_{dof}]',
            'id_mapping': 'Unique ID maps poses to corresponding joint angles',
            'coordinate_convention': 'XYZ Euler angles (Roll-Pitch-Yaw)'
        },
        'kinematics_info': {
            'dh_convention': 'Standard DH parameters',
            'transformation_chain': 'DH transforms chained from base to end-effector',
            'twist_pattern': 'Alternating 0° and 90° link twists for 3D workspace'
        },
        'generation_info': {
            'seed': seed,
            'multiprocessing_used': multiprocessing and mp_config.get('enabled', True) and samples > batch_size,
            'batch_size': batch_size if multiprocessing else samples
        }
    }
    
    return {
        'poses': poses_data,
        'solutions': solutions_data,
        'metadata': metadata
    }

def _generate_with_multiprocessing(joint_angles: np.ndarray, dh_params: List[Dict[str, float]], batch_size: int, max: int) -> Tuple[List[Dict], List[Dict]]:
    
    samples = len(joint_angles)
    
    # Split data into batches
    batches = []
    for i in range(0, samples, batch_size):
        end_idx = min(i + batch_size, samples)
        batch_angles = joint_angles[i:end_idx]
        batches.append((batch_angles, dh_params, i))
    
    print(f"Processing {len(batches)} batches using {max} workers...")
    
    # Process batches in parallel
    with mp.Pool(processes=max) as pool:
        batch_results = list(tqdm(
            pool.imap(compute_sample_batch, batches),
            total=len(batches),
            desc="Computing FK with DH"
        ))
    
    # Flatten results and separate poses from solutions
    poses_data = []
    solutions_data = []
    
    for batch_result in batch_results:
        for sample in batch_result:
            poses_data.append({
                'id': sample['id'],
                'pose': sample['pose']
            })
            solutions_data.append({
                'id': sample['id'],
                'joint_angles': sample['joint_angles']
            })
    
    return poses_data, solutions_data

def _generate_single_threaded(joint_angles: np.ndarray, dh_params: List[Dict[str, float]]) -> Tuple[List[Dict], List[Dict]]:
    """Generate samples using single-threaded processing"""
    
    poses_data = []
    solutions_data = []
    
    for i, angles in enumerate(tqdm(joint_angles, desc="Computing FK with DH")):
        # Compute forward kinematics using DH parameters
        fk_result = forward_kinematics(angles, dh_params)
        
        # Create pose vector [x, y, z, roll, pitch, yaw]
        pose = np.concatenate([
            fk_result['position'],
            fk_result['euler_angles']
        ]).tolist()
        
        poses_data.append({
            'id': i,
            'pose': pose
        })
        
        solutions_data.append({
            'id': i,
            'joint_angles': angles.tolist()
        })
    
    return poses_data, solutions_data

def save_dataset_files(dataset: Dict[str, Any], base_filename: str, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save poses file
    poses_file = output_dir / f"{base_filename}.json"
    poses_data = {
        'data': dataset['poses'],
        'metadata': dataset['metadata']
    }
    save_json(poses_data, poses_file)
    print(f"✓ Saved poses to {poses_file}")
    
    # Save solutions file
    solutions_file = output_dir / f"{base_filename}_solutions.json"
    solutions_data = {
        'data': dataset['solutions'],
        'metadata': dataset['metadata']
    }
    save_json(solutions_data, solutions_file)
    print(f"✓ Saved solutions to {solutions_file}")

def generate_all_datasets(config: Dict[str, Any], project_root: Path, multiprocessing: bool = True):

    # Create data directories
    train_dir = project_root / config['data_dir'] / 'Training'
    test_dir = project_root / config['data_dir'] / 'Testing'
    
    # Set global random seed
    set_random_seed(config['seed'])
    
    # Generate datasets for each DOF
    dof_range = range(config['dof_range'][0], config['dof_range'][1] + 1)
    
    print(f"Dataset Generation Configuration (DH Parameters):")
    print(f"  DOF range: {config['dof_range'][0]} to {config['dof_range'][1]}")
    print(f"  Training samples per DOF: {config['train_samples']}")
    print(f"  Testing samples per DOF: {config['test_samples']}")
    print(f"  Link length (standardized): {config['dh_parameters']['link_length']}")
    print(f"  DH config type: {config['dh_parameters']['config_type']}")
    print(f"  Multiprocessing: {multiprocessing}")
    print(f"  Max workers: {config.get('multiprocessing', {}).get('max', 2)}")
    print(f"  Random seed: {config['seed']}")
    
    for dof in dof_range:
        print(f"\n{'='*60}")
        print(f"Generating data for DOF={dof}")
        print('='*60)
        
        # Training dataset
        print("Generating training data")
        train_dataset = generate_dataset(
            dof=dof,
            samples=config['train_samples'],
            config=config,
            seed=config['seed'] + dof,
            multiprocessing=multiprocessing
        )
        
        # Save training dataset
        save_dataset_files(
            train_dataset, 
            f"{dof}_training", 
            train_dir
        )
        
        # Testing dataset
        print("\nGenerating testing data")
        test_dataset = generate_dataset(
            dof=dof,
            samples=config['test_samples'],
            config=config,
            seed=config['seed'] + dof + 1000,
            multiprocessing=multiprocessing
        )
        
        # Save testing dataset
        save_dataset_files(
            test_dataset, 
            f"{dof}_testing", 
            test_dir
        )
        
        print(f"✓ Completed DOF={dof} data")
    
    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"Training files: {train_dir}")
    print(f"Testing files: {test_dir}")
    print(f"Total DOFs processed: {len(dof_range)}")
    print(f"Using DH parameters with standardized link lengths")

def load_dataset(poses_file: Path, solutions_file: Path) -> Dict[str, Any]:
    with open(poses_file, 'r') as f:
        poses_data = json.load(f)
    
    with open(solutions_file, 'r') as f:
        solutions_data = json.load(f)
    
    # Verify ID alignment
    pose_ids = {sample['id'] for sample in poses_data['data']}
    solution_ids = {sample['id'] for sample in solutions_data['data']}
    
    if pose_ids != solution_ids:
        raise ValueError("Pose and solution IDs do not match!")
    
    return {
        'poses': poses_data['data'],
        'solutions': solutions_data['data'],
        'metadata': poses_data['metadata']
    }

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description='Generate IK datasets using DH parameters')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--project_root',
        type=str,
        default='.',
        help='Project root directory'
    )
    parser.add_argument(
        '--no-multiprocessing',
        action='store_true',
        help='Disable multiprocessing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    project_root = Path(args.project_root)
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Generate datasets
    multiprocessing = not args.no_multiprocessing
    generate_all_datasets(config, project_root, multiprocessing)

if __name__ == '__main__':
    main()