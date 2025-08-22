import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
import sys
import multiprocessing as mp
from functools import partial
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Try relative import first (when used as module), fallback to absolute import
try:
    from .utils import (
        forward_kinematics,
        save_json,
        load_config,
        set_random_seed
    )
except ImportError:
    # Fallback for direct script execution
    from Scripts.utils import (
        forward_kinematics,
        save_json,
        load_config,
        set_random_seed
    )

def compute_sample_batch(args: Tuple[np.ndarray, np.ndarray, int, int]) -> List[Dict[str, Any]]:
    """
    Compute forward kinematics for a batch of samples
    
    Args:
        args: Tuple containing (joint_angles_batch, link_lengths, start_id, batch_size)
    
    Returns:
        List of dictionaries containing sample data
    """
    joint_angles_batch, link_lengths, start_id, batch_size = args
    
    batch_results = []
    for i, joint_angles in enumerate(joint_angles_batch):
        sample_id = start_id + i
        
        # Compute forward kinematics
        fk_result = forward_kinematics(joint_angles, link_lengths)
        
        # Create pose vector [x, y, z, alpha, beta, gamma]
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
    n_dof: int, 
    n_samples: int, 
    seed: int = None,
    use_multiprocessing: bool = True,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Generate IK dataset for a specific DOF configuration with SQL-like structure
    
    Args:
        n_dof: Number of degrees of freedom
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        use_multiprocessing: Whether to use multiprocessing for FK computation
        batch_size: Size of batches for multiprocessing
    
    Returns:
        Dictionary containing:
            - poses: List of pose dictionaries with id and pose
            - solutions: List of solution dictionaries with id and joint_angles
            - metadata: Dataset information
    """
    if seed is not None:
        np.random.seed(seed)
        set_random_seed(seed)
    
    # Hardcoded parameters for IK project
    link_lengths = np.ones(n_dof)  # All links have length 1.0
    angle_limits = (0, 2*np.pi)    # Revolute joints: 0 to 2π radians
    
    # Generate random joint configurations
    joint_angles = np.random.uniform(
        angle_limits[0],
        angle_limits[1],
        size=(n_samples, n_dof)
    )
    
    print(f"Generating {n_samples} samples for DOF={n_dof}...")
    
    # Process samples
    if use_multiprocessing and n_samples > batch_size:
        # Use multiprocessing for larger datasets
        poses_data, solutions_data = _generate_with_multiprocessing(
            joint_angles, link_lengths, batch_size
        )
    else:
        # Single-threaded processing
        poses_data, solutions_data = _generate_single_threaded(
            joint_angles, link_lengths
        )
    
    # Create metadata
    metadata = {
        'n_dof': n_dof,
        'n_samples': n_samples,
        'link_lengths': link_lengths.tolist(),
        'angle_limits': list(angle_limits),
        'units': {
            'angles': 'radians',
            'position': 'meters',
            'orientation': 'radians (ZYX Euler)'
        },
        'description': f'{n_dof}-DOF serial manipulator with unit link lengths',
        'data_format': {
            'pose_structure': '[x, y, z, alpha, beta, gamma]',
            'joint_angles_structure': f'[theta_1, theta_2, ..., theta_{n_dof}]',
            'id_mapping': 'Unique ID maps poses to corresponding joint angles'
        },
        'generation_info': {
            'seed': seed,
            'multiprocessing_used': use_multiprocessing and n_samples > batch_size
        }
    }
    
    return {
        'poses': poses_data,
        'solutions': solutions_data,
        'metadata': metadata
    }

def _generate_with_multiprocessing(
    joint_angles: np.ndarray, 
    link_lengths: np.ndarray, 
    batch_size: int
) -> Tuple[List[Dict], List[Dict]]:
    """Generate samples using multiprocessing"""
    
    n_samples = len(joint_angles)
    n_cores = min(mp.cpu_count(), 4)  # Limit cores for laptop compatibility
    
    # Split data into batches
    batches = []
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_angles = joint_angles[i:end_idx]
        batches.append((batch_angles, link_lengths, i, len(batch_angles)))
    
    print(f"Processing {len(batches)} batches using {n_cores} cores...")
    
    # Process batches in parallel
    with mp.Pool(processes=n_cores) as pool:
        batch_results = list(tqdm(
            pool.imap(compute_sample_batch, batches),
            total=len(batches),
            desc="Computing FK"
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

def _generate_single_threaded(
    joint_angles: np.ndarray, 
    link_lengths: np.ndarray
) -> Tuple[List[Dict], List[Dict]]:
    """Generate samples using single-threaded processing"""
    
    poses_data = []
    solutions_data = []
    
    for i, angles in enumerate(tqdm(joint_angles, desc="Computing FK")):
        # Compute forward kinematics
        fk_result = forward_kinematics(angles, link_lengths)
        
        # Create pose vector [x, y, z, alpha, beta, gamma]
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

def save_dataset_files(
    dataset: Dict[str, Any], 
    base_filename: str, 
    output_dir: Path
):
    """
    Save dataset as separate pose and solution JSON files
    
    Args:
        dataset: Dataset dictionary from generate_dataset
        base_filename: Base filename (e.g., "2_training")
        output_dir: Output directory path
    """
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

def generate_all_datasets(
    config: Dict[str, Any], 
    project_root: Path,
    use_multiprocessing: bool = True
):
    """
    Generate training and testing datasets for all DOF configurations
    
    Args:
        config: Configuration dictionary
        project_root: Root directory of the project
        use_multiprocessing: Whether to use multiprocessing
    """
    # Create data directories
    train_dir = project_root / config['data_dir'] / 'Training'
    test_dir = project_root / config['data_dir'] / 'Testing'
    
    # Set global random seed
    set_random_seed(config['seed'])
    
    # Generate datasets for each DOF
    dof_range = range(config['dof_range'][0], config['dof_range'][1] + 1)
    
    print(f"Dataset Generation Configuration:")
    print(f"  DOF range: {config['dof_range'][0]} to {config['dof_range'][1]}")
    print(f"  Training samples per DOF: {config['train_samples']}")
    print(f"  Testing samples per DOF: {config['test_samples']}")
    print(f"  Multiprocessing: {use_multiprocessing}")
    print(f"  Random seed: {config['seed']}")
    
    for n_dof in dof_range:
        print(f"\n{'='*60}")
        print(f"Generating datasets for DOF={n_dof}")
        print('='*60)
        
        # Training dataset
        print("Generating training dataset...")
        train_dataset = generate_dataset(
            n_dof=n_dof,
            n_samples=config['train_samples'],
            seed=config['seed'] + n_dof,
            use_multiprocessing=use_multiprocessing
        )
        
        # Save training dataset
        save_dataset_files(
            train_dataset, 
            f"{n_dof}_training", 
            train_dir
        )
        
        # Testing dataset
        print("\nGenerating testing dataset...")
        test_dataset = generate_dataset(
            n_dof=n_dof,
            n_samples=config['test_samples'],
            seed=config['seed'] + n_dof + 1000,
            use_multiprocessing=use_multiprocessing
        )
        
        # Save testing dataset
        save_dataset_files(
            test_dataset, 
            f"{n_dof}_testing", 
            test_dir
        )
        
        print(f"✓ Completed DOF={n_dof} datasets")
    
    print(f"\n{'='*60}")
    print("🎉 Dataset generation complete!")
    print(f"📁 Training files: {train_dir}")
    print(f"📁 Testing files: {test_dir}")
    print(f"📊 Total DOFs processed: {len(dof_range)}")

def load_dataset(poses_file: Path, solutions_file: Path) -> Dict[str, Any]:
    """
    Load a complete dataset from pose and solution files
    
    Args:
        poses_file: Path to poses JSON file
        solutions_file: Path to solutions JSON file
    
    Returns:
        Dictionary containing combined dataset
    """
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
    parser = argparse.ArgumentParser(description='Generate IK datasets with SQL-like structure')
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
        help='Disable multiprocessing (use single thread)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    project_root = Path(args.project_root)
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Generate datasets
    use_multiprocessing = not args.no_multiprocessing
    generate_all_datasets(config, project_root, use_multiprocessing)

if __name__ == '__main__':
    main()