import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent))

from .utils import (
    forward_kinematics,
    save_json,
    load_config,
    set_random_seed
)

def generate_dataset(
    n_dof: int,
    n_samples: int,
    angle_limits: tuple = (0, 2*np.pi),
    link_lengths: np.ndarray = None,
    seed: int = None
) -> Dict[str, Any]:
    """
    Generate IK dataset for a specific DOF configuration
    
    Args:
        n_dof: Number of degrees of freedom
        n_samples: Number of samples to generate
        angle_limits: Min and max joint angle limits
        link_lengths: Length of each link (default: all 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing:
            - joint_angles: (n_samples, n_dof) array
            - poses: (n_samples, 6) array [x, y, z, alpha, beta, gamma]
            - metadata: Dataset information
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default link lengths
    if link_lengths is None:
        link_lengths = np.ones(n_dof)
    
    # Generate random joint configurations
    joint_angles = np.random.uniform(
        angle_limits[0],
        angle_limits[1],
        size=(n_samples, n_dof)
    )
    
    # Compute forward kinematics for all samples
    poses = []
    positions = []
    orientations = []
    
    print(f"Generating {n_samples} samples for DOF={n_dof}...")
    for i in tqdm(range(n_samples)):
        fk_result = forward_kinematics(joint_angles[i], link_lengths)
        
        # Combine position and orientation into 6D pose
        pose = np.concatenate([
            fk_result['position'],
            fk_result['euler_angles']
        ])
        poses.append(pose)
        positions.append(fk_result['position'])
        orientations.append(fk_result['euler_angles'])
    
    poses = np.array(poses)
    positions = np.array(positions)
    orientations = np.array(orientations)
    
    # Create dataset dictionary
    dataset = {
        'joint_angles': joint_angles,
        'poses': poses,
        'positions': positions,
        'orientations': orientations,
        'metadata': {
            'n_dof': n_dof,
            'n_samples': n_samples,
            'link_lengths': link_lengths.tolist(),
            'angle_limits': list(angle_limits),
            'units': {
                'angles': 'radians',
                'position': 'meters',
                'orientation': 'radians (ZYX Euler)'
            },
            'description': f'{n_dof}-DOF serial manipulator with unit link lengths'
        }
    }
    
    return dataset

def generate_all_datasets(config: Dict[str, Any], project_root: Path):
    """
    Generate training and testing datasets for all DOF configurations
    
    Args:
        config: Configuration dictionary
        project_root: Root directory of the project
    """
    # Create data directories
    train_dir = project_root / config['data_dir'] / 'Training'
    test_dir = project_root / config['data_dir'] / 'Testing'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    set_random_seed(config['seed'])
    
    # Generate datasets for each DOF
    dof_range = range(config['dof_range'][0], config['dof_range'][1] + 1)
    
    for n_dof in dof_range:
        print(f"\n{'='*50}")
        print(f"Generating datasets for DOF={n_dof}")
        print('='*50)
        
        # Training dataset
        train_dataset = generate_dataset(
            n_dof=n_dof,
            n_samples=config['train_samples'],
            angle_limits=tuple(config['angle_limits']),
            seed=config['seed'] + n_dof  # Different seed for each DOF
        )
        
        # Save training dataset
        train_file = train_dir / f'DOF{n_dof}.json'
        save_json(train_dataset, train_file)
        print(f"✓ Saved training dataset to {train_file}")
        
        # Testing dataset
        test_dataset = generate_dataset(
            n_dof=n_dof,
            n_samples=config['test_samples'],
            angle_limits=tuple(config['angle_limits']),
            seed=config['seed'] + n_dof + 1000  # Different seed for test
        )
        
        # Save testing dataset
        test_file = test_dir / f'DOF{n_dof}.json'
        save_json(test_dataset, test_file)
        print(f"✓ Saved testing dataset to {test_file}")
    
    print(f"\n{'='*50}")
    print("Dataset generation complete!")
    print(f"Training samples per DOF: {config['train_samples']}")
    print(f"Testing samples per DOF: {config['test_samples']}")
    print(f"DOF range: {config['dof_range'][0]} to {config['dof_range'][1]}")

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description='Generate IK datasets')
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
    
    args = parser.parse_args()
    
    # Load configuration
    project_root = Path(args.project_root)
    config_path = project_root / args.config
    config = load_config(config_path)
    
    # Generate datasets
    generate_all_datasets(config, project_root)

if __name__ == '__main__':
    main()