import numpy as np
import json
import yaml
from typing import Dict, List, Tuple, Any
import torch
from pathlib import Path

def set_random_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across numpy and PyTorch
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def forward_kinematics(joint_angles: np.ndarray, link_lengths: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    Compute forward kinematics for serial chain manipulator
    Assumes all joints are revolute and alternate between Z and Y axes
    
    Args:
        joint_angles: Shape (batch_size, n_dof) or (n_dof,) in radians
        link_lengths: Length of each link (default: all 1.0)
    
    Returns:
        Dictionary with:
            - 'position': End-effector position [x, y, z]
            - 'rotation_matrix': 3x3 rotation matrix
            - 'euler_angles': [alpha, beta, gamma] in ZYX convention
    """
    # Ensure 2D shape
    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)
    
    batch_size, n_dof = joint_angles.shape
    
    # Default link lengths
    if link_lengths is None:
        link_lengths = np.ones(n_dof)
    
    # Initialize transformation matrices
    positions = np.zeros((batch_size, 3))
    rotations = np.tile(np.eye(3), (batch_size, 1, 1))
    
    for batch_idx in range(batch_size):
        T = np.eye(4)  # Homogeneous transformation matrix
        
        for i, (angle, length) in enumerate(zip(joint_angles[batch_idx], link_lengths)):
            # Alternate rotation axes: Z for even indices, Y for odd
            if i % 2 == 0:  # Z-axis rotation
                R = rotation_matrix_z(angle)
            else:  # Y-axis rotation
                R = rotation_matrix_y(angle)
            
            # Create link transformation
            T_link = np.eye(4)
            T_link[:3, :3] = R
            T_link[:3, 3] = [length, 0, 0]  # Link along X-axis
            
            # Chain multiplication
            T = T @ T_link
        
        positions[batch_idx] = T[:3, 3]
        rotations[batch_idx] = T[:3, :3]
    
    # Convert rotation matrices to Euler angles
    euler_angles = rotation_matrix_to_euler_zyx(rotations)
    
    # Squeeze output if single sample
    if batch_size == 1:
        positions = positions.squeeze(0)
        rotations = rotations.squeeze(0)
        euler_angles = euler_angles.squeeze(0)
    
    return {
        'position': positions,
        'rotation_matrix': rotations,
        'euler_angles': euler_angles
    }

def rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix for Z-axis rotation"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def rotation_matrix_y(angle: float) -> np.ndarray:
    """Rotation matrix for Y-axis rotation"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (ZYX convention)
    
    Args:
        R: Rotation matrix of shape (..., 3, 3)
    
    Returns:
        Euler angles [alpha, beta, gamma] in radians
    """
    # Handle batch dimension
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    batch_size = R.shape[0]
    
    euler = np.zeros((batch_size, 3))
    
    for i in range(batch_size):
        # ZYX Euler angle extraction
        # Beta (Y rotation)
        sin_beta = -R[i, 2, 0]
        cos_beta = np.sqrt(R[i, 0, 0]**2 + R[i, 1, 0]**2)
        beta = np.arctan2(sin_beta, cos_beta)
        
        # Check for gimbal lock
        if np.abs(cos_beta) < 1e-6:
            # Gimbal lock case
            alpha = 0
            gamma = np.arctan2(-R[i, 0, 1], R[i, 1, 1])
        else:
            # Normal case
            alpha = np.arctan2(R[i, 1, 0] / cos_beta, R[i, 0, 0] / cos_beta)
            gamma = np.arctan2(R[i, 2, 1] / cos_beta, R[i, 2, 2] / cos_beta)
        
        euler[i] = [alpha, beta, gamma]
    
    return euler.reshape(*original_shape, 3)

def euler_zyx_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (ZYX convention) to rotation matrix
    
    Args:
        euler: [alpha, beta, gamma] angles in radians
    
    Returns:
        3x3 rotation matrix
    """
    alpha, beta, gamma = euler
    
    # Z rotation (alpha)
    Rz = rotation_matrix_z(alpha)
    # Y rotation (beta)
    Ry = rotation_matrix_y(beta)
    # X rotation (gamma)
    c, s = np.cos(gamma), np.sin(gamma)
    Rx = np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def compute_jacobian(joint_angles: np.ndarray, link_lengths: np.ndarray = None) -> np.ndarray:
    """
    Compute the Jacobian matrix for the manipulator using numerical differentiation
    
    Args:
        joint_angles: Current joint configuration (n_dof,)
        link_lengths: Length of each link
    
    Returns:
        Jacobian matrix of shape (6, n_dof)
    """
    n_dof = len(joint_angles)
    epsilon = 1e-6
    
    # Get current end-effector pose
    fk_result = forward_kinematics(joint_angles, link_lengths)
    current_pose = np.concatenate([
        fk_result['position'],
        fk_result['euler_angles']
    ])
    
    # Initialize Jacobian
    J = np.zeros((6, n_dof))
    
    # Numerical differentiation
    for i in range(n_dof):
        # Perturb joint i
        joint_angles_plus = joint_angles.copy()
        joint_angles_plus[i] += epsilon
        
        # Compute forward kinematics
        fk_plus = forward_kinematics(joint_angles_plus, link_lengths)
        pose_plus = np.concatenate([
            fk_plus['position'],
            fk_plus['euler_angles']
        ])
        
        # Finite difference
        J[:, i] = (pose_plus - current_pose) / epsilon
    
    return J

def save_json(data: Dict[str, Any], filepath: Path):
    """Save data to JSON file with proper numpy array handling"""
    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_numpy(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    converted_data = convert_numpy(data)
    
    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=2)

def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_position_error(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """
    Compute Euclidean distance between predicted and true positions
    
    Args:
        pred_pos: Predicted position [x, y, z]
        true_pos: True position [x, y, z]
    
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(pred_pos - true_pos)

def compute_orientation_error(pred_euler: np.ndarray, true_euler: np.ndarray) -> float:
    """
    Compute orientation error using geodesic distance on SO(3)
    Simplified version using Euler angle difference
    
    Args:
        pred_euler: Predicted Euler angles [alpha, beta, gamma]
        true_euler: True Euler angles [alpha, beta, gamma]
    
    Returns:
        Orientation error in radians
    """
    # Convert to rotation matrices
    R_pred = euler_zyx_to_rotation_matrix(pred_euler)
    R_true = euler_zyx_to_rotation_matrix(true_euler)
    
    # Compute relative rotation
    R_error = R_true.T @ R_pred
    
    # Extract angle from axis-angle representation
    # angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_error)
    trace = np.clip(trace, -1, 3)  # Numerical stability
    angle = np.arccos((trace - 1) / 2)
    
    return angle

def compute_joint_error(pred_joints: np.ndarray, true_joints: np.ndarray) -> float:
    return np.mean((pred_joints - true_joints) ** 2)