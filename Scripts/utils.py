import numpy as np
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional
import torch
from pathlib import Path

def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def dh_transform_matrix(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    
    return np.array([
        [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
        [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0,        s_alpha,            c_alpha,           d],
        [0,        0,                  0,                 1]
    ])

def generate_dh_parameters(n_dof: int, config: Dict[str, Any]) -> List[Dict[str, float]]:
    config_type = config.get('config_type', 'standard_serial')
    link_length = config.get('link_length', 1.0)
    
    if config_type == 'standard_serial':
        dh_params = []
        
        for i in range(n_dof):
            # Alternate between 0 and pi/2
            if i % 2 == 0:
                alpha = 0.0         # No twist - stay in plane
            else:
                alpha = np.pi/2     # 90° twist - change to perpendicular plane
            
            dh_params.append({
                'd': 0.0,           # No joint offset
                'a': link_length,   # Standardized link length
                'alpha': alpha      # Alternating twist
            })
        
        return dh_params
    
    else:
        raise ValueError(f"Unknown DH config type: {config_type}")

def forward_kinematics(joint_angles: np.ndarray, dh_params: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    if joint_angles.ndim == 1:
        joint_angles = joint_angles.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size, n_dof = joint_angles.shape
    
    if len(dh_params) != n_dof:
        raise ValueError(f"DH parameters length ({len(dh_params)}) must match DOF ({n_dof})")
    
    # Initialize output arrays
    positions = np.zeros((batch_size, 3))
    rotations = np.zeros((batch_size, 3, 3))
    transformations = np.zeros((batch_size, 4, 4)) if not squeeze_output else None
    
    for batch_idx in range(batch_size):
        # Start with identity transformation
        T = np.eye(4)
        
        # Chain DH transformations
        for i in range(n_dof):
            theta = joint_angles[batch_idx, i]
            dh = dh_params[i]
            
            # Compute DH transformation for this joint
            T_i = dh_transform_matrix(dh['d'], dh['a'], dh['alpha'], theta)
            
            # Chain multiply
            T = T @ T_i
        
        # Extract position and orientation
        positions[batch_idx] = T[:3, 3]
        rotations[batch_idx] = T[:3, :3]
        
        if transformations is not None:
            transformations[batch_idx] = T
    
    # Convert rotation matrices to Euler angles (XYZ convention for consistency)
    euler_angles = rotation_matrix_to_euler_xyz(rotations)
    
    # Prepare output
    result = {
        'position': positions,
        'rotation_matrix': rotations,
        'euler_angles': euler_angles
    }
    
    # Squeeze output if single sample
    if squeeze_output:
        result['position'] = positions.squeeze(0)
        result['rotation_matrix'] = rotations.squeeze(0)
        result['euler_angles'] = euler_angles.squeeze(0)
        result['transformation_matrix'] = T
    else:
        result['transformation_matrices'] = transformations
    
    return result

def rotation_matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    original_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    batch_size = R.shape[0]
    
    euler = np.zeros((batch_size, 3))
    
    for i in range(batch_size):
        # XYZ Euler angle extraction
        # Pitch (Y rotation) - clamp to avoid numerical issues
        sin_pitch = np.clip(-R[i, 2, 0], -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)
        
        # Check for gimbal lock
        if np.abs(np.cos(pitch)) < 1e-6:
            # Gimbal lock case - set roll to 0 and solve for yaw
            roll = 0.0
            yaw = np.arctan2(-R[i, 0, 1], R[i, 1, 1])
        else:
            # Normal case
            roll = np.arctan2(R[i, 2, 1], R[i, 2, 2])
            yaw = np.arctan2(R[i, 1, 0], R[i, 0, 0])
        
        euler[i] = [roll, pitch, yaw]
    
    return euler.reshape(*original_shape, 3)

def euler_xyz_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = euler
    
    # Individual rotation matrices
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    
    # X rotation (roll)
    Rx = np.array([
        [1, 0,   0  ],
        [0, c_r, -s_r],
        [0, s_r,  c_r]
    ])
    
    # Y rotation (pitch)
    Ry = np.array([
        [c_p,  0, s_p],
        [0,    1, 0  ],
        [-s_p, 0, c_p]
    ])
    
    # Z rotation (yaw)
    Rz = np.array([
        [c_y, -s_y, 0],
        [s_y,  c_y, 0],
        [0,    0,   1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx

def compute_jacobian(joint_angles: np.ndarray, dh_params: List[Dict[str, float]]) -> np.ndarray:
    n_dof = len(joint_angles)
    J = np.zeros((6, n_dof))
    
    transforms = []
    T = np.eye(4)
    transforms.append(T.copy())
    
    for i in range(n_dof):
        theta = joint_angles[i]
        dh = dh_params[i]
        T_i = dh_transform_matrix(dh['d'], dh['a'], dh['alpha'], theta)
        T = T @ T_i
        transforms.append(T.copy())
    
    # End-effector position
    p_end = transforms[-1][:3, 3]
    
    # Compute Jacobian columns
    for i in range(n_dof):
        # Z-axis of joint i (in base frame)
        z_i = transforms[i][:3, 2]
        
        # Position of joint i (in base frame)
        p_i = transforms[i][:3, 3]
        
        # For revolute joint
        # Linear velocity contribution: z_i × (p_end - p_i)
        J[:3, i] = np.cross(z_i, p_end - p_i)
        
        # Angular velocity contribution: z_i
        J[3:, i] = z_i
    
    return J

def compute_jacobian_pseudoinverse(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    n_dof = J.shape[1]
    
    if n_dof >= 6:
        # Overdetermined or square system - use left pseudoinverse
        # J+ = (J^T J + λI)^(-1) J^T
        JTJ = J.T @ J
        J_pinv = np.linalg.inv(JTJ + damping * np.eye(n_dof)) @ J.T
    else:
        # Underdetermined system - use right pseudoinverse  
        # J+ = J^T (J J^T + λI)^(-1)
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + damping * np.eye(6))
    
    return J_pinv

def detect_singularities(J: np.ndarray, threshold: float = 1e-3) -> Dict[str, Any]:
    U, s, Vt = np.linalg.svd(J)
    
    # Check for singularities
    min_sv = np.min(s)
    condition_number = np.max(s) / (min_sv + 1e-12)
    
    is_singular = min_sv < threshold
    
    return {
        'is_singular': is_singular,
        'min_singular_value': min_sv,
        'condition_number': condition_number,
        'singular_values': s,
        'rank': np.sum(s > threshold)
    }

def save_json(data: Dict[str, Any], filepath: Path):
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
    return np.linalg.norm(pred_pos - true_pos)

def compute_orientation_error(pred_euler: np.ndarray, true_euler: np.ndarray) -> float:
    R_pred = euler_xyz_to_rotation_matrix(pred_euler)
    R_true = euler_xyz_to_rotation_matrix(true_euler)
    
    # Compute relative rotation
    R_error = R_true.T @ R_pred
    
    # Extract angle from axis-angle representation
    # angle = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_error)
    trace = np.clip(trace, -1, 3)  # Numerical stability
    angle = np.arccos((trace - 1) / 2)
    
    return angle

def compute_joint_error(pred_joints: np.ndarray, true_joints: np.ndarray) -> float:
    return np.sqrt(np.mean((pred_joints - true_joints) ** 2))

def wrap_joint_angles(angles: np.ndarray) -> np.ndarray:
    return np.mod(angles, 2 * np.pi)

def pose_distance(pose1: np.ndarray, pose2: np.ndarray, 
                  position_weight: float = 1.0, orientation_weight: float = 1.0) -> float:
    pos_error = compute_position_error(pose1[:3], pose2[:3])
    orient_error = compute_orientation_error(pose1[3:], pose2[3:])
    
    return position_weight * pos_error + orientation_weight * orient_error

def check_joint_limits(joint_angles: np.ndarray, limits: Tuple[float, float]) -> np.ndarray:
    min_angle, max_angle = limits
    return (joint_angles >= min_angle) & (joint_angles <= max_angle)

def compute_manipulability(J: np.ndarray) -> float:
    if J.shape[0] == J.shape[1]:
        return np.abs(np.linalg.det(J))
    else:
        return np.sqrt(np.linalg.det(J @ J.T))

def random_joint_configuration(n_dof: int, limits: Tuple[float, float], 
                              seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    min_angle, max_angle = limits
    return np.random.uniform(min_angle, max_angle, size=n_dof)

def validate_dh_parameters(dh_params: List[Dict[str, float]]) -> bool:
    required_keys = {'d', 'a', 'alpha'}
    
    for i, params in enumerate(dh_params):
        if not isinstance(params, dict):
            raise ValueError(f"DH parameter {i} must be a dictionary")
        
        if not required_keys.issubset(params.keys()):
            missing = required_keys - params.keys()
            raise ValueError(f"DH parameter {i} missing keys: {missing}")
        
        # Check for numeric values
        for key in required_keys:
            if not isinstance(params[key], (int, float, np.number)):
                raise ValueError(f"DH parameter {i}['{key}'] must be numeric")
    
    return True

def workspace_analysis(dh_params: List[Dict[str, float]], 
                      n_samples: int = 1000, 
                      joint_limits: Tuple[float, float] = (0, 2*np.pi)) -> Dict[str, Any]:
    n_dof = len(dh_params)
    
    # Generate random joint configurations
    joint_configs = np.random.uniform(
        joint_limits[0], joint_limits[1], 
        size=(n_samples, n_dof)
    )
    
    # Compute forward kinematics for all samples
    positions = []
    for joints in joint_configs:
        fk_result = forward_kinematics(joints, dh_params)
        positions.append(fk_result['position'])
    
    positions = np.array(positions)
    
    # Compute workspace statistics
    workspace_stats = {
        'center': np.mean(positions, axis=0),
        'std': np.std(positions, axis=0),
        'min_bounds': np.min(positions, axis=0),
        'max_bounds': np.max(positions, axis=0),
        'volume_estimate': np.prod(np.max(positions, axis=0) - np.min(positions, axis=0)),
        'reach': np.max(np.linalg.norm(positions, axis=1)),
        'n_samples': n_samples
    }
    
    return workspace_stats