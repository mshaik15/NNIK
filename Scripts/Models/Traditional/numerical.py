import numpy as np
from scipy.optimize import minimize, least_squares
from typing import List, Dict
import sys
from pathlib import Path

try:
    from ...utils import (
        generate_dh_parameters, forward_kinematics, compute_jacobian,
        compute_jacobian_pseudoinverse, dh_transform_matrix
    )
except ImportError:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from Scripts.utils import (
        generate_dh_parameters, forward_kinematics, compute_jacobian,
        compute_jacobian_pseudoinverse, dh_transform_matrix
    )

def _pose_to_transform(pose: np.ndarray) -> np.ndarray:
    x, y, z = pose[:3]
    roll, pitch, yaw = pose[3:] if len(pose) >= 6 else [0, 0, 0]
    
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch) 
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    
    R = np.array([
        [c_y*c_p, c_y*s_p*s_r - s_y*c_r, c_y*s_p*c_r + s_y*s_r],
        [s_y*c_p, s_y*s_p*s_r + c_y*c_r, s_y*s_p*c_r - c_y*s_r],
        [-s_p,    c_p*s_r,              c_p*c_r]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def _pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> np.ndarray:
    """Compute pose error vector"""
    pos_error = T_desired[:3, 3] - T_current[:3, 3]
    
    R_error = T_desired[:3, :3] @ T_current[:3, :3].T
    trace = np.trace(R_error)
    
    if abs(trace - 3) < 1e-6:
        orient_error = np.zeros(3)
    else:
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        if abs(angle) < 1e-6:
            orient_error = np.zeros(3)
        else:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0], 
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
            orient_error = angle * axis
    
    return np.concatenate([pos_error, orient_error])

def jacobian_ik(n_dof: int, poses: np.ndarray, max_iter: int = 100, tol: float = 1e-3) -> np.ndarray:
    poses = np.atleast_2d(poses)
    if poses.shape[1] < 6:
        poses = np.pad(poses, ((0, 0), (0, 6 - poses.shape[1])), 'constant')
    
    # Use existing DH parameter generation
    dh_config = {'config_type': 'standard_serial', 'link_length': 1.0}
    dh_params = generate_dh_parameters(n_dof, dh_config)
    
    joint_solutions = np.zeros((poses.shape[0], n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = _pose_to_transform(pose)
        q = np.random.uniform(0, 2*np.pi, n_dof)
        
        for _ in range(max_iter):
            # Use existing forward kinematics
            fk_result = forward_kinematics(q, dh_params)
            T_current = fk_result['transformation_matrix']
            
            error = _pose_error(T_current, T_desired)
            
            if np.linalg.norm(error) < tol:
                break
            
            # Use existing Jacobian computation
            J = compute_jacobian(q, dh_params)
            
            # Use existing pseudoinverse computation
            J_pinv = compute_jacobian_pseudoinverse(J, damping=1e-4)
            
            q = q + J_pinv @ error
            q = np.mod(q, 2*np.pi)
        
        joint_solutions[i] = q
    
    return joint_solutions[0] if poses.shape[0] == 1 else joint_solutions

def sdls_ik(n_dof: int, poses: np.ndarray, max_iter: int = 100, tol: float = 1e-3) -> np.ndarray:
    poses = np.atleast_2d(poses)
    if poses.shape[1] < 6:
        poses = np.pad(poses, ((0, 0), (0, 6 - poses.shape[1])), 'constant')
    
    # Use existing DH parameter generation
    dh_config = {'config_type': 'standard_serial', 'link_length': 1.0}
    dh_params = generate_dh_parameters(n_dof, dh_config)
    
    joint_solutions = np.zeros((poses.shape[0], n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = _pose_to_transform(pose)
        
        def objective(q):
            # Use existing forward kinematics
            fk_result = forward_kinematics(q, dh_params)
            T_current = fk_result['transformation_matrix']
            error = _pose_error(T_current, T_desired)
            return np.sum(error**2)
        
        q0 = np.random.uniform(0, 2*np.pi, n_dof)
        bounds = [(0, 2*np.pi) for _ in range(n_dof)]
        
        result = minimize(
            objective, q0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': tol}
        )
        
        joint_solutions[i] = result.x if result.success else q0
    
    return joint_solutions[0] if poses.shape[0] == 1 else joint_solutions