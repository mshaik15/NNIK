import numpy as np
from scipy.optimize import minimize, least_squares
from typing import List, Dict

def _create_robot_dh(n_dof: int) -> List[Dict[str, float]]:
    dh_params = []
    for i in range(n_dof):
        alpha = 0.0 if i % 2 == 0 else np.pi/2
        dh_params.append({
            'd': 0.0, 'a': 1.0, 'alpha': alpha
        })
    return dh_params

def _dh_transform(d: float, a: float, alpha: float, theta: float) -> np.ndarray:
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)
    
    return np.array([
        [c_theta, -s_theta * c_alpha,  s_theta * s_alpha, a * c_theta],
        [s_theta,  c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
        [0,        s_alpha,            c_alpha,           d],
        [0,        0,                  0,                 1]
    ])

def _forward_kinematics(joint_angles: np.ndarray, dh_params: List[Dict[str, float]]) -> np.ndarray:
    """Forward kinematics"""
    T = np.eye(4)
    for i, angle in enumerate(joint_angles):
        if i < len(dh_params):
            dh = dh_params[i]
            T = T @ _dh_transform(dh['d'], dh['a'], dh['alpha'], angle)
    return T

def _compute_jacobian(joint_angles: np.ndarray, dh_params: List[Dict[str, float]]) -> np.ndarray:
    """Compute Jacobian matrix"""
    n_dof = len(joint_angles)
    J = np.zeros((6, n_dof))
    
    transforms = [np.eye(4)]
    T = np.eye(4)
    
    for i in range(n_dof):
        dh = dh_params[i]
        T = T @ _dh_transform(dh['d'], dh['a'], dh['alpha'], joint_angles[i])
        transforms.append(T.copy())
    
    p_end = transforms[-1][:3, 3]
    
    for i in range(n_dof):
        z_i = transforms[i][:3, 2]
        p_i = transforms[i][:3, 3]
        J[:3, i] = np.cross(z_i, p_end - p_i)
        J[3:, i] = z_i
    
    return J

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
    
    dh_params = _create_robot_dh(n_dof)
    joint_solutions = np.zeros((poses.shape[0], n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = _pose_to_transform(pose)
        q = np.random.uniform(0, 2*np.pi, n_dof)
        
        for _ in range(max_iter):
            T_current = _forward_kinematics(q, dh_params)
            error = _pose_error(T_current, T_desired)
            
            if np.linalg.norm(error) < tol:
                break
            
            J = _compute_jacobian(q, dh_params)
            damping = 1e-4
            
            if n_dof >= 6:
                J_pinv = np.linalg.solve(J.T @ J + damping * np.eye(n_dof), J.T)
            else:
                J_pinv = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), np.eye(6))
            
            q = q + J_pinv @ error
            q = np.mod(q, 2*np.pi)
        
        joint_solutions[i] = q
    
    return joint_solutions[0] if poses.shape[0] == 1 else joint_solutions

def sdls_ik(n_dof: int, poses: np.ndarray, max_iter: int = 100, tol: float = 1e-3) -> np.ndarray:
    poses = np.atleast_2d(poses)
    if poses.shape[1] < 6:
        poses = np.pad(poses, ((0, 0), (0, 6 - poses.shape[1])), 'constant')
    
    dh_params = _create_robot_dh(n_dof)
    joint_solutions = np.zeros((poses.shape[0], n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = _pose_to_transform(pose)
        
        def objective(q):
            T_current = _forward_kinematics(q, dh_params)
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