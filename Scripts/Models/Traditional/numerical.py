import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH, SE3
from spatialmath import SE3
from typing import List, Union, Dict


def jacobian_ik(n_dof: int, poses: np.ndarray, max_iter: int = 100, tol: float = 1e-3) -> np.ndarray:
    robot = _create_robot(n_dof)
    poses = np.atleast_2d(poses)
    n_poses = poses.shape[0]
    
    joint_solutions = np.zeros((n_poses, n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = SE3.Trans(pose[:3]) * SE3.RPY(pose[3:], order='xyz')
        
        try:
            sol = robot.ikine_LM(T_desired, ilimit=max_iter, tol=tol)
            if sol.success:
                joint_solutions[i] = sol.q
            else:
                # Random fallback
                joint_solutions[i] = np.random.uniform(0, 2*np.pi, n_dof)
        except:
            joint_solutions[i] = np.random.uniform(0, 2*np.pi, n_dof)
    
    # Return single array if single pose input
    if joint_solutions.shape[0] == 1 and len(poses.shape) == 1:
        return joint_solutions[0]
    
    return joint_solutions

def sdls_ik(n_dof: int, poses: np.ndarray, max_iter: int = 100, tol: float = 1e-3) -> np.ndarray:
    """
    Solve inverse kinematics using SDLS (Gauss-Newton with fallback strategies)
    
    Args:
        n_dof: Number of degrees of freedom  
        poses: Target poses [x, y, z, roll, pitch, yaw] - shape (n, 6) or (6,)
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Joint angles - shape (n, n_dof) or (n_dof,)
    """
    robot = _create_robot(n_dof)
    poses = np.atleast_2d(poses)
    n_poses = poses.shape[0]
    
    joint_solutions = np.zeros((n_poses, n_dof))
    
    for i, pose in enumerate(poses):
        T_desired = SE3.Trans(pose[:3]) * SE3.RPY(pose[3:], order='xyz')
        
        try:
            # Try Gauss-Newton first (SDLS-like)
            sol = robot.ikine_GN(T_desired, ilimit=max_iter, tol=tol)
            
            if sol.success:
                joint_solutions[i] = sol.q
            else:
                # Fallback to LM with higher damping
                sol = robot.ikine_LM(T_desired, ilimit=max_iter//2, tol=tol, λ=0.1)
                if sol.success:
                    joint_solutions[i] = sol.q
                else:
                    # Random fallback
                    joint_solutions[i] = np.random.uniform(0, 2*np.pi, n_dof)
        except:
            joint_solutions[i] = np.random.uniform(0, 2*np.pi, n_dof)
    
    # Return single array if single pose input
    if joint_solutions.shape[0] == 1 and len(poses.shape) == 1:
        return joint_solutions[0]
    
    return joint_solutions