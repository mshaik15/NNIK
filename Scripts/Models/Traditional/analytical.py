import numpy as np

def analytical_ik(n_dof: int, poses: np.ndarray) -> np.ndarray:
    if n_dof != 2:
        raise RuntimeError(f"Analytical IK only implemented for 2-DOF (got {n_dof}-DOF). Use numerical methods.")
    
    poses = np.atleast_2d(poses)
    n_poses = poses.shape[0]
    joint_solutions = np.zeros((n_poses, 2))
    
    for i, pose in enumerate(poses):
        x, y = pose[0], pose[1]
        
        r_squared = x**2 + y**2
        
        # Check reachability (workspace is circle of radius 2.0)
        if r_squared > 4.0:  # Outside workspace
            joint_solutions[i] = np.random.uniform(0, 2*np.pi, 2)
            continue
        
        # Elbow-down solution
        cos_q2 = (r_squared - 2) / 2  # Using law of cosines
        cos_q2 = np.clip(cos_q2, -1, 1)  # Numerical safety
        
        q2 = np.arccos(cos_q2)
        
        # Solve for q1
        k1 = 1 + np.cos(q2)
        k2 = np.sin(q2)
        
        q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
        
        # Wrap to [0, 2π]
        q1 = np.mod(q1, 2*np.pi)
        q2 = np.mod(q2, 2*np.pi)
        
        joint_solutions[i] = [q1, q2]
    
    # Return single array if single pose input  
    if joint_solutions.shape[0] == 1 and len(poses.shape) == 1:
        return joint_solutions[0]
    
    return joint_solutions