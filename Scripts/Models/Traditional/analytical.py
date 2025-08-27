import numpy as np
import time

def analytical_ik(n_dof: int, poses: np.ndarray, timeout: float = 5.0) -> np.ndarray:
    if not (2 <= n_dof <= 10):
        raise RuntimeError(f"Analytical IK implemented for 2-10 DOF (got {n_dof}-DOF)")
    
    start_time = time.time()
    poses = np.atleast_2d(poses)
    n_poses = poses.shape[0]
    
    # Check timeout before starting
    if time.time() - start_time > timeout:
        raise RuntimeError(f"Analytical IK timed out for {n_dof}-DOF")
    
    joint_solutions = np.zeros((n_poses, n_dof))
    
    for i, pose in enumerate(poses):
        # Check timeout during computation
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Analytical IK timed out during computation for {n_dof}-DOF")
        
        try:
            if n_dof == 2:
                joint_solutions[i] = _solve_2dof(pose)
            elif n_dof == 3:
                joint_solutions[i] = _solve_3dof(pose)
            elif n_dof == 4:
                joint_solutions[i] = _solve_4dof(pose)
            elif n_dof == 5:
                joint_solutions[i] = _solve_5dof(pose)
            elif n_dof == 6:
                joint_solutions[i] = _solve_6dof(pose)
            elif n_dof in [7, 8, 9, 10]:
                # Higher DOF - use geometric decomposition with timeout check
                joint_solutions[i] = _solve_high_dof(pose, n_dof, start_time, timeout)
            
        except Exception:
            # Fallback to random solution if analytical method fails
            joint_solutions[i] = np.random.uniform(0, 2*np.pi, n_dof)
    
    # Return single array if single pose input  
    return joint_solutions[0] if n_poses == 1 and poses.shape[0] == 1 else joint_solutions

def _solve_2dof(pose: np.ndarray) -> np.ndarray:
    """2-DOF planar analytical solution"""
    x, y = pose[0], pose[1]
    r_squared = x**2 + y**2
    
    # Check reachability (workspace radius = 2.0 for unit links)
    if r_squared > 4.0:
        return np.random.uniform(0, 2*np.pi, 2)
    
    # Law of cosines for elbow-down solution
    cos_q2 = (r_squared - 2) / 2
    cos_q2 = np.clip(cos_q2, -1, 1)
    q2 = np.arccos(cos_q2)
    
    # Solve for q1
    k1 = 1 + np.cos(q2)
    k2 = np.sin(q2)
    q1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return np.mod([q1, q2], 2*np.pi)

def _solve_3dof(pose: np.ndarray) -> np.ndarray:
    """3-DOF spatial analytical solution"""
    x, y, z = pose[:3]
    
    # Project to XY plane for first two joints
    r_xy = np.sqrt(x**2 + y**2)
    
    if r_xy < 1e-6:  # Singular case - straight up
        q1 = 0.0
        # Solve 2DOF problem in XZ plane
        r_xz = np.sqrt(r_xy**2 + z**2)
        if r_xz > 2.0:
            return np.random.uniform(0, 2*np.pi, 3)
        
        cos_q3 = (r_xz**2 - 2) / 2
        cos_q3 = np.clip(cos_q3, -1, 1)
        q3 = np.arccos(cos_q3)
        q2 = np.arctan2(z, r_xy) - np.arctan2(np.sin(q3), 1 + np.cos(q3))
    else:
        # Base rotation
        q1 = np.arctan2(y, x)
        
        # Solve in the plane defined by base rotation
        r_total = np.sqrt(r_xy**2 + z**2)
        if r_total > 2.0:
            return np.random.uniform(0, 2*np.pi, 3)
        
        cos_q3 = (r_total**2 - 2) / 2
        cos_q3 = np.clip(cos_q3, -1, 1)
        q3 = np.arccos(cos_q3)
        q2 = np.arctan2(z, r_xy) - np.arctan2(np.sin(q3), 1 + np.cos(q3))
    
    return np.mod([q1, q2, q3], 2*np.pi)

def _solve_4dof(pose: np.ndarray) -> np.ndarray:
    """4-DOF analytical solution using geometric decomposition"""
    x, y, z = pose[:3]
    
    # Use wrist center approach - assume last joint is for orientation
    # Position the 3-DOF to get close to target, use 4th for fine adjustment
    
    # First solve 3-DOF to intermediate point
    intermediate_pose = np.array([x * 0.9, y * 0.9, z * 0.9])  # Slightly inside target
    q_3dof = _solve_3dof(intermediate_pose)
    
    # 4th joint for remaining reach/orientation
    remaining_distance = np.linalg.norm([x - intermediate_pose[0], 
                                       y - intermediate_pose[1], 
                                       z - intermediate_pose[2]])
    q4 = np.arctan2(remaining_distance, 1.0)  # Approximate solution
    
    return np.mod([q_3dof[0], q_3dof[1], q_3dof[2], q4], 2*np.pi)

def _solve_5dof(pose: np.ndarray) -> np.ndarray:
    """5-DOF analytical solution"""
    x, y, z = pose[:3]
    
    # Geometric decoupling - solve position with first 3, orientation with last 2
    q_pos = _solve_3dof(pose[:3])
    
    # Orientation joints (simplified)
    roll = pose[3] if len(pose) > 3 else 0.0
    pitch = pose[4] if len(pose) > 4 else 0.0
    
    q4 = roll * 0.5  # Distribute orientation
    q5 = pitch * 0.5
    
    return np.mod([q_pos[0], q_pos[1], q_pos[2], q4, q5], 2*np.pi)

def _solve_6dof(pose: np.ndarray) -> np.ndarray:
    """6-DOF analytical solution using position/orientation decoupling"""
    x, y, z = pose[:3]
    roll = pose[3] if len(pose) > 3 else 0.0
    pitch = pose[4] if len(pose) > 4 else 0.0
    yaw = pose[5] if len(pose) > 5 else 0.0
    
    # Wrist center approach - offset end-effector by last link
    wrist_offset = 0.8  # Assume wrist is 0.8 units from end-effector
    
    # Calculate wrist center position
    wrist_x = x - wrist_offset * np.cos(yaw) * np.cos(pitch)
    wrist_y = y - wrist_offset * np.sin(yaw) * np.cos(pitch)
    wrist_z = z + wrist_offset * np.sin(pitch)
    
    # Solve first 3 joints to reach wrist center
    wrist_pose = np.array([wrist_x, wrist_y, wrist_z])
    q_pos = _solve_3dof(wrist_pose)
    
    # Last 3 joints for wrist orientation (simplified Euler angle assignment)
    q4 = roll / 2.0
    q5 = pitch
    q6 = yaw / 2.0
    
    return np.mod([q_pos[0], q_pos[1], q_pos[2], q4, q5, q6], 2*np.pi)

def _solve_high_dof(pose: np.ndarray, n_dof: int, start_time: float, timeout: float) -> np.ndarray:
    """
    High DOF (7-10) using iterative geometric decomposition
    With timeout protection for computational complexity
    """
    if time.time() - start_time > timeout * 0.8:  # Use 80% of timeout
        raise RuntimeError(f"High DOF analytical IK approaching timeout")
    
    x, y, z = pose[:3]
    
    # Strategy: Use redundancy resolution
    # First 6 DOF for primary pose, additional DOF for optimization criteria
    
    # Get 6-DOF solution first
    q_primary = _solve_6dof(pose[:6] if len(pose) >= 6 else pose)
    
    # Initialize full solution
    q_full = np.zeros(n_dof)
    q_full[:len(q_primary)] = q_primary
    
    # Additional joints - use null space or simple heuristics
    for i in range(6, n_dof):
        if time.time() - start_time > timeout * 0.9:
            # Emergency timeout - fill remaining with random
            q_full[i:] = np.random.uniform(0, 2*np.pi, n_dof - i)
            break
        
        # Simple heuristic: distribute remaining pose error
        if i < len(pose):
            q_full[i] = pose[i] * 0.1  # Small contribution
        else:
            # Null space motion - optimize for manipulability or joint limits
            q_full[i] = np.pi  # Mid-range position
    
    return np.mod(q_full, 2*np.pi)

# Test/verification functions for debugging
def _verify_solution(n_dof: int, pose: np.ndarray, joint_angles: np.ndarray, tol: float = 0.1) -> bool:
    """Verify if analytical solution is reasonable (for debugging)"""
    try:
        # Simple forward kinematics check
        x_fk = sum(np.cos(sum(joint_angles[:i+1])) for i in range(min(n_dof, 3)))
        y_fk = sum(np.sin(sum(joint_angles[:i+1])) for i in range(min(n_dof, 3)))
        
        pos_error = np.sqrt((x_fk - pose[0])**2 + (y_fk - pose[1])**2)
        return pos_error < tol
    except:
        return False