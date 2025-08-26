"""
Simple inverse kinematics solvers using Robotics Toolbox
Just import and use: jacobian_ik(dof, poses) or sdls_ik(dof, poses)
"""

import numpy as np
from typing import List, Union

try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    ROBOTICS_TOOLBOX_AVAILABLE = True
except ImportError:
    ROBOTICS_TOOLBOX_AVAILABLE = False

def _create_robot(n_dof: int):
    """Create a simple serial robot with unit link lengths and alternating twists"""
    if not ROBOTICS_TOOLBOX_AVAILABLE:
        raise ImportError("Install roboticstoolbox-python: pip install roboticstoolbox-python")
    
    links = []
    for i in range(n_dof):
        # Alternate between 0 and π/2 twist for 3D workspace
        alpha = 0.0 if i % 2 == 0 else np.pi/2
        
        link = rtb.DHLink(
            d=0.0,        # No joint offset
            a=1.0,        # Unit link length
            alpha=alpha,  # Alternating twist
            offset=0.0,
            qlim=[0, 2*np.pi]
        )
        links.append(link)
    
    return rtb.DHRobot(links, name=f"{n_dof}DOF_Robot")

# Simple test function
def test_solvers():
    """Test all solvers with a simple example"""
    if not ROBOTICS_TOOLBOX_AVAILABLE:
        print("Robotics Toolbox not installed. Run: pip install roboticstoolbox-python")
        return
    
    print("Testing IK Solvers...")
    
    # Test pose
    test_pose = np.array([1.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # [x, y, z, roll, pitch, yaw]
    
    # Test 3-DOF
    print(f"\nTest pose: {test_pose}")
    
    try:
        q_jacobian = jacobian_ik(3, test_pose)
        print(f"Jacobian IK (3-DOF): {q_jacobian}")
    except Exception as e:
        print(f"Jacobian IK failed: {e}")
    
    try:
        q_sdls = sdls_ik(3, test_pose)
        print(f"SDLS IK (3-DOF): {q_sdls}")
    except Exception as e:
        print(f"SDLS IK failed: {e}")
    
    try:
        q_analytical = analytical_ik(2, test_pose)
        print(f"Analytical IK (2-DOF): {q_analytical}")
    except Exception as e:
        print(f"Analytical IK: {e}")

if __name__ == "__main__":
    test_solvers()