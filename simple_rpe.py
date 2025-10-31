#!/usr/bin/env python3
import numpy as np

def read_kitti_trajectory(filename):
    """Read KITTI format trajectory (12 values per line)"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    poses = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            values = list(map(float, line.strip().split()))
            if len(values) >= 12:
                # Extract translation (positions 3, 7, 11)
                tx, ty, tz = values[3], values[7], values[11]
                poses.append([tx, ty, tz])
    return np.array(poses)

def calculate_rpe(gt_poses, est_poses, delta=1.0):
    """Calculate Relative Pose Error over fixed distances"""
    errors = []
    
    for i in range(1, len(gt_poses)):
        # Calculate distance moved in ground truth
        gt_dist = np.linalg.norm(gt_poses[i] - gt_poses[i-1])
        
        if gt_dist >= delta:
            # Calculate relative motion in both trajectories
            gt_relative = gt_poses[i] - gt_poses[i-1]
            est_relative = est_poses[i] - est_poses[i-1]
            
            # RPE is the difference in relative motions
            error = np.linalg.norm(gt_relative - est_relative)
            errors.append(error)
    
    return np.array(errors)

if __name__ == "__main__":
    gt_poses = read_kitti_trajectory("datasets/KITTI/poses/00.txt")
    est_poses = read_kitti_trajectory("CameraTrajectory.txt")
    
    # Make sure trajectories have same length
    min_len = min(len(gt_poses), len(est_poses))
    gt_poses = gt_poses[:min_len]
    est_poses = est_poses[:min_len]
    
    rpe_errors = calculate_rpe(gt_poses, est_poses, delta=1.0)
    
    print(f"RPE Statistics (delta=1m):")
    print(f"RMSE: {np.sqrt(np.mean(rpe_errors**2)):.6f} m")
    print(f"Mean: {np.mean(rpe_errors):.6f} m")
    print(f"Std:  {np.std(rpe_errors):.6f} m")
    print(f"Max:  {np.max(rpe_errors):.6f} m")
    print(f"Min:  {np.min(rpe_errors):.6f} m")
