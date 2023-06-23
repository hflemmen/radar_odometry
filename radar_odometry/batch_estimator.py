from __future__ import print_function
import numpy as np
import manifpy as mf
import gtsam
from radar_utils.viz import StateCovariances
from gtsam.symbol_shorthand import X, L

"""
Reference frames:
r: All the radar scans
rx: The x'th radar scan
r0: The initial radar scan frame
rn: The current radar scan frame
"""


class Estimator:
    def __init__(self, cfg, timer):
        """
        initial_pose: The NED pose in the format (x, y, theta)
        """
        self.max_iterations = cfg.max_iterations
        self.relative_error_tol = cfg.relative_error_tol
        self.timer = timer
        baseNoise = gtsam.noiseModel.Diagonal.Sigmas(sigmas=np.array([cfg.angular_noise, cfg.range_noise], float))
        self.BEARING_RANGE_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=cfg.huber_k), baseNoise)

        self.graph = gtsam.NonlinearFactorGraph()
        gtsam_initial = gtsam.Pose2(0, 0, 0)
        PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(sigmas=np.array([1.0, 1.0, 1.0], float))
        self.graph.add(gtsam.PriorFactorPose2(X(0), gtsam_initial, PRIOR_NOISE))
        self.initial_estimate = gtsam.Values()
        self.prev_pose = gtsam_initial

    def update(self, pose_num, feature_tracks_t_img, current_active, scan_rn):
        # print(f"Number of active features: {len(current_active)}")
        if len(current_active) < 3:
            print("Too few tracked features")
            raise Exception("Too few tracked features.")
        for id in current_active:
            track_t_img = feature_tracks_t_img[id]
            new_observation = track_t_img[-1]
            if len(track_t_img) > 1 and np.linalg.norm(new_observation - track_t_img[-2]) < 1e-6:
                print('This is strange:', id)
            dist, angle = scan_rn.get_range_angle(new_observation)
            self.graph.add(
                gtsam.BearingRangeFactor2D(X(pose_num), L(id), gtsam.Rot2(angle), dist, self.BEARING_RANGE_NOISE))
            if not L(id) in self.initial_estimate.keys():
                p_r0_r0_ri = scan_rn.unproject_coordinate(new_observation)
                feature_pos_cart = mf.SE2(self.prev_pose.x(), self.prev_pose.y(), self.prev_pose.theta()).act(
                    p_r0_r0_ri)
                self.initial_estimate.insert(L(id), feature_pos_cart)

        self.initial_estimate.insert(X(pose_num), self.prev_pose)

        if pose_num > 0:
            parameters = gtsam.LevenbergMarquardtParams()
            parameters.setRelativeErrorTol(self.relative_error_tol)
            parameters.setMaxIterations(self.max_iterations)
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, parameters)
            self.timer.tic()
            result = optimizer.optimize()
            self.timer.toc()
            self.prev_pose = result.atPose2(X(pose_num))
            est_T_r0_rn = []
            for i in range(0, pose_num):
                pose_temp = result.atPose2(X(i))
                est_T_r0_rn.append(mf.SE2(pose_temp.x(), pose_temp.y(), pose_temp.theta()))

            landmarks_r0 = []
            for i in current_active:
                landmarks_r0.append(result.atVector(L(i)))

            self.initial_estimate = result
            marginals = gtsam.Marginals(self.graph, result)
            marginal_cov = marginals_to_matrices(marginals, pose_num, current_active)
            temp_pose = result.atPose2(X(pose_num))
            T_r0_rn = mf.SE2(temp_pose.x(), temp_pose.y(), temp_pose.theta())
            return T_r0_rn, est_T_r0_rn, landmarks_r0, marginal_cov
        else:
            landmarks_r0 = []
            for id in current_active:
                landmarks_r0.append(self.initial_estimate.atVector(L(id)))
            marginals = None
            return mf.SE2.Identity(), [mf.SE2.Identity()], landmarks_r0, marginals


def marginals_to_matrices(marginals: gtsam.Marginals, pose_num, current_active) -> StateCovariances:
    landmarks = {}
    joint = []
    for i in current_active:
        landmarks[i] = marginals.marginalCovariance(L(i))
        joint.append(marginals.jointMarginalCovariance(gtsam.KeyVector([X(pose_num), L(i)])).fullMatrix())

    poses = []
    for i in range(pose_num + 1):
        poses.append(marginals.marginalCovariance(X(i)))

    return StateCovariances(landmarks, poses, joint)
