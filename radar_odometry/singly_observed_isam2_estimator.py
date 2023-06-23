import gtsam
import numpy as np
import manifpy as mf

from gtsam.symbol_shorthand import X, L
from radar_utils.viz import StateCovariances


# TODO: Solve this problem in a better way after debuging is done. E.g. with a hash function, or with different gtsam symbols.
def uniquifier(id, pose_num):
    return pose_num * int(1e8) + id


class ISAM2Estimator:
    def __init__(self, cfg, timer):
        self.max_iterations = cfg.max_iterations
        self.timer = timer
        base_bearing_range_noise = gtsam.noiseModel.Diagonal.Sigmas(
            sigmas=np.array([cfg.angular_noise, cfg.range_noise], float))
        self.BEARING_RANGE_NOISE = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(k=cfg.huber_k),
            base_bearing_range_noise)
        isam2_param = gtsam.ISAM2Params()

        isam2_param.setRelinearizeThreshold(cfg.relinearize_threshold)
        isam2_param.evaluateNonlinearError = True
        opt_param = gtsam.ISAM2GaussNewtonParams()
        isam2_param.setOptimizationParams(opt_param)
        self.graph = gtsam.ISAM2(isam2_param)

        prior_pose = gtsam.Pose2(0, 0, 0)
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sigmas=np.array([1.0, 1.0, 1.0], dtype=float))
        prior_factor = gtsam.PriorFactorPose2(X(0), prior_pose, prior_noise)
        initial_factors = gtsam.NonlinearFactorGraph()
        initial_factors.add(prior_factor)
        initial_values = gtsam.Values()
        initial_values.insert(X(0), prior_pose)
        self.graph.update(initial_factors, initial_values)
        self.prev_pose = prior_pose

    def update(self, pose_num, feature_tracks_t_img, current_active, scan_rn):
        new_factors = gtsam.NonlinearFactorGraph()
        new_initial_values = gtsam.Values()
        new_landmarks = []
        for id in current_active:
            track_t_img = feature_tracks_t_img[id]
            if len(track_t_img) > 1:
                unique_id = uniquifier(id, pose_num)
                obs_n = track_t_img[-1]
                obs_n1 = track_t_img[-2]
                dist_n, angle_n = scan_rn.get_range_angle(obs_n)
                # This is not strictly correct since it is not scan_rn, but ok as the radar is the same:
                dist_n1, angle_n1 = scan_rn.get_range_angle(obs_n1)
                new_factors.add(gtsam.BearingRangeFactor2D(X(pose_num), L(unique_id), gtsam.Rot2(angle_n), dist_n,
                                                           self.BEARING_RANGE_NOISE))
                new_factors.add(gtsam.BearingRangeFactor2D(X(pose_num - 1), L(unique_id), gtsam.Rot2(angle_n1), dist_n1,
                                                           self.BEARING_RANGE_NOISE))
                new_landmarks.append(unique_id)
                p_r0_r0_ri1 = scan_rn.unproject_coordinate(track_t_img[-2])
                feature_pos_cart = mf.SE2(self.prev_pose.x(), self.prev_pose.y(), self.prev_pose.theta()).act(
                    p_r0_r0_ri1)
                new_initial_values.insert(L(unique_id), feature_pos_cart)
        if pose_num > 0:
            new_initial_values.insert(X(pose_num), self.prev_pose)

        self.timer.tic()
        self.graph.update(new_factors, new_initial_values)
        for i in range(self.max_iterations):
            isam_res = self.graph.update()
            if abs(isam_res.getErrorBefore() - isam_res.getErrorAfter()) < 1e-19:
                break
        self.timer.toc()

        if pose_num > 0:

            result = self.graph.calculateEstimate()

            est_T_r0_rn = []
            for i in range(0, pose_num):
                pose_temp = result.atPose2(X(i))
                est_T_r0_rn.append(mf.SE2(pose_temp.x(), pose_temp.y(), pose_temp.theta()))

            landmarks_r0 = []
            for i in current_active:
                unique_id = uniquifier(i, pose_num)
                landmarks_r0.append(result.atVector(L(unique_id)))

            marginal_cov = marginals_to_matrices(self.graph, pose_num, current_active)

            self.prev_pose = result.atPose2(X(pose_num))
            T_r0_rn = mf.SE2(self.prev_pose.x(), self.prev_pose.y(), self.prev_pose.theta())
            return T_r0_rn, est_T_r0_rn, landmarks_r0, marginal_cov
        else:
            result = self.graph.calculateEstimate()
            landmarks_r0 = []
            for id in current_active:
                p_r0_r0_ri1 = scan_rn.unproject_coordinate(feature_tracks_t_img[id][-1])
                feature_pos_cart = mf.SE2(self.prev_pose.x(), self.prev_pose.y(), self.prev_pose.theta()).act(
                    p_r0_r0_ri1)
                landmarks_r0.append(feature_pos_cart)
                marginals = None
            return mf.SE2.Identity(), [mf.SE2.Identity()], landmarks_r0, marginals


def marginals_to_matrices(marginals: gtsam.ISAM2, pose_num, current_active) -> StateCovariances:
    landmarks = {}
    # joint = []
    for i in current_active:
        id = uniquifier(i, pose_num)
        landmarks[i] = marginals.marginalCovariance(L(id))
    #     joint.append(marginals.jointMarginalCovariance(gtsam.KeyVector([X(pose_num), L(i)])).fullMatrix())
    joint = None
    poses = []
    for i in range(pose_num + 1):
        poses.append(marginals.marginalCovariance(X(i)))

    return StateCovariances(landmarks, poses, joint)
