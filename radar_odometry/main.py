import glob
import os.path
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import radar_utils.viz
from manifpy import SE2
import pymap3d as pm
import radar_odometry.batch_estimator as estimator
import radar_odometry.singly_observed_batch_estimator as singly_observed_batch_estimator
import radar_odometry.singly_observed_isam2_estimator as singly_observed_isam2_estimator
import radar_odometry.evaluation as evaluation
import radar_odometry.isam2_estimator as isam2_estimator
import radar_odometry.of_tracker as of_tracker
from radar_utils.transformations import polarToEuc, fillInRanges
from radar_utils.filter import radar_interference_filter
from radar_utils.scan import PolarScan, EuclideanScan
from radar_utils.viz import TrajectoryVisualizer, UpdateStepData, CartesianVisualizer, CartUpdateData, draw_tracks, \
    FeatureVisualizer, FeatureVisualizerData
from radar_utils.video_creator import VideosSinkArray
import radar_odometry.preprocessing as preprocessing
import radar_odometry.software_config
import radar_odometry.hardware_config


def pad(A, shape):
    # https://stackoverflow.com/questions/61961612/how-to-efficiently-resize-a-numpy-array-to-a-given-shape-padding-with-zeros-if
    shape = np.max([np.shape(A), shape], axis=0)
    out = np.zeros(shape, dtype=np.uint8)
    out[tuple(slice(0, d) for d in np.shape(A))] = A
    return out


def distLatLong(Lat1, Lon1, Lat2, Lon2):
    Lat1 = Lat1 * 180 / np.pi
    Lon1 = Lon1 * 180 / np.pi
    Lat2 = Lat2 * 180 / np.pi
    Lon2 = Lon2 * 180 / np.pi
    latMid = (Lat1 + Lat2) / 2.0

    m_per_deg_lat = 111132.954 - 559.822 * np.cos(2.0 * latMid) + 1.175 * np.cos(4.0 * latMid)
    m_per_deg_lon = (3.14159265359 / 180) * 6367449 * np.cos(latMid)

    deltaLat = abs(Lat1 - Lat2)
    deltaLon = abs(Lon1 - Lon2)

    dist_m = np.sqrt(pow(deltaLat * m_per_deg_lat, 2) + pow(deltaLon * m_per_deg_lon, 2))
    return dist_m


def scan_pos_to_p_w_w_g(initial_pose_e_g, scan_r):
    pose_e_g = scan_r.pose
    pos_w_g = pm.geodetic2ned(pose_e_g[0], pose_e_g[1], 0,
                              initial_pose_e_g[0], initial_pose_e_g[1], 0, deg=False)
    return np.array([pos_w_g[0], pos_w_g[1]])


def scan_pos_to_T_w_g(initial_pose_e_g, scan_r):
    pos = scan_pos_to_p_w_w_g(initial_pose_e_g, scan_r)
    angle = scan_r.pose[2]
    return SE2(pos[0], pos[1], angle)


def set_up_result_folder(
        sw_cfg: radar_odometry.software_config.SoftwareConfig,
        hw_cfg: radar_odometry.hardware_config.HardwareConfig) -> radar_odometry.software_config.SoftwareConfig:
    if sw_cfg.expected_duplicate:
        pass
    else:
        orig_path = sw_cfg.base_path
        iterpath = sw_cfg.get_iteration_path(hw_cfg.in_path)
        i = 0
        if os.path.exists(iterpath):
            while True:
                sw_cfg.base_path = orig_path[:-1] + "_" + str(i) + "/"
                iterpath = sw_cfg.get_iteration_path(hw_cfg.in_path)
                if not os.path.exists(iterpath):
                    break
                else:
                    i += 1
    iterpath = sw_cfg.get_iteration_path(hw_cfg.in_path)
    os.makedirs(iterpath, exist_ok=True)

    project_folder = "/home/henrik/projects/"
    shutil.copy(project_folder + "radar_odometry/radar_odometry/software_config.py",
                iterpath + "software_config.py")
    shutil.copy(project_folder + "radar_odometry/radar_odometry/hardware_config.py",
                iterpath + "hardware_config.py")
    print(f"Storing to folder: {iterpath}")
    return sw_cfg


"""
Reference frames:
r: radar frame (centered at the ships location) [SE2(forward, right, (0=forward, pi/4=right))]
g: gnss frame (NED) [SE2(forward, right, (0=forward, pi/4=right))]
w: Local NED frame around the inital GNSS location [SE2(north, east, (0=north, pi/4=east))]
e: world frame (ECEF) [lat [-pi/2,pi/2], long[-pi/2,pi/2], angle (0=north, pi/4=east))]
x_img: Pixel coordinates of image x.
    t_img: Pixel coordinates of the image used for tracking.

g0 := r0 TODO: This is wrong, they are located at different positions on the ship
"""


def main(sw_cfg, hw_cfg):
    sw_cfg = set_up_result_folder(sw_cfg, hw_cfg)
    full_res_path = sw_cfg.get_iteration_path(hw_cfg.in_path)
    imgpaths = glob.glob(hw_cfg.in_path + "*.bmp")
    imgpaths.sort()
    if len(imgpaths) == 0:
        raise Exception(f"Data not found: {hw_cfg.in_path}")
    if sw_cfg.verbose:
        print(f"Found {len(imgpaths)} images")

    tracker = of_tracker.OpticalFlowTracker(sw_cfg)
    evaluator_initializer = evaluation.EvaluatorInitializer(sw_cfg.rpe_interval,
                                                            sw_cfg.estimator_type,
                                                            sw_cfg.image_layout,
                                                            sw_cfg.show_plots,
                                                            sw_cfg.sub_path,
                                                            hw_cfg.in_path)
    evaluator = evaluation.Evaluator(evaluator_initializer)
    timer = evaluation.Timing()
    video_collector = None
    feature_visualizer = FeatureVisualizer(sw_cfg, hw_cfg)

    est_Ts_r0_rn = []
    est_Ts_w_rn = []
    est = None

    pose_num = 0
    initial_pose_e0_g0 = None
    T_w_g0 = None
    gt_Ts_w_gn = []
    landmarks_hist_r0 = []
    for imgpath in imgpaths:
        if sw_cfg.use_cache:
            scan_polar_rn = PolarScan.load(imgpath, use_cache=True, clockwise=hw_cfg.clockwise)
        else:
            scan_polar_rn = PolarScan.load(imgpath, use_cache=False, clockwise=hw_cfg.clockwise)
            scan_polar_rn = radar_interference_filter(scan_polar_rn)
            scan_polar_rn = radar_interference_filter(scan_polar_rn, 2)

        if sw_cfg.image_layout == sw_cfg.ImageLayout.Cartesian:
            scan_euc_rn = radar_utils.transformations.polarToEuc(scan_polar_rn, new_size=np.array([6000, 6000]),
                                                                use_cache=sw_cfg.use_cache)
            scan_rn = scan_euc_rn
        elif sw_cfg.image_layout == sw_cfg.ImageLayout.Polar:
            scan_rn = scan_polar_rn
            if sw_cfg.resample_ranges:
                scan_rn = fillInRanges(scan_rn)

        # Spatial filtering
        if sw_cfg.size_range > 0 and sw_cfg.sigma_range > 0:
            scan_rn.img = cv2.GaussianBlur(scan_rn.img,
                                           ksize=(sw_cfg.size_angle, sw_cfg.size_range,),
                                           sigmaX=sw_cfg.sigma_angle,
                                           sigmaY=sw_cfg.sigma_range)

        # Filter out weak returns
        scan_rn = preprocessing.prune_weak_returns(scan_rn, sw_cfg.wave_noise_threshold)
        img_color = cv2.cvtColor(scan_rn.img, cv2.COLOR_GRAY2BGR)

        T_g0_r0 = SE2(*hw_cfg.T_g0_r0_tuple)

        if initial_pose_e0_g0 is None:
            initial_pose_e0_g0 = scan_rn.pose
            p_w_w_g0 = scan_pos_to_p_w_w_g(initial_pose_e0_g0[:2], scan_rn)
            T_w_g0 = SE2(p_w_w_g0[0], p_w_w_g0[1], initial_pose_e0_g0[2])
            if sw_cfg.estimator_type == sw_cfg.Estimator.FullHistory:
                est = estimator.Estimator(sw_cfg, timer)
            elif sw_cfg.estimator_type == sw_cfg.Estimator.ISAM2:
                est = isam2_estimator.ISAM2Estimator(sw_cfg, timer)
            elif sw_cfg.estimator_type == sw_cfg.Estimator.SinglyObservedBatch:
                est = singly_observed_batch_estimator.Estimator(sw_cfg, timer)
            elif sw_cfg.estimator_type == sw_cfg.Estimator.SinglyObservedISAM2:
                est = singly_observed_isam2_estimator.ISAM2Estimator(sw_cfg, timer)
            else:
                raise NotImplementedError("This estimator type is not implemented yet:", sw_cfg.estimator_type)
            if sw_cfg.show_visualisation or sw_cfg.record_video:
                traj_vis = TrajectoryVisualizer(T_w_g0, T_g0_r0, full_res_path, sw_cfg.show_visualisation,
                                                sw_cfg.record_video, sw_cfg.use_cache)
                cart_viz = CartesianVisualizer(T_w_g0, T_g0_r0, stable_window_size=17000,
                                               show_visualization=sw_cfg.show_visualisation,
                                               record_video=sw_cfg.record_video)
        if sw_cfg.image_layout == sw_cfg.ImageLayout.Polar:
            res = tracker.update(scan_rn.img)
        elif sw_cfg.image_layout == sw_cfg.ImageLayout.Cartesian:
            res = tracker.update(scan_rn.img, scan_rn.center)
        if res is None:
            continue
        else:
            feature_tracks_t_img, current_ids, removed = res
        if sw_cfg.show_visualisation:
            mask = np.zeros(img_color.shape, dtype='uint8')
            mask = draw_tracks(feature_tracks_t_img, current_ids, mask)
            drawn = cv2.add(img_color, mask)
            if sw_cfg.image_layout == sw_cfg.ImageLayout.Cartesian:
                viz = cv2.resize(drawn, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
            elif sw_cfg.image_layout == sw_cfg.ImageLayout.Polar:
                viz = cv2.resize(drawn, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Img", viz)

        est_T_r0_rn, est_traj_Ts_r0_rn, landmarks_r0, marginals = est.update(pose_num,
                                                                             feature_tracks_t_img,
                                                                             current_ids,
                                                                             scan_rn)
        est_T_g0_rn = T_g0_r0 * est_T_r0_rn
        est_T_w_rn = T_w_g0 * est_T_g0_rn
        est_Ts_r0_rn.append(est_T_r0_rn)
        est_Ts_w_rn.append(est_T_w_rn)
        landmarks_hist_r0.append(landmarks_r0)

        feature_visualizer.update(FeatureVisualizerData(scan_rn.img, feature_tracks_t_img, current_ids, removed))

        gt_Ts_w_gn.append(scan_pos_to_T_w_g(initial_pose_e0_g0, scan_rn))
        if len(landmarks_hist_r0) > 1:
            lm1 = landmarks_hist_r0[-2]
        else:
            lm1 = np.zeros((len(removed), 2))
        if sw_cfg.show_visualisation or sw_cfg.record_video:
            joint_covariance = None if marginals is None else marginals.joint_cov_rn
            update_data = UpdateStepData(scan_polar_rn, est_Ts_r0_rn, gt_Ts_w_gn, landmarks_r0, removed, lm1,
                                         joint_covariance)
            traj_vis_img = traj_vis.viz_trajectory(update_data)

        if len(landmarks_r0) > 1:
            if sw_cfg.verbose:
                print(pose_num, ":Estimate in r0")
                print("\tPos:", est_T_r0_rn.translation())
                print("\tAngle:", est_T_r0_rn.angle() * 180 / np.pi)

                print(pose_num, "Ground truth in r0")
                print("\tPos:", (T_w_g0.inverse() * gt_Ts_w_gn[-1]).translation())
                print("\tAngle:", (T_w_g0.inverse() * gt_Ts_w_gn[-1]).angle() * 180 / np.pi)

            results_for_evaluation = evaluation.Results(est_Ts_w_rn, gt_Ts_w_gn, len(landmarks_r0))
            evaluated_result = evaluator.evaluate(results_for_evaluation, last_run=False, verbose=sw_cfg.verbose)

        if sw_cfg.show_visualisation or sw_cfg.record_video:
            cart_viz_data = CartUpdateData(landmarks_r0, current_ids, est_Ts_r0_rn, marginals)
            cart_viz_img = cart_viz.update(cart_viz_data)
            if video_collector is None:
                video_collector = VideosSinkArray(full_res_path + "all", fps=10)
            video_collector.add([traj_vis_img, cart_viz_img])
            if sw_cfg.show_visualisation:
                cv2.waitKey(1)

        pose_num += 1

    evaluator.store_plots(sw_cfg.get_iteration_path(hw_cfg.in_path))
    timer.store_plots(sw_cfg.get_iteration_path(hw_cfg.in_path))
    if sw_cfg.record_video:
        video_collector.release()
        traj_vis.release()
        feature_visualizer.release()
    if sw_cfg.show_visualisation:
        plt.show()
    plt.close('all')
    return evaluated_result


if __name__ == '__main__':
    import software_config
    import hardware_config

    main(software_config.SoftwareConfig(), hardware_config.HardwareConfig())
