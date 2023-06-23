import cv2
import numpy as np


class OpticalFlowTracker:

    def __init__(self, sw_cfg):
        self.sw_cfg = sw_cfg
        self.old_gray = None  # Previous image(in grayscale)
        self.current_tracks = None  # The location of each tracked feature in the current/previous frame
        self.feature_ids = []  # The id of the features which are available for tracking and not removed
        self.feature_tracks = {}  # The sequence of observations of each feature
        self.next_feature_id = 0

        self.feature_params_initial = dict(maxCorners=sw_cfg.max_features,
                                           qualityLevel=sw_cfg.shi_tomashi_quality_level,
                                           minDistance=sw_cfg.shi_tomashi_min_dist,
                                           blockSize=sw_cfg.shi_tomashi_block_size)

        self.lk_params = dict(winSize=(sw_cfg.klt_window_size, sw_cfg.klt_window_size),
                              maxLevel=sw_cfg.klt_max_pyr_level,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                              minEigThreshold=1e-3)

    def update(self, frame_gray, center=None):
        """

        @param frame_gray:
        @param center:
        @return: (feature_tracks, active_ids, removed_features)
        removed_features: An array with a number indicating why it was removed. If the number = 1 the minEigThreshold was
         violated, if number  = 2 then the backtrack difference was too large and if number = 3 then both tests failed.
        """

        if self.old_gray is None:
            # Take first frame and find corners in it
            self.old_gray = frame_gray
            self.add_new_features(frame_gray, center)
            return self.feature_tracks.copy(), self.feature_ids.copy(), np.zeros_like(self.feature_ids)

        # Add new features
        len_old_active_features = len(self.current_tracks)
        if len(self.feature_ids) < self.sw_cfg.min_features:
            self.add_new_features(frame_gray, center)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.current_tracks, None, **self.lk_params)
        p2, st2, err = cv2.calcOpticalFlowPyrLK(frame_gray, self.old_gray, p1, None, **self.lk_params)

        # Verify optical flow
        diff = p2 - p1
        dists = np.linalg.norm(diff, axis=2)

        failed_backtrack = dists > self.sw_cfg.klt_max_dist_backtrack
        to_keep = np.logical_and(st == 1, np.logical_not(failed_backtrack))
        removed = 1 * (st != 1) + 2 * failed_backtrack
        # Select good points
        temp_current_tracks = p1[to_keep]
        self.feature_ids = self.feature_ids[to_keep.flatten() == 1]
        removed = removed[:len_old_active_features]
        self.current_tracks = temp_current_tracks.reshape(-1, 1, 2)

        # Build feature tracks
        for (id, point) in zip(self.feature_ids, self.current_tracks):
            if id in self.feature_tracks:
                self.feature_tracks[id].append(point.reshape(-1))
            else:
                self.feature_tracks[id] = [point.reshape(-1)]
        self.old_gray = frame_gray
        ids_out = self.feature_ids.copy()
        self.feature_ids = self.feature_ids.reshape(-1)
        return self.feature_tracks.copy(), ids_out, removed

    def add_new_features(self, frame_gray, center=None):

        mask = np.ones_like(frame_gray, dtype='uint8') * 255
        for id in self.feature_ids:
            p = self.feature_tracks[id][-1]
            cv2.circle(mask, (int(p[0]), int(p[1])), self.sw_cfg.exclusion_zone_around_existing_features, color=0,
                       thickness=-1)

        self.feature_params_initial['maxCorners'] = self.sw_cfg.max_features - len(self.feature_ids)
        temp_features = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params_initial)

        new_tracks = []
        for f in temp_features:
            if self.sw_cfg.image_layout == self.sw_cfg.ImageLayout.Polar:
                if f[0][0] > self.sw_cfg.exclusion_zone_around_ship:
                    new_tracks.append(f)
            elif center is not None and self.sw_cfg.image_layout == self.sw_cfg.ImageLayout.Cartesian:
                if np.linalg.norm(f[0] - center) > self.sw_cfg.exclusion_zone_around_ship:
                    new_tracks.append(f)
            else:
                raise ValueError(
                    f"You either need to pass 'center', or otherwise check your parameters. Center: {center}, image_layout: {self.sw_cfg.image_layout}, f[0]:{f[0]}, norm(f[0]):{np.linalg.norm(f[0])}")
        if self.current_tracks is None:
            self.current_tracks = np.array(new_tracks)
        elif len(new_tracks) == 0:
            print(f"There was no new features found")
        else:
            self.current_tracks = np.concatenate((self.current_tracks, np.array(new_tracks)), axis=0)

        new_feature_ids = [i for i in range(self.next_feature_id, self.next_feature_id + len(new_tracks))]
        new_feature_ids = np.array(new_feature_ids, dtype='int32').reshape(-1)
        self.feature_ids = np.concatenate((self.feature_ids, new_feature_ids)).astype('int32')
        self.next_feature_id = max(self.feature_ids) + 1
        # Build feature tracks
        for (id, point) in zip(new_feature_ids, new_tracks):
            if id in self.feature_tracks:
                self.feature_tracks[id].append(point.reshape(-1))
            else:
                self.feature_tracks[id] = [point.reshape(-1)]
