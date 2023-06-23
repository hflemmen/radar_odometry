from dataclasses import dataclass
import enum

"""
This file contains parameters which define the behaviour of the radar odometry. Some of can be changed to test different
configurations. Others must be tuned to the data you have."""


@dataclass
class SoftwareConfig:
    class Estimator(enum.Enum):
        FullHistory = 0
        ISAM2 = 1  # <-- The one used in the paper.
        SinglyObservedBatch = 2  # <-- Not recommended
        SinglyObservedISAM2 = 3  # <-- Not recommended

    class ImageLayout(enum.Enum):
        Polar = 0  # <-- The one used in the paper
        Cartesian = 1  # <-- Not recommended. Works poorly with the current feature detection and management

    estimator_type = Estimator.ISAM2

    image_layout = ImageLayout.Polar

    ## Preprocessing
    resample_ranges = True  # True if it should resample the far away distances to a uniform scale.

    # Spatial bluring
    sigma_range = 0.6
    sigma_angle = 0.8
    size_range = 15
    size_angle = size_range

    ## Feature detection and tracking
    max_features = 84
    min_features = 38
    shi_tomashi_block_size = 10
    shi_tomashi_min_dist = 30
    shi_tomashi_quality_level = 0.46
    klt_window_size = 15
    klt_max_pyr_level = 4
    klt_max_dist_backtrack = 40  # Threshold for maximum error in eucledian distance for forward/backward klt.
    exclusion_zone_around_existing_features = 10
    exclusion_zone_around_ship = 10  # [pixels] Exclude features too close to the ship.
    # Set the value for how much we subtract from the raw image to reduce noise.
    wave_noise_threshold = 10

    ## Estimation
    range_noise = 10  # [m]
    angular_noise = 0.02  # [rad]

    relinearize_threshold = 0.1  # Only for isam2

    relative_error_tol = 2e-4  # Only for full history optimization

    max_iterations = 200
    huber_k = 0.1  # The huber cutoff threshold

    ## Evaluation
    rpe_interval = 5  # [iterations]
    home_folder = "/home/henrik/"
    base_path = home_folder + 'Results/fig_2300623_release/'
    sub_path = ''
    verbose = True
    show_visualisation = True
    show_plots = False
    record_video = True
    expected_duplicate = False

    ## Other
    use_cache = False

    def get_iteration_path(self, dataset_path):
        return f"{self.base_path}{dataset_path.split('/')[-3]}/{str(self.estimator_type).split('.')[-1]}/{str(self.image_layout).split('.')[-1]}/{self.sub_path}"
