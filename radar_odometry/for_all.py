import concurrent.futures
import copy
import glob
import multiprocessing
import os
import radar_odometry.main


def safe_main_func(sw_cfg, hw_cfg, main_func=None):
    try:
        return main_func(sw_cfg, hw_cfg)
    except Exception as e:
        iter_path = sw_cfg.get_iteration_path(hw_cfg.in_path)
        error_file_path = iter_path + "error.txt"
        os.makedirs(iter_path, exist_ok=True)
        with open(error_file_path, 'w') as f:
            print("Error occured")
            f.write(str(e))


def unpack(args):
    return safe_main_func(*args)


def for_all_datasets(sw_cfg, hw_cfg, main_func=None, datasets_sel=None):
    if datasets_sel is None:
        datasets_sel = [
            '2018-06-23-22_22_30/',
            '2018-06-25-08_17_00/',
            '2018-06-17-13_42_00/',
            '2018-06-20-20_05_30/',
            '2018-06-24-01_05_00/',
            '2018-07-13-13_00_00/',
            '2018-06-22-22_18_00/',
            '2018-06-15-17_41_30/',
            '2018-06-23-08_28_30/',
        ]
    datase_folder = "/work/Data/polarlys_both/"
    datasets = glob.glob(datase_folder + "*/Radar*", recursive=True)
    if main_func is None:
        main_func = radar_odometry.main.main
    cfgs = []
    for dataset in datasets:
        if any(d in dataset for d in datasets_sel):
            print(f"Starting new dataset: {dataset}")
            new_sw_cfg = copy.copy(sw_cfg)
            new_hw_cfg = copy.copy(hw_cfg)
            new_hw_cfg.in_path = dataset + '/'
            os.makedirs(new_sw_cfg.get_iteration_path(new_hw_cfg.in_path), exist_ok=True)
            cfgs.append((new_sw_cfg, new_hw_cfg, main_func))
        else:
            print(f"Skipping dataset: {dataset}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as p:
        res = p.map(unpack, cfgs)
        res = list(res)
        print(f"res: {len(res)}")
    return res


def for_all_configurations(sw_cfg, hw_cfg, main_func=None, paralell=True):
    if main_func is None:
        main_func = radar_odometry.main.main
    print("Starting for all configurations")
    cfgs = []
    for image_layout in sw_cfg.ImageLayout:
        for estimator in sw_cfg.Estimator:
            if estimator == sw_cfg.Estimator.SinglyObservedBatch or estimator == sw_cfg.Estimator.SinglyObservedISAM2:
                print(f"Skipping estimator: {estimator}")
                continue
            new_sw_cfg = copy.copy(sw_cfg)
            new_hw_cfg = copy.copy(hw_cfg)
            new_sw_cfg.image_layout = image_layout
            new_sw_cfg.estimator_type = estimator
            cfgs.append((new_sw_cfg, new_hw_cfg, main_func))
            print(f"Added new configuration with image_layout: {image_layout} and estimator: {estimator}")

    if paralell:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as p:
            res = p.map(unpack, cfgs)
            res = list(res)
            print(f"res: {len(res)}")
    else:
        res = []
        for cfg in cfgs:
            res.append(safe_main_func(*cfg))
    return res


def for_parameter_range(range: list, sw_cfg, hw_cfg, main_func=None):
    if main_func is None:
        main_func = radar_odometry.main.main
    cfgs = []
    for value in range:
        new_sw_cfg = copy.copy(sw_cfg)
        new_hw_cfg = copy.copy(hw_cfg)
        new_sw_cfg.exclusion_zone_around_existing_features = value
        new_sw_cfg.sub_path = f"exclusion_zone_around_existing_features_{value}/"
        cfgs.append((new_sw_cfg, new_hw_cfg))

    with multiprocessing.Pool(8) as p:
        p.starmap(main_func, cfgs, chunksize=1)


def run_parameter_range():
    import software_config
    import hardware_config
    range = list(range(10, 210, 10))
    sw_cfg = software_config.SoftwareConfig()
    sw_cfg.base_path = 'figa_221024_exclusionon_zon/'
    sw_cfg.show_visualisation = False
    sw_cfg.show_plots = False
    sw_cfg.record_video = False
    sw_cfg.verbose = False
    for_parameter_range(range, sw_cfg, hardware_config.HardwareConfig())


def run_for_all_both():
    import software_config
    import hardware_config
    sw_cfg = software_config.SoftwareConfig()
    sw_cfg.base_path = '/work/Results/fig_230220_all_both_bug_fixed/'
    sw_cfg.show_visualisation = False
    sw_cfg.show_plots = False
    sw_cfg.record_video = True
    sw_cfg.verbose = True
    sw_cfg.expected_duplicate = True
    hw_cfg = hardware_config.HardwareConfig()
    main_func = for_all_datasets
    for_all_configurations(sw_cfg, hw_cfg, main_func, paralell=True)


if __name__ == '__main__':
    # run_parameter_range()
    # import software_config
    # import hardware_config

    # sw_cfg = software_config.SoftwareConfig()
    # sw_cfg.base_path = '/work/Results/fig_230202_all_datasets_all_modes/'
    # sw_cfg.expected_duplicate = True
    # for_all_datasets(sw_cfg=sw_cfg, hw_cfg=hardware_config.HardwareConfig())
    # for_all_configurations(sw_cfg=sw_cfg, hw_cfg=hardware_config.HardwareConfig())
    run_for_all_both()
