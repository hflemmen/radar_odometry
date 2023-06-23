import os
import time
from dataclasses import dataclass
from typing import List, overload
import manifpy as mf
import numpy as np
import matplotlib.pyplot as plt

import radar_odometry.software_config


@dataclass
class Results:
    Ts_w_r: List[mf.SE2]
    Ts_w_g: List[mf.SE2]
    num_of_landmarks: int


@dataclass
class CombinedResults:
    rpes_rot: List[float]
    rpes_trans: List[float]
    ates_rot: List[float]
    ates_trans: List[float]

    def __init__(self):
        self.rpes_rot = []
        self.rpes_trans = []
        self.ates_rot = []
        self.ates_trans = []

    def append(self, rpe_rot, rpe_trans, ate_rot, ate_trans):
        self.rpes_rot.append(rpe_rot)
        self.rpes_trans.append(rpe_trans)
        self.ates_rot.append(ate_rot)
        self.ates_trans.append(ate_trans)


@dataclass
class EvaluationOutput:
    rmse_rpe_rot: float
    rmse_rpe_trans: float
    rmse_ate_rot: float
    rmse_ate_trans: float
    max_rpe_rot: float
    max_rpe_trans: float
    max_ate_rot: float
    max_ate_trans: float


@dataclass
class EvaluatorInitializer:
    rpe_interval: int
    estimator_type: radar_odometry.software_config.SoftwareConfig.Estimator
    image_layout: radar_odometry.software_config.SoftwareConfig.ImageLayout
    show_plots: bool
    sub_path: str
    dataset_path: str


class Evaluator:
    def __init__(self, initializer: EvaluatorInitializer):
        self.rpe_interval = initializer.rpe_interval
        self.estimator_type = initializer.estimator_type
        self.image_layout = initializer.image_layout
        self.show_plots = initializer.show_plots
        self.disp_rpe_trans = ErrorDisp(0, "RPE trans", "Norm of translational RPE [m]", self.show_plots)
        self.disp_rpe_rot = ErrorDisp(1, "RPE rot", "Norm of rotational RPE [deg]", self.show_plots)
        self.disp_ate_trans = ErrorDisp(2, "ATE trans", "Norm of translational ATE [m]", self.show_plots)
        self.disp_ate_rot = ErrorDisp(3, "ATE rot", "Norm of rotational ATE [deg]", self.show_plots)
        self.num_landmark_hist = []
        self.hist = CombinedResults()
        self.results: Results = None

    def evaluate(self, res: Results, last_run=False, verbose=False):
        rmse_rpe_rot, rmse_rpe_trans, rmses_rpe_rots, rmses_rpe_trans = calc_rpe(res, self.rpe_interval)
        if verbose:
            print("RMSE RPE ROT:", rmse_rpe_rot)
            print("RMSE RPE TRANS:", rmse_rpe_trans)
        self.disp_rpe_trans.disp_error(rmses_rpe_trans)
        self.disp_rpe_rot.disp_error(rmses_rpe_rots)
        if last_run:
            self.disp_rpe_trans.store_plot('fig/', 'RPE_trans')
            self.disp_rpe_rot.store_plot('fig/', 'RPE_rot')

        rmse_ate_rot, rmse_ate_trans, rmses_ate_rots, rmses_ate_trans = calc_ate(res)
        if verbose:
            print("RMSE ATE ROT:", rmse_ate_rot)
            print("RMSE ATE TRANS:", rmse_ate_trans)
        self.disp_ate_trans.disp_error(rmses_ate_trans)
        self.disp_ate_rot.disp_error(rmses_ate_rots)
        if last_run:
            self.disp_ate_trans.store_plot('fig/', 'ATE_trans')
            self.disp_ate_rot.store_plot('fig/', 'ATE_rot')

        self.num_landmark_hist.append(res.num_of_landmarks)

        self.hist.append(rmse_rpe_rot, rmse_rpe_trans, rmse_ate_rot, rmse_ate_trans)

        self.results = res

        res = self.evaluation_results()
        return res

    def evaluation_results(self):
        rmse_rpe_rot = np.sqrt(
            sum([(x * (180.0 / np.pi)) ** 2 for x in self.hist.rpes_rot]) / len(self.hist.rpes_rot))
        rmse_rpe_trans = np.sqrt(sum([x ** 2 for x in self.hist.rpes_trans]) / len(self.hist.rpes_trans))
        rmse_ate_rot = np.sqrt(
            sum([(x * (180.0 / np.pi)) ** 2 for x in self.hist.ates_rot]) / len(self.hist.ates_rot))
        rmse_ate_trans = np.sqrt(sum([x ** 2 for x in self.hist.ates_trans]) / len(self.hist.ates_trans))
        max_rpe_rot = max(self.hist.rpes_rot) * (180.0 / np.pi)
        max_rpe_trans = max(self.hist.rpes_trans)
        max_ate_rot = max(self.hist.ates_rot) * (180.0 / np.pi)
        max_ate_trans = max(self.hist.ates_trans)

        return EvaluationOutput(rmse_rpe_rot, rmse_rpe_trans, rmse_ate_rot, rmse_ate_trans, max_rpe_rot, max_rpe_trans,
                                max_ate_rot, max_ate_trans)

    def store_plots(self, local_path):
        fullpath = local_path
        self.disp_rpe_trans.store_plot(fullpath, 'RPE_trans')
        self.disp_rpe_rot.store_plot(fullpath, 'RPE_rot')
        self.disp_ate_trans.store_plot(fullpath, 'ATE_trans')
        self.disp_ate_rot.store_plot(fullpath, 'ATE_rot')

        with open(fullpath + "res.txt", 'w') as f:
            f.write("Rot RPE[deg]:\n")
            rmse_rpe_rot = np.sqrt(
                sum([(x * (180.0 / np.pi)) ** 2 for x in self.hist.rpes_rot]) / len(self.hist.rpes_rot))
            f.write(f"RMSE:{rmse_rpe_rot}\n")
            f.write(f"MAX:{max(self.hist.rpes_rot) * (180.0 / np.pi)}\n")
            f.write("Trans RPE:\n")
            rmse_rpe_trans = np.sqrt(sum([x ** 2 for x in self.hist.rpes_trans]) / len(self.hist.rpes_trans))
            f.write(f"RMSE:{rmse_rpe_trans}\n")
            f.write(f"MAX:{max(self.hist.rpes_trans)}\n")
            f.write("Rot ATE[deg]:\n")
            rmse_ate_rot = np.sqrt(
                sum([(x * (180.0 / np.pi)) ** 2 for x in self.hist.ates_rot]) / len(self.hist.ates_rot))
            f.write(f"RMSE:{rmse_ate_rot}\n")
            f.write(f"MAX:{max(self.hist.ates_rot) * (180.0 / np.pi)}\n")
            f.write("Trans ATE:\n")
            rmse_ate_trans = np.sqrt(sum([x ** 2 for x in self.hist.ates_trans]) / len(self.hist.ates_trans))
            f.write(f"RMSE:{rmse_ate_trans}\n")
            f.write(f"MAX:{max(self.hist.ates_trans)}\n")

        with open(fullpath + "traj.csv", 'w') as f:
            f.write("rx,ry,xyaw,gx,gy,gyaw\n")
            for i in range(len(self.results.Ts_w_r)):
                f.write(
                    f"{self.results.Ts_w_r[i].x()},{self.results.Ts_w_r[i].y()},{self.results.Ts_w_r[i].angle()},{self.results.Ts_w_g[i].x()},{self.results.Ts_w_g[i].y()},{self.results.Ts_w_g[i].angle()}\n")


def read_from_traj_csv(path, rpe_interval) -> Evaluator:
    with open(path, 'r') as f:
        f.readline()
        lines = f.readlines()
        Ts_w_r = []
        Ts_w_g = []
        eval_init = EvaluatorInitializer(rpe_interval, None, None, False, "from_file/", None)
        eval = Evaluator(eval_init)
        for line in lines:
            line = line.split(',')
            Ts_w_r.append(mf.SE2(float(line[0]), float(line[1]), float(line[2])))
            Ts_w_g.append(mf.SE2(float(line[3]), float(line[4]), float(line[5])))
            res = Results(Ts_w_r, Ts_w_g, 0)
            eval.evaluate(res, False, False)
    return eval


class MultiEvaluator:

    def __init__(self, initializers: List[Evaluator], names: List[str]):
        self.evaluators = []
        self.names = names
        for initializer in initializers:
            self.evaluators.append(initializer)

    def evaluate(self, ress: List[Results], last_run=False, verbose=False):
        print("If you use this, remove this print statement.")
        for i, res in enumerate(ress):
            self.evaluators[i].evaluate(res, last_run, verbose)

    def add_evaluator(self, evaluator, name):
        self.evaluators.append(evaluator)
        self.names.append(name)

    def store_error_plots(self, path, title):
        ### RPE
        fig = plt.figure()
        # plt.rcParams["text.usetex"] = "True"

        plt.title(title)
        axs = plt.subplot(2, 1, 1)
        for name, evaluator in zip(self.names, self.evaluators):
            axs.plot(evaluator.hist.rpes_rot, label=name)
        axs.set_title(f"RPE (interval: {self.evaluators[0].rpe_interval})")
        axs.set_ylabel("Rotational [deg]")
        # axs.set_xlabel("# iterations")
        # axs.set_yscale('log')
        plt.legend()
        axs = plt.subplot(2, 1, 2)
        for name, evaluator in zip(self.names, self.evaluators):
            axs.plot(evaluator.hist.rpes_trans, label=name)
        # axs.set_title("RPE")
        axs.set_ylabel("Translational [m]")
        axs.set_xlabel("# iterations")
        # axs.set_yscale('log')
        plt.legend()
        plt.draw()
        plt.savefig(path + "RPE" + title + ".pdf")
        # fig.close()
        # plt.show()

        ### ATE
        fig = plt.figure()
        # plt.rcParams["text.usetex"] = "True"
        plt.title(title)
        axs = plt.subplot(2, 1, 1)
        for name, evaluator in zip(self.names, self.evaluators):
            axs.plot(evaluator.hist.ates_rot, label=name)
        axs.set_title(f"ATE")
        axs.set_ylabel("Rotational [deg]")
        # axs.set_xlabel("# iterations")
        # axs.set_yscale('log')
        plt.legend()
        axs = plt.subplot(2, 1, 2)
        for name, evaluator in zip(self.names, self.evaluators):
            axs.plot(evaluator.hist.ates_trans, label=name)
        # axs.set_title("ATE")
        axs.set_ylabel("Translational [m]")
        axs.set_xlabel("# iterations")
        # axs.set_yscale('log')
        plt.legend()
        plt.draw()
        plt.savefig(path + "ATE" + title + ".pdf")
        # fig.close()
        # plt.show()

    def store_xy_plots(self, path, title):
        # plt.figure()
        # plt.title(title)
        names = []
        Ts_w_r = []
        for name, evaluator in zip(self.names, self.evaluators):
            names.append(name)
            Ts_w_r.append(evaluator.results.Ts_w_r)
        # names.append("Yeti")
        # names.append("Ours")
        names.append("Ships pos ref")
        Ts_w_r.append(self.evaluators[1].results.Ts_w_g)
        # names.append("Ground truth yeti")
        # Ts_w_r.append(self.evaluators[0].results.Ts_w_g)
        store_xy(Ts_w_r, names, path, title)
        # plt.draw()
        # plt.savefig(path + "xy" + title + ".pdf")
        # plt.show()


def store_xy(res: List[List[mf.SE2]], names, path, title):
    rxs = []
    rys = []
    starting_points = []
    for alg in res:
        starting_points.append(alg[0])
    for e in range(len(res)):
        rxs.append([])
        rys.append([])
        for i in range(len(res[e])):
            rxs[e].append((starting_points[e].inverse() * res[e][i]).x())
            rys[e].append((starting_points[e].inverse() * res[e][i]).y())
    plt.figure()
    for e in range(len(rxs)):
        plt.plot(rxs[e], rys[e], label=names[e])

    plt.legend()
    plt.title(f"Dataset {title}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.draw()
    # plt.show()
    plt.savefig(path + "xy.pdf")


def read_evaluation_results(path):
    with open(path, 'r') as f:
        f.readline()
        rmse_rpe_rot = float(f.readline().split(":")[1])
        max_rpe_rot = float(f.readline().split(":")[1])
        f.readline()
        rmse_rpe_trans = float(f.readline().split(":")[1])
        max_rpe_trans = float(f.readline().split(":")[1])
        f.readline()
        rmse_ate_rot = float(f.readline().split(":")[1])
        max_ate_rot = float(f.readline().split(":")[1])
        f.readline()
        rmse_ate_trans = float(f.readline().split(":")[1])
        max_ate_trans = float(f.readline().split(":")[1])
    return EvaluationOutput(rmse_rpe_rot, rmse_rpe_trans, rmse_ate_rot, rmse_ate_trans, max_rpe_rot, max_rpe_trans,
                            max_ate_rot, max_ate_trans)


def calc_rpe(res: Results, offset: int) -> (float, float, List[float], List[float]):
    """
    Assumes the poses and ground truth lists are equally long and that each index corresponds to the same timestamp.
    @param res:
    @param offset:
    @return:
    """
    rpes = []
    for i in range(len(res.Ts_w_r) - offset):
        T_w_ri = res.Ts_w_r[i]
        T_w_ri1 = res.Ts_w_r[i + offset]
        T_w_gi = res.Ts_w_g[i]
        T_w_gi1 = res.Ts_w_g[i + offset]
        rpes.append((T_w_gi.inverse() * T_w_gi1).inverse() * (T_w_ri.inverse() * T_w_ri1))

    rpe_rots = [abs(x.angle()) for x in rpes]
    rpe_transs = [np.linalg.norm(x.translation()) for x in rpes]

    rmse_rot = np.sqrt(sum([x.angle() ** 2 for x in rpes]) / len(res.Ts_w_r))
    rmse_trans = np.sqrt(sum([np.linalg.norm(x.translation()) ** 2 for x in rpes]) / len(res.Ts_w_r))

    return rmse_rot, rmse_trans, rpe_rots, rpe_transs


def calc_ate(res: Results) -> (float, float, List[float], List[float]):
    T_wr_r0 = res.Ts_w_r[0]
    T_wg_g0 = res.Ts_w_g[0]

    T_wg_wr = T_wg_g0 * T_wr_r0.inverse()  # Vi definerer g0 == r0

    ates_rot = []
    ates_trans = []
    for i in range(len(res.Ts_w_r)):
        T_wr_ri = res.Ts_w_r[i]
        T_wg_gi = res.Ts_w_g[i]
        ate = T_wg_gi.inverse() * T_wg_wr * T_wr_ri
        ates_rot.append(abs(ate.angle()))
        ates_trans.append(np.linalg.norm(ate.translation()))

    ate_rmse_rot = np.sqrt(sum([x ** 2 for x in ates_rot]) / len(ates_rot))
    ate_rmse_trans = np.sqrt(sum(x ** 2 for x in ates_trans) / len(ates_trans))

    return ate_rmse_rot, ate_rmse_trans, ates_rot, ates_trans


class ErrorDisp:

    def __init__(self, fignum, title, ylabel, show_plots):
        self.fignum = fignum
        self.title = title
        self.ylabel = ylabel
        self.show_plots = show_plots
        self.prev_error = None

    def disp_error(self, error: List[float]):
        self.prev_error = error
        if self.show_plots:
            fig = plt.figure(num=self.fignum)
            fig.clf()
            plt.title(self.title)
            plt.xlabel("# iterations")
            plt.ylabel(self.ylabel)
            plt.plot(error)

            plt.draw()
            plt.pause(0.0001)

    def store_plot(self, fullpath, error_type):
        os.makedirs(fullpath, exist_ok=True)
        filename = f"{error_type}.pdf"
        fig = plt.figure(num=self.fignum)
        if not self.show_plots:
            fig.clf()
            plt.title(self.title)
            plt.xlabel("# iterations")
            plt.ylabel(self.ylabel)
            plt.plot(self.prev_error)
        plt.draw()
        plt.savefig(fullpath + filename, dpi=800, format='pdf')


class Timing:
    def __init__(self):
        # self.name = name
        self.results = []

    def tic(self):
        self.start = time.perf_counter_ns()

    def toc(self):
        now = time.perf_counter_ns()
        if self.start is None:
            raise Exception("Time is not ticed")
        self.results.append(now - self.start)
        self.start = None

    def get_stats(self):
        max_time = max(self.results)
        mean_time = sum(self.results) / len(self.results)
        return mean_time, max_time

    def store_plots(self, local_path):
        mean_time_ns, max_time_ns = self.get_stats()
        mean_time = mean_time_ns / 1e9
        max_time = max_time_ns / 1e9
        with open(local_path + "timing.txt", "w") as f:
            f.write(f"Mean time: {mean_time}\n")
            f.write(f"Max time: {max_time}\n")
