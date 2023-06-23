import sys
from dataclasses import dataclass


@dataclass
class ResultData:
    RPE_ROT_RMSE: float
    RPE_ROT_MAX: float
    RPE_TRANS_RMSE: float
    RPE_TRANS_MAX: float
    ATE_ROT_RMSE: float
    ATE_ROT_MAX: float
    ATE_TRANS_RMSE: float
    ATE_TRANS_MAX: float


def read_res(path: str) -> ResultData:
    print("Deceprecated. Use the same function in evaluation.py", file=sys.stderr)
    with open(path, 'r') as f:
        f.readline()
        RPE_ROT_RMSE = float(f.readline().split(':')[-1])
        RPE_ROT_MAX = float(f.readline().split(':')[-1])
        f.readline()
        RPE_TRANS_RMSE = float(f.readline().split(':')[-1])
        RPE_TRANS_MAX = float(f.readline().split(':')[-1])
        f.readline()
        ATE_ROT_RMSE = float(f.readline().split(':')[-1])
        ATE_ROT_MAX = float(f.readline().split(':')[-1])
        f.readline()
        ATE_TRANS_RMSE = float(f.readline().split(':')[-1])
        ATE_TRANS_MAX = float(f.readline().split(':')[-1])
    res = ResultData(RPE_ROT_RMSE, RPE_ROT_MAX, RPE_TRANS_RMSE, RPE_TRANS_MAX, ATE_ROT_RMSE, ATE_ROT_MAX,
                     ATE_TRANS_RMSE, ATE_TRANS_MAX)

    return res
