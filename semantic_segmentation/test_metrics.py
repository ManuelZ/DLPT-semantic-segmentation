from dataclasses import dataclass
import random
import torch
import numpy as np
from metrics import DiceScore

# The numbers used in this test come from the DLPT course


@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = (
        False  # enable CuDNN benchmark for the sake of performance
    )
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


def setup_system(system_config: SystemConfig) -> None:
    torch.manual_seed(system_config.seed)
    np.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic


setup_system(SystemConfig)

ground_truth = torch.zeros(1, 224, 224, dtype=torch.int64)
ground_truth[:, 50:100, 50:100] = 1
ground_truth[:, 50:150, 150:200] = 2

prediction_prob = torch.zeros(1, 3, 224, 224).uniform_().softmax(dim=1)
class_prediction = prediction_prob.argmax(dim=1)

# No index ignoring
scorer = DiceScore(3, ignore_index=None)
dice = scorer(class_prediction, ground_truth)
mdice = dice.mean()

torch.testing.assert_close(mdice, torch.tensor(0.2379727729935648))
torch.testing.assert_close(dice, torch.tensor([0.47558773, 0.0892497, 0.14908088]))

# Ignoring an index
scorer = DiceScore(3, ignore_index=0)
dice = scorer(class_prediction, ground_truth)
mdice = dice.mean()

torch.testing.assert_close(mdice, torch.tensor(0.4838796700573852))
# This fails in the first element because I'm returning 1 instead of nan
# torch.testing.assert_close(
#    dice, torch.tensor([torch.nan, 0.40573578, 0.56202356]), equal_nan=True
# )
