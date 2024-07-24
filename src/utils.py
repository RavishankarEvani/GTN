import os
import torch
import random
import numpy as np
from typing import Tuple
from config import GraphTextureNetworkConfig


def seed_everything(
    seed: int = 0
) -> None:
    """
    Sets the seed for various random number generators and 
    configurations to ensure that the results are reproducible. It affects 
    Python's `random` module, the environment variable `PYTHONHASHSEED`, 
    NumPy, and PyTorch.

    Args:
        seed (int, optional): The seed value to use. Default is 0.

    Returns:
        None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_tuple(
    x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a single value into a tuple containing two copies of the value.

    Args:
        x (torch.Tensor): The input value to be converted into a tuple.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two copies of the input value.
    """
    return (x, x)

def percentage_acc(
    correct: int, 
    total: int
) -> float:
    """
    Calculate the percentage accuracy of predictions.

    This function calculates the accuracy as the percentage of correct predictions
    out of the total predictions.

    Args:
        correct (int): The number of correct predictions.
        total (int): The total number of predictions.

    Returns:
        percentage_acc (float): The percentage accuracy of the predictions.
    """
    percentage_acc = 100. * correct / total
    
    return percentage_acc
    


def save_log(
    log_str: str, 
    cfg: GraphTextureNetworkConfig,
    mode: str = 'train'
) -> None:
    
    """
    Save training and testing performance metrics to a text file by appending a log string to a specified log file. 

    Args:
        log_str (str): The log string to be saved to the file.
        cfg (GraphTextureNetworkConfig): Configuration object containing paths and other settings.
        mode (str, optional): The mode of operation ('train' or 'test'). Default is 'train'.

    Returns:
        None
    """
    # If file exists, append a new record to the file.
    with open(os.path.join(cfg.paths.log, f'split_{cfg.training.split}.txt') , 'a') as file:
        file.write(f'({mode.upper()}) ' + log_str + '\n')


def folder_setup(
    cfg: GraphTextureNetworkConfig
) -> None:
    
    """
    Updates the paths (based on experiment and backbone) in the configuration object and ensures that the necessary 
    directories for logging and snapshots exist.

    Args:
        cfg (GraphTextureNetworkConfig): Configuration object containing paths, experiment name and backbone information.

    Returns:
        None
    """
    
    cfg.paths.data = os.path.join(cfg.paths.data, cfg.experiment + os.path.sep)
    cfg.paths.log = os.path.join(cfg.paths.log, cfg.experiment , cfg.backbone + os.path.sep)
    cfg.paths.snapshot = os.path.join(cfg.paths.snapshot, cfg.experiment , cfg.backbone + os.path.sep)
    
    os.makedirs(cfg.paths.log, exist_ok=True)
    os.makedirs(cfg.paths.snapshot, exist_ok=True)

