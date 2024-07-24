import sys
sys.path.append("..") # Add higher directory to python modules path.
 
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import mlflow
import os
from typing import Tuple, Dict
from .utils import percentage_acc
from config import GraphTextureNetworkConfig
 
def test_epoch(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    criterion: torch.nn, 
    device: torch.device, 
    cfg: GraphTextureNetworkConfig,
    epoch: int,
    test_acc: list,
    test_loss: dict
) -> Tuple[Dict[int, float], Dict[int, float]]:
    
    
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to run the evaluation on.
        cfg (GraphTextureNetworkConfig): Configuration object.
        epoch (int): The current epoch number.
        test_acc (dict): Dictionary to store test accuracy for each epoch.
        test_loss (dict): Dictionary to store test loss for each epoch.

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: Updated test accuracy and loss dictionaries.
    """
    
    model.eval()
    current_total_test_loss, correct, total = 0, 0, 0
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    prog_bar = tqdm(range(len(test_loader)), file=sys.stdout, bar_format=bar_format)
 
    data_loader = iter(test_loader)
 
    with torch.no_grad():
        for batch_idx in prog_bar:
            data, target = next(data_loader)
            if cfg.accelerator.cuda:
                data, target = data.to(device), target.to(device)
            output, _, _ = model(x = data)
            current_total_test_loss += criterion(output, target).item()

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().numpy()
            total += target.size(0)
 
            current_test_acc = percentage_acc(correct = correct, 
                                               total = total)
            
            print_str = f'Epoch: {epoch}/{cfg.training.num_epochs}  ' \
                        + f'Iter: {batch_idx + 1}/{len(test_loader)}  ' \
                        + f'Loss: {current_total_test_loss / (batch_idx + 1):.3f} |' \
                        + f'Accuracy: {current_test_acc:.3f}% ({correct}/{total})'
                        
            prog_bar.set_description(print_str)

 
    test_acc[epoch] = current_test_acc
    test_loss[epoch] = current_total_test_loss / (batch_idx + 1)
 
    
    return test_acc, test_loss