import sys
sys.path.append("..") # Add higher directory to python modules path.
 
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import torch
import mlflow
from .utils import percentage_acc
from config import GraphTextureNetworkConfig
 
def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader, 
    criterion: torch.nn, 
    optimizer: torch.optim, 
    device: torch.device, 
    cfg: GraphTextureNetworkConfig,
    epoch: int,
    train_acc: dict,
    train_loss: dict
) -> Tuple[Dict[str, int], Dict[str, int]]:
    
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        criterion (torch.nn): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to run the training on.
        cfg (GraphTextureNetworkConfig): Configuration object.
        epoch (int): The current epoch number.
        train_acc (dict): Dictionary to store training accuracy for each epoch.
        train_loss (dict): Dictionary to store training loss for each epoch.

    Returns:
        Tuple[Dict[int, float], Dict[int, float]]: Updated training accuracy and loss dictionaries.
    """
    
    model.train()
    current_total_train_loss, correct, total = 0, 0, 0
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    prog_bar = tqdm(range(len(train_loader)), file=sys.stdout, bar_format=bar_format)
 
    data_loader = iter(train_loader)
 
    for batch_idx in prog_bar:
        data, target = next(data_loader)
        if cfg.accelerator.cuda:
            data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, _,_ = model(x = data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
 
        current_total_train_loss += loss.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum().numpy()
        total += target.size(0)
        
        current_train_acc = percentage_acc(correct = correct, 
                                               total = total)
        
        print_str = f'Epoch: {epoch}/{cfg.training.num_epochs}  ' \
                        + f'Iter: {batch_idx + 1}/{len(train_loader)}  ' \
                        + f'Loss: {current_total_train_loss / (batch_idx + 1):.3f} |' \
                        + f'Accuracy: {current_train_acc:.3f}% ({correct}/{total})'
        
        prog_bar.set_description(print_str)
 
    train_acc[epoch] = current_train_acc
    train_loss[epoch] = current_total_train_loss / (batch_idx + 1)
 
    return train_acc, train_loss