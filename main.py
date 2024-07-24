# --------------------------------------------------------
# Multiscale Graph Texture Network (GTN)
# Author: Ravishankar Evani
# --------------------------------------------------------
 
import hydra
from hydra.core.config_store import ConfigStore
from config import GraphTextureNetworkConfig
import mlflow
import torch
import torch.optim as optim
import dataloader
from src.loss import PreAveragedFocalLoss
from src.utils import seed_everything, folder_setup
from src.build import build_model
from src.train import train_epoch
from src.test import test_epoch
 
# Registering the Config class with the name 'graph_texture_net_config'.
cs = ConfigStore.instance()
cs.store(name='graph_texture_net_config', node=GraphTextureNetworkConfig)
 
 
def setup(
    cfg: GraphTextureNetworkConfig
    ) -> GraphTextureNetworkConfig:
    
    """
    Sets up the MLflow experiment tracking, PyTorch autologging, folder structure, 
    random seed, and CUDA availability based on the given configuration.

    Args:
        cfg (GraphTextureNetworkConfig): Configuration object containing various experimental settings.

    Returns:
        GraphTextureNetworkConfig: Updated configuration object based on CUDA availability.
    """
    
    # Set the registry URI for MLflow
    mlflow.set_registry_uri(cfg.tracking.uri)
    
    # Set the experiment name in MLflow with experimental details
    mlflow.set_experiment(cfg.experiment + '_' + f'seed_{cfg.training.seed}_{cfg.backbone}_layers_{cfg.common.layers}\
                            _aug_{cfg.common.additional_augmentation}')
    
    # Disable PyTorch autologging in MLflow
    mlflow.pytorch.autolog(disable=True)
    
    # Setup necessary folders for the experiment
    folder_setup(cfg)
    
    # Set the random seed for reproducibility
    seed_everything(seed=cfg.training.seed)
    
    # Check if CUDA is available and set the corresponding flag in the configuration object
    cfg.accelerator.cuda = cfg.accelerator.cuda and torch.cuda.is_available()
    
    return cfg
    
    
@hydra.main(version_base=None, config_path="conf/dataset", config_name="dtd")
def main(
    cfg: GraphTextureNetworkConfig
) -> None:
    
    """
    Main function to set up the experiment, initialize MLflow tracking, 
    prepare the data loaders, model, loss function, optimizer, and scheduler, 
    and run the training and evaluation loops.

    Args:
        cfg (GraphTextureNetworkConfig): Configuration object containing all the necessary 
                                         settings for the experiment.
    """
    
    # Set up experiment and folder and update configurtion object based on CUDA availability
    cfg = setup(cfg)
    
    # Create a unique run name based on the architecture, backbone, and device
    run_name = cfg.common.architecture_name + '_' + cfg.backbone + '_' + torch.cuda.get_device_name(int(cfg.accelerator.device))
    
    # Add split information to the run name
    run_name += '_' + str(cfg.training.split)
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        
        # Set MLflow tag (backbone used) and log parameters
        mlflow.set_tag("backbone", cfg.backbone)
        mlflow.log_params(cfg.common[cfg.backbone])
        
        # Set MLflow tag (split information)
        mlflow.set_tag("split", cfg.training.split)
        
        train_acc = {}
        test_acc = {}
        train_loss = {}
        test_loss = {}
 
        # Initialize the dataloader
        dl = dataloader.dtd.Dataloader(cfg)
            
        # Get the classes, train and test loaders from the dataloader
        classes, train_loader, test_loader = dl.getloader()
 
        # Instantiate model
        model = build_model(cfg = cfg, 
                              num_classes = len(classes))
 
        # Instantiate loss function
        criterion = PreAveragedFocalLoss(alpha = cfg.loss.alpha, 
                                         gamma = cfg.loss.gamma)
        
        # Instantiate optimizer
        optimizer = optim.AdamW(params = model.parameters(), 
                                lr=cfg.optimizer.lr,
                                weight_decay=cfg.optimizer.weight_decay)
        
        # Instantiate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, 
                                                                eta_min = cfg.optimizer.lr * cfg.scheduler.eta_min_scaling_factor,
                                                                T_max = cfg.training.num_epochs)
 
        device = torch.device(f"cuda:{cfg.accelerator.device}" if cfg.accelerator.cuda else "cpu")
        model.to(device)
    
        # Train and evaluate GTN
        for epoch in range(1, cfg.training.num_epochs + 1):
            print('Epoch:', epoch)
            
            # Train the model for one epoch
            train_acc, train_loss = train_epoch(model = model,
                                                train_loader = train_loader, 
                                                criterion = criterion, 
                                                optimizer = optimizer, 
                                                device = device, 
                                                cfg = cfg,
                                                epoch = epoch,
                                                train_acc = train_acc,
                                                train_loss = train_loss)
            
            # Evaluate the model for one epoch
            test_acc, test_loss = test_epoch(model = model,
                                            test_loader = test_loader, 
                                            criterion = criterion, 
                                            device = device, 
                                            cfg = cfg,
                                            epoch = epoch,
                                            test_acc = test_acc,
                                            test_loss = test_loss)
            
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                'train_acc': train_acc[epoch],
                'test_acc': test_acc[epoch],
                'train_loss': train_loss[epoch],
                'test_loss': test_loss[epoch],
                'epoch': epoch
            }, step=epoch)
            
            
            # Step the scheduler
            scheduler.step()
            
        # Log the trained model to MLflow
        mlflow.pytorch.log_model(pytorch_model = model, 
                                 artifact_path = "model_artifact")
        
        # End the MLflow run
        mlflow.end_run()
 
if __name__ == "__main__":
    main()