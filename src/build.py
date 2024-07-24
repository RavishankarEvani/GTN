from src.model import GTN
from config import GraphTextureNetworkConfig
import torch.nn as nn

def build_model(
        cfg: GraphTextureNetworkConfig, 
        num_classes: int
) -> nn.Module:
        
    """
    Returns a Graph Texture Network (GTN) model based on the given configuration.

    Args:
        cfg (GraphTextureNetworkConfig): Configuration object containing model and training settings.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: An instance of the GTN model configured as per settings in the configuration object.
    """

    cfg_model = cfg.common[cfg.backbone]

    model = GTN(
            cfg = cfg,
            n_classes = num_classes,
            depth_dims = cfg_model.depth_dims,
            spatial_dims = cfg_model.spatial_dims,
            embedding_dim = cfg_model.embedding_dim,
            depth_compression_ratio = cfg_model.depth_compression_ratio,
            backbone_name = cfg.backbone,
            fine_tune_backbone = cfg_model.fine_tune_backbone
            )
    
    return model