from typing import List, Tuple, Dict, Union, Optional, Callable
import torch
import torch.nn as nn
import timm


class ConvNeXt(nn.Module):
    
    """
    ConvNeXt class.

    This class defines a ConvNeXt backbone for feature extraction. It allows extraction
    of feature maps from specified layers of the backbone.

    Attributes:
        model (nn.Module): The ConvNeXt backbone.
        init (list): A list containing a single stem layer used to extract initial feature maps.
        layers (list): A list of subsequent layers used to extract additional feature maps.
        feature_maps (dict): A dictionary for storing feature maps extracted from the selected layers.

    Methods:
        forward_hook(name: str) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
            Creates a hook to store feature maps from the specified layer.
        forward(x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            Forward pass through the model to extract feature maps.
    """
    
    def __init__(
        self,
        in_channels: int,
        model_name: str = 'convnext_tiny'
    ):
        super(ConvNeXt, self).__init__()
        
        """
        Initializes the ConvNeXt class.

        Args:
            in_channels (int): Number of input channels.
            model_name (str, optional): Name of the ConvNeXt model. Default is 'convnext_tiny'.
        """
        
        # Instantiate model
        self.model = timm.create_model(model_name,  pretrained=True, in_chans=in_channels)
        
        # Define the layers you want to extract feature maps from
        self.init = ['stem']
        self.layers = [0,1,2,3]

        # Create hooks to store feature maps for the selected layers
        self.feature_maps = {name: None for name in self.init + self.layers}
        for name in self.init:
            hook = getattr(self.model, name).register_forward_hook(self.forward_hook(name))
            setattr(self, f"{name}_hook", hook)

        for name in self.layers:
            hook = getattr(self.model.stages, str(name)).register_forward_hook(self.forward_hook(name))
            setattr(self, f"stage{name}_hook", hook)
        
    def forward_hook(
        self, 
        name: str
    ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        """
        Creates a hook to store feature maps from the specified layer.

        Args:
            name (str): The name of the layer to create a hook for.

        Returns:
            Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]: The hook function to store the feature maps.
        """
        def hook(
            module: nn.Module, 
            input: Tuple[torch.Tensor, ...], 
            output: torch.Tensor
        ) -> None:
            
            self.feature_maps[name] = output
            
        return hook
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model to extract feature maps.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
                Feature maps from the specified layers.
        """
        
        _ = self.model(x)

        a0 = self.feature_maps['stem']  # stem
        x1 = self.feature_maps[0]  # stage0
        x2 = self.feature_maps[1]  # stage1
        x3 = self.feature_maps[2]  # stage2
        x4 = self.feature_maps[3]  # stage3
        
        return a0, x1, x2, x3, x4
    
    
def backbone_selection(
    backbone_name: str = 'convnext_tiny'
) -> ConvNeXt:
    """
    Selects and returns the specified pre-trained ConvNeXt backbone.

    Args:
        backbone_name (str, optional): The name of the ConvNeXt backbone to use. Default is 'convnext_tiny'.

    Returns:
        ConvNeXt: The selected ConvNeXt model.

    Raises:
        ValueError: If the provided backbone name is not in the list of valid backbones.
    """
    valid_backbones = ['convnext_nano', 'convnext_tiny', 'convnext_base', 'convnext_large']

    if backbone_name not in valid_backbones:
        raise ValueError(f"Unknown backbone model name: {backbone_name}")
    
    return ConvNeXt(in_channels=3, model_name=backbone_name)