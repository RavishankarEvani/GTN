# --------------------------------------------------------
# Multiscale Graph Texture Network (GTN)
# Author: Ravishankar Evani
# --------------------------------------------------------

import torch
import torch.nn as nn
from src.backbone import backbone_selection
from typing import List, Tuple, Union, Optional
from config import GraphTextureNetworkConfig
import torch_geometric
from torch_geometric.nn import aggr
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


def reshape_tensor(
    x : torch.Tensor
) -> torch.Tensor:
    """
    Reshapes a tensor of shape BxCxHxW to BxCx(H*W).

    Args:
        x (torch.Tensor): Input tensor of shape BxCxHxW.

    Returns:
        torch.Tensor: Reshaped tensor of shape BxCx(H*W).
    """
    # Get the shape of the input tensor
    B, C, H, W = x.shape

    # Reshape the tensor to BxCx(H*W)
    return x.view(B, C, H * W)


class GTNCreateNode(nn.Module):
    """
    Graph Texture Network Create Node (GTNCreateNode) class.

    Creates graph nodes from input feature maps extracted from each layer of the backbone. 
        1. 1x1 convolution
        2. Batch normalization
        3. GELU activation
        4. Dropout
        5. Reshaped to form graph nodes

    Attributes:
        conv1x1 (nn.Conv2d): 1x1 convolution layer to compress input features.
        batch_norm (nn.BatchNorm2d): Batch normalization layer.
        act (nn.GELU): GELU activation function.
        dropout (nn.Dropout): Dropout layer.
    
    Methods:
        forward(x):
            Forward pass through the module to generate graph nodes.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout_prob: float = 0.6
    ):
        """
        Initializes the GTNCreateNode class.

        Args:
            in_channels (int): Number of input channels for 1x1 convolution.
            out_channels (int): Number of output channels from 1x1 convolution.
            dropout_prob (float, optional): Dropout probability. Default is 0.6.
        """
        super(GTNCreateNode, self).__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, 
                                 out_channels, 
                                 kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Forward pass to create graph nodes.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H*W), where H*W represents the flattened spatial dimensions.
        """
        
        x = self.conv1x1(x)
        x = self.batch_norm(x)
        x = self.act(x)
        x = self.dropout(x)
        
        B,C,_,_ = x.shape
        out = x.reshape(B,C,-1)
        
        return out



class GTNMessageAgg(MessagePassing):
    """
    Graph Texture Network Message Passing and Aggregation (GTNMessageAgg) class.

    This GTNMessageAgg module performs the following operations at each layer of the backbone:   
        1. Creation of masked edge indices
        2. Computation of edge weights
        3. Message passing and aggregation
        4. Residual connection and orderless aggregation.

    Attributes:
        in_channels (int): Number of input channels/nodes.
        act (nn.Module): Activation function.
        diag_weight (nn.Parameter): Learnable parameter for self-loop re-clibration.
        masking_matrix (nn.Parameter): Learnable masking matrix for adjacency matrix.
        ln (nn.LayerNorm): Layer normalization.
        edge_index (torch.Tensor): Edge indices for the graph.
        edge_weight (torch.Tensor): Edge weights for the graph.
    
    Methods:
        reset_parameters():
            Resets learnable parameters.
        forward(x):
            Forward pass through the GTN module.
        residual_aggregation(x_res, x_prop):
            Applies residual connection, activation, normalisation and orderless aggregation.
        create_edge_index(x):
            Creates masked edge indices.
        create_edge_weight(edge_index, num_nodes):
            Creates the edge weights between graph vertices and re-calibreates self-loops.
        message(x_j, edge_weight):
            Computes the messages using neighborhood node features and re-calibrated edge weights.
        get_edge_weights():
            Retrieves the edge weights.
        get_edge_index():
            Retrieves the edge indices.
    """
    def __init__(
        self, 
        num_node: int, 
        node_dim: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = aggr.MultiAggregation(['mean', 'std']),
        **kwargs
    ):
        
        """
        Initializes the GTNMessageAgg class.

        Args:
            num_node (int): Number of nodes in the graph.
            node_dim (int): Dimension of each node.
            aggr (Union[str, List[str], Aggregation], optional): Aggregation method. Default is MultiAggregation(['mean', 'std']).
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(aggr, **kwargs)

        self.num_node = num_node
        self.act = nn.GELU()

        # Learnable parameter for self-loop re-calibration
        self.sr_weight = nn.Parameter(torch.Tensor([0.5]), 
                                      requires_grad=True)

        # Learnable parameter for masking edge indices
        self.masking_matrix = nn.Parameter(torch.randn(num_node* num_node), 
                                           requires_grad=True)

        self.ln = nn.LayerNorm((num_node, node_dim))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()


    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Forward pass through the GTN module: 
            1. Create masked adjacency matrix (edge index + edge weights)
            2. Message Passing + Aggregation
            3. Residual Connection + Orderless Aggregation
        
        Args:
            x (torch.Tensor): Original graph vertices.

        Returns:
            torch.Tensor: Orderless graph vertex representations.
        """
        
        res = x
        
        num_nodes = x.size(-2) # self.node_dim is also -2
        
        # Create Adjacency Matrix
        self.edge_index = self.create_edge_index(x)
        self.edge_weight = self.create_edge_weight(self.edge_index, 
                                                   num_nodes)

        # Message Passing + Aggregation
        prop = self.propagate(edge_index = self.edge_index, 
                              x=x, 
                              edge_weight=self.edge_weight)

        # Residual Connection + Orderless Aggregation
        out = self.residual_aggregation(x_res = res, 
                                        x_prop = prop)

        return out

    def residual_aggregation(
        self, 
        x_res: torch.Tensor, 
        x_prop: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Applies residual connection, activation, normalisation and orderless aggregation.

        Args:
            x_res (torch.Tensor): Original graph vertices.
            x_prop (torch.Tensor): Enhanced graph vertices after message passing and aggregation.

        Returns:
            torch.Tensor: Orderless graph vertex representations.
        """
        
        
        # Add residual/bias (x_prop)
        x = x_prop + x_res
        
        # Activation
        x = self.act(x)
        
        # Norm, orderless aggregation
        out = self.ln(x).mean(-1)
        
        return out

    def create_edge_index(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        
        """
        Creates masked edge indices.

        Args:
            x (torch.Tensor): Original graph vertices.

        Returns:
            torch.Tensor: Masked edge indices.
        """
        
        # First initialise an edge index for a fully connected graph
        edge_index = torch.tensor([[i, j] for i in range(x.shape[1]) for j in range(x.shape[1])], 
                                    dtype=torch.long, 
                                    device=x.device).t()
        
        row, col = edge_index[0], edge_index[1]
        
        # Apply mask to edge indices
        mask = (torch.sigmoid(self.masking_matrix)>0.5).bool()
        row = row[mask]
        col = col[mask]
        
        edge_index = torch.stack([row, col], dim=0)
        
        return edge_index


    def create_edge_weight(
        self, 
        edge_index: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        
        """
        Creates the edge weights between graph vertices and re-calibreates self-loops.

        Args:
            edge_index (torch.Tensor): Masked edge indices.
            num_nodes (int): Number of graph vertices.

        Returns:
            torch.Tensor: Normalisation followed by re-calibration of edge weights.
        """
        
        # Extract row and column indices from edge_index
        row, col = edge_index[0], edge_index[1]
        
        # Compute the inverse of the degree for each node in the graph
        deg_inv = 1. / degree(col, num_nodes=num_nodes).clamp_(1.)

        # Initialize edge weights based on the inverse degree.
        edge_weight = deg_inv[col]
        
        # Apply a sigmoid activation to a learnable parameter
        sr_weight = torch.sigmoid(self.sr_weight)
            
        # Re-calibrate self-loops.
        edge_weight[row == col] += sr_weight * edge_weight[row == col]  
        
        return edge_weight

    def message(
        self, 
        x_j: torch.Tensor, 
        edge_weight: torch.Tensor
    ) -> torch.Tensor:

        """
        Computes the messages using neighborhood node features and re-calibrated edge weights.

        Args:
            x_j (torch.Tensor): Neighborhood node feature.
            edge_weight (torch.Tensor): Re-calibrated edge weights.

        Returns:
            torch.Tensor: Message matrix.
        """

        message_mtrix = edge_weight.view(-1, 1) * x_j

        return message_mtrix

    def get_edge_weights(
        self
    ) -> torch.Tensor:
        """
        Retrieves re-calibrated edge weights.

        Returns:
            torch.Tensor: Re-calibrated edge weights.
        """
        return self.edge_weight
    
    def get_edge_index(
        self
    ) -> torch.Tensor:
        """
        Retrieves the edge indices after applying learnable mask.

        Returns:
            torch.Tensor: Masked edge indices.
        """
        return self.edge_index

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_node}, '
                f'{self.node_dim}, self_loop_recalibration_weight={torch.sigmoid(self.sr_weight)})')

        
    
class GTN(nn.Module):
    """
    Graph Texture Network (GTN) class.

    Attributes:
        cfg: Configuration object.
        backbone: ConvNeXt backbone network for feature extraction.
        cgn_list: List of CreateGraphNode modules (one for each layer of the backbone).
        gl_list: List of GTN modules (one for each layer of the backbone).
        classifier: Classifier layer for final classification.
    
    Methods:
        forward(x):
            Forward pass through the network.
        get_edge_weights():
            Retrieves the edge weights from the GTN modules.
        get_edge_indices():
            Retrieves the edge indices from the GTN modules.
    """
    def __init__(
        self,
        cfg: GraphTextureNetworkConfig,
        n_classes: int,
        depth_dims: List[int],
        spatial_dims: List[int],
        depth_compression_ratio: Union[int, None],
        embedding_dim: int,
        backbone_name: str,
        fine_tune_backbone: bool = False
    ):
        super(GTN, self).__init__()

        """
        Initializes the GTN class.

        Args:
            cfg (GraphTextureNetworkConfig): Configuration object.
            n_classes (int): Number of texture/material categories.
            depth_dims (List[int]): List of depth dimensions for each layer.
            spatial_dims (List[int]): List of spatial dimensions for each layer.
            depth_compression_ratio (Union[int, None]): Ratio for depth compression.
            embedding_dim (int): Dimension of the embedding space.
            backbone_name (str): Name of the backbone network.
            fine_tune_backbone (bool, optional): Whether to fine-tune the backbone network. Default is False.
        """

        self.cfg = cfg
        
        # Backbone selection
        self.backbone = backbone_selection(backbone_name)

        depth_dims = depth_dims[-self.cfg.common.layers:]
        spatial_dims = spatial_dims[-self.cfg.common.layers:]

        # Assign min compression ratio as 1
        if depth_compression_ratio is None:
            depth_compression_ratio = 1

        # Define graph node creation and GTN modules
        self.cgn_list = nn.ModuleList()
        self.gl_list = nn.ModuleList()

        for depth_dim, spatial_dim in zip(depth_dims, spatial_dims):
            num_node = depth_dim // depth_compression_ratio
            node_dim = spatial_dim * spatial_dim

            # Create graph nodes
            cgn_module = GTNCreateNode(in_channels = depth_dim, 
                                       out_channels = num_node)
            
            # Message Passing and Aggregation
            gl_module = GTNMessageAgg(num_node=num_node, 
                                        node_dim=node_dim, 
                                        aggr='mean') # aggr='mean' => rho = 1/N(i)U(i). aggr='sum' => rho = 1

            self.cgn_list.append(cgn_module)
            self.gl_list.append(gl_module)


        # Define the classifier
        self.classifier = torch_geometric.nn.dense.linear.Linear(embedding_dim, 
                                                                 n_classes, 
                                                                 bias=False, 
                                                                 weight_initializer='glorot')

        # Only use the following layers for gradient computation
        exclude_names = ['cgn_list', 'gl_list', 'classifier']

        # Freeze backbone
        if fine_tune_backbone == False:
            for name, param in self.named_parameters():
                if not any([exclude_name in name for exclude_name in exclude_names]):
                    param.requires_grad = False

    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        
        """
        Forward pass through GTN.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            out (torch.Tensor): Output tensor after Softmax operation.
            vertex_list (List[torch.Tensor]): List of graph vertices.
            og_list (List[torch.Tensor]]): List of graph vertices after message passing, message aggregation and orderless aggregation.
        """
        
        # Get list of feature maps from the backbone
        x_list = self.backbone(x)
        
        x_list = x_list[-self.cfg.common.layers:]
        
        # Create Graph Verices
        vertex_list = [cgn(feature) for cgn, feature in zip(self.cgn_list, x_list)]
        
        # Generate orderless graph representations
        og_list = [gl(graph_node) for gl, graph_node in zip(self.gl_list, vertex_list)]
    
        # Concatenate multiscale outputs
        x = torch.cat(og_list, dim=1)

        # Output from classification layer
        out = self.classifier(x)
        # Useful for our specific use case but generally not necessary and can be commented out
        out = nn.functional.softmax(out, dim=1)

        return out, vertex_list, og_list
    
    def get_edge_weights(
        self
    ) -> List[torch.Tensor]:
        """
        Retrieves list of edge weights from the GTNMessageAgg modules.

        Returns:
            List[torch.Tensor]: List of edge weights.
        """
        edge_weights = []
        for gl_module in self.gl_list:
            edge_weights.append(gl_module.get_edge_weights())
        return edge_weights

    def get_edge_indices(
        self
    ) -> List[torch.Tensor]:
        """
        Retrieves a list of edge indices from the GTNMessageAgg modules.

        Returns:
            List[torch.Tensor]: List of edge indices.
        """
        edge_indices = []
        for gl_module in self.gl_list:
            edge_indices.append(gl_module.get_edge_index())
        return edge_indices