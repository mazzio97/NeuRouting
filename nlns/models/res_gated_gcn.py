from typing import Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, to_dense_adj
from sklearn.utils.class_weight import compute_sample_weight


class TSPGCNLayer(gnn.MessagePassing):
    def __init__(self, hidden_dim: int = 300):
        """
        GCN Layer for computing graph information as defined in [1].

        [1] https://arxiv.org/pdf/1906.01227.pdf
        Args:
            hidden_dim (int, optional): Node and edges embedded dimension. Defaults to 300.
        """
        super().__init__(aggr="sum")

        # edge_U, edge_V, edge_W -> W_4, W_5, W_3 in equation (5)
        self.edge_U = nn.Linear(hidden_dim, hidden_dim)
        self.edge_V = nn.Linear(hidden_dim, hidden_dim)
        self.edge_W = nn.Linear(hidden_dim, hidden_dim)

        # edge_U, edge_V -> W_1, W_2 in equation (4)
        self.node_U = nn.Linear(hidden_dim, hidden_dim)
        self.node_V = nn.Linear(hidden_dim, hidden_dim)

        self.batch_norm_edge = nn.BatchNorm1d(hidden_dim)
        self.batch_norm_node = nn.BatchNorm1d(hidden_dim)

    def forward(self, node_embedding: torch.Tensor, edge_embedding: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass on a batch of graphs.

        Args:
            node_embedding (torch.Tensor): 
                Node information, embedded as explained in the reference paper.
            edge_embedding (torch.Tensor): 
                Edge information, embedded as explained in the reference paper.
            edge_index (torch.Tensor): 
                Tensor containing edge information on connected nodes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of node and edges.
        """
        # compute edge features
        edge_gate = self.edge_W(edge_embedding) + \
            self.edge_updater(edge_index, x=node_embedding)
        edge_norm = torch.relu(self.batch_norm_edge(edge_gate))
        edge_features = edge_embedding + edge_norm

        # compute node features
        node_gate = self.node_U(node_embedding) + \
            self.propagate(edge_index, x=node_embedding, edge=edge_embedding)
        node_norm = torch.relu(self.batch_norm_node(node_gate))
        node_features = node_embedding + node_norm

        return node_features, edge_features

    def edge_update(self, x_j: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
        """
        Perform aggregation function on neighbouring nodes.
        Partially computes the content of equation (5) in the reference paper.

        Args:
            x_j (torch.Tensor): "Left" neighbour node
            x_i (torch.Tensor): "Right" neighbour node

        Returns:
            torch.Tensor: Computes the partial edge embeddings
        """
        return self.edge_U(x_j) + self.edge_U(x_i)

    def message(self, 
                x_j: torch.Tensor, 
                x_i: torch.Tensor, 
                edge: torch.Tensor, 
                epsilon: float = 1e-20) -> torch.Tensor:
        """
        Compute the message that will be aggregated later.

        Args:
            x_j (torch.Tensor): Neighbour node
            x_i (torch.Tensor): Central node
            edge (torch.Tensor): Edge connecting the two nodes
            epsilon (float, optional): Small epsilon value, used to prevent division by 0. Defaults to 1e-20.

        Returns:
            torch.Tensor: Computed message based on neighbour nodes.
        """
        sigma_edge = torch.sigmoid(edge)
        
        norm_factor = sigma_edge.sum(dim=0)
        norm_factor += torch.full_like(norm_factor, 1e-20)
        eta = sigma_edge / norm_factor

        return eta * self.node_V(x_j)


class ResGatedGCN(pl.LightningModule):
    def __init__(self, 
                 num_neighbors: int = 20,
                 hidden_dim: int = 300, 
                 mlp_layers: int = 3,
                 gcn_layers: int = 30,
                 initial_learning_rate: float = 0.001,
                 learning_rate_decay_patience: int = 0, 
                 compute_weights: bool = True):
        """
        Residual Gated GCN Model as explained in [1].

        [1] https://arxiv.org/pdf/1906.01227.pdf

        Args:
            num_neighbors (int): Number of neighbors used to embed the instances.
            hidden_dim (int, optional): 
                Node and edges embedding dimension. Defaults to 300.
            mlp_layers (int, optional): 
                Number of mlp layers used for classification. Defaults to 3.
            gcn_layers (int, optional): 
                Number of convolution layers used for feature computation. Defaults to 30.
            steps_per_epoch (int): Number of steps in an epoch for the LR scheduler. Defaults to 100.
            compute_weights (bool): Compute class weights for balanced BCE. Defaults to True.
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_neighbors = num_neighbors
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate_decay_patience = learning_rate_decay_patience
        self.compute_weights = compute_weights

        self.node_embedding = nn.Linear(4, hidden_dim)
        self.edge_distance_embedding = nn.Linear(1, hidden_dim // 2)
        self.edge_type_embedding = nn.Linear(1, hidden_dim // 2)

        self.gcn_layers = nn.ModuleList([
            TSPGCNLayer(hidden_dim) for _ in range(gcn_layers)
        ])

        # define MLP layers as list of Linear layers with ReLU activation
        # and last layer as classification layer with a sigmoid activation
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            for _ in range(mlp_layers - 1)
        ])
        self.classification = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        
    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass on batch of graphs.

        Args:
            data (Data): Input graphs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                Tuple containing predictions and loss value computed on the input
                data.
        """
        node = self.node_embedding(data.x)
        
        # compute embedding on the distance (|E|, 1) and the type (|E|, 1)
        edge_d = self.edge_distance_embedding(data.edge_attr[:, 0].reshape(-1, 1))
        edge_t = self.edge_type_embedding(data.edge_attr[:, 1].reshape(-1, 1))
        edge = torch.cat((edge_d, edge_t), dim=1)

        for gcn_l in self.gcn_layers:
            node, edge = gcn_l(node, edge, data.edge_index)

        # compute the probability of an edge being in the final tour
        for mlp_l in self.mlp_layers:
            edge = mlp_l(edge)
        pred = self.classification(edge)

        # compute predicted adjacency matrix
        pred_adj = to_dense_adj(data.edge_index, data.batch, pred.reshape(-1))
        pred_adj = [split[0, :, :] for split in pred_adj.split(1)]

        weights = None
        if self.compute_weights:
            cw = compute_sample_weight("balanced", y=data.y.cpu().numpy())
            weights = torch.tensor(cw).to(pred.device).float()

        loss = nn.functional.binary_cross_entropy(pred.reshape(-1), data.y, weight=weights)
        return pred_adj, loss

    def predict(self, data: Batch) -> Tuple[torch.Tensor, nn.BCELoss]:
        """
        Compute the heatmap of a graph.

        Args:
            data (Batch): The input graph, batched by torch_geometric.

        Returns:
            Tuple[torch.Tensor, nn.BCELoss]: Tuple containing the heatmap for each batch and the loss.
        """
        return self(data)

    def training_step(self, data: Batch, batch_idx: int) -> nn.BCELoss:
        """
        Perform a training step.

        Args:
            data (Batch): Input data.
            batch_idx (int): Batch index.

        Returns:
            Tuple[nn.BCELoss]: Loss produced by the heatmap model.
        """
        _, loss = self.predict(data)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, data: Batch, batch_idx: int) -> Tuple[nn.BCELoss]:
        """
        Perform a validation step.

        Args:
            data (Batch): Input data.
            batch_idx (int): Batch index.

        Returns:
            Tuple[nn.BCELoss]: Loss produced by the heatmap model.
        """
        with torch.no_grad():
            pred, loss = self.predict(data)
        self.log("valid/loss", loss, batch_size=len(pred))
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Instatiante the optimizer for the heatmap model.

        Returns:
            torch.optim.Adam: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.99, patience=0)
        return [optimizer], [{ "scheduler": scheduler , "monitor": "valid/loss", "interval": "epoch", "strict": False, "frequency": 5 }]
