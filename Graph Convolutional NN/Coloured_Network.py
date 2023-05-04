import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool

dev = torch.device('cuda')

class Colored_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.Convol_layer = GraphConv(1,64).to(dev)
        self.Message_passing = GraphConv(64,32).to(dev)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.MLP = nn.Sequential(nn.Linear(32, 16, device=dev),
                                   nn.ReLU(),
                                   nn.Dropout(0.15),
                                   nn.Linear(16, 8, device=dev),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(8, 1, device=dev))


    def forward(self, feat, graph, b):
        feat = feat.to(dtype=torch.float32)
        conv = self.Convol_layer(feat, graph)
        conv = self.relu(conv)
        conv = self.drop(conv)
        conv = self.Message_passing(conv, graph)
        conv = self.relu(conv)
        conv = self.drop(conv)
        last_layer = self.MLP(conv)
        last_layer = global_mean_pool(last_layer, b)
        last_layer = last_layer.squeeze()
        output = torch.sigmoid(last_layer)

        return output
