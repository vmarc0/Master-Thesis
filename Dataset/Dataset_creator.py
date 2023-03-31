import torch
from time import time
from random import random
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from Library import MakePlanted
from Library import MakeRandGraph

class Generating(InMemoryDataset):

    def __init__(self, root, N, painted, connectivity, n_graphs = 10000):
        self.coloured_dataset = painted
        self.N = N
        self.conn = connectivity
        self.path_file = root
        self.n_graphs = n_graphs
        super(Phase_tr, self).__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        # List of the raw files
        return

    @property
    def processed_file_names(self):
        if self.coloured_dataset:
            return f'Coloured_Dataset_c{int(self.conn)}_N{self.N}.pt'
        else:
            return f'Unpainted_Dataset_c{int(self.conn)}_N{self.N}.pt'


    def process(self):
        data_list = []
        self.n_graphs
        Q = 5

        tempo = time()

        if self.coloured_dataset:

            for k in range(self.n_graphs):
                nodef = torch.zeros(self.N, dtype=torch.float32)

                if (random() > 0.5):
                    Y = torch.tensor([1], dtype=torch.float32)
                    grafo, mapp, _, _ = MakePlanted(self.N, int(self.conn*self.N/2), Q)

                    #Coloro il dataset
                    for i in range(self.N):
                        nodef[mapp[i]] = int(i*Q/self.N)

                else:
                    Y = torch.tensor([0], dtype=torch.float32)
                    grafo, _, _ = MakeRandGraph(self.N, int(self.conn*self.N/2))

                    #Metto colori a caso essendo il grafo non colorabile
                    for i in range(self.N):
                        nodef[i] = int(random()*Q)

                grafo = to_undirected(grafo)
                nodef = nodef.view(-1, 1)
                graph_obj = Data(nodef, edge_index = grafo, y = Y)
                data_list.append(graph_obj)

        else:

            for k in range(self.n_graphs):

                if (random() > 0.5):
                    Y = torch.tensor([1], dtype=torch.float32)
                    grafo, _, _, nodef = MakePlanted(self.N, int(self.conn*self.N/2), Q)

                else:
                    Y = torch.tensor([0], dtype=torch.float32)
                    grafo, _, nodef = MakeRandGraph(self.N, int(self.conn*self.N/2))

                grafo = to_undirected(grafo)
                nodef = nodef.view(-1, 1)
                graph_obj = Data(nodef, edge_index = grafo, y = Y)
                data_list.append(graph_obj)
        tp = time() - tempo
        print(f"Dataset generato in {int(tp/60):2d} min e {tp%60:.1f} s\n")
        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
