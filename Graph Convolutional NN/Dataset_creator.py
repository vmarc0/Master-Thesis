import torch
from time import time
from random import random
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected
from Library import MakePlanted
from Library import MakeRandGraph

class Phase_tr(InMemoryDataset):

    def __init__(self, root, N, painted, connectivity, n_graphs = 10000):
        self.N = N
        self.coloured_dataset = painted
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
        N = self.N
        Q = 5
        where = self.path_file + "Dataset_info.txt"
        g = open(where, "w")
        print(f"Quantità di grafi:\t{self.n_graphs}", file=g)
        print(f"\nNumero di nodi per grafo:\t{N}", file=g)
        print(f"\nConnettività:\t{self.conn}", file=g)
        print(f"\nQ = {Q}", file=g)
        g.close()

        tempo = time()

        if self.coloured_dataset:

            for k in range(self.n_graphs):
                nodef = torch.zeros(N, dtype=torch.float32)

                if (random() > 0.5):
                    Y = torch.tensor([1], dtype=torch.float32)
                    grafo, mapp, _, _ = MakePlanted(N, int(self.conn*N/2), Q)

                    #Coloro il dataset
                    for i in range(N):
                        nodef[mapp[i]] = int(i*Q/N)

                else:
                    Y = torch.tensor([0], dtype=torch.float32)
                    grafo, _, _ = MakeRandGraph(N, int(self.conn*N/2))

                    #Metto colori a caso essendo il grafo non colorabile
                    for i in range(N):
                        nodef[i] = int(random()*Q)

                grafo = to_undirected(grafo)
                nodef = nodef.view(-1, 1)
                graph_obj = Data(nodef, edge_index = grafo, y = Y)
                data_list.append(graph_obj)

        else:

            for k in range(self.n_graphs):

                if (random() > 0.5):
                    Y = torch.tensor([1], dtype=torch.float32)
                    grafo, _, _, nodef = MakePlanted(N, int(self.conn*N/2), Q)

                else:
                    Y = torch.tensor([0], dtype=torch.float32)
                    grafo, _, nodef = MakeRandGraph(N, int(self.conn*N/2))

                grafo = to_undirected(grafo)
                nodef = nodef.view(-1, 1)
                graph_obj = Data(nodef, edge_index = grafo, y = Y)
                data_list.append(graph_obj)
        tp = time() - tempo
        print(f"Dataset generato in {int(tp/60):2d} min e {tp%60:.1f} s\n")
        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
