from torch.utils.data import Dataset
from Data_Loading_Routines import load_batched


dataset = 'D:/Python/Miei programmi/Data/Graph data/adversarial-training/'
small_dataset = 'D:/Python/Miei programmi/Data/Graph data/meta_test/'

class CustomGraphDataset_batched(Dataset):
    def __init__(self, Data_len, root = dataset, batch_size = 8):
        
        self.adj_data = []
        self.col_data = []
        self.nv_data = []
        self.nc_data = []
        

        for adj, col, nv, nc in load_batched(Data_len, root, N_BATCH = batch_size):
            self.adj_data.append(adj)
            self.col_data.append(col)
            self.nv_data.append(nv)
            self.nc_data.append(nc)

    def __len__(self):
        return len(self.adj_data)

    def __getitem__(self, idx):
        matrices = self.adj_data[idx]
        cols = self.col_data[idx]
        verts = self.nv_data[idx]
        ncols = self.nc_data[idx]
        return matrices, cols, verts, ncols