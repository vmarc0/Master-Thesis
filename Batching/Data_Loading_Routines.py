import numpy as np
import os
import torch

dataset = 'D:/Python/Miei programmi/Data/Graph data/adversarial-training/'

def load_adversarial(name='0.graph', basefolder = dataset):
    
    DATAPATH = f'{basefolder}{name}'
    
    with open(DATAPATH, 'r') as f:
        _ = f.readline()
        data = f.readline().strip()
        n = [int(x) for x in data.split() if x.isdigit()][0]
        
        header = 'EDGE_DATA_SECTION'
        
        while data != header:
            data = f.readline().strip()
        
        adj = np.zeros((n,n))
        
        while data != '-1':
            data = f.readline().strip()
            if data != '-1':
                e1,e2 = [int(x) for x in data.split() if x.isdigit()]
                adj[e1][e2] = 1 #non scrive anche [e2][e1] poiché edge_index è in forma undirect
            else:
                break
        
        while data != 'DIFF_EDGE':
            data = f.readline().strip()
        
        data = f.readline().strip()
        adj_adv = adj.copy()
        e1,e2 = [int(x) for x in data.split() if x.isdigit()]
        adj_adv[e1][e2] = 1
        adj_adv[e2][e1] = 1

        while data != 'CHROM_NUMBER':
            data = f.readline().strip()
        
        data = f.readline().strip()
        col = [int(x) for x in data.split() if x.isdigit()][0]
        
    return np.array([adj, adj_adv], dtype=np.float32), col
    
#load_adversarial(basefolder='D:/Python/Miei programmi/Data/Graph data/meta_test/')[0][1][40,47]


def load_batched(root = dataset, N_BATCH=8):

    adj_data = []
    col_data = []

    #costruisce un batch mettendo in lista i grafi 
    for fname in os.listdir(root):
        adj, col = load_adversarial(fname)
        adj_data.append(adj)
        col_data.append(col)
        
#in ogni batch ci sono le due adj
        if len(adj_data) == N_BATCH:
#quindi in un n_batch = 8 trovo 16 matrici che andranno impilate in mvv
            mvv = []
            mvc = []
            ns = []
            cs = []
            for i, adj in enumerate(adj_data):
                mvv.append(adj[0,::])
                mvv.append(adj[1,::])
        #adj[1 è adj_adv con un link extra come indicato sul file.graph

                n = adj[0,::].shape[0]
                cols = col_data[i]
                
                ns.append(n)
                ns.append(n)
                mvc.append(np.ones((n,cols),dtype=np.float32))
                mvc.append(np.ones((n,cols),dtype=np.float32))
                cs.append(cols)
                cs.append(cols)

            batch_mvv = np.zeros((0,0), dtype=np.float32)
            batch_mvc = np.zeros((0,0), dtype=np.float32)
#in batch_mvv vengono messe in un'unica matrice le 16 matrici, adj e adj_adv insieme una dopo l'altra
            for i, mat in enumerate(mvv):
                n = ns[i]
                c = cs[i]

                batch_mvv = np.pad(batch_mvv, [(0,n),(0,n)])
                batch_mvc = np.pad(batch_mvc, [(0,n),(0,c)])

                batch_mvv[-n:,-n:] = mvv[i]
                batch_mvc[-n:,-c:] = mvc[i]

            adj_data = []
            col_data = []

            yield torch.from_numpy(batch_mvv), torch.from_numpy(batch_mvc), torch.from_numpy(np.array(ns)), torch.from_numpy(np.array(cs))