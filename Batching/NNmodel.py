import torch
import torch.nn as nn

dev = torch.device('cuda')

def to_MLP(tens, detail):
    n_max = torch.max(detail)
    n_max = n_max.item()
    new_t = torch.zeros(detail.size(0), n_max, tens.size(1), dtype=torch.float32, device=dev)
    ind = 0
    for b, v in enumerate(detail):
        f_ind = ind + v.item()
        new_t[b, :v.item(), :] = tens[ind:f_ind, :]
        ind += v.item()
        
    return new_t

def removing_z(mat):
    row, column = torch.nonzero(mat, as_tuple=True)
    row = torch.max(row)
    row = row.item() + 1
    column = torch.max(column)

    if not(column.item() == mat.size(1)-1):
        print(f"\nnonzero_col.size:\t{column.item()+1}\n\n")
        print(f"\nmat_colons.size:\t{mat.size(1)}\n\n")
        print("\nQualcosa non va nelle dimensioni dei dati\n\n")
    new_mat = torch.zeros(row, mat.size(1), dtype=torch.float32)
    new_mat = mat[:row,:]
    return new_mat

def to_LSTM (tensor):
    l = []
    for i in range(tensor.size(0)):
        zeros_removed = removing_z(tensor[i])
        l.append(zeros_removed)
    out = torch.vstack(tuple(l))
    out = out.unsqueeze(0)

    return out
"""
def divide (tens, detail):
    new_t = torch.zeros(detail.size(0), dtype=torch.float32, device=dev)
    ind = 0
    for b, v in enumerate(detail):
        f_ind = ind + v.item()
        somma = tens[ind:f_ind]
        ind += v.item()
        somma = torch.sum(somma)
        new_t[b] = somma.item()/v.item()
    return new_t
"""
class Graph_Net_art(nn.Module):
    def __init__(self, batch_size, n_iters=32, mlp_dim=32):
        super().__init__()

        self.mlp_dim = mlp_dim
        self.n_iters = n_iters

        # Define initializers; il 2 di fronte a batch_s è per l'adversarial
        self.starter = torch.ones(2*batch_size, 1,  dtype=torch.float32, device=dev)
        self.starter.requires_grad = False

        self.V_init = nn.Linear(1, mlp_dim, device=dev)
        self.C_init = nn.Linear(1, mlp_dim, device=dev)
        self.C_init.requires_grad = False

        # Vertex_mlp
        self.V_mlp = nn.Sequential(nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.ReLU(),
                                   nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.ReLU(),
                                   nn.Linear(mlp_dim, mlp_dim, device=dev))
        # Color MLP
        self.C_mlp = nn.Sequential(nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.ReLU(),
                                   nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.ReLU(),
                                   nn.Linear(mlp_dim, mlp_dim, device=dev))

        # Voting Layer
        self.vote = nn.Sequential(nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.Linear(mlp_dim, mlp_dim, device=dev),
                                   nn.Linear(mlp_dim, 1, device=dev))

        self.V_update = nn.LSTM(2*mlp_dim, mlp_dim, device=dev)
        self.C_update = nn.LSTM(mlp_dim, mlp_dim, device=dev)

    def forward(self, Adj, Mvc, nvert, ncol):

#        total_nodes = int(torch.sum(nvert))
#        total_cols = int(torch.sum(ncol))

        N = Adj.size(1)
        print(f"\nN vale:\t{N}\n")
        Col_tot = Mvc.size(2)

        V_embed = self.V_init(self.starter)
        V_embed = V_embed.unsqueeze(1)
        V_embed = V_embed.repeat(1, N, 1)
        
        C_embed = self.C_init(self.starter)
        C_embed = C_embed.unsqueeze(1)
        C_embed = C_embed.repeat(1, Col_tot, 1)
        
        """
        print("\nV_embed_:\t",V_embed.size(),"\n")
        print("\nC_embed_:\t",C_embed.size(),"\n")
        print("\nMat:\t",Adj.size(),"\n")
        """
#        cb = self.Q*(Mat.size(0))      Utilizzo sum(ncol)
        V_embed = to_LSTM(V_embed)
        C_embed = to_LSTM(C_embed)

        V_embed = to_MLP(V_embed.squeeze(0), nvert)
        C_embed = to_MLP(C_embed.squeeze(0), ncol)

        #Preparo V_state e C_state per la LSTM (così come uscirebbe da LSTM-output)
        V_st = to_LSTM(V_embed)
        C_st = to_LSTM(C_embed)
        """
        print("\nV_embed_toLSTM:\t",V_st.size(),"\n")
        print("\nC_embed_toLSTM:\t",C_st.size(),"\n")
        """
        V_state = (V_st, torch.zeros(1, V_st.size(1), self.mlp_dim, device=dev))
        C_state = (C_st, torch.zeros(1, C_st.size(1), self.mlp_dim, device=dev))

        for _ in range(self.n_iters):
            V = V_state[0]
            V = to_MLP(V.squeeze(0), nvert)  #(nbatch x) N x mlp_dim

            C = C_state[0]
            C = to_MLP(C.squeeze(0), ncol)   #allo stesso modo (nbatch x) Col_tot x mlp_dim

            """
            print("\nV_for_MLP:\t",V.size(),"\n")
            print("\nC_for_MLP:\t",C.size(),"\n")
            """

            V_msg = self.C_mlp(C) # Vertex message comes from colors
                                    #V_msg: (nbatch) x ncol x mlp_dim

            V_msg = torch.bmm(Mvc, V_msg) #(nbatch x nvert x ncol) x V_msg

            C_msg = self.V_mlp(V) # Color message comes from vertices
            #C_msg: (nbatch x) nvert x mlp_dim

            C_msg = torch.bmm(Mvc.transpose(1,2), C_msg) #(nbatch x ncol x nvert) x C_msg

            #V_msg: (nbatch x) nvert x mlp_dim
            #C_msg: (nbatch x) ncol x mlp_dim


            #MVVxV: (nbatch x) nvert x mlp_dim
            #cat(dim=2): (nbatch x) x (nvert) x 2mlp_dim   OBSOLETO
            #After_unsq  cat(dim=2): (nbatch x) x seq_len(=1) x (nvert) x 2mlp_dim   OBSOLETO
            #V_state[0]: (nbatch x) nvert x mlp_dim
            """
        Per Rnn_inp è stato scelto -preso da colab-     nel corretto ordine
            Sequence lenght = 1
            Number of batch = numero di nodi totale di tutti i grafi nel batch
            Input size = 2d (a seguito del cat: d+d)

        Per la tupla V_state    nel corretto ordine
            number of layers = 1
            Number of batch = come sopra (per C_state è lo stesso ma per #colori)
            hidden state size = d (nota che è l'output dell'LSTM)

            Nota che tali numeri si riferiscono alla dimensionalità dei tensori in ingresso
            """
            rnn_inp = torch.cat([torch.bmm(Adj, V), V_msg], dim=2)

        #modifico la forma dell'input per la LSTM
            rnn_inp = to_LSTM(rnn_inp)
            C_msg = to_LSTM(C_msg)

            
            #to printing variables
            hxx = V_state[0]
            cell = V_state[1]
            """
            print("input_Vrnn:\t", rnn_inp.size())
            print("\ninput_Cnn:\t", C_msg.size())
            
            print(f"\nTotal real nodes nel batch:\t{total_nodes}\n")
            print(f"\nTotal real colors nel batch:\t{total_cols}\n")
            
            print("\nhxx:\t",hxx.size(),"\ncell:\t", cell.size())
            """

            _, V_state = self.V_update(rnn_inp, V_state)

            _, C_state = self.C_update(C_msg, C_state)

        logits = V_state[0].squeeze(0)
        logits = self.vote(logits)   #votes: [nbatch x nvert, 1] ovvero 1 voto per vertice
        logits = logits.squeeze()
        print(logits.size())
#[index:f_ind]  da incollare dopo logits
        output = torch.cat([torch.mean(x) for x in torch.split(logits, list(nvert))])
        output = torch.sigmoid(logits)
        #output deve essere come Y ovvero tensore [nbatch]
        return output
