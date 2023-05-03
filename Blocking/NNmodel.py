import torch
import torch.nn as nn

dev = torch.device('cuda')

class Graph_Net_art(nn.Module):
    def __init__(self, batch_size, n_iters=32, mlp_dim=32):
        super().__init__()

        self.mlp_dim = mlp_dim
        self.n_iters = n_iters

        # Define initializers; il 2 di fronte a batch_s è per l'adversarial
        self.starter = torch.tensor([1], dtype=torch.float32, device=dev)
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

        N = int(torch.sum(nvert))

        Col = int(torch.sum(ncol))

        V_embed = self.V_init(self.starter)
        V_embed = V_embed.unsqueeze(0)
        V_embed = V_embed.unsqueeze(1)
        V_embed = V_embed.repeat(1, N, 1)
        
        C_embed = self.C_init(self.starter)
        C_embed = C_embed.unsqueeze(0)
        C_embed = C_embed.unsqueeze(1)
        C_embed = C_embed.repeat(1, Col, 1)
        
        """
        print("\nV_embed_:\t",V_embed.size(),"\n")
        print("\nC_embed_:\t",C_embed.size(),"\n")
        print("\nMat:\t",Adj.size(),"\n")
        """
        #Preparo V_state e C_state per la LSTM (così come uscirebbe da LSTM-output)
        """
        print("\nV_embed_toLSTM:\t",V_st.size(),"\n")
        print("\nC_embed_toLSTM:\t",C_st.size(),"\n")
        """
        V_state = (V_embed, torch.zeros(1, N, self.mlp_dim, device=dev))
        C_state = (C_embed, torch.zeros(1, Col, self.mlp_dim, device=dev))

        for _ in range(self.n_iters):
            V = V_state[0]
            V = V.squeeze(0)  #(nbatch x) N x mlp_dim

            C = C_state[0]
            C = C.squeeze(0)   #allo stesso modo (nbatch x) Col_tot x mlp_dim

            """
            print("\nV_for_MLP:\t",V.size(),"\n")
            print("\nC_for_MLP:\t",C.size(),"\n")
            """

            V_msg = self.C_mlp(C) # Vertex message comes from colors
                                    #V_msg: (nbatch) x ncol x mlp_dim

            V_msg = torch.mm(Mvc, V_msg) #(nbatch x nvert x ncol) x V_msg

            C_msg = self.V_mlp(V) # Color message comes from vertices
            #C_msg: (nbatch x) nvert x mlp_dim

            C_msg = torch.mm(Mvc.transpose(0,1), C_msg) #(nbatch x ncol x nvert) x C_msg

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
            rnn_inp = torch.cat([torch.mm(Adj, V), V_msg], dim=1)
            
            rnn_inp = rnn_inp.unsqueeze(0)
            C_msg = C_msg.unsqueeze(0)
            
            #to printing variables
            hxx = V_state[0]
            cell = V_state[1]
            """
            print("input_Vrnn:\t", rnn_inp.size())
            print("\ninput_Cnn:\t", C_msg.size())
            
            
            print("\nhxx:\t",hxx.size(),"\ncell:\t", cell.size())
            """

            _, V_state = self.V_update(rnn_inp, V_state)

            _, C_state = self.C_update(C_msg, C_state)

        logits = V_state[0].squeeze(0)
        logits = self.vote(logits)   #post-vote: [nvert, 1] ovvero 1 voto per vertice

#        logits = logits.squeeze()
        output = torch.cat([torch.mean(x).unsqueeze(0) for x in torch.split(logits, list(nvert))])

#[index:f_ind]  da incollare dopo logits
        output = torch.sigmoid(output)
        #output deve essere come Y ovvero tensore [nbatch]
        return output
