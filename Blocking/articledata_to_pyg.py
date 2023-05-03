import torch

dev = torch.device('cuda')

def extract(blockmat, verts):   #da usare solo con blocchi quadrati
    n = torch.max(verts)
    n = n.item()
    adj = torch.zeros(verts.size(0), n, n, dtype=torch.float32, device=dev)
    ind = 0
    for b, v in enumerate(verts):
        f_ind = ind + v.item()
        adj[b,:v.item(),:v.item()] = blockmat[ind:f_ind,ind:f_ind]
        ind += v.item()
    return adj

def build_MVC(colors, nodes):
    C = torch.max(colors)
    C = C.item()
    N = torch.max(nodes)
    N = N.item()
    mat = torch.zeros(colors.size(0), N, C, dtype=torch.float32, device=dev)
    
    for i, c in enumerate(colors):
        sub_mat = torch.ones(nodes[i], c,  dtype=torch.float32, device=dev)
        mat[i, :nodes[i], :c] = sub_mat[:,:]
        
    return mat
        