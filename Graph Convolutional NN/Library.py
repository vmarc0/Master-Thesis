import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def MakePlanted (n,m, Q):
    NoverQ = n/Q
    degree = torch.zeros(n, dtype=torch.int)
#    adj = torch.zeros(n,n, dtype = torch.float)          #qui vedere come si scrive una matrice in torch
    mapp = list(range(n))
    random.shuffle(mapp)

    neigh = [] #ciascuna colonna è una lista di vicini riferita al nodo n. colonna
    for k in range(n):
        neigh.append([])

    graph = torch.zeros(2, m, dtype=torch.int64)   #ogni coppia in questo array rappresenta il link fra i due nodi

    for i in range(m):
        var1 = random.random()
        var1 *= n
        var1 = int(var1)
        var2 = var1
        while (int(var1/NoverQ) == int(var2/NoverQ)):
            var2 = random.random()
            var2 *= n
            var2 = int(var2)
        var1 = mapp[var1]
        var2 = mapp[var2]
        graph[0][i] = var1
        graph[1][i] = var2

        neigh[var1].append(var2)
        neigh[var2].append(var1)

        degree[var1] += 1
        degree[var2] += 1

    return graph, mapp, neigh, degree

def MakeRandGraph (N,M):

    degree = torch.zeros(N, dtype=torch.int)
#    adj = torch.zeros(N,N, dtype = torch.float)

    neigh = [] #ciascuna colonna è una lista di vicini riferita al nodo n. colonna
    for k in range(N):
        neigh.append([])

    graph = torch.zeros(2, M, dtype=torch.int64)   #ogni coppia in questo array rappresenta il link fra i due nodi

    for j in range(M):
        var1 = random.random()
        var1 *= N
        var1 = int(var1)
        var2 = var1
        while (var2 == var1):
            var2 = random.random()
            var2 *= N
            var2 = int(var2)
        graph[0][j] = var1
        graph[1][j] = var2

        neigh[var1].append(var2)
        neigh[var2].append(var1)

#        adj[var1][var2] = 1
#        adj[var2][var1] = 1

        degree[var1] += 1
        degree[var2] += 1

    return graph, neigh, degree #, adj
#    return is_col(graph, degree, neigh, M, N), adj

def printer():

    data = np.loadtxt("loss.txt")

    plt.figure()
    plt.plot(data[:,0], data[:,1], label="Train")
    plt.plot(data[:,0], data[:,3], label="Test")
    plt.xlabel("epochs")
    plt.title("Loss")
    plt.legend(loc="upper right")
    plt.yscale("log")
    plt.savefig("Loss.png")
    plt.close()

    plt.figure()
    plt.plot(data[:,0], data[:,2], label="Train")
    plt.plot(data[:,0], data[:,4], label="Test")
    plt.plot(data[:,0], data[:,5], label="False_Positive")
    plt.plot(data[:,0], data[:,6], label="False_Negative")
    plt.xlabel("epochs")
    plt.title("Accuracy")
    plt.ylim(top=1)
    plt.legend(loc="lower right")
    plt.savefig("Accuracy.png")
    plt.close()
