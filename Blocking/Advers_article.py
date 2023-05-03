import torch
import os
import sys
import time
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from Dataset_loading import CustomGraphDataset_batched
from NN_fv_onArt import Graph_Net_art

start_time = time.time()
run = 1 + len(os.listdir('old_loss/'))

dev = torch.device('cuda')
if not torch.cuda.is_available():
    sys.exit("GPU non disponibile")

torch.cuda.empty_cache()

os.mkdir(f"old_loss/run{run}")
test_root = 'D:/Python/Miei programmi/Data/Graph data/adversarial-testing/'

#dimensione embedding per V e C
d = 32
Q = 2

#n_iterations of the LSTM inside the Net: T=32 in article
T = 32

#dimensioni singolo grafo dipendenti dal dataset; Batch_size:
b_size = 5
n_data = 500

num_epochs = 1000000   #Nell'articolo sono 5300
learning_rate = 1e-5
check = 1 + int(num_epochs/50)
#n_iters = 50      numero di batches
data_time = time.time()
Data = CustomGraphDataset_batched(n_data, batch_size = b_size)
Data_test = CustomGraphDataset_batched(n_data, root=test_root, batch_size= n_data)

model = Graph_Net_art(b_size, T, d)
#model = example(d)
print(f"\nCaricato il dataset in {time.time() - data_time:.3f} s")

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

g = open(f"old_loss/run{run}/Hyperparameters.txt", "w")

print(f"Grandezza dataset:\t{n_data}", file=g)
print(f"Batch size:\t{b_size}\n", file=g)
print(f"Epoche per il training:\t{num_epochs}", file=g)
print(f"Learning rate:\t{learning_rate}\n", file=g)
print(f"Cicli interni alla rete:\t{T}", file=g)
print(f"Dimensione embedding d:\t{d}", file=g)

g.close()
#M_VC = torch.ones(b_size, N, Q, dtype=torch.float32, device=dev)

pred = torch.zeros(2*b_size, dtype = torch.float32, device=dev)

total_out = []
c = 0

pair = torch.tensor([1, 0], dtype = torch.float32, device=dev)
y = pair.repeat(b_size)
y_t = pair.repeat(n_data)

testing = next(iter(Data_test))

f = open(f"old_loss/run{run}/loss.txt", "w")

for epoch in range(num_epochs):
    ep_time = time.time()
    correct = 0
    corr_t = 0
    model.train()
    
    with torch.no_grad():
        Mvv, Mvc, nvert, ncol = testing
        out = model(Mvv.to(dev), Mvc.to(dev), nvert, ncol)
        
        pred = out > 0.5
        for k,p in enumerate(pred):
            if p.item():
                pred[k] = 1.0
            else:
                pred[k] = 0.0
                
        corr_t += int((pred == y_t).sum())
        loss_tr = loss_fn(out, y_t)
        
        
    for data in Data:
        Mvv, Mvc, nvert, ncol = data
        
        out = model(Mvv.to(dev), Mvc.to(dev), nvert, ncol)

        """
        print ("\nRisult finale:\n\n", out)
        print("\nLabels:\n", y, "\n")
        """
        # Accuracy
        pred = out > 0.5
        for k,p in enumerate(pred):
            if p.item():
                pred[k] = 1.0
            else:
                pred[k] = 0.0
        
        correct += int((pred == y).sum())
        

        loss = loss_fn(out, y)
        
        
#        for i in model.parameters(): print (i,"\n\n")

                            # Backward and optimize
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

    if epoch%check == 0:
        c += 1
        torch.save(model.state_dict(), f'old_loss/run{run}/check_weights{c}.pt')
    train_acc = correct/(2*n_data)
    test_acc = corr_t/(2*n_data)
    t = time.time() - ep_time
    if loss.item() < 1e-3:
        break
    print(epoch, loss.item(), train_acc, loss_tr.item(), test_acc, file=f, flush=True)
    
#    if epoch%1 == 0:      #buono %400 per riferimento articolo
"""
print(f'\nEpoch: {epoch+1:03d}, Train Acc: {train_acc:.3f}')
print(f"Epoca svolta in {int(t/60):2d} min e {t%60:.3f} s")
print('Loss:\t', loss.item(), f'\nTrain Acc: {train_acc:.3f}')
"""
    
f.close()

data = np.loadtxt(f'old_loss/run{run}/loss.txt')
plt.plot(data[:,0], data[:,1])
plt.yscale("log")
plt.savefig("grafico.png")
plt.clf()

plt.plot(data[:,0], data[:,2])
plt.savefig("acc.png")
plt.clf()


torch.save(model.state_dict(), f'old_loss/run{run}/model_weights.pt')
t = time.time() - start_time
print(f'\n\nFinished training in {int(t/60):2d} min and {t%60:.1f} s')


#print("Colorable fraction:\t", colorable/n_data)