import torch
import sys
import time
import torch.nn as nn
from Dataset_loading import CustomGraphDataset_batched
from NN_fv_onArt import Graph_Net_art
from articledata_to_pyg import extract
from articledata_to_pyg import build_MVC

start_time = time.time()

dev = torch.device('cuda')
if not torch.cuda.is_available():
    sys.exit("GPU non disponibile")

#dimensione embedding per V e C
d = 16
Q = 2

#n_iterations of the LSTM inside the Net: T=32 in article
T = 8

#dimensioni singolo grafo dipendenti dal dataset; Batch_size:
b_size = 5
n_data = 2*b_size

num_epochs = 15   #Nell'articolo sono 5300
#n_iters = 50      numero di batches
data_time = time.time()
Data = CustomGraphDataset_batched(batch_size = b_size)

model = Graph_Net_art(b_size, T, d)
#model = example(d)
print(f"\nCaricato il dataset in {time.time() - data_time:.3f} s")

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#M_VC = torch.ones(b_size, N, Q, dtype=torch.float32, device=dev)

pred = torch.zeros(n_data, dtype = torch.float32, device=dev)

total_out = []

pair = torch.tensor([1, 0], dtype = torch.float32, device=dev)
y = pair.repeat(b_size)

for epoch in range(num_epochs):
    ep_time = time.time()
    correct = 0
    model.train()
#    print("\nEPOCH:\t", epoch+1, "\n\n")
    for data in Data:
        
        Mvv, _, nvert, ncol = data
        Mvv = extract(Mvv, nvert)
        Mvc = build_MVC(ncol, nvert)
        
        
        out = model(Mvv, Mvc, nvert, ncol)


#        print ("\nRisult finale:\n\n", out)
#        print("\nLabels:\n", data.y, "\n")
        
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

    train_acc = correct/n_data
    t = time.time() - ep_time
    if epoch%2 == 0:      #buono %400 per riferimento articolo
        print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}')
        print(f"\nEpoca svolta in {int(t/60):2d} min e {t%60:.3f} s")
        print('\nLoss:\t', loss.item())
t = time.time() - start_time
print(f'\n\nFinished training in {int(t/60):2d} min e {t%60:.1f} s')
TOT = sum(total_out)/len(total_out)
print ("\nOutput medio:\t", TOT.item())

#print("Colorable fraction:\t", colorable/n_data)