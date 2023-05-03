import torch
import torch.nn as nn
from NN_fv_onArt import Graph_Net_art
import os
from Dataset_loading import CustomGraphDataset_batched

dev = torch.device('cuda')
path = 'D:/Python/Miei programmi/Data/Graph data/adversarial-training/'

run = len(os.listdir('old_loss/'))

d = 16
T = 8
b_size = 10

num_data = 10

model = Graph_Net_art(b_size, T, d)
model.load_state_dict(torch.load(f'old_loss/run{run}/model_weights.pt'))

Data = CustomGraphDataset_batched(num_data, root=path, batch_size = b_size)

pair = torch.tensor([1, 0], dtype = torch.float32, device=dev)
y = pair.repeat(b_size)

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model.eval()

correct = 0
for data in Data:
    Mvv, Mvc, nvert, ncol = data
        
    out = model(Mvv.to(dev), Mvc.to(dev), nvert, ncol)
    
    print ("\nRisult finale:\n\n", out)
    
        # Accuracy
    pred = out > 0.5
    for k,p in enumerate(pred):
        if p.item():
            pred[k] = 1.0
        else:
            pred[k] = 0.0
        
    correct += int((pred == y).sum())
        

    loss = loss_fn(out, y)
    print (f"La loss\t{loss.item()}\n")
        
#        for i in model.parameters(): print (i,"\n\n")

Acc = correct/(2*num_data)
print (f"Accuracy:\t{Acc}")