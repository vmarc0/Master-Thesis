import torch
from time import time
from torch_geometric.loader import DataLoader
from Coloured_Network import Colored_Net
from Dataset_creator import Phase_tr
from Library import printer
import os

dev = torch.device('cuda')

# n_graphs default: 10000
n_data = 10000
seed = False
# Q = 5
painted = False
connectivity = 15
N = 100

Acc_lim = 0.85
num_ep = 3500
check = 1000 #ogni quante epoche stampare grafici e salvare il training
b_size = 30
learning_rate = 1e-4

tot_run = 6

if seed:
    my_seed = 0
    torch.manual_seed(my_seed)

origin = os.getcwd()

root_dataset_train = "/home/veithmar/dataset/Mydataset/train/"
root_dataset_test = "/home/veithmar/dataset/Mydataset/test/"

#root_dataset_test = "D:/Python/Miei programmi/Data/New_dataset/small/test/"
#root_dataset = "D:/Python/Miei programmi/Data/Generated_dataset/"


for conn in range(7):
    connect = connectivity + conn

    Dataset_training = Phase_tr(root_dataset_train, N, painted, connect)
    Dataset_testing = Phase_tr(root_dataset_test, N, painted, connect)

    loader_train = DataLoader(Dataset_training, batch_size = b_size)
    loader_test = DataLoader(Dataset_testing, batch_size = b_size)

    print("Dataset pronto\n")

    loss_fn = torch.nn.BCELoss()
    evalu = False

    for n_run in range(tot_run):
        nlear = 0
        if n_run > 2:
            evalu = True
        os.chdir(origin)
        percorso = "runs/"
        if seed:
            percorso += "manseed_"

        if not evalu:
            percorso += "noeval_"

        if painted:
            percorso += "col_"
        else:
            percorso += "uncol_"

        percorso += f"c_{connect}/"

        if not os.path.exists(percorso):
            os.mkdir(percorso)
        run = 1 + len(os.listdir(percorso))
        percorso += f"run{run}"

        os.mkdir(percorso)

        os.chdir(percorso)

        g = open("Hyperparameters.txt", "w")

        print(f"Batch size:\t{b_size}\n", file=g)
        print(f"Epoche per il training:\t{num_ep}", file=g)
        print(f"\nLearning rate:\t{learning_rate}\n", file=g)

        print(f"\nDataset colorato:\t{painted}", file=g)
        print(f"\nConnettività:\t{connect}", file=g)
        print(f"\nDimensioni Dataset:\t{n_data}", file=g)
        if seed:
            print(f"\nMy seed:\t{my_seed}", file=g)

        g.close()

        repeat = True
        while(repeat):

            repeat = False
            model = Colored_Net()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            f = open("loss.txt", "w")

            tempo = time()
            train_time = 0
            last = []
            a = 1
            for epoch in range(num_ep):

                model.train()

                correct = 0
                loss = 0
                t0 = time()
                for tr_dat in loader_train:
                    tr_dat.to(dev)

                    out = model(tr_dat.x, tr_dat.edge_index, tr_dat.batch)

                    # Accuracy
                    predic = out > 0.5
                    for k,p in enumerate(predic):
                        if p.item():
                            predic[k] = 1.0
                        else:
                            predic[k] = 0.0


                    correct += int((predic == tr_dat.y).sum())

                    loss_b = loss_fn(out, tr_dat.y)
                    loss += loss_b
                                        # Backward and optimize
                    loss_b.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                t1 = time() - t0
                train_time += t1
                if evalu:
                    model.eval()
                with torch.no_grad():
                    corr_t = 0
                    loss_test = 0
                    false_pos = 0
                    false_neg = 0
                    for test_dat in loader_test:
                        test_dat.to(dev)

                        out = model(test_dat.x, test_dat.edge_index, test_dat.batch)

                        # Accuracy
                        predic = out > 0.5
                        for k,p in enumerate(predic):
                            if p.item():
                                predic[k] = 1.0
                            else:
                                predic[k] = 0.0

                        bool_acc = (predic == test_dat.y)
                        corr_t += int(bool_acc.sum())

                        for k,boolean in enumerate(bool_acc):
                            if not boolean.item():
                                if predic[k] == 1:
                                    false_pos += 1
                                else:
                                    false_neg += 1

                        loss_b = loss_fn(out, test_dat.y)
                        loss_test += loss_b

                loss /= len(loader_train)
                loss_test /= len(loader_test)

                train_acc = correct/n_data
                test_acc = corr_t/n_data

                false_pos /= n_data
                false_neg /= n_data

                print(epoch, loss.item(), train_acc, loss_test.item(), test_acc, false_pos, false_neg, file=f, flush=True)

                if epoch == 10*a and a<4:
                    last.append(loss.item())
                    last.append(train_acc)
                    a += 1

                if (epoch+1)%int(check) == 0:
                    torch.save(model.state_dict(), f'check_weights_ep{epoch+1}.pt')
                    tp = time() - tempo
                    print(f"Svolte {epoch+1} epoche in {int(tp/60):2d} min and {tp%60:.1f} s")
                    printer()

                if a == 4:
                    a += 1
                    if (last[0] == last[2] == last[4] and last[1] == last[3] == last[5]):
                        nlear += 1
                        if nlear < 20:
                            repeat = True
                        break
                if train_acc > Acc_lim:
                    ep_end = epoch+1
                    break

        printer()
        torch.save(model.state_dict(), 'model_weights.pt')

        tp = time() - tempo
        print(f"Training concluso in {int(tp/60):2d} min and {tp%60:.1f} s")
        f.close()

        g = open("Hyperparameters.txt", "a")
        print(f"\nTentativi andati a vuoto:\t{nlear}", file=g)
        if train_acc > Acc_lim:
            print(f"\nEpoche effettive svolte per il training:\t{ep_end}", file=g)
        else:
            print("\nIl training si è svolto per intero\n")
        print(f"Training concluso in {int(train_time/60):2d} min and {train_time%60:.1f} s", file=g)
        g.close()
