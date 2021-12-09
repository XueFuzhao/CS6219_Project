import torch.utils.data as data
import torch.nn as nn
import torch
from training_dataloader import TrainingDataLoader
#from sru import SRU, SRUCell
import sys
import time
import torch.nn.functional as F

data_train_file = sys.argv[1]
label_train_file  = sys.argv[2]
data_val_file = sys.argv[3]
label_val_file  = sys.argv[4]
init_lr = float(sys.argv[5])
rigth_windew_length = int(sys.argv[6])
seq_len = 120


learning = {'rate' : init_lr,
            'singFeaDim' : 4,#*10,
            'minEpoch' : 30,
            'batchSize' : 128,
            'timeSteps' : 10,
            'left' : 0,
            'right': rigth_windew_length,
            'hiddenDim' : 256,
            'layerNum': 3,
            'targetDim':4
           }



feaDim = (learning['left'] + learning['right']+1) * learning['singFeaDim'] + 1
trDataset = TrainingDataLoader(data_train_file, label_train_file, learning['batchSize'], learning['timeSteps'], learning['singFeaDim'] ,seq_len, learning['left'],learning['right'])
cvDataset = TrainingDataLoader(data_val_file, label_val_file, learning['batchSize'], seq_len, learning['singFeaDim'] ,seq_len, learning['left'],learning['right'])


trGen = data.DataLoader(trDataset,batch_size=1,shuffle=False,num_workers=0)
cvGen = data.DataLoader(cvDataset,batch_size=1,shuffle=False,num_workers=0)

from sru import SRU, SRUCell

class lstm(nn.Module):
    def __init__(self,input_size = feaDim, hidden_size = 128 , output_size = 4 ):
        super(lstm,self).__init__()
        self.hidden_size = hidden_size
        #self.lstm_layer2 = SRU(input_size, hidden_size, num_layers=learning["layerNum"],dropout=0.1)
        self.lstm_layer2 = nn.GRU(input_size, hidden_size, num_layers=learning["layerNum"],dropout=0.1)
        self.Dense_layer3 = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self,x,c_in):
        b, t, h = x.size()
        x = torch.transpose(x,1,0)
        x,c = self.lstm_layer2(x,c_in)
        x = torch.transpose(x,1,0)
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer3(x)
        return x,c
    
model = lstm(input_size=feaDim, hidden_size= learning['hiddenDim'],output_size=learning['targetDim']).cuda()

loss_function = nn.CrossEntropyLoss()
loss_function_dtw = SoftDTW(use_cuda=True, gamma=0.1)

optimizer = torch.optim.Adam( model.parameters(),lr=learning['rate'])



def train(model, train_loader, my_loss, optimizer, epoch, hidden1):

    model.train()
    acc = 0

    for batch_idx, (x,y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda().squeeze(0)
        y = y.cuda().squeeze(0)
        b, t, h = x.size()

        if b == learning['batchSize']:
            optimizer.zero_grad()
            if batch_idx%(seq_len/learning['timeSteps']) == 0:
                hidden1_before = hidden1.cuda()
            else:
                hidden1_before = hidden1_after.cuda()


            output, hidden1_after = model(x,hidden1_before)

            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()

            loss = my_loss(output, y)
            loss.backward()

            optimizer.step()


            _, pred = torch.max(output.data, 1)


            hidden1_after = hidden1_after.detach()
            

            acc += ((pred == y).sum()).cpu().numpy()
            if (batch_idx % 4000 == 0):
                print("train:     epoch:%d ,step:%d, total loss:%f "%(epoch+1,batch_idx,loss))

    print(acc)
    print(trDataset.numFeats)
    print("Accuracy: %f"%(acc/trDataset.numFeats))


def val(model, train_loader, my_loss, optimizer, epoch, hidden1):

    model.eval()
    acc = 0
    val_loss = 0
    strands = 0
    correct_strands = 0
    val_loss_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda().squeeze(0)
        y = y.cuda().squeeze(0)


        b, t, h = x.size()
        if b == learning['batchSize']:
            optimizer.zero_grad()
            if batch_idx%(seq_len/learning['timeSteps']) == 0:
                hidden1_before = hidden1.cuda()
            else:
                hidden1_before = hidden1_after.cuda()
            hidden1_before = torch.zeros_like(hidden1).cuda()
            with torch.no_grad():
                output, hidden1_after= model(x, hidden1_before)

            _, pred = torch.max(output.data, 1)
            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()
            loss = my_loss(output, y)
            val_loss += float(loss.item())
            val_loss_list.append(val_loss)


            acc += ((pred == y).sum()).cpu().numpy()
            strands += b
            seq_pred = (pred == y).view(y_batch_size,-1).sum(dim=1).float()/(seq_len)
            seq_y = torch.ones_like(seq_pred)
            correct_strands += ((seq_pred == seq_y).sum()).cpu().numpy()
            if (batch_idx % 200 == 0):#1000/60
                print("val:        epoch:%d ,step:%d, total loss:%f"%(epoch+1,batch_idx,loss))

    print(acc)
    print(cvDataset.numFeats)
    print("Accuracy: %f" % (acc / cvDataset.numFeats))
    print("Strand Accuracy: %f" % (correct_strands/ strands))
    print("LOSS: %f" % (val_loss / len(val_loss_list)))
    return float(val_loss / len(val_loss_list))

val_loss_before = 10000
count_epoch = 0


for epoch in range(100):
    print("=====================================================================")

    h1 = torch.zeros(learning['layerNum'], learning['batchSize'], learning['hiddenDim'])


    time_start = time.time()
    train(model, trGen, loss_function, optimizer, epoch, h1)
    time_cost = time.time() - time_start
    print("Train Time Cost : %f"%(time_cost))
    time_val_start = time.time()
    val_loss_after = val(model, cvGen, loss_function, optimizer, epoch, h1)

    if(val_loss_before - val_loss_after < 0) and (count_epoch > 2):
        val_loss_before = 10000
        count_epoch = 0
    else:
        val_loss_before = val_loss_after
        count_epoch += 1

    torch.save(model.state_dict(), 'sru.pth')
    time_end = time.time()
    time_cost = time_end - time_val_start
    print("Val Time Cost : %f"%(time_cost))



