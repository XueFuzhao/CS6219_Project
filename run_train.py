import torch.utils.data as data
import torch.nn as nn
import torch
from training_dataloader import TrainingDataLoader
from sru import SRU, SRUCell



data_train_file = sys.argv[1]
label_train_file  = sys.argv[2]
data_val_file = sys.argv[3]
label_val_file  = sys.argv[4]
init_lr = float(sys.argv[5])


learning = {'rate' : init_lr,
            'singFeaDim' : 4, 
            'minEpoch' : 30,
            'batchSize' : 128,#40 at first
            'timeSteps' : 20,
            'left' : 0,
            'right': 4,
            'hiddenDim' : 128,
            'layerNum': 3,
            'targetDim':4
           }



feaDim = (learning['left'] + learning['right']+1) * learning['singFeaDim']
trDataset = TrainingDataLoader(data_train_file, label_train_file, learning['batchSize'], learning['timeSteps'], learning['singFeaDim'] ,learning['left'],learning['right'])

trGen = data.DataLoader(trDataset,batch_size=learning['batchSize'],shuffle=False,num_workers=0)


class lstm(nn.Module):
    def __init__(self,input_size = feaDim, hidden_size = 1024 , output_size = 1095 ):
        super(lstm,self).__init__()
        self.Dense_layer1 = nn.Sequential(Dense(input_size,hidden_size))
        self.lstm_layer2 = nn.Sequential(SRU(hidden_size, hidden_size,num_layers=3))
        self.Dense_layer3 = nn.Sequential(Dense(hidden_size, output_size))

    def forward(self,x,c):
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer1(x)
        x = torch.reshape(x, (b, t, 1024))
        x,c = self.lstm_layer2(x)
        b, t, h = x.size()
        x = torch.reshape(x, (b*t, h))
        x = self.Dense_layer3(x)
        return x,c
    
model = lstm(input_size=feaDim, hidden_size= learning['hiddenDim'],output_size=learning['targetDim']).cuda()

loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD( model.parameters(),lr=0.25,momentum=0.5,weight_decay=0.001)
optimizer = torch.optim.Adam( model.parameters(),lr=learning['rate'],weight_decay=0.001)

#cvDataset = TrainingDataLoader(data_cv, ali_cv, gmm,learning['timeSteps'], feaDim,learning['left'],learning['right'])


def train(model, train_loader, my_loss, optimizer, epoch, hidden1, hidden2, ADJ_node, ADJ_id, current_node):

    model.train()
    acc = 0

    for batch_idx, (x,y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()
        #f = f.cuda()
        #print(f)
        b, t, h = x.size()

        if b == learning['batchSize']:
            optimizer.zero_grad()
            if batch_idx == 0:
                hidden1_before = hidden1.cuda()
            else:
                hidden1_before = hidden1_after.cuda()


            output, hidden1_after = model(x,hidden1_before)

            y_batch_size, y_time_steps = y.size()
            y = torch.reshape(y, tuple([y_batch_size * y_time_steps]))
            y = y.long()

            #print(kl_loss)

            loss = my_loss(output, y) 
            loss.backward()

            optimizer.step()
            _, pred = torch.max(output.data, 1)


            hidden1_after = hidden1_after.detach()
            

            acc += ((pred == y).sum()).cpu().numpy()
            if (batch_idx % 100 == 0):

                print("train:     epoch:%d ,step:%d, total loss:%f "%(epoch+1,batch_idx,loss))

    print(acc)
    print(trDataset.numFeats)
    print("Accuracy: %f"%(acc/trDataset.numFeats))


def val(model, train_loader, my_loss, optimizer, epoch, hidden1, hidden2, ADJ_node, ADJ_id, current_node):

    model.eval()
    acc = 0
    val_loss = 0
    val_loss_list = []

    for batch_idx, (x, y) in enumerate(train_loader):
        # If you run this code on CPU, please remove the '.cuda()'
        x = x.cuda()
        y = y.cuda()

        b, t, h = x.size()
        if b == learning['batchSize']:
            optimizer.zero_grad()
            if batch_idx == 0:
                hidden1_before = hidden1.cuda()
            else:
                hidden1_before = hidden1_after.cuda()
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
            if (batch_idx % 100 == 0):#1000/60
                print("val:        epoch:%d ,step:%d, total loss:%f"%(epoch+1,batch_idx,loss,kl_loss1, kl_loss2))

    print(acc)
    print(cvDataset.numFeats)
    print("Accuracy: %f" % (acc / cvDataset.numFeats))
    print("LOSS: %f" % (val_loss / len(val_loss_list)))
    return float(val_loss / len(val_loss_list))

val_loss_before = 10000
count_epoch = 0

#init_cell4lstm2 = torch.zeros(1, batch_size, hidden_size).cuda()
#init_hidden4lstm2 = torch.zeros(1, batch_size, hidden_size).cuda()

for epoch in range(30):
    print("=====================================================================")

    h1 = torch.zeros(learning['layerNum'], learning['batchSize'], learning['hiddenDim'])


    time_start = time.time()
    train(model, trGen, loss_function, optimizer, epoch, h1)

    #print(scheduler.get_lr())

    val_loss_after = val(model, cvGen, loss_function, optimizer, epoch, h1, h2, ADJ_node, ADJ_id, current_node)

    if(val_loss_before - val_loss_after < 0) and (count_epoch > 2):
        scheduler.step()
        val_loss_before = 10000
        count_epoch = 0
    else:
        val_loss_before = val_loss_after
        count_epoch += 1

    torch.save(model.state_dict(), 'sru.pth')
    time_end = time.time()
    time_cost = time_end - time_start
    print("Time Cost : %f"%(time_cost))
    #if (float(scheduler.get_lr()[0]) < 0.001):
    #    break


'''print("The second ladder is coming!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

for epoch in range(90):
    print("=====================================================================")


    h1 = torch.zeros(learning['layerNum'], 1, learning['batchSize'], learning['hiddenDim'])
    h2 = torch.zeros(learning['layerNum2'], 1, learning['batchSize'], learning['hiddenDim'])
    ADJ_node = torch.zeros(learning['batchSize'], learning['window_size'], learning['graph_dim'])
    current_node = torch.zeros(learning['batchSize'], learning['hiddenDim'])
    ADJ_id = []
    for i in range(learning['batchSize']):
        ADJ_id.append(0)

    time_start = time.time()
    train(model, trGen, loss_function, optimizer2, epoch, h1, h2 , ADJ_node, ADJ_id, current_node)

    print(scheduler2.get_lr())

    val_loss_after = val(model, cvGen, loss_function, optimizer2, epoch, h1, h2 , ADJ_node, ADJ_id, current_node)


    scheduler2.step()
    torch.save(model.state_dict(), exp + '/dnn.nnet.pth')
    time_end = time.time()
    time_cost = time_end - time_start
    print("Time Cost : %f"%(time_cost))
    if float(scheduler2.get_lr()[0]) < 1e-6:
        break'''