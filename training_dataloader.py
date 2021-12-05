import torch.utils.data as data
import torch
import torch.nn.functional as F




class TrainingDataLoader(data.Dataset):

    def __init__(self, data_path, label_path=None, batchsize=40, timeSteps=20, inputDim=195, sequence_len=120, left=0, right=4):

        self.left = left
        self.right = right
        self.timesteps = timeSteps
        self.batchsize = batchsize
        self.sequence_len = sequence_len
        self.samples_per_sequence = self.sequence_len//self.timesteps


        training_data_file  = open(data_path,'r')
        training_data_file_lines = training_data_file.readlines()
        one_training_data_sample_list = []
        reverse_one_training_data_sample_list = []
        self.dataset_list = []
        self.reverse_dataset_list = []
        for line in training_data_file_lines:

            line = line.strip()
            if len(line) > 5:
                sample = self.map_dna_sequence_to_one_hot(line,sequence_len)
                #reverse_sample = self.map_dna_sequence_to_one_hot(line[::-1], sequence_len)
                reverse_sample = torch.flip(self.map_dna_sequence_to_one_hot(line[::-1], sequence_len), [1])
                sample = torch.cat([sample[:sequence_len//2+1],reverse_sample[sequence_len//2+1:]])
                #reverse_sample = torch.flip(self.map_dna_sequence_to_one_hot(line[::-1],sequence_len), [1])
                #sample = torch.cat([sample,reverse_sample],dim=)

                one_training_data_sample_list.append(sample)
                #reverse_one_training_data_sample_list.append(reverse_sample)
            elif len(one_training_data_sample_list) > 1:
                #print(torch.cat(one_training_data_sample_list,dim=0).float())
                while len(one_training_data_sample_list)<10:

                    one_training_data_sample_list.append(torch.zeros(1,sequence_len,4,dtype=torch.long))
                    #reverse_one_training_data_sample_list.append(torch.zeros(1,sequence_len,4,dtype=torch.long))

                one_sample = torch.mean(torch.cat(one_training_data_sample_list, dim=0).float(), dim=0)
                #reverse_one_sample = torch.mean(torch.cat(reverse_one_training_data_sample_list, dim=0).float(), dim=0)
                one_sample = self.map_to_left_window(one_sample,self.left,self.right)[:sequence_len]
                #reverse_one_sample = self.map_to_left_window(reverse_one_sample, self.left, self.right)[:sequence_len]
                #print("================================")
                #print(one_sample,reverse_one_sample)
                one_sample = self.position_embedding(one_sample)
                #reverse_one_sample = self.position_embedding(reverse_one_sample)
                self.dataset_list.append(one_sample)
                #self.reverse_dataset_list.append(reverse_one_sample)
                one_training_data_sample_list = []
                #reverse_one_training_data_sample_list = []

        print("Length of two dataset_list: ",len(self.dataset_list), len(self.reverse_dataset_list))

        #init label
        self.label_list = []



        if label_path:
            label_data_file = open(label_path,'r')
            label_data_file_lines = label_data_file.readlines()

            for line in label_data_file_lines:
                line = line.strip()
                if len(line) > 5:
                    
                    one_sample = self.map_dna_sequence_to_one_hot(line,sequence_len, True ) #.squeeze(0)
                    self.label_list.append(one_sample)


        self.numFeats = len(self.label_list)*self.sequence_len
        
        print("Num Features:")
        print(self.numFeats)
        print("Num Samples:")
        print(len(self.label_list))

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()

    def position_embedding (self, input_data):
        seq_len,_ = input_data.size()
        pos_embeding = torch.ones(seq_len,1).float()
        pos_embeding = torch.cumsum(pos_embeding,dim=0)/seq_len
        #print(pos_embeding)
        return torch.cat([input_data,pos_embeding],dim=1)

    def map_dna_sequence_to_one_hot(self, input_data,sequence_len=120, label = False):
        dna_dict = {'A': 0, 'T' : 1, 'G' : 2, 'C' : 3 }
        list_here = []
        for ch in input_data:
            list_here.append(dna_dict.get(ch))
        int_tensor = torch.FloatTensor(list_here) 
        if label:
            if int_tensor.size(0)-sequence_len > 0:
                int_tensor= int_tensor[:sequence_len]
            elif int_tensor.size(0)-sequence_len < 0:
                int_tensor = torch.cat([int_tensor,torch.zeros(sequence_len-len(input_data))])
            return int_tensor.to(torch.long)
        one_hot_tensor = F.one_hot(int_tensor.to(torch.int64),4)
        if one_hot_tensor.size(0)-sequence_len > 0:
            one_hot_tensor = one_hot_tensor[:sequence_len]
        elif one_hot_tensor.size(0)-sequence_len < 0:
            #print(one_hot_tensor,torch.zeros(sequence_len-len(input_data) , 4,dtype=torch.long) )
            one_hot_tensor = torch.cat([ one_hot_tensor, torch.zeros(sequence_len-len(input_data),4,dtype=torch.long) ])
        return one_hot_tensor.unsqueeze(0)
    
    def map_to_left_window(self,input_data,left = 4,right=4):
        data_list = []

        data_list.append(input_data)
        for i in range(left):
            #data_list.append(torch.cat([ torch.zeros(i + 1, 40), input_data[:-i-1]], dim=0))
            data_list.append(torch.cat([torch.zeros(i + 1, 4), input_data[:-i - 1]], dim=0))
        for i in range(right):
            #data_list.append(torch.cat([input_data[i+1:], torch.zeros(i+1,4)],dim=0))
            #data_list.append(torch.cat([input_data[i + 1:], torch.zeros(i + 1, 40)], dim=0))
            data_list.append(torch.cat([input_data[i + 1:], torch.zeros(i + 1, 4)], dim=0))
        #if input
        return torch.cat(data_list, dim=1)




    def __len__(self):
        return self.numFeats // (self.timesteps*self.batchsize)


    def __getitem__(self, item):
        #print(item)
        row_index = self.batchsize*(item//self.samples_per_sequence)
        col_index = item%self.samples_per_sequence
        #print(self)
        #print(row_index,col_index)
        data_sequences_list = self.dataset_list[row_index:row_index+self.batchsize]
        #reverse_data_sequences_list = self.reverse_dataset_list[row_index:row_index + self.batchsize]
        data_sequences = torch.stack(data_sequences_list)
        #reverse_data_sequences = torch.stack(reverse_data_sequences_list)
        xMini = data_sequences[:,col_index*self.timesteps:(col_index+1)*self.timesteps]
        #reverse_xMini = reverse_data_sequences[:, col_index * self.timesteps:(col_index + 1) * self.timesteps]
        
        label_sequences_list = self.label_list[row_index:row_index+self.batchsize]
        label_sequences = torch.stack(label_sequences_list)
        yMini = label_sequences[:,col_index*self.timesteps:(col_index+1)*self.timesteps]



        return (xMini, yMini)

