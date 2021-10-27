import torch.utils.data as data
import torch
import torch.nn.functional as F


#dna_dict = {'A': 0, 'T' : 1, 'G' : 2, 'C' : 3 }

class TrainingDataLoader(data.Dataset):

    def __init__(self, data_path, label_path=None, batchsize=40, timeSteps=20, inputDim=195, sequence_len=120, left=0, right=4):
        #self.data = data
        #self.ali = ali
        #self.exp = exp

        #self.lable_list = [0]
        self.left = left
        self.right = right
        self.timesteps = timeSteps
        self.batchsize = batchsize
        self.sequence_len = sequence_len
        self.samples_per_sequence = self.sequence_len//self.timesteps

         
        
        
        #init training data
        training_data_file  = open(data_path,'r')
        training_data_file_lines = training_data_file.readlines()
        #print(training_data_file_lines)
        one_training_data_sample_list = []
        
        self.dataset_list = []
        for line in training_data_file_lines:

            line = line.strip()
            if len(line) > 5:
                one_training_data_sample_list.append(self.map_dna_sequence_to_one_hot(line))#+self.right)) + sample_template)
            elif len(one_training_data_sample_list) > 1:
                #print(one_training_data_sample_list)
                #print(torch.cat(one_training_data_sample_list,dim=0).size())
                one_sample = torch.sum(torch.cat(one_training_data_sample_list,dim=0),dim=0)
                one_sample = self.map_to_left_window(one_sample,self.right)[:sequence_len]
                #print(one_sample.size())
                #print(asdasda)
                self.dataset_list.append(one_sample)
                #self.dataset_list.append()
                #one_training_data_sample_list = []
                one_training_data_sample_list = []
                
        
        #init label
        self.label_list = []



        if label_path:
            label_data_file = open(label_path,'r')
            label_data_file_lines = label_data_file.readlines()
            #print(len(label_data_file_lines))
            for line in label_data_file_lines:
                line = line.strip()
                if len(line) > 5:
                    
                    one_sample = self.map_dna_sequence_to_one_hot(line,sequence_len, True ) #.squeeze(0)
                    #print(one_sample.size())
                    #print(asdasd)
                    self.label_list.append(one_sample)


        self.numFeats = len(self.label_list)*self.sequence_len
        
        print("Num Features:")
        print(self.numFeats)

        print("Num Samples:")
        print(len(self.label_list))
        #print(len(self.dataset_list))

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()


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
            one_hot_tensor = torch.cat([one_hot_tensor,torch.zeros(sequence_len-len(input_data) , 4)])
        return one_hot_tensor.unsqueeze(0)
    
    def map_to_left_window(self,input_data,left=4):
        data_list = []
        #data_list.append()
        #print(input_data.size())

        data_list.append(input_data)
        for i in range(left):
            #data_list.append(input_data)
            #print(input_data[i+1:].size())
            #print(torch.zeros(i+1,4).size())
            data_list.append(torch.cat([input_data[i+1:], torch.zeros(i+1,4)],dim=0))
        return torch.cat(data_list, dim=1)




    def __len__(self):
        #return int(len(self.dataset_list)*self.samples_per_sequence)
        return self.numFeats // (self.timesteps*self.batchsize)


    def __getitem__(self, item):
        
        row_index = item//(self.batchsize*self.samples_per_sequence)
        col_index = item%self.samples_per_sequence
        #self.samples_per_sequence

        data_sequences_list = self.dataset_list[row_index:row_index+self.batchsize]
        
        #data_sequences = torch.FloatTensor(data_sequences)
        #data_sequences = torch.cat(data_sequences_list,dim=0).view(self.timesteps,self.batchsize,-1)
        data_sequences = torch.stack(data_sequences_list)
        xMini = data_sequences[:,col_index:col_index+self.timesteps]
        
        label_sequences_list = self.label_list[row_index:row_index+self.batchsize]
        #data_sequences = torch.FloatTensor(data_sequences)
        #label_sequences = torch.cat(label_sequences_list,dim=1).view(self.timesteps,self.batchsize,-1)
        label_sequences = torch.stack(label_sequences_list)
        yMini = label_sequences[:,col_index:col_index+self.timesteps]
        
        #item += self.batchsize

        return (xMini, yMini)

