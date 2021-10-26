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
        self.timeSteps = timeSteps
        self.batchsize = batchsize
        self.sequence_len = sequence_len
        self.samples_per_sequence = self.sequence_len//self.timesteps

        
        
        
        #init training data
        training_data_file  = open(data_path,'r')
        training_data_file_lines = training_data_file.readlines()
        one_training_data_sample_list = []
        
        self.dataset_list = []
        for line in training_data_file_lines:
            line = line.strip()
            if len(line) < 5:
                one_training_data_sample_list.append(self.map_dna_sequence_to_one_hot(line,sequence_len+self.right)) #+ sample_template)
            else:
                one_sample = torch.sum(torch.cat(one_training_data_sample_list,dim=0),dim=0)
                one_sample = map_to_left_window(one_sample)[:sequence_len]
                self.dataset_list.append(one_sample)
                #self.dataset_list.append()
                #one_training_data_sample_list = []
                one_training_data_sample_list = []
                
        
        #init label
        self.label_list = []
        if label_path:
            label_data_file = open(data_path,'r')
            label_data_file_lines = label_data_file.readlines()
            for line in label_data_file_lines:
                line = line.strip()
                one_sample = self.map_dna_sequence_to_one_hot(line,sequence_len)
                self.label_list.append(one_sample)




        #self.batchsize = batchsize
        #self.batchID = 1

        ## Number of utterances loaded into RAM.
        ## Increase this for speed, if you have more memory.
        #self.maxSplitDataSize = 1000

        ## Parameters for initialize the iteration
        #self.item_counter = 0
        #self.timeSteps_Num = 0


        #self.labelDir = tempfile.TemporaryDirectory()
        #aliPdf = self.labelDir.name + '/alipdf.txt'
        #aliPdf = '/home/glen/alipdf.txt'
        ## Generate pdf indices
        #Popen (['ali-to-pdf', exp + '/final.mdl',
        #            'ark:gunzip -c %s/ali.*.gz |' % ali,
        #            'ark,t:' + aliPdf]).communicate()

        ## Read labels
        #with open (aliPdf) as f:
        #    labels, self.numFeats = self.readLabels (f)

        ## Determine the number of steps
        ## need to re calculate The last patch will be deleted


        #self.numSteps = -(-self.numFeats // ( self.timeSteps))
      
        #self.inputFeatDim = inputDim ## IMPORTANT: HARDCODED. Change if necessary.
        #self.singleFeatDim = inputDim//(1+self.left+self.right)
        #self.outputFeatDim = self.readOutputFeatDim()
        #self.splitDataCounter = 0
        #print out the configuration
        #print ("NumFeats:%d"%(self.numFeats))
        #print("NumSteps:%d" % (self.numSteps))
        #print ("FeatsDim:%d"%(self.inputFeatDim))
        #print ("TimeSteps:%d"%(self.timeSteps))
        #print("OutputFeatDim:%d"%(self.outputFeatDim))
        
        #self.x = numpy.empty ((0, self.inputFeatDim), dtype=numpy.float32)
        #self.y = numpy.empty (0, dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        #self.f = numpy.empty(0, dtype=numpy.uint16)
        #self.batchPointer = 0
        #self.doUpdateSplit = True

        ## Read number of utterances
        #with open (data + '/utt2spk') as f:
        #    self.numUtterances = sum(1 for line in f)
        #self.numSplit = - (-self.numUtterances // self.maxSplitDataSize)
        #print("numUtterances:%d"%(self.numUtterances))
        #print("numSplit:%d" % (self.numSplit))

        ## Split data dir per utterance (per speaker split may give non-uniform splits)
        #if os.path.isdir (data + 'split' + str(self.numSplit)):
        #    shutil.rmtree (data + 'split' + str(self.numSplit))
        #Popen (['utils/split_data.sh', '--per-utt', data, str(self.numSplit)]).communicate()
        #print(labels)
        ## Save split labels and delete label
        #self.splitSaveLabels(labels)

    ## Clean-up label directory
    def __exit__ (self):
        self.labelDir.cleanup()


    def map_dna_sequence_to_one_hot(self, input_data,sequence_len=120):
        dna_dict = {'A': 0, 'T' : 1, 'G' : 2, 'C' : 3 }
        list_here = []
        for ch in input_data:
            list_here.append(dna_dict.get(ch))
        int_tensor = torch.FloatTensor(list_here)        
        one_hot_tensor = F.one_hot(input_data)
        if len(input_data)-sequence_len > 0:
            one_hot_tensor = one_hot_tensor[:sequence_len]
        elif len(input_data)-sequence_len < 0:
            one_hot_tensor = torch.cat([one_hot_tensor,torch.zeros(sequence_len-len(input_data) , 4)])
        return one_hot_tensor.unsqueeze(0)
    
    def map_to_left_window(self,input_data,left=8):
        data_list = []
        for i in range(left):
            data_list.append(torch.cat(input_data[i+1:], torch.zeros(i+1,4)))
        return torch.cat(data_list, dim=1)



    '''## Determine the number of output labels

    ## Load labels into memory
    def readLabels (self, aliPdfFile):
        labels = {}
        numFeats = 0
        FilledNumFeats = 0
        for line in aliPdfFile:
            line = line.split()
            numFeats += len(line)-1

            if (len(line)-1)%self.timeSteps!=0:
                FilledNumFeats += (self.timeSteps -(len(line)-1)%self.timeSteps) 
            
            labels[line[0]] = numpy.array([int(i) for i in line[1:]], dtype=numpy.uint16) ## Increase dtype if dealing with >65536 classes
        return labels, numFeats+FilledNumFeats
    
    ## Save split labels into disk
    def splitSaveLabels (self, labels):
        for sdc in range (1, self.numSplit+1):
            splitLabels = {}
            with open (self.data + '/split' + str(self.numSplit) + 'utt/' + str(sdc) + '/utt2spk') as f:
                for line in f:
                    uid = line.split()[0]
                    if uid in labels:
                        #print(uid)
                        splitLabels[uid] = labels[uid]
            with open (self.labelDir.name + '/' + str(sdc) + '.pickle', 'wb') as f:
                pickle.dump (splitLabels, f)


    ## Return split of data to work on
    ## There
    def getNextSplitData (self):
        p1 = Popen (['apply-cmvn','--print-args=false','--norm-vars=true',
                '--utt2spk=ark:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/utt2spk',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/cmvn.scp',
                'scp:' + self.data + '/split' + str(self.numSplit) + 'utt/' + str(self.splitDataCounter) + '/feats.scp','ark:-'],
                stdout=PIPE, stderr=subprocess.DEVNULL)
        #print("Here is the p1 stdout")
        #print(p1.stdout.readlines())
        p2 = Popen (['splice-feats','--print-args=false','--left-context='+str(self.left),'--right-context='+str(self.right),'ark:-','ark:-'], stdin=p1.stdout, stdout=PIPE)
        
        p1.stdout.close()	
        

        with open (self.labelDir.name + '/' + str(self.splitDataCounter) + '.pickle', 'rb') as f:
            labels = pickle.load (f)
        #print(labels)
        featList = []
        labelList = []
        flaglist = []

        while True:
            #print(p2.stdout)
            uid, featMat = kaldiIO.readUtterance (p2.stdout)
            #print("========================")
            #print(uid)
            #print(featMat)
            if uid == None:
                self.lable_list = labelList
                #print("=====111111111111111111111===================")
                return (numpy.vstack(featList), numpy.hstack(labelList), numpy.hstack(flaglist))
            if uid in labels:
                row,col = featMat.shape
                fillNum = self.timeSteps - (row % self.timeSteps)
                fillRight = fillNum//2
                fillLeft = fillNum - fillRight
                featMat = numpy.concatenate([numpy.tile(featMat[0],(fillLeft,1)), featMat, numpy.tile(featMat[-1],(fillRight,1))])
                labels4uid = labels[uid]
                labels4uid = numpy.concatenate([numpy.tile(labels4uid[0],(fillLeft,)), labels4uid, numpy.tile(labels4uid[-1],(fillRight,))])
                flags4uid = numpy.zeros(labels4uid.shape)
                flags4uid[-1] = 1
                flaglist.append(flags4uid)
                featList.append (featMat)
                labelList.append (labels4uid)
    '''

    def __len__(self):
        return int(len(self.dataset_list)*self.samples_per_sequence)

    def __getitem__(self, item):
        
        row_index = item//(self.batchsize*self.samples_per_sequence)*self.batchsize
        col_index = item%self.samples_per_sequence
        #self.samples_per_sequence
        data_sequences = self.dataset_list[row_index:row_index+self.batchsize]
        data_sequences = torch.FloatTensor(data_sequences)
        xMini = data_sequences[,col_index:col_index+self.time_steps]
        
        data_sequences = self.dataset_list[row_index:row_index+self.batchsize]
        data_sequences = torch.FloatTensor(data_sequences)
        yMini = data_sequences[,col_index:col_index+self.time_steps]
        
        return (xMini, yMini)
        '''while (self.item_counter >= self.timeSteps_Num):
            #print(self.item_counter)
            #print(self.timeSteps_Num)
            if not self.doUpdateSplit:
                self.doUpdateSplit = True
                #print('==============================')
                # return the last group of data, may repeated several times but not matter
                return (self.xMini,self.yMini,self.fMini)
                # break

            self.splitDataCounter += 1
            x, y, f = self.getNextSplitData()
            self.split_counter = 0
            self.batchPointer = len(self.x) - len(self.x) % self.timeSteps
            self.timeSteps_Num = self.batchPointer//self.timeSteps
            self.x = numpy.concatenate((self.x[self.batchPointer:], x))
            self.y = numpy.concatenate((self.y[self.batchPointer:], y))
            self.f = numpy.concatenate((self.f[self.batchPointer:], f))
            self.item_counter = 0
            self.batchnum = (len(self.x) - len(self.x) % (self.timeSteps)) // (self.timeSteps * self.batchsize)
            #print(self.batchnum)
            if self.splitDataCounter == self.numSplit:
                self.splitDataCounter = 0
                self.doUpdateSplit = False
        #print(item)


        item = item % ((len(self.x) - len(self.x) % self.timeSteps)//self.timeSteps)

        item = (item % self.batchsize) * self.batchnum + (item // self.batchsize)
        #print(item)

        self.xMini = self.x[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.yMini = self.y[item * self.timeSteps:item * self.timeSteps +  self.timeSteps]
        self.fMini = self.f[item * self.timeSteps:item * self.timeSteps + self.timeSteps]
        self.item_counter += 1

        self.xMini = torch.from_numpy(self.xMini)
        self.yMini = self.yMini.astype(numpy.int16)
        self.yMini = torch.from_numpy(self.yMini)
        self.fMini = self.fMini.astype(numpy.int16)
        self.fMini = torch.from_numpy(self.fMini)


        return (self.xMini, self.yMini, self.fMini)'''

