import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import WeightedRandomSampler
import pandas as pd
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
        super(Net,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.batch_norm = nn.BatchNorm2d(1)
        h0 = torch.zeros(self.num_layers,16,self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,16,self.hidden_size).to(device)
        self.h0 = nn.Parameter(h0,requires_grad=True)
        self.c0 = nn.Parameter(c0,requires_grad=True)
        
        
        
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.5) 
        # self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(hidden_size, num_classes) 
    
    def forward(self,x):

        
        # _, (hn, _) = self.lstm(x,(self.h0,self.c0))
        # _,hn = self.rnn(x,self.h0)
        _, (hn, _) = self.lstm(x,(self.h0,self.c0))
            
        hn = hn[-1]
        out = F.dropout(hn,p=0.5,training=self.training)
        out = self.fc2(out)
        return F.log_softmax(out)
    
    def print_h0(self):
        print(self.h0,self.c0)

def load_model(input_size,hidden_size,num_layers,num_classes,device,weights_path = None):
    net = Net(input_size,hidden_size,num_layers,num_classes,device)
    if weights_path is not None:
        net.load_state_dict(torch.load(weights_path))
    net.to(device)
    net.eval()
    return net

def find_local_maxima(array:list, min_value = 9, local_max_value=0):
    """
    returns: list with same size as input array, with 1 where the local maxima (higher than min_value) is and 0 elsewhere.
    """
    loc_max = [0 for x in array]
    for i in range(1,len(array)-1):
        if array[i] >= array[i-1] and array[i] >= array[i+1] and array[i] >= min_value: 
            loc_max[i] = local_max_value
            if array[i-1] > min_value:
                loc_max[i-1] = local_max_value
            if array[i+1] > min_value:
                loc_max[i+1] = local_max_value
        else:
            if loc_max[i] != local_max_value:
                loc_max[i] = 0
    return loc_max

def add_samples_max_overlap(ds,X_list,y_list):
    """
    Processes csv files to add keypoints from window into X_list and corresponding labels to y_list.
    Window is considered to have gesture in it if it has more frames with gesture than neighboring windows i.e. is a local maxima, 
    and then neighboring windows may have a gesture if they have at least 7 frames with gesture.
    """
    
    #Define window_size and overlap among windows.
    window_size = 30
    overlap = window_size//2
    
    X = ds.values
    tmp_labels = ds.iloc[:,127].values
    labels_in_window = []
    #Iterate through dataset and write down number of frames with gesture in window to labels_in_window and hands position for each frame in window to X_list.
    for i in range(ds.shape[0]//overlap-1):
        start_idx = i*overlap
        end_idx = start_idx + window_size
        
        labels_in_window.append(sum(tmp_labels[start_idx:end_idx] > 0.5))
        X_list.append(X[start_idx:end_idx,:126])
            
    # Convert number of labels in window a.k.a labels_in_window to labels for each window.
    label = np.max(X[:,127]).astype(np.int32)
    y = find_local_maxima(labels_in_window,min_value=7,local_max_value=label)
    for l in y:
        y_list.append(l)
        
def fill_lists(path_to_data,X_list,y_list):
    """
    Fills list with data for follow-up processing.
    """
    for file in os.listdir(path_to_data):
        add_samples_max_overlap(pd.read_csv(path_to_data+'/'+file),X_list,y_list)

def load_dataset(path:str,unbalanced_dataset:bool) -> torch.utils.data.DataLoader:
    """
    Loads a data from a folder with csv files each csv file contains keypoints processed via mediapipe and classlabel.
    Balances dataset using WeightedRandomSampler depending on unbalanced_dataset value.
    Arguments:
        path: str, path to dataset.
        unbalanced_dataset: bool, whether to balance dataset.
    Returns:
        torch.utils.data.DataLoader.
    """ 
    X,y = [], []
    fill_lists(path,X,y)
    labels = F.one_hot(torch.tensor(y,dtype=torch.int64),num_classes=9)
    X = torch.tensor(X)
    ds = torch.utils.data.TensorDataset(X,labels)
    if unbalanced_dataset:
        df = pd.DataFrame(ds[:][1].argmax(dim=1).numpy())
        samples = df.shape[0]
        w = 1./(df.value_counts().sort_index().values)
        weights = []
        for x in labels.argmax(dim=1):
            if x.numpy() == 0:
                weights.append(w[x.numpy()])
            
        sampler = WeightedRandomSampler(torch.from_numpy(np.array(weights)),samples)
        dataloader = torch.utils.data.DataLoader(ds,sampler=sampler,batch_size=16,drop_last=True)
    else:
        return torch.utils.data.DataLoader(ds,batch_size=16,drop_last = True) 
    
    return dataloader
   