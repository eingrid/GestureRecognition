import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes,device):
        super(Net,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        h0 = torch.zeros(self.num_layers,16,self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,16,self.hidden_size).to(device)
        self.h0 = nn.Parameter(h0,requires_grad=True)
        self.c0 = nn.Parameter(c0,requires_grad=True)
        
        
        
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,dropout=0.5) 
        self.fc = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(hidden_size, num_classes) 
    
    def forward(self,x):
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