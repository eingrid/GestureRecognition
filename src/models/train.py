import torch
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import argparse
import sys
import os
sys.path.append(os.path.abspath('/mnt/Dev/WORK/hand_gesture'))
from src.utils import load_dataset
from net import load_model 


 
def train_epoch(net,dataloader_train,optimizer,confmat,acc,f1,epoch,device,log_time):
    """
    Trains model on a PyTorch dataloader and prints metrics (accuracy,f1_score,confusion matrix) for model on a train dataset.
    Arguments:
        net: network class
        dataloader: torch.DataLoader
        optimizer: torch.optim
        confmat: torchmetric.ConfusionMatrix
        acc: torchmetric.Accuracy
        f1: torchmetric.F1Score
        epoch: int, current epoch number.
    """ 
    
    net.train()
    correct = 0
    epoch_loss = 0
    ConfusionMatrix = torch.zeros(size=(9,9)).to(device)
    f1_score = 0
    for (X_train, y_train) in dataloader_train:
        X_train, y_train = X_train.float().to(device), y_train.to(device)
        optimizer.zero_grad()
        output = net(X_train)
        loss = F.nll_loss(output,y_train.argmax(dim=1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pred = torch.exp(output)
        ConfusionMatrix += confmat(pred.argmax(dim=1).to(device),y_train.argmax(dim=1).to(device))
        correct += acc(pred.to(device),y_train.argmax(dim=1))
        f1_score += f1(pred.argmax(dim=1).to(device),y_train.argmax(dim=1))
        
    if epoch % log_time == 0 :
        print(f'TRAIN Epoch {epoch} Loss: {epoch_loss/len(dataloader_train)} Accuracy: {correct/len(dataloader_train)} F1 : {f1_score/len(dataloader_train)}')
        print(f'TRAIN Epoch {epoch} CONF MAT \n {ConfusionMatrix}')
    
def test_epoch(net,dataloader,confmat,acc,f1,epoch,device,log_time):
    """
    Calculates and prints metrics (accuracy,f1_score,confusion matrix) for model on a given dataloader.
    Arguments:
        net: network class
        dataloader: torch.DataLoader
        confmat: torchmetric.ConfusionMatrix
        acc: torchmetric.Accuracy
        f1: torchmetric.F1Score
        epoch: int, current epoch number.
    """
    net.train(False)
    epoch_loss = 0 
    accuracy = 0
    ConfusionMatrix = torch.zeros(size=(9,9)).to(device)
    f1_score = 0
    for (X_train, y_train) in dataloader:
        X_train, y_train = X_train.to(device), y_train.to(device)
        output = net(X_train.float())
        pred = output[-1].squeeze(0).argmax()
        loss = F.nll_loss(output,y_train.argmax(dim=1))
        epoch_loss += loss.item()
        pred = torch.exp(output)
        ConfusionMatrix += confmat(pred.argmax(dim=1).to(device),y_train.argmax(dim=1).to(device))
    
        accuracy += acc(pred.to(device),y_train.argmax(dim=1).to(device))
        f1_score += f1(pred.argmax(dim=1).to(device),y_train.argmax(dim=1).to(device))
    if epoch % log_time == 0 :
        print(f'TEST Epoch {epoch} Loss: {epoch_loss/len(dataloader)}, Accuracy: {accuracy/len(dataloader)} F1 : {f1_score/len(dataloader)} \n')
        print(f'TEST Epoch {epoch} CONF MAT {ConfusionMatrix}\n')
    
def train(epochs,net,unbalanced_dataset,train_ds_path,valid_ds_path,device,log_time,save_path):
    dataloader_train = load_dataset(train_ds_path,unbalanced_dataset)
    if valid_ds_path is not None:
        dataloader_valid = load_dataset(valid_ds_path,False)
        
    optimizer = optim.Adam(net.parameters(),lr=3e-5,weight_decay=2e-2)
    acc = torchmetrics.Accuracy().to(device)
    f1 = torchmetrics.F1Score(num_classes=9,average='macro').to(device)
    confmat = torchmetrics.ConfusionMatrix(num_classes=9).to(device)
    
    for epoch in range(epochs):
        train_epoch(net,dataloader_train,optimizer,confmat,acc,f1,epoch,device,log_time)
        if valid_ds_path is not None:
            test_epoch(net,dataloader_valid,confmat,acc,f1,epoch,device,log_time)
    torch.save(net.state_dict(),save_path)

def main(args):
    input_size = 126
    num_layers = 2
    hidden_size = 96
    num_classes = 9
    
    model_path = args['model_path']
    epoch_number = args['epoch_number']
    unbalanced_dataset = args['unbalanced_dataset']
    train_dataset = args['train_dataset']
    validation_dataset = args['validation_dataset']
    log_time = args['log_time']
    device = args['device']
    save_path = args['save_path']
    net = load_model(input_size,hidden_size,num_layers,num_classes,device,model_path) 
    train(epoch_number,net,unbalanced_dataset,train_dataset,validation_dataset,device,log_time,save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type = str, required=False, help='Loads checkpoint of the model. Trains model from zero if not specified ')
    parser.add_argument('--epoch_number',type = int, required=False, help='Epoch number to train for',default=200)
    parser.add_argument('--unbalanced_dataset',type = bool, required=True, help='Uses undersampling if set to true')
    parser.add_argument('--train_dataset',type = str, required=True, help='Path to train dataset')
    parser.add_argument('--validation_dataset',type = str, required=False, help='Path to validation dataset')
    parser.add_argument('--log_time',type = int, required=False, help='Print metrics every log_time epochs',default=40)
    parser.add_argument('--device',type = str, required=False, help='Device to use while training',default='cuda')
    parser.add_argument('--save_path',type = str, required=False, help='Path where model will be saved',default='model.pth')
    args = parser.parse_args()
    args = vars(args)
    main(args)
