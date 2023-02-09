import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
import random
from tqdm import tqdm


from model.models import ResNet, BasicBlock, Bottleneck
from loader.dataset import ScenesDataset, get_transform, get_df
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import classification_report

def parse_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, required=True, default='resnet34')
  parser.add_argument('--trainCsvPath', type=str, required=True)
  parser.add_argument('--valCsvPath', type=str, required=True)
  parser.add_argument('--epoch', type=str, required=True, default=2)


  args, _ = parser.parse_known_args()
  return args

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model():
    args = parse_args()
    if args.model_name == 'resnet34':
        model = ResNet(BasicBlock, [3,4,6,3])
        return model
    if args.model_name == 'resnet50':
        model = ResNet(Bottleneck, [3,4,6,3])
        return model

def train_epoch(epoch,model, loader, optimizer, loss_func, device):
    model.train()
    running_loss = 0.0
    reporting_step = 50
    for i,(images,labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % reporting_step == reporting_step-1:
            print(f'Epoch {i} step {i} avg loss {running_loss/reporting_step: .4f}')
            running_loss = 0.0

def test_epoch(epoch,model, loader, device):
    ytrue = []
    ypred = []
    with torch.no_grad():
        model.eval()
        for i, (images, labels) in enumerate(loader):
            #i=0
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs, dim=1)
            ytrue+= list(labels.cpu().numpy())
            ypred += list(predicted.cpu().numpy())
    return ypred, ytrue


if __name__ == '__main__':
    args = parse_args()
    device = torch.device("cuda:0")

    train_df,out_dim = get_df(args.trainCsvPath)
    val_df,_ = get_df(args.valCsvPath)
    transform_train, transform_val = get_transform(image_size=112)
    
    trainset = ScenesDataset(train_df, transforms=transform_train, mode='train')
    valset = ScenesDataset(val_df, transforms=transform_val, mode='val')
    
    train_loader = DataLoader(train_df,batch_size =32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_df, batch_size =32, shuffle=False, num_workers=2)
    
    
    model = get_model().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters, lr= 0.001, weight_decay= 5e-4)

    for epoch in range (args.epoch):
        train_epoch(epoch, model, train_loader, loss_func, optimizer, device=device)
        ypred, ytrue = test_epoch(epoch, model, val_loader, device = device)
        print(classification_report(ytrue, ypred))



