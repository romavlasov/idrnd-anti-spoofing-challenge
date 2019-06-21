import os
import time
import datetime
import argparse
import yaml

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

import metrics.classification as metrics

import models
import losses

from data.datasets import idrnd
from data.transform import Transforms

from utils.handlers import AverageMeter
from utils.handlers import MetaData

from utils.storage import save_weights
from utils.storage import load_weights


cv2.setNumThreads(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):   
    model = getattr(models, config['encoder'])(device=device,
                                                 out_features=config['out_features'],
                                                 pretrained=config['pretrained'])
    
    start_epoch = 0
    if config['snapshot']['use']:
        load_weights(model, config['prefix'], 'model', config['snapshot']['epoch'])
        start_epoch = config['snapshot']['epoch']
    
    if torch.cuda.is_available() and config['parallel']:
        model = nn.DataParallel(model)
        
    criterion = getattr(losses, config['loss'])()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        factor=0.5,
                                                        patience=2,
                                                        min_lr=1e-6)
    
    train_df, test_df = idrnd.load_dataset(config['train']['folder'], test_size=0.05)
    
    train_loader = DataLoader(idrnd.TrainAntispoofDataset(
        train_df, Transforms(input_size=config['input_size'], train=True)),
                              batch_size=config['batch_size'],
                              num_workers=config['num_workers'],
                              shuffle=True)
    
    test_loader = DataLoader(idrnd.TrainAntispoofDataset(
        test_df, Transforms(input_size=config['input_size'], train=False), config['tta']),
                             batch_size=config['batch_size'],
                             num_workers=config['num_workers'],
                             shuffle=False)
    
    thresholds = np.linspace(0.001, 0.6, num=config['thresholds'])
    best_threshold = 0.5
    best_epoch = 0
    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(start_epoch, config['num_epochs']):
        if epoch == 0:
            opt = optim.Adam(model.module.linear_params(), lr=config['learning_rate'])
            train(train_loader, model, criterion, opt, epoch, config)
        else:
            train(train_loader, model, criterion, optimizer, epoch, config)
        
        loss, accuracy, score = validation(test_loader, model, criterion, thresholds)
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(' Validation:'
              ' Time: {}'
              ' Epoch: {}'
              ' Loss: {:.4f}'.format(current_time, epoch + 1, loss))
            
        best_index = np.argmin(score)
        print(' Threshold: {:.4f}'
              ' Accuracy: {:.5f}'
              ' Score: {:.5f}'.format(thresholds[best_index], accuracy[best_index], score[best_index]))

        if best_loss > loss:
            best_threshold = thresholds[best_index]
            best_score = score[best_index]
            best_loss = loss
            best_epoch = epoch + 1
            save_weights(model, config['prefix'], 'model', 'best', config['parallel'])
        
        if epoch != 0:
            lr_scheduler.step(loss)
        
        save_weights(model, config['prefix'], 'model', epoch + 1, config['parallel'])
        
    print(' Best threshold: {:.4f}'
          ' Best score: {:.5f}'
          ' Best loss: {:.4f}'
          ' Best epoch: {}'.format(best_threshold, best_score, best_loss, best_epoch))
        

def train(data_loader, model, criterion, optimizer, epoch, config):
    model.train()
    
    loss_handler = AverageMeter()
    accuracy_handler = AverageMeter()
    score_handler = AverageMeter()
    
    tq = tqdm(total=len(data_loader) * config['batch_size'])
    tq.set_description('Epoch {}, lr {:.2e}'.format(epoch + 1, 
                                                    get_learning_rate(optimizer)))
    
    for i, (image, target) in enumerate(data_loader):
        image = image.to(device)
        target = target.to(device)

        output = model(image).view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        batch_size = image.size(0)
        
        if (i + 1) % config['step'] == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        pred = torch.sigmoid(output) > 0.5
        target = target > 0.5
        
        accuracy = metrics.accuracy(pred, target)
        score = metrics.min_c(pred, target)

        loss_handler.update(loss)
        accuracy_handler.update(accuracy)
        score_handler.update(score)
        
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_lr = get_learning_rate(optimizer)
        
        tq.update(batch_size)
        tq.set_postfix(loss='{:.4f}'.format(loss_handler.avg),
                       accuracy='{:.5f}'.format(accuracy_handler.avg),
                       score='{:.5f}'.format(score_handler.avg))
    tq.close()


def validation(data_loader, model, criterion, thresholds):
    model.eval()
    
    loss_handler = AverageMeter()
    accuracy_handler = [AverageMeter() for _ in thresholds]
    score_handler = [AverageMeter() for _ in thresholds]
    
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device)
            target = target.to(device)
            
            output = model(image).view(-1)
            
            loss = criterion(output, target)
            loss_handler.update(loss)
    
            target = target.byte()
            for i, threshold in enumerate(thresholds):
                pred = torch.sigmoid(output) > threshold
            
                accuracy = metrics.accuracy(pred, target)
                score = metrics.min_c(pred, target)

                accuracy_handler[i].update(accuracy)
                score_handler[i].update(score)

    return (loss_handler.avg, 
            [i.avg for i in accuracy_handler], 
            [i.avg for i in score_handler])


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train code')
    parser.add_argument('--config', required=True, help='configuration file')
    args = parser.parse_args()
    
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    main(config)
