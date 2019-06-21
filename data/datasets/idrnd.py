import os
import json
import datetime

import numpy as np
import pandas as pd

import cv2
import torch

import torch.utils.data as data

from sklearn.model_selection import train_test_split


class TrainAntispoofDataset(data.Dataset):
    def __init__(self, df, transform=None, tta=1):
        self.df = df
        self.tta = tta
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.df.iloc[index % len(self.df)]

        image = self._load_image(item['path'])
        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(item['label']).float()

    def __len__(self):
        return len(self.df) * self.tta
    
    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image / 255.
    
    
class TestAntispoofDataset(data.Dataset):
    def __init__(self, df, transform=None, tta=4):
        self.df = df
        self.tta = tta
        self.transform = transform
    
    def __getitem__(self, index):
        item = self.df.iloc[index % len(self.df)]

        image = self._load_image(item['path'])
        if self.transform is not None:
            image = self.transform(image)

        return image, item['id'], item['frame']

    def __len__(self):
        return len(self.df) * self.tta
    
    def _load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return image / 255.
    

def load_dataset(path_data, test_size=0.1):
    path_images = []
    
    for label in ['2dmask', 'real', 'printed', 'replay']:
        videos = os.listdir(os.path.join(path_data, label))
        for video in videos:
            frames = os.listdir(os.path.join(path_data, label, video))
            for frame in frames:
                path_images.append({
                    'path': os.path.join(path_data, label, video, frame),
                    'label': int(label != 'real'),
                    'video': video})
                
    videos = list(set(x['video'] for x in path_images))
    videos_tr, videos_ts = train_test_split(videos, test_size=test_size, random_state=123)
    
    train_path_images = pd.DataFrame([x for x in path_images if x['video'] in videos_tr])
    test_path_images = pd.DataFrame([x for x in path_images if x['video'] in videos_ts])

    return train_path_images, test_path_images
    

def load_test_dataset(path_data):
    path_images = []
    
    for label in ['live', 'spoof']:
        videos = os.listdir(os.path.join(path_data, label))
        for video in videos:
            frames = os.listdir(os.path.join(path_data, label, video))
            for frame in frames:
                if frame.endswith('_120.jpg'):
                    path_images.append({
                        'path': os.path.join(path_data, label, video, frame),
                        'label': int(label != 'live'),
                        'video': video,
                        'frame': frame})
                
    return pd.DataFrame(path_images)