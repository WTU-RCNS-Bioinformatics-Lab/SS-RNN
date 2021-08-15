import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
import torch
from torchvision import transforms
from utils import data
from core import model
from collections import Counter
from torch.utils.data import DataLoader
import torch.nn as nn


class Breast(torch.utils.data.Dataset):
    def __init__(self, trainset=True, transform=None):
        self.transform = transform
        # parent_path = os.path.abspath(os.path.dirname(__file__))
        dataset_path = "/home/cwj/dataset/breast"
        x_path = dataset_path
        y_path = dataset_path
        if trainset==True:
            x_path = os.path.join(x_path, 'train_x.npy')
            y_path = os.path.join(y_path, 'train_y.npy')
        else:
            x_path = os.path.join(x_path, 'test_x.npy')
            y_path = os.path.join(y_path, 'test_y.npy')

        x = np.load(x_path)
        y = np.load(y_path)

        x = x.astype(np.float32)
        y = y.astype(np.long)

        print(x.dtype, y.dtype)
        self.x = x
        self.y = y



    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.x[idx,:]
        labels = self.y[idx]
        data = (inputs, labels)

        if self.transform:
            data = self.transform(data)

        return data

def preprocess():
    parent_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(parent_path, '..', 'dataset', 'breast')
    df = pd.read_csv(os.path.join(dataset_path, 'BreastCancer.csv'))
    le = LabelEncoder()
    le.fit(df['Classification'])
    df['Classification']  = le.transform(df['Classification'])
    print(Counter(df['Classification']))
    # print(df.head())
    #
    train_set, test_set = train_test_split(df, stratify=df['Classification'], random_state=42)
    train_set = train_set.values
    test_set = test_set.values

    train_x = train_set[:,:-1]
    train_y = train_set[:,-1]
    test_x = test_set[:, :-1]
    test_y = test_set[:, -1]

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(train_x)
    train_x = min_max_scaler.transform(train_x)
    test_x = min_max_scaler.transform(test_x)

    np.save(os.path.join(dataset_path, 'train_x.npy'), train_x)
    np.save(os.path.join(dataset_path, 'train_y.npy'), train_y)
    np.save(os.path.join(dataset_path, 'test_x.npy'), test_x)
    np.save(os.path.join(dataset_path, 'test_y.npy'), test_y)

def count():
    parent_path = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(parent_path, '..', 'dataset', 'breast')
    df = pd.read_csv(os.path.join(dataset_path, 'BreastCancer.csv'))
    print(df.shape)

    # print(train_set.shape, test_set.shape)

if __name__ == '__main__':
    # preprocess()
    # train_dataset = Breast(trainset=True, transform=transforms.Compose([data.ToTensor()]))
    # test_dataset = Breast(trainset=False, transform=transforms.Compose([data.ToTensor()]))
    #
    # train_data_loader = DataLoader(train_dataset, batch_size=10, drop_last=False,
    #                                num_workers=2, pin_memory=True, shuffle=True)
    #
    # test_data_loader = DataLoader(test_dataset, batch_size=10, drop_last=False,
    #                               num_workers=2, pin_memory=True, shuffle=True)
    #
    # model = model.Normal(in_features=train_dataset.x.shape[-1], skip=3, out_features=len(Counter(train_dataset.y)))
    # criterion = nn.CrossEntropyLoss()
    # for data in train_data_loader:
    #     input,label = data
    #     output = model(input)
    #     loss = criterion(output, label)
    #     print(input.shape, label.shape, output.shape)
    #     print(input.dtype, label.dtype, output.dtype)
    #     break
    print(count())

