##
import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
##
from torch.utils.data import DataLoader
from utils import data
from collections import Counter
from torchvision import transforms
from core import model

def preprocess():
    np.random.seed(75)
    raw_data_path = '/root/cwj/SKIP20200308/dataset/diabetes/diabetes_data_upload.csv'
    df = pd.read_csv(raw_data_path)
    permutation = np.random.permutation(len(df))
    df = df.iloc[permutation]
    df = df.reset_index(drop=True)
    le = LabelEncoder()
    le.fit(['Male', 'Female'])
    df['Gender'] = le.transform(df['Gender'])
    le.fit(['Yes', 'No'])

    df['Polyuria'] = le.transform(df['Polyuria'])
    df['Polydipsia'] = le.transform(df['Polydipsia'])
    df['sudden weight loss'] = le.transform(df['sudden weight loss'])
    df['weakness'] = le.transform(df['weakness'])
    df['Polyphagia'] = le.transform(df['Polyphagia'])
    df['Genital thrush'] = le.transform(df['Genital thrush'])
    df['visual blurring'] = le.transform(df['visual blurring'])
    df['Itching'] = le.transform(df['Itching'])
    df['Irritability'] = le.transform(df['Irritability'])
    df['delayed healing'] = le.transform(df['delayed healing'])
    df['partial paresis'] = le.transform(df['partial paresis'])
    df['muscle stiffness'] = le.transform(df['muscle stiffness'])
    df['Alopecia'] = le.transform(df['Alopecia'])
    df['Obesity'] = le.transform(df['Obesity'])
    le.fit(['Positive', 'Negative'])
    df['class'] = le.transform(df['class'])

    ##
    # data = df.values

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=23)

    ##
    mean = train_data['Age'].mean()
    std = train_data['Age'].std()
    ##
    train_data = train_data.copy()
    train_data.loc[:, 'Age'] = (train_data.loc[:, 'Age'] - mean) / std

    test_data = test_data.copy()
    test_data.loc[:, 'Age'] = (test_data.loc[:, 'Age'] - mean) / std

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    save_path = '/root/cwj/SKIP20200308/dataset/diabetes/'
    train_data.to_csv(os.path.join(save_path, 'train_data.csv'), index_label=False)
    test_data.to_csv(os.path.join(save_path, 'test_data.csv'), index_label=False)




class DiabetesDataset(torch.utils.data.Dataset):
    def __init__(self, trainset=True, transform=None):
        data_path = '/home/cwj/dataset/diabetes'
        if trainset==True:
            data_path = os.path.join(data_path, 'train_data.csv')
        else:
            data_path = os.path.join(data_path, 'test_data.csv')
        data = pd.read_csv(data_path).values
        self.x = data[:,:-1]
        self.y = data[:,-1]

        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.x[idx,:]
        labels = self.y[idx]
        data = (inputs, labels)

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        inputs, labels = data

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs, labels


def count():
    raw_data_path = os.path.join(os.path.dirname(__file__),'..','dataset/diabetes','diabetes_data_upload.csv')
    df = pd.read_csv(raw_data_path)
    print(df.shape)

if __name__ == '__main__':
    # train_dataset = DiabetesDataset(trainset=True, transform=transforms.Compose([data.ToTensor()]))
    # test_dataset = DiabetesDataset(trainset=False, transform=transforms.Compose([data.ToTensor()]))
    #
    # train_data_loader = DataLoader(train_dataset, batch_size=10, drop_last=False,
    #                                num_workers=2, pin_memory=True, shuffle=True)
    #
    # test_data_loader = DataLoader(test_dataset, batch_size=10, drop_last=False,
    #                               num_workers=2, pin_memory=True, shuffle=True)
    #
    # model = model.Normal(in_features=train_dataset.x.shape[-1], skip=3, out_features=len(Counter(train_dataset.y)))
    # for data in train_data_loader:
    #     input, label = data
    #     output = model(input)
    #     print(input.shape, label.shape, output.shape)
    #     print(input.dtype, label.dtype, output.dtype)
    #
    #     break

    count()