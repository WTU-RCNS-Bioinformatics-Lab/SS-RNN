import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
import pandas as pd
from sklearn import preprocessing
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from sklearn import preprocessing
from utils import tool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN,SMOTETomek
from utils.diabetes_data import DiabetesDataset
from utils.breast_data import Breast


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = self.x[idx,:,:]
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


# totensor = ToTensor()
# totensor()


class MitEegDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        path = '/media/cwj/File/chb-mit-eeg-merge-mat'
        x_file_path = None
        y_file_path = None
        if dataset == 'trainset':
            x_file_path = os.path.join(path, 'train_x.pkl')
            y_file_path = os.path.join(path, 'train_y.pkl')
        else:
            x_file_path = os.path.join(path, 'test_x.pkl')
            y_file_path = os.path.join(path, 'test_y.pkl')

        x_file = open(x_file_path, 'rb')
        y_file = open(y_file_path, 'rb')
        mean = 0.22180476472313504
        std = 67.9172308441852

        x = pickle.load(x_file)
        y = pickle.load(y_file)
        # print(Counter(y))
        smote = SMOTE(random_state=21, n_jobs=2)
        x,y = smote.fit_resample(x, y)
        # print(Counter(y))
        x = (x - mean) / std
        x = x.reshape((x.shape[0], 256, 17))
        x_file.close()
        y_file.close()
        self.x = x
        self.y = y



def get_eeg_data():
    path = '/home/cwj/dataset/eeg_data.csv'
    eeg_data = pd.read_csv(path, index_col=0)
    # print(Counter(eeg_data['y']))
    le = LabelEncoder()
    classes = [1,2,3,4,5]
    le.fit(classes)
    eeg_data['y'] = le.transform(eeg_data['y'])

    train_x,test_x,train_y,test_y = train_test_split(
        eeg_data.iloc[:,0:-1].values, eeg_data.iloc[:,-1].values,
        test_size=0.2, random_state=55
    )

    mean = train_x.mean()
    std = train_x.std()

    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    train_x = train_x.reshape((-1, 178, 1))
    test_x = test_x.reshape((-1, 178, 1))

    # print(eeg_data.head())


    return train_x,train_y,test_x,test_y


def count():
    path = os.path.join(os.path.dirname(__file__), '..', 'dataset/chb-mit-eeg-merge-mat')

    train_x_file = open(os.path.join(path, 'train_y.pkl'), 'rb')
    test_x_file = open(os.path.join(path, 'test_y.pkl'), 'rb')

    train_x = pickle.load(train_x_file)
    test_x = pickle.load(test_x_file)
    from collections import Counter
    print(train_x.shape[0] + test_x.shape[0])
    print(Counter(test_x))


def get_ecg_data():
    path = '/home/cwj/dataset/ecg_data.csv'
    ecg_data = pd.read_csv(path, index_col=0)

    ecg_data = ecg_data.reset_index(drop=True)
    permutation = np.random.permutation(len(ecg_data))
    ecg_data = ecg_data.iloc[permutation]

    le = LabelEncoder()
    classes = ['N', 'Q', 'V', 'S', 'F']
    le.fit(classes)
    ecg_data['label'] = le.transform(ecg_data['label'])


    train_set, test_set = train_test_split(ecg_data, random_state=73, stratify=ecg_data['label'], test_size=0.2)


    train_x = train_set.drop(columns=['label']).values
    train_y = train_set['label'].values
    print(Counter(train_y))
    smote = SMOTE(random_state=21, n_jobs=8)
    train_x, train_y = smote.fit_resample(train_x, train_y)
    print(Counter(train_y))

    train_x = train_x.reshape((-1, 280, 1))

    # train_y = train_y.reshape((-1,1))

    test_x = test_set.drop(columns=['label']).values
    test_x = test_x.reshape((-1, 280, 1))
    test_y = test_set['label'].values
    print(Counter(test_y))


    return train_x,train_y,test_x,test_y

def get_dataloader(x, y, batch_size=100, shuffle=True, drop_last=False, num_workers=4):

    dataset = Dataset(x, y, transform=transforms.Compose([ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                        num_workers=num_workers, pin_memory=True, shuffle=shuffle)

    return loader

def get_dataset(args):
    train_dataset,test_dataset = None,None
    print('Loading data...')
    if args.dataset == 'ecg':
        train_x, train_y, test_x, test_y = get_ecg_data()
        train_dataset = Dataset(train_x, train_y, transform=transforms.Compose([ToTensor()]))
        test_dataset = Dataset(test_x, test_y, transform=transforms.Compose([ToTensor()]))
    elif args.dataset == 'eeg':
        train_x, train_y, test_x, test_y = get_eeg_data()
        train_dataset = Dataset(train_x, train_y, transform=transforms.Compose([ToTensor()]))
        test_dataset = Dataset(test_x, test_y, transform=transforms.Compose([ToTensor()]))
    elif args.dataset == 'miteeg':
        train_dataset = MitEegDataset(dataset='trainset', transform=transforms.Compose([ToTensor()]))
        test_dataset = MitEegDataset(dataset='testset', transform=transforms.Compose([ToTensor()]))
    elif args.dataset == 'diabetes':
        train_dataset = DiabetesDataset(trainset=True, transform=transforms.Compose([ToTensor()]))
        test_dataset = DiabetesDataset(trainset=False, transform=transforms.Compose([ToTensor()]))
    elif args.dataset == 'breast':
        train_dataset = Breast(trainset=True, transform=transforms.Compose([ToTensor()]))
        test_dataset = Breast(trainset=False, transform=transforms.Compose([ToTensor()]))
    return train_dataset,test_dataset


def get_dataloader_parallel(rank, world_size, x, y, batch_size, shuffle, num_workers=16):
    dataset = Dataset(x, y, transform=transforms.Compose([ToTensor()]))
    sampler = DistributedSampler(
        dataset=dataset, rank=rank, num_replicas=world_size, shuffle=shuffle
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            sampler=sampler,
                            num_workers=2,
                            pin_memory=True)
    return dataloader


def prepare_data(data, device):
    inputs, labels = data

    inputs = inputs.to(device)
    labels = labels.to(device)
    return inputs, labels

if __name__ == '__main__':
    # path = '/root/kistoff2/dataset/chb-mit-eeg-merge-mat'
    # train_x_file = open(os.path.join(path, 'train_x.pkl'), 'rb')
    # train_y_file = open(os.path.join(path, 'train_y.pkl'), 'rb')
    # x = pickle.load(train_x_file)
    # y = pickle.load(train_y_file)
    #
    # # get_eeg_data()
    #
    # ##
    # from collections import Counter
    # print(Counter(y))

    # eeg_train_dataset = MitEegDataset(dataset='trainset', transform=transforms.Compose([ToTensor()]))
    count()
