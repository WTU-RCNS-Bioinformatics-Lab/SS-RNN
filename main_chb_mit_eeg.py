import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data import get_dataloader,get_dataloader_parallel,get_ecg_data,get_eeg_data
from utils import tool
from core.run import train_loop,test_loop
from utils.tool import get_model_path,get_result_path,save_result,save_model
import os
import time
from tensorboardX import SummaryWriter
from collections import Counter
import pandas as pd
from torchvision import transforms
import scipy.io as scio


tool.set_seed(547)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

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
def get_mit_eeg_dataloader(x, y, batch_size=100, shuffle=True, drop_last=False, num_workers=4):

    dataset = Dataset(x, y, transform=transforms.Compose([ToTensor()]))
    loader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last,
                        num_workers=num_workers, pin_memory=True, shuffle=shuffle)

    return loader

def train(args):

    runs_path = os.path.join(get_result_path(args), 'runs')
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)
    dataset_path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'
    writer = SummaryWriter(runs_path)
    # train_data_loader = get_dataloader(
    #     x=train_x, y=train_y,
    #     batch_size=args.batch_size, shuffle=True, num_workers=8
    # )
    #
    # test_data_loader = get_dataloader(
    #     x=test_x, y=test_y,
    #     batch_size=args.batch_size, shuffle=True, num_workers=8
    # )
    # train_file_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
    #                    'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
    #                    'chb11', 'chb12', 'chb13', 'chb14', 'chb15',
    #                    'chb16', 'chb18', 'chb19', 'chb20']
    # test_file_list = ['chb21','chb22','chb23','chb24']
    train_file_list = ['chb18']
    test_file_list = ['chb18']
    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    ModelClass, model_name = args.model
    model = ModelClass(in_features=17, skip=args.skip, out_features=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=10, gamma=0.1
    )

    train_loss_list = []
    test_loss_list = []
    test_y_true_list = []
    test_y_pred_list = []

    mean = 0.22180476472313504
    deviation = 67.9172308441852

    for epoch_id in range(args.epoch):
        for dir in train_file_list:
            dir = 'chb18'
            # print('training sub file: ', dir)
            dir_path = os.path.join(dataset_path, dir)
            for filename in os.listdir(dir_path):
                filename = 'chb18_19.mat'
                print('training sub file: ', filename)
                print('loading dataset...')
                sub_file_path = os.path.join(dir_path, filename)
                print(sub_file_path)
                file = scio.loadmat(sub_file_path)
                train_x = file['inputs']
                train_x = (train_x - mean) / deviation
                train_y = file['label']
                train_y = train_y.reshape(train_y.shape[1])
                print('dataset is ready!')
                train_data_loader = get_dataloader(
                    x=train_x, y=train_y,
                    batch_size=args.batch_size, shuffle=False, num_workers=0
                )
                train_loss = train_loop(
                    model=model, optimizer=optimizer, criterion=criterion,
                    device=device, data_loader=train_data_loader,
                    epoch_id=epoch_id, epoch=args.epoch, lr_scheduler=lr_scheduler
                )
        for dir in test_file_list:
            dir_path = os.path.join(dataset_path, dir)
            for filename in os.listdir(dir_path):
                print('testing sub file: ', filename)
                print('loading dataset...')
                sub_file_path = os.path.join(dir_path, filename)
                file = scio.loadmat(sub_file_path)
                test_x = file['inputs']
                test_x = (test_x - mean) / deviation
                test_y = file['label']
                print('dataset is ready!')
                test_data_loader = get_dataloader(
                    x=test_x, y=test_y,
                    batch_size=args.batch_size, shuffle=False, num_workers=0
                )
                test_loss,y_true,y_pred = test_loop(
                    model=model, criterion=criterion, device=device, data_loader=test_data_loader,
                    epoch_id=epoch_id, epoch=args.epoch
                )
                train_loss_list.append(train_loss)
                test_loss_list.append(test_loss)
                test_y_true_list.append(y_true)
                test_y_pred_list.append(y_pred)

        # writer.add_scalar('train loss', train_loss, global_step=epoch_id)
        # writer.add_scalar('test loss', test_loss, global_step=epoch_id)

        if (epoch_id + 1) % 10 == 0:
            #saving model
            # print('Saving result...')
            model_path = get_model_path(args)
            save_model(model_path, model)
            result_path = get_result_path(args)
            save_result(result_path, train_loss_list, test_loss_list, test_y_true_list, test_y_pred_list)
            # print('Result is saved!')


if __name__ == '__main__':
    args = tool.get_args()
    print(args)
    start = time.time()
    # print('Loading data...')
    # train_x, train_y, test_x, test_y = None,None,None,None
    # if args.dataset == 'ecg':
    #     train_x, train_y, test_x, test_y  = get_ecg_data()
    # elif args.dataset == 'eeg':
    #     train_x, train_y, test_x, test_y = get_eeg_data()
    # aa = Counter(train_y)
    # print('Dataset is ready!')

    ##
    train(args)

    ##

    total_time = time.strftime('%H:%M:%S', time.localtime(time.time()-start))
    print('Total time:{}'.format(total_time))

