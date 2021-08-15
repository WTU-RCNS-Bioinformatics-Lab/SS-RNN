import random
import numpy as np
import torch
import numpy as np
import argparse
import os
from tqdm import tqdm
from core import model as ModelPackage
import pandas as pd



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=2, type=int, help='epoch')
    parser.add_argument('--batch_size', default=2000, type=int, help='')
    # parser.add_argument('--data_url', default='/root/kistoff2/dataset/ecg_data.csv', type=str,
    #                     help='the training and validation data path')
    parser.add_argument('--result_url', default='/root/cwj/SKIP20200308/result/', type=str,
                        help='the path to save training outputs')
    parser.add_argument('--debug', default=True, type=str2bool, help='')
    parser.add_argument('--version', default=2.0, type=float, help='')
    parser.add_argument('--cuda', default=0, type=int, help='cuda')
    parser.add_argument('--model', default='ModelF', type=arg_model)
    parser.add_argument('--multigpu', default=False, type=str2bool)
    parser.add_argument('--skip', default=2, type=int)
    parser.add_argument('--dataset', default='breast', type=str)# ecg eeg mit_eeg diabetes breast
    args = parser.parse_args()

    return args


def get_param_from_path(path):
    experiment_path_list = []
    for item in os.listdir(path):
        bbb = item.split('_')
        item_dic = {}
        pp = os.path.join(path, item)
        item_dic['path'] = pp
        experiment_path_list.append(item_dic)
        for i in bbb:
            j = i.split('=')
            item_dic[j[0]] = j[1]


    data = pd.DataFrame(experiment_path_list)

    return data

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('Yes', 'YES', 'yes', 'True', 'TRUE', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('No', 'NO', 'no', 'False', 'FALSE', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_model(m):
    if not isinstance(m, str):
        raise argparse.ArgumentTypeError('String value expected.')
    try:
        ModelClass = getattr(ModelPackage, m)
        return (ModelClass, m)
    except AttributeError:
        raise argparse.ArgumentTypeError('String value must be a name of model from pakage core model')
    else:
        raise argparse.ArgumentTypeError('Unkonw error.')


def get_result_path(args):
    result_path = os.path.join(os.path.dirname(__file__), '..', 'result')
    if args.debug is True:
        result_path = os.path.join(result_path, 'debug')
    else:
        result_path = os.path.join(result_path, 'release')

    result_path = os.path.join(result_path, str(args.version))
    _, model_name = args.model
    folder = 'epoch=' + str(args.epoch) + \
             '_batchsize='+ str(args.batch_size) + \
             '_model=' + model_name + \
             '_skip=' + str(args.skip) + \
             '_dataset=' + args.dataset

    result_path = os.path.join(result_path, folder)


    if os.path.exists(result_path) is False:
        os.makedirs(result_path)

    return result_path

def get_model_path(args):
    father_path = get_result_path(args)
    model_path = os.path.join(father_path, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path


def get_pic_path(args):
    father_path = get_result_path(args)
    pic_path = os.path.join(father_path, 'pic')
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)

    return pic_path

def save_result(save_path, train_loss_list, test_loss_list, test_y_true_list, test_y_pred_list):
    save_path = os.path.join(save_path, 'outputs')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    np.save(
        file=os.path.join(save_path, 'train_loss.npy'),
        arr=train_loss_list
    )

    np.save(
        file=os.path.join(save_path, 'test_loss.npy'),
        arr=test_loss_list
    )

    np.save(
        file=os.path.join(save_path, 'test_y_true.npy'),
        arr=test_y_true_list
    )

    np.save(
        file=os.path.join(save_path, 'test_y_pred.npy'),
        arr=test_y_pred_list
    )

def save_model(model_path, model):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, 'model.pth'))

class ProgressBar:
    def __init__(self, total, ncols, desc, position, show=True):
        self.show = show
        if self.show == True:
            self.pbar = tqdm(total=total, ncols=ncols, ascii=True, position=position)
            self.pbar.set_description(desc)

    def set_postfix(self, loss):
        if self.show == True:
            self.pbar.set_postfix(loss='{:5.2f}'.format(loss))
    def update(self, step=1): #step=1 shi epoch mei ci zeng zhang 1
        if self.show == True:
            self.pbar.update(step)

    def close(self):
        if self.show == True:
            self.pbar.close()


def show_log(device):
    show_log = device == torch.device('cuda:0') or \
               device == torch.device('cuda') or \
               device == torch.device('cuda:1') or \
               device == torch.device('cpu') or \
               device == 0

    return show_log


if __name__ == '__main__':
    print(os.path.dirname(__file__))

