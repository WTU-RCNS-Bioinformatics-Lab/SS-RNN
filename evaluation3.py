import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.tool import get_model_path,get_result_path,get_args,get_param_from_path
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
import seaborn as sns


def test1():
    # compute importance of each model
    parent_path = os.path.dirname(__file__)
    dataset = 'miteeg'
    path = os.path.join(parent_path, 'result/release', '4.0')

    params = get_param_from_path(path)

    params_group = params.sort_values(['model', 'skip']).groupby('model')

    accuracy = {}

    for name, group in params_group:
        accuracy_list = []

        for i, row in group.iterrows():
            sub_path = row['path']
            y_pred = np.load(os.path.join(sub_path, 'outputs/test_y_pred.npy'))
            y_pred = F.softmax(torch.tensor(y_pred), dim=2)
            y_pred = y_pred.argmax(dim=2).numpy()
            y_true = np.load(os.path.join(sub_path, 'outputs/test_y_true.npy'))
            model_accuracy = []
            for i in range(len(y_true)):
                tmp_y_true = y_true[i]
                tmp_y_pred = y_pred[i]
                tmp_report = classification_report(tmp_y_true, tmp_y_pred, output_dict=True)
                model_accuracy.append(tmp_report['accuracy'])

            model_accuracy = np.array(model_accuracy)
            accuracy_list.append(model_accuracy.mean())
            # print(y_true.shape, y_pred.shape)
        accuracy[name] = np.array(accuracy_list).mean()
        accuracy_list.clear()

        print(accuracy)

    ##
    acc = accuracy.values()
    acc = np.array(list(acc))
    save_path = os.path.join(parent_path, 'avg_acc', dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'acc.npy'), acc)

def plot_():
    parent_path = os.path.dirname(__file__)
    path = os.path.join(parent_path, 'avg_acc')

    acc_list = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        acc = np.load(os.path.join(file_path, 'acc.npy'))
        acc = acc.tolist()
        acc_list.append(acc)

    accuracy = np.array(acc_list)

    avg_accuracy = accuracy.mean(axis=0)

    acc_gru = []
    acc_bilstm = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        acc_gru_ = np.load(os.path.join(file_path, 'acc_gru.npy'))
        acc_bilstm_ = np.load(os.path.join(file_path, 'acc_bilstm.npy'))
        acc_bilstm.append(acc_bilstm_.mean())
        acc_gru.append(acc_gru_.mean())

    acc_bilstm = np.array(acc_bilstm).mean()
    acc_gru = np.array(acc_gru).mean()
    print(acc_gru)
    print(acc_bilstm)


    avg_accuracy = np.insert(avg_accuracy, 0, acc_bilstm)
    avg_accuracy = np.insert(avg_accuracy, 0, acc_gru)
    ##
    plt.ylabel('Average accuracy', fontsize=15)
    sns.set_theme(style="whitegrid")
    x = ['GRU', 'BiLSTM','LSTM', 'SkipA', 'SkipB', 'SkipC', 'SkipD', 'SkipE', 'SkipF']
    y = avg_accuracy
    ax = sns.barplot(x=x, y=y)

    for i in range(len(x)):
        ax.text(i, y[i], '{:1.3f}'.format(y[i]), ha='center')

    plt.savefig(os.path.join(parent_path, 'pic/avg_acc.jpg'), dpi=600)
    plt.show()


    # sns.set_theme(style="whitegrid")
    #
    # tips = sns.load_dataset("tips")
    #
    # ax = sns.barplot(x="day", y="total_bill", data=tips)
    # plt.legend()

def compute_acc():
    root_path = '/home/cwj/PycharmProjects/SKIP-Compare/result/release/1.0/epoch=50_batchsize=2000_model=NormalBiLSTM_skip=2_dataset=diabetes'
    dataset = 'diabetes'
    model_name = 'bilstm'
    y_pred = np.load(os.path.join(root_path, 'outputs/test_y_pred.npy'))
    y_pred = F.softmax(torch.tensor(y_pred), dim=2)
    y_true = np.load(os.path.join(root_path, 'outputs/test_y_true.npy'))
    y_pred = y_pred.argmax(dim=2).numpy()

    report = []
    acc_list = []
    for i in range(len(y_true)):
        tmp_y_true = y_true[i]
        tmp_y_pred = y_pred[i]
        tmp_report = classification_report(tmp_y_true, tmp_y_pred, output_dict=True)
        report.append(tmp_report)
        acc_list.append(tmp_report['accuracy'])


    accuracy = np.array(acc_list)
    parent_path = os.path.dirname(__file__)
    save_path = os.path.join(parent_path, 'avg_acc', dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'acc_'+model_name+'.npy'), accuracy)



if __name__ == '__main__':
    plot_()
    #test1()
    #compute_acc()