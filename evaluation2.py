import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.tool import get_model_path,get_result_path,get_args,get_param_from_path
import os
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from collections import Counter

dpi = 300

def plot_train_test_loss(params, pic_path, dataset):

    def _plot(params):
        fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        for (i,row),ax in zip(params.iterrows(), axs.flat):
            file_path = row['path']
            model_name = row['model']
            skip = row['skip']
            test_loss = np.load(os.path.join(file_path, 'outputs', 'test_loss.npy'))
            train_loss = np.load(os.path.join(file_path, 'outputs', 'train_loss.npy'))
            train_loss_line = ax.plot(train_loss, label='train loss', linestyle='-')
            test_loss_line = ax.plot(test_loss, label='test loss', linestyle=':')
            ax.set_xlabel('Train step')
            ax.set_ylabel('Loss')

            if model_name.startswith('Model'):
                ax.set_title('Model' + model_name[5] + '-' + skip)
            else:
                ax.set_title('Model' + model_name[4] + '-' + skip)
            ax.legend()
        fig.show()
        sub_pic_path = os.path.join(pic_path, dataset)
        if not os.path.exists(sub_pic_path):
            os.makedirs(sub_pic_path)

        fig.savefig(os.path.join(sub_pic_path, 'loss_'+model_name+'.jpg'), dpi=dpi)

    skip_a_params = params.query("model=='ModelA' or model=='SkipAModel'")
    skip_a_params = skip_a_params.sort_values(['skip'])
    _plot(skip_a_params)

    skip_b_params = params.query("model=='ModelB' or model=='SkipBModel'")
    skip_b_params = skip_b_params.sort_values(['skip'])
    _plot(skip_b_params)

    skip_c_params = params.query("model=='ModelC' or model=='SkipCModel'")
    skip_c_params = skip_c_params.sort_values(['skip'])
    _plot(skip_c_params)

    skip_d_params = params.query("model=='ModelD' or model=='SkipDModel'")
    skip_d_params = skip_d_params.sort_values(['skip'])
    _plot(skip_d_params)

    skip_e_params = params.query("model=='ModelE' or model=='SkipEModel'")
    skip_e_params = skip_e_params.sort_values(['skip'])
    _plot(skip_e_params)

    skip_e_params = params.query("model=='ModelF' or model=='SkipFModel'")
    skip_e_params = skip_e_params.sort_values(['skip'])
    _plot(skip_e_params)





def plot_violin(data, pic_path, dataset, title, label_list):
    plt.figure(figsize=(20,12))

    widths = 3
    positions = [i for i in range(0, len(data) * widths, widths)]
    violin = plt.violinplot(
            data,widths=2,
            positions=positions, showextrema=False
            )

    for patch in violin['bodies']:
        patch.set_facecolor('#D43F3A')
        patch.set_edgecolor('black')
        patch.set_alpha(1)

    for i,d in enumerate(data):
        min_value,quantile1,median,quantile3,max_value = np.percentile(data[i], [0,25,50,75,100])
        plt.scatter(positions[i], median, color='white', zorder=4)
        plt.vlines(positions[i], quantile1, quantile3, lw=9, zorder=3, color='black')
        plt.vlines(positions[i], min_value, max_value, zorder=2, color='black')
    plt.ylabel(title[0].upper()+title[1:].lower(), fontsize=fontsize)
    plt.xticks(ticks=positions, labels=label_list, rotation=45, ha='right',rotation_mode='anchor')
    plt.tick_params(labelsize=fontsize)
    sub_pic_path = os.path.join(pic_path, dataset)
    if not os.path.exists(sub_pic_path):
        os.makedirs(sub_pic_path)
    plt.savefig(os.path.join(sub_pic_path, title.lower()+'_violin.jpg'), dpi=dpi)

    plt.show()


def get_result_data(params):
    # params = params.sort_values(['model','skip'])
    loss_list = []
    label_list = []
    report_list = []
    for i, row in params.iterrows():
        file_path = row['path']
        model_name = row['model']
        skip = row['skip']
        dataset = row['dataset']

        loss = np.load(os.path.join(file_path, 'outputs', 'test_loss.npy'))
        loss_list.append(loss)

        y_pred = np.load(os.path.join(file_path, 'outputs', 'test_y_pred.npy'))
        y_pred = F.softmax(torch.tensor(y_pred), dim=2)
        y_pred = y_pred.argmax(dim=2).numpy()
        y_true = np.load(os.path.join(file_path, 'outputs', 'test_y_true.npy'))

        # print(Counter(y_true.flatten()))
        # print(Counter(y_pred.flatten()))
        report = []
        for i in range(len(y_true)):
            tmp_y_true = y_true[i]
            tmp_y_pred = y_pred[i]
            tmp_report = classification_report(tmp_y_true, tmp_y_pred, output_dict=True)
            report.append(tmp_report)

        report_list.append(report)
        label = ''
        if model_name == 'ANormal' or model_name == 'NormalModel':
            label = label + 'Original'
        else:
            if model_name.startswith('Model'):
                label = label  + 'Skip'+  model_name[5] + '-' + skip
            else:
                label = label + 'Skip' + model_name[4] + '-' + skip

        label_list.append(label)

    return loss_list,label_list,report_list


def plot_eval(report_list, pic_path, dataset):
    acc_list = []
    for report, label in zip(report_list, label_list):
        acc = []

        for r in report:
            tmp_acc = r['accuracy']
            acc.append(tmp_acc)

        acc_list.append(acc)

    plot_violin(acc_list, pic_path, dataset, 'acc', label_list)

def plot_result(result_path, pic_path):


    params = get_param_from_path(result_path)

    ecg_params = params.query("dataset=='ecg'")
    # loss_list, label_list = get_result_data(ecg_params)
    # plot_violin(loss_list, pic_path, 'ecg', 'loss', label_list)
    # plot_train_test_loss(ecg_params, pic_path,'ecg')

    eeg_params = params.query("dataset=='eeg'")
    eeg_params = eeg_params.sort_values(['model','skip'])
    loss_list, label_list, report_list = get_result_data(eeg_params)
    plot_violin(loss_list, pic_path, 'eeg', 'loss', label_list)
    plot_train_test_loss(eeg_params, pic_path,'eeg')
    plot_eval(report_list, pic_path, 'eeg')

    mit_eeg_params = params.query("dataset=='miteeg'")


if __name__ == '__main__':
    fontsize = 20
    parent_path = os.path.dirname(__file__)
    pic_path = os.path.join(parent_path, 'pic')

    dataset_name = ['ecg', 'eeg', 'miteeg', 'diabetes', 'breast']
    local_path = ['2.0', '3.0', '4.0', '6.0', '7.0']



    for dataset, path in zip(dataset_name, local_path):
        result_path = os.path.join(parent_path, 'result/release', path)

        params = get_param_from_path(result_path)

        eeg_params = params.query("dataset=='{}'".format(dataset))
        eeg_params = eeg_params.sort_values(['model', 'skip'])
        loss_list, label_list, report_list = get_result_data(eeg_params)
        plot_violin(loss_list, pic_path, dataset, 'loss', label_list)
        plot_train_test_loss(eeg_params, pic_path, dataset)
        plot_eval(report_list, pic_path, dataset)




