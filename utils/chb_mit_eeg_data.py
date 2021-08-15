##
import wfdb
import os
from IPython.display import display
import scipy.io as scio
from tqdm import tqdm
import pandas as pd
import numpy as np
import pyedflib
import re
import sys
import pickle


##
def extract():
    path = '/root/kistoff2/dataset/chb-mit-scalp-eeg-database-1.0.0/'

    def get_record_list(path):
        record_path = os.path.join(path, 'RECORDS')
        with open(record_path, encoding='UTF-8') as f:
            # read all the document into a list of strings (each line a new string)
            content = f.readlines()

        return content

    records_list = get_record_list(path)

    ##
    persons_list = sorted(list(set([record.split('/')[0] for record in records_list])))

    ##
    def get_content(person):
        filename = os.path.join(path, person, person+'-summary.txt')
        with open(filename, encoding='UTF-8') as f:
            content = f.readlines()

        return content

    # print(get_content(persons_list[0]))
    ##


    part_info_dict = {}


    def info_dict(content):
        line_nos = len(content)
        line_no = 1

        channels = []
        file_name = []
        file_info_dict = {}

        for line in content:

            # if there is Channel in the line...
            if re.findall('Channel \d+', line):
                # split the line into channel number and channel reference
                channel = line.split(': ')
                # get the channel reference and remove any new lines
                channel = channel[-1].replace("\n", "")
                # put into the channel list
                channels.append(channel)

            # if the line is the file name
            elif re.findall('File Name', line):
                # if there is already a file_name
                if file_name:
                    # flush the current file info to it
                    part_info_dict[file_name] = file_info_dict

                # get the file name
                file_name = re.findall('\w+\d+_\d+|\w+\d+\w+_\d+', line)[0]

                file_info_dict = {}
                # put the channel list in the file info dict and remove duplicates
                file_info_dict['Channels'] = list(set(channels))
                # reset the rest of the options
                file_info_dict['Start Time'] = ''
                file_info_dict['End Time'] = ''
                file_info_dict['Seizures Window'] = []

            # if the line is about the file start time
            elif re.findall('File Start Time', line):
                # get the start time
                file_info_dict['Start Time'] = re.findall('\d+:\d+:\d+', line)[0]

            # if the line is about the file end time
            elif re.findall('File End Time', line):
                # get the start time
                file_info_dict['End Time'] = re.findall('\d+:\d+:\d+', line)[0]

            elif re.findall('Seizure Start Time|Seizure End Time|Seizure \d+ Start Time|Seizure \d+ End Time', line):
                file_info_dict['Seizures Window'].append(int(re.findall('\d+', line)[-1]))

            # if last line in the list...
            if line_no == line_nos:
                # flush the file info to it
                part_info_dict[file_name] = file_info_dict

            line_no += 1


    for persion in persons_list:
        content = get_content(persion)
        info_dict(content)

    ##
    display(part_info_dict['chb01_18'])

    ##


    all_channels = []
    pbar = tqdm(total=len(part_info_dict))
    for key in part_info_dict:
        pbar.update()
        filename = os.path.join(
            path, key[:5], key + '.edf'
        )
        # print(filename)
        f = pyedflib.EdfReader(filename)
        channel_names = f.getSignalLabels()
        all_channels.extend(channel_names)

        # try:
        #     ccc = channel_names.index('T8-P8')
        # except:
        #     t8_p8_count = t8_p8_count + 1




    pbar.close()
        # if raw_data is None:
        #     continue

    # turn the list into a pandas series
    all_channels = pd.Series(all_channels)

    # count how many times the channels appear in each participant
    channel_counts = all_channels.value_counts()
    print(channel_counts)

    ##
    threshold = len(part_info_dict.keys())
    channel_keeps = list(channel_counts[channel_counts == 672].index)
    # channel_keeps = list(channel_keeps[channel_keeps <=1316].index)
    # print(channel_keeps)

    ##
    EXAMPLE_FILE = records_list[17].rstrip('\n')
    EXAMPLE_ID = EXAMPLE_FILE.split('/')[1].split('.')[0]

    ##

    def data_load(file, selected_channels=[]):
        try:

            f = pyedflib.EdfReader(file)

            # get a list of the EEG channels
            if len(selected_channels) == 0:
                selected_channels = f.getSignalLabels()

            # get the names of the signals
            channel_names = f.getSignalLabels()
            # get the sampling frequencies of each signal
            channel_freq = f.getSampleFrequencies()

            # make an empty file of 0's
            sigbufs = np.zeros((f.getNSamples()[0], len(selected_channels)))
            # for each of the channels in the selected channels
            for i, channel in enumerate(selected_channels):
                # add the channel data into the array
                sigbufs[:, i] = f.readSignal(channel_names.index(channel))

            # turn to a pandas df and save a little space
            df = pd.DataFrame(sigbufs, columns=selected_channels).astype('float32')

            # get equally increasing numbers upto the length of the data depending
            # on the length of the data divided by the sampling frequency
            index_increase = np.linspace(0,
                                         len(df) / channel_freq[0],
                                         len(df), endpoint=False)

            # round these to the lowest nearest decimal to get the seconds
            seconds = np.floor(index_increase).astype('uint16')

            # make a column the timestamp
            df['Time'] = seconds

            # make the time stamp the index
            df = df.set_index('Time')

            # name the columns as channel
            df.columns.name = 'Channel'

            return df, channel_freq[0]

        except:
            OSError
            return None, None


    ##

    save_path = '/root/kistoff2/dataset/chb-mit-eeg-mat'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pbar = tqdm(total=len(part_info_dict))
    inputs_list = []
    labels_list = []


    file_count = 0
    for key in part_info_dict:
        pbar.update()
        ecg_file_path = os.path.join(
            path, key.split('_')[0], key + '.edf'
        )
        seizures_window = part_info_dict[key]['Seizures Window']
        raw_data, freq = data_load(ecg_file_path, channel_keeps)
        if raw_data is None:
            continue
        label = np.zeros((len(raw_data) // freq), dtype=np.long)

        if seizures_window:
            for i in range(0,len(seizures_window),2):
                label[seizures_window[i]:seizures_window[i+1] + 1] = 1


        raw_data = raw_data.values
        raw_data = raw_data.reshape((-1, 256, 17))


        dataset = {'inputs':raw_data,'label':label}
        ecg_file_save_path = os.path.join(save_path, key.split('_')[0])
        if not os.path.exists(ecg_file_save_path):
            os.makedirs(ecg_file_save_path)
        ecg_file_save_path = os.path.join(ecg_file_save_path, key + '.mat')
        scio.savemat(ecg_file_save_path, dataset)

        file_count = file_count + 1

    pbar.close()

    print('file count:',file_count)


##

def merge():
    path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'

    file_count = 0
    for file_path in os.listdir(path):
        sub_file_path = os.path.join(path, file_path)
        file_count = file_count + len(os.listdir(sub_file_path))

    save_path = '/root/kistoff2/dataset/chb-mit-eeg-merge-mat/'
    print('merging...')
    pbar = tqdm(total=file_count)

    for file_path in os.listdir(path):
        sub_file_path = os.path.join(path, file_path)
        sub_save_path = os.path.join(save_path, file_path)
        tmp_inputs_list = []
        tmp_labels_list = []
        for p in os.listdir(sub_file_path):
            pbar.update()

            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            labels = file['label']
            inputs = inputs.reshape((inputs.shape[0], -1))
            labels = labels.flatten()

            inputs = pd.DataFrame(inputs)
            labels = pd.Series(labels)

            tmp_inputs_list.append(inputs)
            tmp_labels_list.append(labels)

            if len(tmp_inputs_list) > 10:
                if not os.path.exists(sub_save_path):
                    os.makedirs(sub_save_path)
                inputs_ = pd.concat(tmp_inputs_list)
                tmp_inputs_list.clear()
                labels_ = pd.concat(tmp_labels_list)
                tmp_labels_list.clear()
                inputs_.to_csv(os.path.join(sub_save_path, 'inputs.csv'), mode='a', header=False)
                labels_.to_csv(os.path.join(sub_save_path, 'labels.csv'), mode='a', header=False)

        if len(tmp_inputs_list) > 0:
            if not os.path.exists(sub_save_path):
                os.makedirs(sub_save_path)
            inputs_ = pd.concat(tmp_inputs_list)
            tmp_inputs_list.clear()
            labels_ = pd.concat(tmp_labels_list)
            tmp_labels_list.clear()
            inputs_.to_csv(os.path.join(sub_save_path, 'inputs.csv'), mode='a', header=False)
            labels_.to_csv(os.path.join(sub_save_path, 'labels.csv'), mode='a', header=False)


    pbar.close()



def merge_train_set():
    path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'
    train_file_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
                       'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
                       'chb11', 'chb12', 'chb13', 'chb14', 'chb15',
                       'chb16', 'chb18', 'chb19', 'chb20']
    save_path = '/root/kistoff2/dataset/chb-mit-eeg-merge-mat/'
    file_count = 0
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        file_count = file_count + len(os.listdir(sub_file_path))

    pbar = tqdm(total=file_count)
    tmp_inputs_list = []
    tmp_labels_list = []
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        for p in os.listdir(sub_file_path):
            pbar.update()

            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            labels = file['label']
            inputs = inputs.reshape((inputs.shape[0], -1))
            labels = labels.flatten()

            # inputs = pd.DataFrame(inputs)
            # labels = pd.Series(labels)
            tmp_inputs_list.append(inputs)
            tmp_labels_list.append(labels)

            if len(tmp_inputs_list) > 50:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                inputs_ = np.concatenate(tmp_inputs_list, axis=0)
                tmp_inputs_list.clear()
                labels_ = np.concatenate(tmp_labels_list, axis=0)
                tmp_labels_list.clear()
                train_x_file = open(os.path.join(save_path, 'train_x.pkl'), 'ab')
                train_y_file = open(os.path.join(save_path, 'train_y.pkl'), 'ab')
                pickle.dump(inputs_, train_x_file)
                pickle.dump(labels_, train_y_file)
                train_x_file.close()
                train_y_file.close()

    if len(tmp_inputs_list) > 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        inputs_ = np.concatenate(tmp_inputs_list, axis=0)
        tmp_inputs_list.clear()
        labels_ = np.concatenate(tmp_labels_list, axis=0)
        tmp_labels_list.clear()
        train_x_file = open(os.path.join(save_path, 'train_x.pkl'), 'ab')
        train_y_file = open(os.path.join(save_path, 'train_y.pkl'), 'ab')
        pickle.dump(inputs_, train_x_file)
        pickle.dump(labels_, train_y_file)
        train_x_file.close()
        train_y_file.close()


    pbar.close()

def merge_test_set():
    path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'
    train_file_list = ['chb21', 'chb22', 'chb23', 'chb24']
    save_path = '/root/kistoff2/dataset/chb-mit-eeg-merge-mat/'
    file_count = 0
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        file_count = file_count + len(os.listdir(sub_file_path))

    pbar = tqdm(total=file_count)
    tmp_inputs_list = []
    tmp_labels_list = []
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        for p in os.listdir(sub_file_path):
            pbar.update()

            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            labels = file['label']
            inputs = inputs.reshape((inputs.shape[0], -1))
            labels = labels.flatten()

            # inputs = pd.DataFrame(inputs)
            # labels = pd.Series(labels)
            tmp_inputs_list.append(inputs)
            tmp_labels_list.append(labels)

            if len(tmp_inputs_list) > 50:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                inputs_ = np.concatenate(tmp_inputs_list, axis=0)
                tmp_inputs_list.clear()
                labels_ = np.concatenate(tmp_labels_list, axis=0)
                tmp_labels_list.clear()
                train_x_file = open(os.path.join(save_path, 'test_x.pkl'), 'ab')
                train_y_file = open(os.path.join(save_path, 'test_y.pkl'), 'ab')
                pickle.dump(inputs_, train_x_file)
                pickle.dump(labels_, train_y_file)
                train_x_file.close()
                train_y_file.close()

    if len(tmp_inputs_list) > 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        inputs_ = np.concatenate(tmp_inputs_list, axis=0)
        tmp_inputs_list.clear()
        labels_ = np.concatenate(tmp_labels_list, axis=0)
        tmp_labels_list.clear()
        train_x_file = open(os.path.join(save_path, 'test_x.pkl'), 'ab')
        train_y_file = open(os.path.join(save_path, 'test_y.pkl'), 'ab')
        pickle.dump(inputs_, train_x_file)
        pickle.dump(labels_, train_y_file)
        train_x_file.close()
        train_y_file.close()


    pbar.close()


def compute_train_set_mean_deviation():
    print('calculating mean...')
    train_file_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05',
                       'chb06', 'chb07', 'chb08', 'chb09', 'chb10',
                       'chb11', 'chb12', 'chb13', 'chb14', 'chb15',
                       'chb16', 'chb18', 'chb19', 'chb20']

    path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'
    sum = 0
    num = 0
    train_count = 0
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        train_count = train_count + len(os.listdir(sub_file_path))
    pbar = tqdm(total=train_count)
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)

        for p in os.listdir(sub_file_path):
            pbar.update()
            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            inputs = inputs.flatten()
            sum = sum + inputs.sum()
            num = num + len(inputs)

    pbar.close()

    mean = sum / num
    print('calculating deviation')

    sum = 0
    num = 0
    pbar = tqdm(total=train_count)
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)

        for p in os.listdir(sub_file_path):
            pbar.update()
            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            inputs = inputs.flatten()
            sum = sum + np.square(inputs - mean).sum()
            num = num + len(inputs)

    pbar.close()

    deviation = np.sqrt(sum / (num - 1))

    return mean,deviation


def get_chb_mit_eeg_data():
    path = '/root/kistoff2/dataset/chb-mit-eeg-mat/'
    train_file_list = ['chb01','chb02','chb03','chb04','chb05',
                       'chb06','chb07','chb08','chb09','chb10',
                       'chb11','chb12','chb13','chb14','chb15',
                       'chb16','chb18','chb19','chb20']
    test_file_list = ['chb21','chb22','chb23','chb24']

    train_inputs_list = []
    train_labels_list = []
    train_count = 0
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        train_count = train_count + len(os.listdir(sub_file_path))
    pbar = tqdm(total=train_count, desc='Train set')
    # train_x = np.empty(shape=(0, 256, 17))
    # train_y = np.empty(shape=(0))
    for file_path in train_file_list:
        sub_file_path = os.path.join(path, file_path)
        tmp_x_list = []
        tmp_y_list = []
        for p in os.listdir(sub_file_path):
            pbar.update()
            filename = os.path.join(sub_file_path, p)
            file = scio.loadmat(filename)
            inputs = file['inputs']
            labels = file['label']
            labels = labels.reshape(labels.shape[1])
            # train_x = np.append(train_x, inputs, axis=0)
            # train_y = np.append(train_y, labels, axis=0)
            tmp_x_list.append(inputs)
            tmp_y_list.append(labels)
        train_inputs_list.append(np.concatenate(tmp_x_list, axis=0))
        train_labels_list.append(np.concatenate(tmp_y_list, axis=0))



    pbar.close()

    # train_x = np.concatenate(train_inputs_list)
    # train_y = np.concatenate(train_labels_list)
    train_x = train_inputs_list
    train_y = train_labels_list

    # test_count = 0
    # for file_path in test_file_list:
    #     sub_file_path = os.path.join(path, file_path)
    #     test_count = test_count + len(os.listdir(sub_file_path))
    #
    # pbar = tqdm(total=test_count, desc='Test set')
    # test_x = np.empty(shape=(0, 256, 17))
    # test_y = np.empty(shape=(0))
    # for file_path in test_file_list:
    #     sub_file_path = os.path.join(path, file_path)
    #     for p in os.listdir(sub_file_path):
    #         pbar.update()
    #         filename = os.path.join(sub_file_path, p)
    #         file = scio.loadmat(filename)
    #         inputs = file['inputs']
    #         labels = file['label']
    #         labels = labels.reshape(labels.shape[1])
    #         test_x = np.append(test_x, inputs, axis=0)
    #         test_y = np.append(test_y, labels, axis=0)
    #
    # pbar.close()

    # return train_x,train_y,test_x,test_y
    return train_x, train_y


if __name__ == '__main__':
    merge_test_set()
    # merge_train_set()
    #0.22180476472313504 67.9172308441852
    # mean,deviation = compute_train_set_mean_deviation()
    # print(mean, deviation)
    # train_x,train_y= get_chb_mit_eeg_data()
    #
    #
    # ##
    # xxxx = np.concatenate(train_x)




