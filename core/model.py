import torch
import torch.nn as nn
from utils import tool
from utils import data
from core.rnn import LSTM,LSTMCell,SkipBCell,SkipACell,SkipCCell,SkipDCell,SkipECell,SkipFCell
from tqdm import tqdm
# from utils.data import get_dataset
from torch.utils.data import DataLoader
from collections import Counter

class BaseLSTM(nn.Module):
    def __init__(self, **kwargs):
        super(BaseLSTM, self).__init__()
        in_features = kwargs.pop('in_features')
        out_features = kwargs.pop('out_features')
        cell = kwargs.pop('cell')
        self.layer1 = nn.Sequential(
            LSTM(
                cell=cell(**kwargs), input_size=in_features, hidden_size=16,
                bias=True, batch_first=True
            )

        )
        self.layer2 = nn.Sequential(
            LSTM(
                cell=cell(**kwargs), input_size=16, hidden_size=8,
                bias=True, batch_first=True
            )
        )
        self.layer3 = nn.Sequential(
            LSTM(
                cell=cell(**kwargs), input_size=8, hidden_size=5,
                bias=True, batch_first=True
            )
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=5, out_features=out_features)
        )

    def forward(self, input):
        output, (h_n, c_n)= self.layer1(input)
        output, (h_n, c_n) = self.layer2(output)
        output, (h_n, c_n) = self.layer3(output)
        h_n = h_n.squeeze()
        output = self.layer4(h_n)

        return output

class NormalModel(BaseLSTM):
    def __init__(self, **kwargs):
        # kwargs = {'in_features': in_features, 'cell':LSTMCell}
        kwargs['cell'] = LSTMCell
        super(NormalModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(NormalModel, self).forward(input)

class SkipAModel(BaseLSTM):
    def __init__(self, **kwargs):
        # in_features= kwargs['in_features']
        # skip = kwargs['skip']
        kwargs['cell'] = SkipACell
        super(SkipAModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipAModel, self).forward(input)

class SkipBModel(BaseLSTM):
    def __init__(self, **kwargs):
        kwargs['cell'] = SkipBCell
        super(SkipBModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipBModel, self).forward(input)


class SkipCModel(BaseLSTM):
    def __init__(self, **kwargs):
        kwargs['cell'] = SkipCCell
        super(SkipCModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipCModel, self).forward(input)


class SkipDModel(BaseLSTM):
    def __init__(self, **kwargs):
        kwargs['cell'] = SkipDCell
        super(SkipDModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipDModel, self).forward(input)


class SkipEModel(BaseLSTM):
    def __init__(self, **kwargs):
        kwargs['cell'] = SkipECell
        super(SkipEModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipEModel, self).forward(input)



class SkipFModel(BaseLSTM):
    def __init__(self, **kwargs):
        kwargs['cell'] = SkipFCell
        super(SkipFModel, self).__init__(**kwargs)

    def forward(self, input):
        return super(SkipFModel, self).forward(input)


class EncoderDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(EncoderDecoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer6 = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.layer7 = nn.Sequential(
            nn.Linear(in_features=64, out_features=out_features),
            nn.BatchNorm1d(num_features=out_features),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = self.layer6(output)
        output = self.layer7(output)

        return output


class Base(nn.Module):
    def __init__(self, **kwargs):
        super(Base, self).__init__()
        in_features = kwargs['in_features']
        # out_features = kwargs['out_features']
        self.fc = EncoderDecoder(in_features=in_features, out_features=128)
        kwargs['in_features'] = 4
        self.kwargs = kwargs
        # print(kwargs)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        outputs = outputs.reshape((-1, 32, 4))
        return outputs


class Normal(Base):
    def __init__(self, **kwargs):
        super(Normal, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = NormalModel(**kwargs)

    def forward(self, inputs):
        outputs = super(Normal, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs

class ModelA(Base):
    def __init__(self, **kwargs):
        super(ModelA, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipAModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelA, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs


class ModelB(Base):
    def __init__(self, **kwargs):
        super(ModelB, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipBModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelB, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs

class ModelC(Base):
    def __init__(self, **kwargs):
        super(ModelC, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipCModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelC, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs

class ModelD(Base):
    def __init__(self, **kwargs):
        super(ModelD, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipDModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelD, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs


class ModelE(Base):
    def __init__(self, **kwargs):
        super(ModelE, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipEModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelE, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs


class ModelF(Base):
    def __init__(self, **kwargs):
        super(ModelF, self).__init__(**kwargs)
        kwargs = self.kwargs
        self.core = SkipFModel(**kwargs)

    def forward(self, inputs):
        outputs = super(ModelF, self).forward(inputs)
        outputs = self.core(outputs)
        return outputs




if __name__ == '__main__':
    ##
    args = tool.get_args()

    train_x, train_y, test_x, test_y = data.get_ecg_data()
    device = torch.device('cuda')
    train_dataloader = data.get_dataloader(train_x, train_y, batch_size=1000)
    model = SkipEModel(in_features=train_x.shape[-1], skip=args.skip).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    ##
    pbar = tqdm(total=len(train_dataloader))
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)




        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update()
    pbar.close()
    pass

    # args = tool.get_args()
    #
    # train_dataset, test_dataset = get_dataset(args)
    #
    # train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,
    #                                num_workers=2, pin_memory=True, shuffle=True)
    #
    # test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
    #                               num_workers=2, pin_memory=True, shuffle=True)
    # print('Dataset is ready!')
    # normal = Normal(in_features=train_dataset.x.shape[-1], out_features=len(Counter(train_dataset.y)), skip=2)
    # for data in train_data_loader:
    #     inputs,labels = data
    #     outputs = normal(inputs)
    #     print(inputs.shape, labels.shape, outputs.shape)
    #
    #     break