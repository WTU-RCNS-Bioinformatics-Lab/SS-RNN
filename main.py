import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from core import parallel
from utils.data import get_dataloader,get_dataloader_parallel,get_ecg_data,get_eeg_data
from utils import tool
from core.run import train_loop,test_loop
from utils.tool import get_model_path,get_result_path,save_result,save_model
import os
import time
from tensorboardX import SummaryWriter
from collections import Counter
from utils.data import Dataset,DataLoader,ToTensor,MitEegDataset,get_dataset
from torchvision import transforms



tool.set_seed(547)


def train(args):

    runs_path = os.path.join(get_result_path(args), 'runs')
    if not os.path.exists(runs_path):
        os.makedirs(runs_path)

    writer = SummaryWriter(runs_path)

    train_dataset, test_dataset = get_dataset(args)

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,
                        num_workers=2, pin_memory=True, shuffle=True)

    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False,
                                   num_workers=2, pin_memory=True, shuffle=True)
    print('Dataset is ready!')

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
    ModelClass, model_name = args.model
    model = ModelClass(in_features=train_dataset.x.shape[-1], skip=args.skip, out_features=len(Counter(train_dataset.y)))
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


    for epoch_id in range(args.epoch):
        train_loss = train_loop(
            model=model, optimizer=optimizer, criterion=criterion,
            device=device, data_loader=train_data_loader,
            epoch_id=epoch_id, epoch=args.epoch, lr_scheduler=lr_scheduler
        )
        test_loss,y_true,y_pred = test_loop(
            model=model, criterion=criterion, device=device, data_loader=test_data_loader,
            epoch_id=epoch_id, epoch=args.epoch
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_y_true_list.append(y_true)
        test_y_pred_list.append(y_pred)

        writer.add_scalar('train loss', train_loss, global_step=epoch_id)
        writer.add_scalar('test loss', test_loss, global_step=epoch_id)

        if (epoch_id + 1) % 10 == 0:
            #saving model
            # print('Saving +...')
            model_path = get_model_path(args)
            save_model(model_path, model)
            result_path = get_result_path(args)
            save_result(result_path, train_loss_list, test_loss_list, test_y_true_list, test_y_pred_list)
            # print('Result is saved!')


if __name__ == '__main__':
    args = tool.get_args()
    print(args)
    start = time.time()
    train(args)
    total_time = time.strftime('%H:%M:%S', time.localtime(time.time()-start))
    print('Total time:{}'.format(total_time))

