from utils import tool
from utils.data import prepare_data
import torch
from tqdm import tqdm

def train_loop(model, optimizer, criterion, device, data_loader,
             epoch_id, epoch, lr_scheduler):
    model.train()
    train_loss = tool.AverageMeter()
    is_show_log = tool.show_log(device)
    # pbar = tool.ProgressBar(
    #     total=len(data_loader), ncols=70,
    #     desc='Epoch: {}/{}'.format(epoch_id + 1, epoch),
    #     show=True, position=device:
    # )

    text = 'Train Epoch: {}/{}'.format(epoch_id + 1, epoch)
    pbar = tqdm(
        total=len(data_loader), desc=text, ascii=True,
        ncols=80, disable=bool(1-is_show_log)
    )
    for i, data in enumerate(data_loader):
        inputs, labels = prepare_data(data, device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()  # compute gradient
        optimizer.step()  # update weight : W = W + learn_rate * gradient
        # pbar.update(1)
        train_loss.update(loss.item(), len(inputs))
        pbar.update(1)

    lr_scheduler.step()
    pbar.close()

    return train_loss.avg

def test_loop(model, criterion, device, data_loader,
              epoch_id, epoch):
    model.eval()
    test_loss = tool.AverageMeter()
    is_show_log = tool.show_log(device)
    text = 'Test Epoch: {}/{}'.format(epoch_id + 1, epoch)
    pbar = tqdm(
        total=len(data_loader), desc=text, ascii=True,
        ncols=80, disable=bool(1 - is_show_log)
    )
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = prepare_data(data, device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss.update(loss.item(), len(inputs))

            y_true_list.append(labels)
            y_pred_list.append(outputs)
            pbar.update()
        pbar.close()

    y_true = torch.cat(y_true_list, dim=0).detach().cpu().numpy()
    y_pred = torch.cat(y_pred_list, dim=0).detach().cpu().numpy()


    return test_loss.avg,y_true,y_pred