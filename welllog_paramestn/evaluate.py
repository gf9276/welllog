"""
evaluate（其实还有部分reuse code） 直接嵌入到train和predict里就行
"""
import math

import torch
from Utils.common import sample_to_device
from torchmetrics.functional.regression import r2_score


def common_forward(net, batch, criterion):
    """
    前向传播
    复用代码，不知道放哪里好，直接放在这里了，因为训练预测评估都要用到这个
    :param net:
    :param batch:
    :param criterion:
    :return:
    """
    cur_device = next(net.parameters()).device
    this_batch = sample_to_device(batch, cur_device)
    features = torch.swapaxes(this_batch["features"], 1, 2)
    label = this_batch["label"]

    output = net(features).squeeze()
    loss = criterion(output, label)

    predicted = output.clone().detach()  # 获取标签

    return loss, predicted, label


def evaluate(net, test_loader, criterion):
    """
    评估代码
    :param net:
    :param test_loader:
    :param criterion:
    :return:
    """
    net.eval()  # 很重要
    net.training = False
    total_loss = []
    label_nbr = 0
    eq_nbr = 0
    cur_device = next(net.parameters()).device
    all_label = []
    all_predicted = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            loss, predicted, label = common_forward(net, batch, criterion)
            label_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除
            total_loss.append(loss.item())  # 记录损失函数

            all_label.append(label)
            all_predicted.append(predicted)

        all_label = torch.cat(tuple(all_label), dim=0)
        all_predicted = torch.cat(tuple(all_predicted), dim=0)

    return r2_score(all_predicted, all_label).item(), sum(total_loss) / len(total_loss), all_label, all_predicted
