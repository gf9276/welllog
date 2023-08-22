# 有的损失函数没有，我自己来写
import torch


def rmse_loss(predicted, tgt):
    loss_func = torch.nn.MSELoss()
    mse = loss_func(tgt, predicted)
    rmse = torch.sqrt(mse)
    return rmse
