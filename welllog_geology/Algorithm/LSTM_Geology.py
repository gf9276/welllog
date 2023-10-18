import torch
import torch.nn as nn
import numpy as np

# training config
slicelength = 97
epoch = 1
batchsize = 64
learningrate = 0.02
lossfunc = 0

# network config
featurenum = 6
label = ['34', '19', '3', '4', '25', '2', '1', '0', '23', '35', '32', '33', '31', '30', '28', '29', '26', '27', '22',
         '21', '12', '13', '14', '15', '16', '17',
         '18', '24', '11', '20', '5', '6', '7', '8', '9', '10']
hiddensize = 2
numlayers = 2

# -------- Do not delete this line, the configuration ends here. --------

# --------- 变量映射 & 二次处理 & 通用函数定义, 我不喜欢原先的变量命名. ---------
model_name = 'LSTM'
features_num = featurenum
hidden_size = hiddensize
num_layers = numlayers
label_classes = label
label_num = len(label_classes)
batch_size = batchsize
loss_dict = {0: torch.nn.CrossEntropyLoss(), 1: torch.nn.CrossEntropyLoss()}  # 为了兼容命令,暂时保留1
loss_func_idx = lossfunc
learning_rate = learningrate


def loss_and_opt(net):
    loss_func = loss_dict[loss_func_idx]
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    return loss_func, opt, exp_lr


# ---------------------------- 模型的具体内容 ----------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rnn = nn.LSTM(
            input_size=features_num,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # self.fc = torch.nn.Linear(hidden_size, label_num)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, label_num)
        )

    def forward(self, x):
        x = np.swapaxes(x, 2, 1)
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.fc(r_out[:, -1, :])
        return out
