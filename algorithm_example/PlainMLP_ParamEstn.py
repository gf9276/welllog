import torch
import torch.nn as nn

# training config
slicelength = None
epoch = None
batchsize = 64
learningrate = None
lossfunc = None

# network config
featurenum = None
fclayers = [None]
label = [None]

# -------- Do not delete this line, the configuration ends here. --------

# --------- 变量映射 & 二次处理 & 通用函数定义, 我不喜欢原先的变量命名. ---------
model_name = 'PlainMLP'
features_num = featurenum
fc_layers = fclayers
label_classes = label
label_num = len(label_classes)
batch_size = batchsize
loss_dict = {0: torch.nn.MSELoss(), 1: torch.nn.MSELoss()}  # 为了兼容命令,暂时保留1
loss_func_idx = lossfunc
learning_rate = learningrate


def loss_and_opt(net):
    loss_func = loss_dict[loss_func_idx]
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    exp_lr = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
    return loss_func, opt, exp_lr


# ---------------------------- 模型的具体内容 ----------------------------

def make_fclayers(cfg, dropout=True):
    layers = []
    in_channels = fc_layers[0]
    for v in cfg:
        linear = nn.Linear(in_channels, v)
        if dropout:
            layers += [linear, nn.Dropout(), nn.ReLU(inplace=True)]
        else:
            layers += [linear, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc_input = nn.Linear(features_num * slicelength, fc_layers[0])
        self.fc = make_fclayers(fc_layers[1:])
        self.cls = nn.Linear(fc_layers[-1], 1)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # x.size(0)指x的batchsize
        x = self.fc_input(x)
        x = self.fc(x)
        x = self.cls(x)
        return x
