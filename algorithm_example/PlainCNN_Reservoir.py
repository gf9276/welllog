import torch
import torch.nn as nn

# training config
slicelength = NAN
epoch = NAN
batchsize = 64
learningrate = NAN
lossfunc = NAN

# network config
featurenum = NAN
convlayers = [NAN]
fclayers = [NAN]
label = [NAN]

# -------- Do not delete this line, the configuration ends here. --------

# --------- 变量映射 & 二次处理 & 通用函数定义, 我不喜欢原先的变量命名. ---------
model_name = 'PlainCNN'
features_num = featurenum
fc_layers = fclayers
conv_layers = convlayers
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

def make_cnnlayers(cfg, batch_norm=True):
    layers = []
    in_channels = features_num
    for v in cfg:
        conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv1d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


def make_fclayers(cfg, dropout=True):
    layers = []
    # in_channels = int(convLayers[-1])
    in_channels = fc_layers[0]
    for v in cfg[1:]:
        linear = nn.Linear(in_channels, v)
        if dropout:
            layers += [linear, nn.ReLU(inplace=True), nn.Dropout()]
        else:
            layers += [linear, nn.ReLU(inplace=True)]
        in_channels = v

    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_layer = make_cnnlayers(conv_layers)

        # conv1d添加平均全局池化
        self.avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(conv_layers[-1] * 1, fc_layers[0])

        self.fc = make_fclayers(fc_layers)
        self.cls = nn.Linear(fc_layers[-1], label_num)

    def forward(self, x):
        x = self.conv_layer(x)
        # 添加的平均全局池化层
        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)  # x.size(0)指x的batchsize
        x = self.fc1(x)
        x = self.fc(x)
        x = self.cls(x)
        return x
