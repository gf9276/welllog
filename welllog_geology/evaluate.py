"""
evaluate（其实还有部分reuse code） 直接嵌入到train和predict里就行
"""
import torch
from Utils.common import sample_to_device


def common_forward(net, batch, criterion):
    """
    前向传播
    复用代码，不知道放哪里好，直接放在这里了，因为训练预测评估都要用到这个
    :param net: 网络模型
    :param batch: 批大小
    :param criterion: 损失函数
    :return: 损失函数 预测值
    """
    if hasattr(net, 'model_type') and net.model_type == "transformer":
        cur_device = next(net.parameters()).device
        this_batch = sample_to_device(batch, cur_device)
        features = this_batch["features"]
        features = features.reshape(features.size()[0], -1, net.patch_height * features.shape[2])
        transformer_batch = net.get_batch(features, None, 926219)
        multi_label = this_batch["multi_label"]
        multi_label = multi_label.reshape(multi_label.size()[0], -1, net.patch_height)
        multi_label = multi_label[:, :, multi_label.shape[2] // 2].long()

        output = net.forward(src=transformer_batch.src, src_mask=transformer_batch.src_mask, just_encoder=True)
        # output = output[:, output.shape[1] // 2, :]  # 我懒，直接取中间值得了
        multi_label = multi_label.reshape(-1)
        output = output.reshape(-1, output.shape[2])
        loss = criterion(output, multi_label)

        _, predicted = output.max(1)  # 获取标签

    else:
        cur_device = next(net.parameters()).device
        this_batch = sample_to_device(batch, cur_device)
        features = torch.swapaxes(this_batch["features"], 1, 2)
        label = this_batch["label"].long()

        output = net(features)
        loss = criterion(output, label)

        _, predicted = output.max(1)  # 获取标签
        multi_label = label

    return loss, predicted, multi_label


def evaluate(net, test_loader, criterion):
    """
    评估代码
    :param net: 网络模型
    :param test_loader: 测试集 dataloader对象
    :param criterion: 损失函数
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
            eq_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
            total_loss.append(loss.item())  # 记录损失函数

            all_label.append(label)
            all_predicted.append(predicted)

        all_label = torch.cat(tuple(all_label), dim=0)
        all_predicted = torch.cat(tuple(all_predicted), dim=0)

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss), all_label, all_predicted
