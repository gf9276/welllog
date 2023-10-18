"""
功能多了，代码就会复杂起来，没办法的
慢慢就会失控了。。。
"""
import os
import sys
import time
from multiprocessing import cpu_count

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import argparse
from importlib import import_module
import json
import torch
from pathlib import Path
from evaluate import common_forward, evaluate
from Utils.common import set_seeds, save_fig
from Utils.analysis import plt_confusion_matrix
from MyDataloader.h5_dataloader import setup_dataloaders


def parse_args():
    """
    提供训练参数
    :return:
    """
    parser = argparse.ArgumentParser(description='Train a model')
    # 模型配置文件
    parser.add_argument('--config', default="LSTM_Geology", help='模型的名字')
    # 文件和路径相关
    parser.add_argument('--logging_filepath', default="./Log/Train/logging.json", help='日志文件路径')
    parser.add_argument('--model_save_path', default="./Log/Train/output.pth", help='模型文件保存路径')
    parser.add_argument('--train_filepath', default="./Data/train.h5", help='训练集路径')
    parser.add_argument('--val_filepath', default="./Data/test.h5", help='验证集路径')
    # 一些开关
    parser.add_argument('--pretrained', default="True", help='是否要预训练')
    parser.add_argument('--checkpoint', default='./Log/Train/output.pth', help='模型权重路径')
    parser.add_argument('--draw_plt', default="True", help='是否绘图')
    # 其他
    parser.add_argument('--gpu_id', default="0", help='gpu_id')

    args = parser.parse_args()
    return args


def run_epoch(net, train_loader, optimizer, criterion):
    net.train()  # 很重要
    total_loss = []
    label_nbr = 0  # 标签总数
    eq_nbr = 0  # equal number
    cur_device = next(net.parameters()).device
    print_period = 10
    start = time.time()

    for batch_idx, batch in enumerate(train_loader):
        net.zero_grad()
        optimizer.zero_grad()

        loss, predicted, label = common_forward(net, batch, criterion)
        label_nbr += len(label)  # 这是考虑到整体数据量不能被batch size整除

        loss.backward()
        optimizer.step()

        eq_nbr += predicted.eq(label).sum().item()  # 记录标签准确数量
        total_loss.append(loss.item())  # 记录损失函数

        if batch_idx % print_period == 0:
            use_time = time.time() - start
            start = time.time()
            print("Batch: {} / {}, Sliced: {} / {}, Loss: {:.4f}, Acc: {:.4f}%, Speed: {:.4f} sliced/s".format(
                batch_idx, len(train_loader),
                label_nbr, len(train_loader.dataset),
                sum(total_loss) / len(total_loss),
                100 * eq_nbr / label_nbr,
                label_nbr / use_time))

    return eq_nbr / label_nbr, sum(total_loss) / len(total_loss)


def eval_val(net, val_loader, criterion):
    val_acc, val_loss, all_label, all_predicted = evaluate(net, val_loader, criterion)
    return val_acc, val_loss, all_label, all_predicted


def main(args):
    # --------------------O(∩_∩)O-------------- 成功第一步，固定随机数种子 ----------------------------------
    set_seeds()
    # --------------------O(∩_∩)O---------------- 成功第二步，配置参数 -------------------------------------
    # 01 获取 args 配置参数, 训练脚本的相关参数
    cfg = "Algorithm." + args.config
    logging_filepath = str(Path(args.logging_filepath))
    log_dir = str(Path(logging_filepath).parent)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = str(Path(args.model_save_path))
    train_h5filepath = str(Path(args.train_filepath))
    val_h5filepath = str(Path(args.val_filepath))
    checkpoint = str(Path(args.checkpoint))
    pretrained = args.pretrained
    draw_plt = args.draw_plt
    gpu_id = args.gpu_id
    device = torch.device('cuda:' + gpu_id if torch.cuda.is_available() else 'cpu')
    # 02 获取模型配置文件里的参数, 与模型有关的训练参数
    x = import_module(cfg)
    epoch_nbr = x.epoch
    batch_size = x.batch_size
    label_classes = x.label_classes
    # --------------------O(∩_∩)O-------------- 成功第三步，加载数据集 ----------------------------------
    train_dataset, train_loader = setup_dataloaders(
        train_h5filepath,
        label_classes,
        batch_size,
        num_workers=3 * cpu_count() // 4,
        shuffle=True)
    val_dataset, val_loader = setup_dataloaders(
        val_h5filepath,
        label_classes,
        batch_size,
        num_workers=3 * cpu_count() // 4,
        shuffle=False)
    # ---------------O(∩_∩)O--------------- 成功第四步，加载模型（if need） ------------------------------
    net = x.Net()
    if pretrained == "True" and Path(checkpoint).exists():
        pretrained_filepath = checkpoint
        loaded_model = torch.load(pretrained_filepath, map_location=torch.device("cpu"))
        net_dict = net.state_dict()

        # 直接判断同名model的尺寸是否一样就可以了，不一样的不加载
        pretrained_dict = {k: v for k, v in loaded_model.items() if k in net_dict and net_dict[k].shape == v.shape}

        net_dict.update(pretrained_dict)  # 更新一下。。
        net.load_state_dict(net_dict, strict=False)

    net = net.to(device)
    criterion, optimizer, exp_lr = x.loss_and_opt(net)
    # --------------------O(∩_∩)O-------------- 成功第五步，训练加验证 ----------------------------------
    result_dict = {"trainloss": [], "valloss": [], "trainaccuracy": [], "valaccuracy": []}  # 用于保存记录到json里面的字典
    print('---------------- 训练前测试一波 ----------------')
    val_acc, val_loss, all_label, all_predicted = eval_val(net, val_loader, criterion)
    # best_acc = val_acc
    # best_epoch = -1
    best_acc = 0
    best_epoch = 0
    print("测试集准确率: {:.4f}%, 测试集loss: {:.4f}".format(100 * val_acc, val_loss))
    for cur_epoch in range(epoch_nbr):
        print('---------------- 当前epoch: {} ----------------'.format(cur_epoch))
        print('当前lr:', optimizer.state_dict()['param_groups'][0]['lr'])
        train_acc, train_loss = run_epoch(net, train_loader, optimizer, criterion)
        val_acc, val_loss, all_label, all_predicted = eval_val(net, val_loader, criterion)
        exp_lr.step()  # 学习率迭代

        # 记录结果，并打印
        result_dict['trainaccuracy'].append(train_acc)
        result_dict['valaccuracy'].append(val_acc)
        result_dict['trainloss'].append(train_loss)
        result_dict['valloss'].append(val_loss)
        print("训练集准确率: {:.4f}%, 测试集准确率: {:.4f}%, 训练集loss: {:.4f}, 测试集loss: {:.4f}"
              .format(100 * train_acc, 100 * val_acc, train_loss, val_loss))

        # 保存验证集表现最好的模型
        if best_acc < val_acc:
            best_acc = val_acc
            best_epoch = cur_epoch
            print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))
            print('save model')
            torch.save(net.state_dict(), model_save_path)  # 转到cpu，再保存参数
            if draw_plt == "True":
                plt_confusion_matrix(all_predicted,
                                     all_label,
                                     label_classes,
                                     save_path=str(Path(log_dir) / "confusion_matrix.png"))

        # 保存json和图片
        with open(logging_filepath, "w") as f:
            json.dump(result_dict, f, indent=2)

        if draw_plt == "True":
            save_fig(result_dict['trainaccuracy'], "accuracy of train set", str(Path(log_dir) / "train_acc.png"))
            save_fig(result_dict['valaccuracy'], "accuracy of val set", str(Path(log_dir) / "val_acc.png"))
            save_fig(result_dict['trainloss'], "train loss", str(Path(log_dir) / "train_loss.png"))
            save_fig(result_dict['valloss'], "val loss", str(Path(log_dir) / "val_loss.png"))

    # --------------------O(∩_∩)O-------------- 成功第六步，结束 ----------------------------------
    print('---------------- 训练结束，完结撒花 ----------------')
    print('best acc: {:.4f}%, best epoch: {}'.format(100 * best_acc, best_epoch))


if __name__ == '__main__':
    # 不要把东西都写到这里面，容易造成全局变量泛滥
    # 在 if __name__ == '__main__': 里面写一长串代码和裸奔没区别
    my_args = parse_args()
    main(my_args)
