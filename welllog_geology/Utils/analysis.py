"""
分析结果的代码
"""
import matplotlib

matplotlib.use("Agg")  # 必须放在这个位置
from matplotlib import pyplot as plt
import torch


def plt_confusion_matrix(predicted, gth, label_classes_name, percentage=False, save_path="confusion_matrix.png"):
    """

    :param predicted: shape = (n,)
    :param gth: shape = (n,)
    :param label_classes_name:
    :param percentage: 是否百分比化
    :param save_path: 保存路径
    :return:
    """
    # 基本参数
    label_classes_nbr = len(label_classes_name)  # 总类别数量
    if percentage:
        conf_matrix = make_percentage_confusion_matrix(predicted, gth, label_classes_nbr)  # 直接生成混淆矩阵
    else:
        conf_matrix = make_confusion_matrix(predicted, gth, label_classes_nbr)  # 直接生成混淆矩阵

    # 绘原始热力图
    fig_size = max(label_classes_nbr // 2, 6)  # 自动适应大小，我真聪明
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    ax.plot([0, 1], [1, 0], "m:", transform=ax.transAxes)
    font_dict = dict(fontsize=8, color='b', weight='light', style='italic')

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(label_classes_nbr):
        for y in range(label_classes_nbr):
            # 注意这里的matrix[y, x]不是matrix[x, y]！
            info = conf_matrix[y, x]
            if percentage:
                info = format(info, '.2f')
            else:
                info = int(info)
            if info != '0.00':
                plt.text(x, y, info,
                         font_dict,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if float(info) > thresh else "black")

    plt.yticks(range(label_classes_nbr), label_classes_name, fontsize=8)
    plt.ylabel("predict")
    plt.xticks(range(label_classes_nbr), label_classes_name, fontsize=8, rotation=45)
    plt.xlabel("ground truth")
    if percentage:
        plt.title("confusion matrix with percentage")
    else:
        plt.title("confusion matrix")

    plt.tight_layout()  # 保(证图不)重叠
    plt.savefig(save_path)
    plt.close()


def make_confusion_matrix(predicted, gth, label_classes_nbr):
    """
    矩阵的第0个维度，预测值；第一个维度，表示真实值。比如（3, 5）表示第三行第五列，预测出来是3，实际是5的数量
    和plt的x，y轴规则规则恰好相反
    :param predicted: 预测出来的标签，数字类型的
    :param gth: ground truth
    :param label_classes_nbr: 数字对应的标签名字
    :return:
    """
    conf_matrix = torch.zeros(label_classes_nbr, label_classes_nbr)
    for y, x in zip(predicted, gth):
        conf_matrix[y, x] += 1

    return conf_matrix.cpu().numpy()


def make_percentage_confusion_matrix(predicted, gth, label_classes_nbr):
    """
    处理成百分比格式的
    :param predicted:
    :param gth:
    :param label_classes_nbr:
    :return:
    """
    conf_matrix = make_confusion_matrix(predicted, gth, label_classes_nbr)
    total_num = conf_matrix.sum(1)  # 对行求和, 得到预测出来的每一个类别的数量
    for n in range(len(total_num)):  # 计算百分比
        if total_num[n] != 0:
            conf_matrix[n] = (conf_matrix[n] / total_num[n] * 100)
        elif total_num[n] == 0:
            conf_matrix[n] = 0
    return conf_matrix


def test():
    predict = torch.randint(0, 10, (100,))
    gth = torch.randint(0, 10, (100,))
    label_classes_name = ["a", 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    plt_confusion_matrix(predict, gth, label_classes_name, percentage=True)


if __name__ == '__main__':
    test()
