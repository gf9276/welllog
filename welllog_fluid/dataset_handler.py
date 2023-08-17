# 油的数据
import argparse
import json
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))  # 加入当前路径，直接执行有用
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pathlib
import random
import time
from multiprocessing import cpu_count

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")  # 必须放在这个位置
from matplotlib import pyplot as plt


def parse_args():
    """
    输入参数的，虽然大部分参数都写在json里，但是json的路径你得告诉我
    :return:
    """
    parser = argparse.ArgumentParser(description='处理样本集')
    parser.add_argument('--json', default="定边/定边预探井130_全井段_地质分层20230725/pre_proc.json", help='json文件的路径')
    parser.add_argument('--draw_plt', default="False", help='是否绘图')
    args = parser.parse_args()
    return args


def set_seeds(seed=42):
    """
    Set Python random seeding and PyTorch seeds.
    固定随机数种子

    Parameters
    ----------
    seed: int, default: 42
        Random number generator seeds for PyTorch and python
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


def get_random_range(length):
    random_range = np.arange(length)
    np.random.shuffle(random_range)  # 打乱
    return random_range


def read_json(json_cfg_filepath):
    with open(json_cfg_filepath) as f:
        content = json.load(f)
    return content


def sava_data2h5(h5_filepath, sliced_dataset, grp_name):
    """
    保存数据到h5里
    :param h5_filepath:
    :param sliced_dataset:
    :param grp_name:
    :return:
    """
    try:
        h5_file = h5py.File(h5_filepath, "w-")  # 写模式与w类似，但是如果h5文件已经存在的话，会报错
    except:
        h5_file = h5py.File(h5_filepath, "r+")  # 读+写，如果存在了就直接写呗

    for well_name in sliced_dataset:
        data = sliced_dataset[well_name][grp_name]  # 具体的特征数据
        if data is None:
            continue
        try:
            well_grp = h5_file.create_group(well_name)
        except:
            well_grp = h5_file[well_name]
        well_grp.create_dataset(grp_name, data=data)

    h5_file.close()


class WellDatasetCtrls:
    def __init__(self, json_cfg_filepath, draw_plt):
        """
        根据json文件路径和初始化
        :param json_cfg_filepath:
        :param draw_plt:
        """
        self.json_cfg_filepath = json_cfg_filepath
        self.draw_plt = draw_plt
        self.cfg_param = read_json(json_cfg_filepath)  # 读取配置
        self.proc_nbr = cpu_count()  # 全速前进！

        # ----------------------------------------------------------这些都是具体的配置---------------------------------------------------------
        # 切片长度，默认为1
        self.slice_length = 1
        if "slice_length" in self.cfg_param.keys():
            self.slice_length = self.cfg_param["slice_length"]

        # 切片步长，默认为切片长度的2/3，但是最小值为1
        self.slice_step = (2 * self.slice_length) // 3
        if "slice_step" in self.cfg_param.keys():
            self.slice_step = self.cfg_param["slice_step"]
        self.slice_step = max(1, self.slice_step)

        # 路径
        self.dataset_filepath = self.cfg_param["dataset_filepath"] if "dataset_filepath" in self.cfg_param.keys() else None
        self.output_dir = self.cfg_param["output_dir"] if "output_dir" in self.cfg_param.keys() else None

        # 特征和标签
        self.features_name = self.cfg_param["features_name"] if "features_name" in self.cfg_param.keys() else None
        self.label_name = self.cfg_param["label_name"] if "label_name" in self.cfg_param.keys() else None

        # 保留异常标签？
        self.keep_outlier_label = False
        if "keep_outlier_label" in self.cfg_param.keys():
            self.keep_outlier_label = self.cfg_param["keep_outlier_label"]
        if self.label_name is None:
            self.keep_outlier_label = False

        # 加一份中位数的标签？ 默认用的是中间值  mode / median
        self.label_type = "median"
        if "label_type" in self.cfg_param.keys():
            self.label_type = self.cfg_param["label_type"]

        # by wells / by slice 默认 by slice
        self.dataset_divide_method = "by slice"
        if "dataset_divide_method" in self.cfg_param.keys():
            self.dataset_divide_method = self.cfg_param["dataset_divide_method"]

        self.train_set_list = list(set(self.cfg_param["train_set_list"])) if "train_set_list" in self.cfg_param.keys() else []
        self.val_set_list = list(set(self.cfg_param["val_set_list"])) if "val_set_list" in self.cfg_param.keys() else []

        # 处理参数
        self.proc_info = self.cfg_param["proc_info"] if "proc_info" in self.cfg_param.keys() else {}
        self.add_padding = self.cfg_param["add_padding"] if "add_padding" in self.cfg_param.keys() else False

        # 默认的处理方式-->生成测试集还是训练集
        self.proc_method = "train"
        if "proc_method" in self.cfg_param.keys():
            self.proc_method = self.cfg_param["proc_method"]
        if self.label_name is None:
            self.proc_method = "test"
        if self.proc_method == "test":
            self.slice_step = 1
            self.add_padding = True  # 预测强制默认加padding

    def extend_dataset(self, sliced_dataset):
        for well_name in sliced_dataset.keys():
            label = sliced_dataset[well_name]["label"]
            values, cnt = np.unique(label, return_counts=True)
            mean_cnt = sum(cnt) / len(cnt)
            for i in range(len(values)):
                cur_cnt = cnt[i]
                cur_value = values[i]
                expansion_factor = round(mean_cnt / cur_cnt)
                if expansion_factor > 50 or expansion_factor < 2:
                    continue

                idx = (sliced_dataset[well_name]["label"] == cur_value).squeeze()
                for key in sliced_dataset[well_name].keys():
                    expend_arr = np.tile(sliced_dataset[well_name][key][idx], (expansion_factor, 1, 1))
                    sliced_dataset[well_name][key] = np.concatenate((sliced_dataset[well_name][key], expend_arr), 0)

        return sliced_dataset

    def analysis_label(self, sliced_dataset, title="unknown"):
        if not self.draw_plt:
            return
        for well_name in sliced_dataset.keys():
            label = sliced_dataset[well_name]["label"]
            values, cnt = np.unique(label, return_counts=True)
            # 少，才能画直方图
            if len(cnt) < 50:
                fig, ax_bar = plt.subplots()

                # 绘制直方图
                rects = ax_bar.bar([str(round(x)) for x in values], list(cnt), label='label')
                ax_bar.set_ylabel('label number')
                ax_bar.set_xlabel('label name')
                ax_bar.set_title('label distribution')
                ax_bar.legend()
                ax_bar.bar_label(rects, label_type='edge', fontsize=6, rotation=45)
                plt.xticks(fontsize=6, rotation=45)
                plt.savefig(str(pathlib.Path(self.output_dir) / (title + "_" + well_name + "_分布.svg")), dpi=600, format='svg')
                # plt.show()
                plt.close()
            else:
                fig, ax_liner = plt.subplots()
                # 绘制折线图
                ax_liner.plot([x for x in values], list(cnt), label='label')
                ax_liner.set_ylabel('label number')
                ax_liner.set_xlabel('label name')
                ax_liner.set_title('label distribution')
                ax_liner.legend()
                plt.xticks(fontsize=6, rotation=45)
                plt.savefig(str(pathlib.Path(self.output_dir) / (title + "_" + well_name + "_分布.svg")), dpi=600, format='svg')
                # plt.show()
                plt.close()

    def proc_dataset(self):
        """
        处理数据集全流程
        :return:
        """

        start_time = time.time()
        dataset = self.read_dataset()
        end_time = time.time()
        print("读取数据耗费时间：{}".format(end_time - start_time))

        start_time = time.time()
        merged_dataset = self.merge_features_and_label(dataset)
        merged_dataset, merged_dataset_outlier = self.proc_merged_dataset(merged_dataset)
        sliced_dataset = self.slice_dataset(merged_dataset, merged_dataset_outlier)
        end_time = time.time()
        print("切片数据耗费时间：{}".format(end_time - start_time))

        start_time = time.time()
        if self.proc_method == "train":
            # train_sliced_dataset, val_sliced_dataset = self.proc_sliced_dataset(sliced_dataset, remove_outlier_sliced=True)
            train_sliced_dataset, val_sliced_dataset = self.proc_sliced_dataset(sliced_dataset)
            # train_sliced_dataset = self.extend_dataset(train_sliced_dataset)
            self.save_dataset2h5("train", train_sliced_dataset)
            self.save_dataset2h5("val", val_sliced_dataset)
            self.analysis_label(train_sliced_dataset, "train")
            self.analysis_label(val_sliced_dataset, "val")
        elif self.proc_method == "test":
            test_sliced_dataset = self.proc_sliced_dataset(sliced_dataset)
            self.save_dataset2h5("test", test_sliced_dataset)
            self.analysis_label(test_sliced_dataset, "test")
        end_time = time.time()
        print("保存数据耗费时间：{}".format(end_time - start_time))

    def read_dataset(self):
        """
        读取数据集
        :return:
        """
        dataset = {}

        dataset_file = h5py.File(self.dataset_filepath, "r")
        for well_name in dataset_file.keys():
            # 第一层，是井次
            dataset[well_name] = {}
            for key in dataset_file[well_name]:
                # 第二层是曲线名字，只读取特征和标签
                if (key in self.features_name) or (self.label_name is not None and key == self.label_name):
                    dataset[well_name][key] = dataset_file[well_name][key][:].astype("float32").reshape(-1, 1)  # 变成二维后面好拼接
                    dataset[well_name][key][np.isnan(dataset[well_name][key])] = -99999  # 非法值直接变成-99999
                    dataset[well_name][key][np.isinf(dataset[well_name][key])] = -99999  # 非法值直接变成-99999
        dataset_file.close()
        return dataset

    def merge_features_and_label(self, dataset):
        """
        拼接数据集，前面都是特征，最后一列是标签
        :param dataset:
        :return:
        """
        merged_dataset = {}
        for well_name in dataset.keys():
            # ----------------------------------------------------------单井次数据---------------------------------------------------------
            a_well_dataset = dataset[well_name]

            # --------------------------------------------进行同长度处理，如果不是同长度，不是我的问题-------------------------------------------
            min_length = None
            for key in a_well_dataset.keys():
                min_length = a_well_dataset[key].shape[0] if min_length is None else min(min_length, a_well_dataset[key].shape[0])
            for key in a_well_dataset:
                if min_length < a_well_dataset[key].shape[0]:
                    a_well_dataset[key] = a_well_dataset[key][0:min_length, :]

            # --------------------------------------------合并所有数据，前面n列是特征，最后一列是标签-------------------------------------------
            features = np.concatenate(tuple(a_well_dataset[key] for key in self.features_name), 1)
            if self.label_name is not None:
                label = a_well_dataset[self.label_name]
                features_and_labels = np.concatenate((features, label), 1)
            else:
                features_and_labels = features
            # 为了test的时候，能给每一个深度点生成标签，所以这里需要我们自己手动填充
            # 训练的时候，如果数据集够大，这里应该是没有影响的，当然，训练的时候可以不填充
            if self.add_padding:
                # 复制比填充好
                opening_features_and_labels = np.copy(features_and_labels[0:self.slice_length // 2])
                ending_features_and_labels = np.copy(features_and_labels[-(self.slice_length - self.slice_length // 2 - 1):])
                features_and_labels = np.concatenate((opening_features_and_labels, features_and_labels, ending_features_and_labels), 0)

                # features_and_labels = np.pad(features_and_labels,
                #                              ((self.slice_length // 2, self.slice_length - self.slice_length // 2 - 1), (0, 0)),
                #                              constant_values=-99999)

            merged_dataset[well_name] = features_and_labels
        return merged_dataset

    def __cat_merged_dataset__(self, merged_dataset):
        """
        把所有井的数据合到一起，合成一个矩阵
        :return:
        """
        dataset_list = []
        for well_name in list(merged_dataset.keys()):
            dataset_list.append(merged_dataset[well_name])
        return np.concatenate(tuple(dataset_list), 0)

    def proc_merged_dataset(self, merged_dataset):
        """
        处理数据
        :return:
        """
        matrix_dataset = self.__cat_merged_dataset__(merged_dataset)
        merged_dataset_outlier = {}

        for well_name in list(merged_dataset.keys()):
            # 生成一坨false，用于记录，该深度点数据是否存在异常值
            merged_dataset_outlier[well_name] = np.zeros((merged_dataset[well_name].shape[0])) != 0
            # 先对深度进行处理
            # if "DEPTH" in self.features_name:
            #     # 获取DEPTH的idx
            #     depth_idx = self.features_name.index("DEPTH")
            #     # 使用全局的数值，进行归一化
            #     outlier_idx = (abs(matrix_dataset[:, depth_idx] - (-99999)) < 1) | np.isnan(matrix_dataset[:, depth_idx])  # 剔除异常值
            #     max_value = matrix_dataset[:, depth_idx][~outlier_idx].max()  # 获取最大值
            #     min_value = matrix_dataset[:, depth_idx][~outlier_idx].min()  # 获取最小值
            #     merged_dataset[well_name][:, depth_idx] = (merged_dataset[well_name][:, depth_idx] - min_value) / (max_value - min_value)

            # 然后对每一个进行处理
            for key in self.proc_info:
                # 获取在矩阵中的位置
                key_idx = None
                if key in self.features_name:
                    # 检索具体处理内容
                    key_idx = self.features_name.index(key)
                elif self.label_name is not None and key == self.label_name:
                    key_idx = len(self.features_name)  # 只有一个标签，当然是最后一个了

                # 开始处理
                if key_idx is not None:
                    # 可能有多个处理过程
                    # 和下面那个位置，二选一
                    outlier_idx = (abs(merged_dataset[well_name][:, key_idx] - (-99999)) < 10)
                    merged_dataset_outlier[well_name] |= (abs(merged_dataset[well_name][:, key_idx] - (-99999)) < 10)
                    for i in range(len(self.proc_info[key])):
                        # # 找到异常值，必须放在这里，有的异常值经过一次处理，就不会再处理第二次了
                        # outlier_idx = (abs(merged_dataset[well_name][:, key_idx] - (-99999)) < 10)
                        # -----------------------------------------------------------------------------------------------------------------------------------
                        if self.proc_info[key][i][0] == "mn":
                            if len(self.proc_info[key][i]) >= 3:
                                # 直接使用他的参数。为了防止，最后面有一个l之类的符号
                                min_value = float(self.proc_info[key][i][1] if self.proc_info[key][i][1][-1].isdigit() else self.proc_info[key][i][1][:-1])
                                max_value = float(self.proc_info[key][i][2] if self.proc_info[key][i][2][-1].isdigit() else self.proc_info[key][i][2][:-1])
                            else:
                                all_well_dataset_outlier_idx = (abs(matrix_dataset[:, key_idx] - (-99999)) < 1) | np.isnan(matrix_dataset[:, key_idx])
                                min_value = matrix_dataset[:, key_idx][~all_well_dataset_outlier_idx].min()
                                max_value = matrix_dataset[:, key_idx][~all_well_dataset_outlier_idx].max()
                            if outlier_idx.sum() != 0:
                                # 如果存在异常值，不管数量是多少，直接把异常值变成最小值
                                merged_dataset[well_name][:, key_idx][outlier_idx] = min_value
                            merged_dataset[well_name][:, key_idx] -= min_value
                            if abs(max_value - min_value) > 1e-3:
                                # 为了防止一整列都是-99999的情况。。。
                                merged_dataset[well_name][:, key_idx] /= (max_value - min_value)
                        # -----------------------------------------------------------------------------------------------------------------------------------
                        elif self.proc_info[key][i][0] == "sd":
                            if len(self.proc_info[key][i]) >= 3:
                                # 直接使用他的参数。为了防止，最后面有一个l之类的符号
                                mean_value = float(self.proc_info[key][i][1] if self.proc_info[key][i][1][-1].isdigit() else self.proc_info[key][i][1][:-1])
                                std_value = float(self.proc_info[key][i][2] if self.proc_info[key][i][2][-1].isdigit() else self.proc_info[key][i][2][:-1])
                            else:
                                all_well_dataset_outlier_idx = (abs(matrix_dataset[:, key_idx] - (-99999)) < 1) | np.isnan(matrix_dataset[:, key_idx])
                                mean_value = matrix_dataset[:, key_idx][~all_well_dataset_outlier_idx].mean()
                                std_value = matrix_dataset[:, key_idx][~all_well_dataset_outlier_idx].std()
                            if outlier_idx.sum() != 0:
                                merged_dataset[well_name][:, key_idx][outlier_idx] = mean_value
                            merged_dataset[well_name][:, key_idx] -= mean_value
                            if abs(std_value - 0) > 1e-3:
                                # 为了防止一整列都是-99999的情况。。。
                                merged_dataset[well_name][:, key_idx] /= std_value
                        # -----------------------------------------------------------------------------------------------------------------------------------
                        elif self.proc_info[key][i][0] == "log":
                            # outlier_idx |= merged_dataset[well_name][:, key_idx] <= 0  # 既然要log，小于等于0的也是异常值
                            merged_dataset[well_name][:, key_idx][merged_dataset[well_name][:, key_idx] <= 0] = 1  # 要log的甚至有负数
                            if outlier_idx.sum() != 0:
                                merged_dataset[well_name][:, key_idx][outlier_idx] = 1
                            merged_dataset[well_name][:, key_idx] = np.log10(merged_dataset[well_name][:, key_idx])

        return merged_dataset, merged_dataset_outlier

    def slice_a_well_dataset(self, merged_dataset, merged_dataset_outlier, well_name):
        sliced_features_list = []  # 切片后的特征
        sliced_label_list = []  # 切片后的标签
        sliced_mul_label_list = []  # 切片后的标签
        sliced_outlier_list = []  # 切片后的标签
        cur_idx = 0  # 当前的idx
        # 判断是否能切
        while cur_idx + self.slice_length <= merged_dataset[well_name].shape[0]:
            # ----------------------------------------------------------获取每个切片的特征---------------------------------------------------------
            cur_feature = np.expand_dims(merged_dataset[well_name][cur_idx: cur_idx + self.slice_length, 0:len(self.features_name)], axis=0)
            sliced_features_list.append(cur_feature)

            if self.label_name is not None:
                # ----------------------------------------------------------获取每个切片的单个标签---------------------------------------------------------
                if self.label_type == "mode":
                    mode_label_list = []  # 众数 --> mode，我也很奇怪，众数的英文居然是mode
                    for i in range(merged_dataset[well_name][cur_idx: cur_idx + self.slice_length, len(self.features_name):].shape[1]):
                        u, c = np.unique(merged_dataset[well_name][cur_idx: cur_idx + self.slice_length, len(self.features_name):][:, i], return_counts=True)
                        mode_label_list.append(u[c.argmax()])
                        mode_label = np.array(mode_label_list).reshape(1, 1, -1)
                        sliced_label_list.append(mode_label)
                elif self.label_type == "median":
                    median_label = merged_dataset[well_name][cur_idx: cur_idx + self.slice_length, len(self.features_name):][self.slice_length // 2, :]
                    median_label = median_label.reshape(1, 1, -1)
                    sliced_label_list.append(median_label)

                # ----------------------------------------------------------获取每个切片的一对一标签---------------------------------------------------------
                cur_mul_label = merged_dataset[well_name][cur_idx: cur_idx + self.slice_length, len(self.features_name):].reshape(1, self.slice_length, -1)
                sliced_mul_label_list.append(cur_mul_label)

                # ----------------------------------------------------------获取每个切片是否存在异常值---------------------------------------------------------
                cur_outlier = (merged_dataset_outlier[well_name][cur_idx: cur_idx + self.slice_length]).sum() > (self.slice_length // 10)
                sliced_outlier_list.append(np.asarray(cur_outlier).reshape(-1))  # 异常值大于指定行数，我就认为你是异常切片

            # ----------------------------------------------------------刷新步长---------------------------------------------------------
            cur_idx += self.slice_step

        sliced_features = np.concatenate(tuple(sliced_features_list), 0) if len(sliced_features_list) != 0 else None  # n * 97 * 5
        sliced_label = np.concatenate(tuple(sliced_label_list), 0) if len(sliced_label_list) != 0 else None  # n * 1 * 1
        sliced_mul_label = np.concatenate(tuple(sliced_mul_label_list), 0) if len(sliced_mul_label_list) != 0 else None  # n * 97 * 1
        sliced_outlier = np.concatenate(tuple(sliced_outlier_list), 0) if len(sliced_outlier_list) != 0 else None

        return {"features": sliced_features, "label": sliced_label, "multi_label": sliced_mul_label, "outlier": sliced_outlier}

    def slice_dataset(self, merged_dataset, merged_dataset_outlier):
        """
        多线程切片
        :param merged_dataset:
        :param merged_dataset_outlier:
        :return:
        """
        # ----------------------------------------------------------处理---------------------------------------------------------
        sliced_dataset = {}

        for well_name in merged_dataset.keys():
            sliced_dataset[well_name] = self.slice_a_well_dataset(merged_dataset, merged_dataset_outlier, well_name)

        return sliced_dataset

    def proc_sliced_dataset(self, sliced_dataset, remove_outlier_sliced=False):
        """
        处理切片完的数据集，主要是标签异常值是否保留
        :param sliced_dataset:
        :param remove_outlier_sliced:
        :return:
        """
        # ----------------------------------------------------------是否保留异常值---------------------------------------------------------
        if self.keep_outlier_label is False and self.label_name is not None:
            for well_name in list(sliced_dataset.keys()):
                normal_value_idx = np.zeros((sliced_dataset[well_name]["label"].shape[0])) != 0  # 生成一坨false
                for i in range(sliced_dataset[well_name]["label"].shape[1]):
                    # 这里其实只有一行，不用for也可以
                    normal_value_idx |= (abs(sliced_dataset[well_name]["label"][:, i] - -99999) < 1).reshape(-1)
                normal_value_idx = ~normal_value_idx
                if sliced_dataset[well_name]["label"][normal_value_idx, :].shape[0] == 0:
                    # 有的井一个数据都用不了，这一类直接删除
                    sliced_dataset.pop(well_name)
                else:
                    # 存在可用数据就更新一下
                    sliced_dataset[well_name]["label"] = sliced_dataset[well_name]["label"][normal_value_idx, :]
                    sliced_dataset[well_name]["features"] = sliced_dataset[well_name]["features"][normal_value_idx, :]
                    sliced_dataset[well_name]["multi_label"] = sliced_dataset[well_name]["multi_label"][normal_value_idx, :]
                    sliced_dataset[well_name]["outlier"] = sliced_dataset[well_name]["outlier"][normal_value_idx]

        if remove_outlier_sliced:
            for well_name in list(sliced_dataset.keys()):
                keep_sliced_idx = ~sliced_dataset[well_name]["outlier"]
                if sliced_dataset[well_name]["label"][keep_sliced_idx, :].shape[0] == 0:
                    sliced_dataset.pop(well_name)
                else:
                    sliced_dataset[well_name]["label"] = sliced_dataset[well_name]["label"][keep_sliced_idx, :]
                    sliced_dataset[well_name]["features"] = sliced_dataset[well_name]["features"][keep_sliced_idx, :]
                    sliced_dataset[well_name]["multi_label"] = sliced_dataset[well_name]["multi_label"][keep_sliced_idx, :]
                    sliced_dataset[well_name]["outlier"] = sliced_dataset[well_name]["outlier"][keep_sliced_idx]

        # ----------------------------------------------------------训练集还是测试集---------------------------------------------------------
        if self.proc_method == "train":
            if len(self.train_set_list) == 0 and len(self.val_set_list) == 0:
                train_sliced_dataset, val_sliced_dataset = self.partition_dataset(sliced_dataset)
            else:
                train_sliced_dataset, val_sliced_dataset = self.specify_partition_dataset(sliced_dataset)

            return train_sliced_dataset, val_sliced_dataset

        elif self.proc_method == "test":
            return sliced_dataset

    def specify_partition_dataset(self, sliced_dataset):
        """
        指定划分数据集
        :param sliced_dataset:
        :return:
        """
        wells_name = list(sliced_dataset.keys())
        train_sliced_dataset = {}
        val_sliced_dataset = {}
        if len(self.val_set_list) == 0:
            self.val_set_list = list(set(wells_name) - set(self.train_set_list))
            if len(self.val_set_list) == 0:
                self.val_set_list.append(self.train_set_list[0])
        elif len(self.train_set_list) == 0:
            self.train_set_list = list(set(wells_name) - set(self.val_set_list))
            if len(self.train_set_list) == 0:
                self.train_set_list.append(self.val_set_list[0])
        for well_name in self.train_set_list:
            train_sliced_dataset[well_name] = {"features": sliced_dataset[well_name]["features"],
                                               "label": sliced_dataset[well_name]["label"],
                                               "multi_label": sliced_dataset[well_name]["multi_label"],
                                               "outlier": sliced_dataset[well_name]["outlier"]}
        for well_name in self.val_set_list:
            val_sliced_dataset[well_name] = {"features": sliced_dataset[well_name]["features"],
                                             "label": sliced_dataset[well_name]["label"],
                                             "multi_label": sliced_dataset[well_name]["multi_label"],
                                             "outlier": sliced_dataset[well_name]["outlier"]}

        return train_sliced_dataset, val_sliced_dataset

    def partition_dataset(self, sliced_dataset):
        """
        划分数据集
        :param sliced_dataset:
        :return:
        """
        train_sliced_dataset = {}
        val_sliced_dataset = {}
        if self.dataset_divide_method == "by wells":
            # 按井分训练集和预测集
            wells_name = list(sliced_dataset.keys())
            wells_nbr = len(list(sliced_dataset.keys()))
            wells_dict = dict(zip(list(range(wells_nbr)), wells_name))
            random_range = get_random_range(wells_nbr)
            train_wells_nbr = int(wells_nbr * 0.9)
            for i in range(wells_nbr):
                well_name = wells_dict[random_range[i]]
                if i < train_wells_nbr:
                    train_sliced_dataset[well_name] = {"features": sliced_dataset[well_name]["features"],
                                                       "label": sliced_dataset[well_name]["label"],
                                                       "multi_label": sliced_dataset[well_name]["multi_label"],
                                                       "outlier": sliced_dataset[well_name]["outlier"]}
                else:
                    val_sliced_dataset[well_name] = {"features": sliced_dataset[well_name]["features"],
                                                     "label": sliced_dataset[well_name]["label"],
                                                     "multi_label": sliced_dataset[well_name]["multi_label"],
                                                     "outlier": sliced_dataset[well_name]["outlier"]}

        else:
            # 按照切片分预测值
            all_sliced_features = np.concatenate(tuple(sliced_dataset[well_name]["features"] for well_name in sliced_dataset.keys()), 0)
            all_sliced_label = np.concatenate(tuple(sliced_dataset[well_name]["label"] for well_name in sliced_dataset.keys()), 0)
            all_sliced_multi_label = np.concatenate(tuple(sliced_dataset[well_name]["multi_label"] for well_name in sliced_dataset.keys()), 0)
            all_sliced_outlier = np.concatenate(tuple(sliced_dataset[well_name]["outlier"] for well_name in sliced_dataset.keys()), 0)
            # 打乱所有切片顺序
            random_range = get_random_range(all_sliced_features.shape[0])
            all_sliced_features = all_sliced_features[random_range]
            all_sliced_label = all_sliced_label[random_range]
            all_sliced_multi_label = all_sliced_multi_label[random_range]
            all_sliced_outlier = all_sliced_outlier[random_range]
            # 划分训练集和验证集
            train_dataset_size = int(all_sliced_features.shape[0] * 0.9)
            train_sliced_dataset = {"all_well": {"features": all_sliced_features[:train_dataset_size],
                                                 "label": (all_sliced_label[: train_dataset_size]),
                                                 "multi_label": (all_sliced_multi_label[: train_dataset_size]),
                                                 "outlier": (all_sliced_outlier[: train_dataset_size])}}
            val_sliced_dataset = {"all_well": {"features": all_sliced_features[train_dataset_size:],
                                               "label": (all_sliced_label[train_dataset_size:]),
                                               "multi_label": (all_sliced_multi_label[train_dataset_size:]),
                                               "outlier": (all_sliced_outlier[train_dataset_size:])}}

        return train_sliced_dataset, val_sliced_dataset

    def save_dataset2h5(self, file_stem, sliced_dataset):
        """
        保存数据集到h5文件中
        :param file_stem:
        :param sliced_dataset:
        :return:
        """
        # 测试路径
        dir_path = pathlib.Path(self.output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)
        h5_filepath = pathlib.Path(self.output_dir) / (file_stem + ".h5")
        if h5_filepath.exists():
            h5_filepath.unlink()

        with h5py.File(h5_filepath, 'w') as h5_file:
            h5_file.attrs["slice_length"] = self.slice_length
            h5_file.attrs["slice_step"] = self.slice_step

        # 保存数据
        for i in sliced_dataset[list(sliced_dataset.keys())[0]].keys():
            sava_data2h5(str(h5_filepath), sliced_dataset, i)


if __name__ == "__main__":
    # 初始化种子
    set_seeds()
    my_args = parse_args()
    json_filepath = my_args.json
    print("\nHi, {}".format(json_filepath))
    my_dataset_ctrl = WellDatasetCtrls(json_filepath, (my_args.draw_plt == "True"))
    my_dataset_ctrl.proc_dataset()
