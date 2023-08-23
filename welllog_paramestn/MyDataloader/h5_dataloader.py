import json
import random
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, h5filepath: str, label_classes: list, which_wells=None, noise=False):
        """
        label_classes: 比如{'3', '5', '1', '0', '6', '4', '2', '-99999'}
        label_classes_dict: 比如 {3: 0, 5: 1, 1: 2, 0: 3, 6: 4, 4: 5, 2: 6, -99999: 7}
        注意string和int的变化
        """

        # a-->b b-->a
        self.h5filepath = h5filepath
        self.noise = noise
        self.have_label = True

        # 直接全部加载到内存里，其实可以在__getitem__里面重新打开文件，会不会起冲突我就不知道了，没试过
        self.dataset = {}
        self.wells_name = []  # 保存井次信息 --> 必须list，python里的dict貌似没有顺序，新版本可能有
        self.wells_size = []  # 保存每个wells对应的长度

        self.features_h5file = h5py.File(h5filepath, "r")
        self.slice_length = self.features_h5file.attrs["slice_length"]
        self.slice_step = self.features_h5file.attrs["slice_step"]
        self.proc_info = json.loads(self.features_h5file.attrs["proc_info"])
        self.label_name = self.features_h5file.attrs["label_name"]

        all_well_features = []
        all_well_label = []
        all_well_multi_label = []
        for well_name in self.features_h5file.keys():
            if which_wells is not None and well_name not in which_wells:
                continue
            self.wells_name.append(well_name)
            self.wells_size.append(len(self.features_h5file[well_name]["features"][:]))

            # 按照float32的格式读取
            all_well_features.append(self.features_h5file[well_name]["features"][:].astype("float32"))
            if "label" in self.features_h5file[well_name].keys() and "multi_label" in self.features_h5file[well_name].keys():
                all_well_label.append(self.features_h5file[well_name]["label"][:].astype("float32"))
                all_well_multi_label.append(self.features_h5file[well_name]["multi_label"][:].astype("float32"))
                self.have_label = True
            else:
                self.have_label = False

        self.features_h5file.close()

        self.dataset["features"] = np.concatenate(tuple(all_well_features), 0)
        if self.have_label:
            self.dataset["label"] = np.concatenate(tuple(all_well_label), 0)
            self.dataset["multi_label"] = np.concatenate(tuple(all_well_multi_label), 0)

        self.features_nbr = self.dataset["features"].shape[2]
        self.slice_length = self.dataset["features"].shape[1]

        self.features_mean = np.mean(self.dataset["features"].reshape(-1, self.features_nbr), axis=0)
        self.features_std = np.std(self.dataset["features"].reshape(-1, self.features_nbr), axis=0)

    def __len__(self):
        # 返回数据集长度
        return len(self.dataset["features"])

    def __getitem__(self, idx):
        """
        原来聪明的dataloader，会自动转化为tensor，那我就不多此一举了
        """
        features = np.copy(self.dataset["features"][idx])
        if self.noise:
            # 这个加噪声的方式有问题，所以先不加
            for i in range(self.features_nbr):
                noise = 0.1 * np.random.normal(loc=self.features_mean[i], scale=self.features_std[i], size=features[:, i].shape)
                features[:, i] += noise

        if self.have_label:
            label = self.dataset["label"][idx].squeeze()
            multi_label = self.dataset["multi_label"][idx].squeeze()
            return {"features": features, "label": label, "multi_label": multi_label, **self.get_data_well(idx)}
        else:
            return {"features": features, **self.get_data_well(idx)}

    def get_data_well(self, idx):
        """
        解析获得数据对应的井次和井次里的idx
        其实这个用处不大
        :return:
        """
        output = {}
        well_idx = idx
        for i in range(len(self.wells_name)):
            if well_idx - self.wells_size[i] < 0:
                output = {'well': self.wells_name[i], "idx": well_idx}
                break
            else:
                well_idx -= self.wells_size[i]

        return output


def setup_dataloaders(h5filepath: str,
                      label_classes: list,
                      batch_size: int,
                      num_workers=0,
                      shuffle=True,
                      which_wells=None,
                      noise=False):
    """
    Prepare datasets for training, validation and test.
    """

    def _worker_init_fn(worker_id):
        """
        Worker init fn to fix the seed of the workers
        用来固定数据加载过程中的线程随机数种子的
        """

        seed = torch.initial_seed() % 2 ** 32 + worker_id  # worker_id 可以不加，每个epoch都不一样，**优先级很高的
        np.random.seed(seed)
        random.seed(seed)

    dataset = MyDataSet(h5filepath, label_classes, which_wells, noise)
    sampler = None  # 这个是在多GPU上用的，我没搞多GPU
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        sampler=sampler,
        drop_last=False
    )

    return dataset, loader  # 数据集获取成功，外面一般用的都是 train_loader
