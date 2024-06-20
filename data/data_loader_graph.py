# ========================================================================================
# Dataloader for LPGT
# ========================================================================================


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random

from data.AMIL import AMIL
from utils.config import cfg
from torch_geometric.data import Data, Batch

datasets = {"AMIL": AMIL}


class GMDataset(Dataset):
    def __init__(self, name, length, **args):
        self.name = name
        self.ds = datasets[name](**args)
        self.true_epochs = length is None
        self.length = (self.ds.total_size if self.true_epochs else length)
        if self.true_epochs:
            print(f"Initializing {self.ds.sets}-set with all {self.length} examples.")
        else:
            print(f"Initializing {self.ds.sets}-set. Randomly sampling {self.length} examples.")
        self.img_resize = self.ds.img_resize
        self.classes = self.ds.classes
        self.cls = None
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(cfg.NORM_MEANS, cfg.NORM_STD)])

    def set_cls(self, cls):
        if cls == "none":
            cls = None
        self.cls = cls
        if self.true_epochs:
            self.length = self.ds.total_size if cls is None else self.ds.size_by_cls[cls]

    def build_graphs(self, n: int, n_pad: int = None, edge_pad: int = None):
        A = np.ones((n, n))
        edge_num = int(np.sum(A, axis=(0, 1)))

        idxs = np.nonzero(A)

        if n_pad is None:
            n_pad = n
        if edge_pad is None:
            edge_pad = edge_num
        assert n_pad >= n
        assert edge_pad >= edge_num

        return np.array(idxs, dtype=np.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx if self.true_epochs else None

        anno_dict = self.ds.get_1_sample(idx, cls=self.cls)

        label_sizes_norm = anno_dict['label_sizes_norm']
        anchors_norm = anno_dict['anchors_norm']
        gt_label_pos_norm = anno_dict['gt_label_pos_norm']
        n_label = anchors_norm.shape[0]

        node_features = np.concatenate([anchors_norm, label_sizes_norm], axis=1)

        edge_indices = self.build_graphs(n_label)
        anchors_dis_norm = anchors_norm[edge_indices[0]] - anchors_norm[edge_indices[1]]
        edge_features = np.hstack((anchors_dis_norm, label_sizes_norm[edge_indices[0]], label_sizes_norm[edge_indices[1]]))

        target = np.concatenate([anno_dict['gt_disp_norm'], anchors_norm, label_sizes_norm], axis=1)
        gt_label_dis_norm = gt_label_pos_norm[edge_indices[0]] - gt_label_pos_norm[edge_indices[1]]

        graph = Data(
            x=torch.tensor(node_features).to(torch.float32),
            y=torch.tensor(target).to(torch.float32),
            edge_attr=torch.tensor(edge_features).to(torch.float32),
            edge_y=torch.tensor(gt_label_dis_norm).to(torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.long)
        )

        ret_dict = {
            'n_nodes': torch.tensor(n_label, dtype=torch.long),  # label数量
            'im_sizes': torch.tensor(anno_dict['im_size'], dtype=torch.long),  # 图像大小
            'L_pcks': torch.tensor(anno_dict['L_pck'], dtype=torch.float32),  # pck阈值
            'graphs': graph,
            'images': self.trans(anno_dict['image'])
        }

        return ret_dict


def collate_fn(data: list):
    """
    Create mini-batch data for training.
    :param data: data dict
    :return: mini-batch
    """

    def pad_tensor(inp):
        assert type(inp[0]) == torch.Tensor
        it = iter(inp)
        t = next(it)
        max_shape = list(t.shape)
        while True:
            try:
                t = next(it)
                for i in range(len(max_shape)):
                    max_shape[i] = int(max(max_shape[i], t.shape[i]))
            except StopIteration:
                break
        max_shape = np.array(max_shape)

        padded_ts = []
        for t in inp:
            pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
            pad_pattern[::-2] = max_shape - np.array(t.shape)
            pad_pattern = tuple(pad_pattern.tolist())
            padded_ts.append(F.pad(t, pad_pattern, "constant", 0))

        return padded_ts

    def stack(inp):
        if type(inp[0]) == list:
            ret = []
            for vs in zip(*inp):
                ret.append(stack(vs))
        elif type(inp[0]) == dict:
            ret = {}
            for kvs in zip(*[x.items() for x in inp]):
                ks, vs = zip(*kvs)
                for k in ks:
                    assert k == ks[0], "Key value mismatch."
                ret[k] = stack(vs)
        elif type(inp[0]) == torch.Tensor:
            new_t = pad_tensor(inp)
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == np.ndarray:
            new_t = pad_tensor([torch.from_numpy(x) for x in inp])
            ret = torch.stack(new_t, 0)
        elif type(inp[0]) == str:
            ret = inp
        elif type(inp[0]) == Data:  # Graph from torch.geometric, create a batch
            ret = Batch.from_data_list(inp)
        else:
            raise ValueError("Cannot handle type {}".format(type(inp[0])))
        return ret

    ret = stack(data)
    return ret


def worker_init_fix(worker_id):
    """
    Init dataloader workers with fixed seed.
    """
    random.seed(cfg.RANDOM_SEED + worker_id)
    np.random.seed(cfg.RANDOM_SEED + worker_id)


def worker_init_rand(worker_id):
    """
    Init dataloader workers with torch.initial_seed().
    torch.initial_seed() returns different seeds when called from different dataloader threads.
    """
    random.seed(torch.initial_seed())
    np.random.seed(torch.initial_seed() % 2 ** 32)


def get_dataloader(dataset, batch_size, fix_seed=True, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        worker_init_fn=worker_init_fix if fix_seed else worker_init_rand,
    )
