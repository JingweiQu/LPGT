# ========================================================================================
# AMIL class
# ========================================================================================


import json
import os
import random
import numpy as np
from PIL import Image
from itertools import groupby
from operator import itemgetter
from torchvision import transforms

from utils.config import cfg


ann_path = cfg.AMIL.ROOT_DIR + "/ImageAnnotation"
layout_path = cfg.AMIL.ROOT_DIR + "/Layout"
image_path = cfg.AMIL.ROOT_DIR + "/PNGImage-P"

sets_translation_dict = dict(train="trn", test="test")


class AMIL:
    def __init__(self, sets, img_resize):
        """
        :param sets: 'train' or 'test'
        :param img_resize: resized object size
        """
        self.sets = sets_translation_dict[sets]
        self.ann_files = open(os.path.join(layout_path, self.sets + ".txt"), "r").read().split("\n")
        self.ann_files = self.ann_files[: len(self.ann_files) - 1]
        self.ann_path = ann_path
        self.image_path = image_path
        self.classes = cfg.AMIL.CLASSES
        self.classes.sort()
        self.img_resize = img_resize
        self.ann_files_cls_dict = {cls: list(filter(lambda x: x.endswith(f':{cls}'), self.ann_files)) for cls in self.classes}
        self.total_size = len(self.ann_files)
        self.size_by_cls = {cls: len(ann_list) for cls, ann_list in self.ann_files_cls_dict.items()}
        self.trans = transforms.Compose([transforms.GaussianBlur(kernel_size=5, sigma=2.0)])

    def get_1_sample(self, idx, cls=None):
        """
        :param idx: Index of datapoint to sample, None for random sampling
        :param cls: None for random class, or specify for a certain set
        """
        if cls is None:
            ann_files = self.ann_files
        elif type(cls) == int:
            cls = self.classes[cls]
            ann_files = self.ann_files_cls_dict[cls]
        else:
            assert type(cls) == str
            ann_files = self.ann_files_cls_dict[cls]

        assert len(ann_files) > 0
        if idx is None:
            file_name = random.choice(ann_files)
        else:
            file_name = ann_files[idx]
        file_name, category = file_name.split(':')
        ann_file = file_name + '.json'
        with open(os.path.join(self.ann_path, category, ann_file)) as f:
            annotation = json.load(f)

        if cls is not None:
            assert cls == category

        sorted_data = sorted(annotation['shapes'], key=itemgetter('label'))
        grouped_data = {key: list(group) for key, group in groupby(sorted_data, key=itemgetter('label'))}

        sorted_text = sorted(grouped_data['Text'], key=itemgetter('group_id'))
        gt_label_bboxes = np.array(list(map(itemgetter('points'), sorted_text))).astype(float)
        x1_y1 = gt_label_bboxes[:, 0, :]
        x2_y2 = gt_label_bboxes[:, 1, :]
        x1 = np.minimum(x1_y1[:, 0], x2_y2[:, 0])
        x2 = np.maximum(x1_y1[:, 0], x2_y2[:, 0])
        y1 = np.minimum(x1_y1[:, 1], x2_y2[:, 1])
        y2 = np.maximum(x1_y1[:, 1], x2_y2[:, 1])
        lbb_left_top = np.column_stack((x1, y1))
        lbb_right_bottom = np.column_stack((x2, y2))
        gt_label_pos = (lbb_right_bottom + lbb_left_top) / 2
        label_sizes = lbb_right_bottom - lbb_left_top

        sorted_anchor = sorted(grouped_data['Anchor'], key=itemgetter('group_id'))
        if len(sorted_text) != len(sorted_anchor):
            anchors = []
            grouped_anchor = {key: list(group) for key, group in groupby(sorted_anchor, key=itemgetter('group_id'))}
            grouped_anchor = list(grouped_anchor.values())

            for dict_list in grouped_anchor:
                if len(dict_list) == 1:
                    anchors.append(np.array(dict_list[0]['points'][0]))
                else:
                    multi_anchors = np.array(list(map(itemgetter('points'), dict_list)))[:, 0, :]
                    group_id = dict_list[0]['group_id']
                    label_center = gt_label_pos[group_id - 1]
                    nearest_anchor = self.find_nearest_anchor(label_center, multi_anchors)
                    anchors.append(nearest_anchor)
            anchors = np.array(anchors)
        else:
            anchors = np.array(list(map(itemgetter('points'), sorted_anchor)))[:, 0, :]

        img, im_size = self.get_image(os.path.join(self.image_path, category, file_name + '-P.png'))

        gt_label_pos_norm = self.normalize_points(gt_label_pos, im_size)
        label_sizes_norm = self.normalize_points(label_sizes, im_size)
        anchors_norm = self.normalize_points(anchors, im_size)

        gt_disp_norm = gt_label_pos_norm - anchors_norm

        L_pck = np.max(lbb_right_bottom.max(0) - lbb_left_top.min(0))

        anno_dict = dict(gt_label_pos_norm=gt_label_pos_norm,
                         anchors_norm=anchors_norm,
                         gt_disp_norm=gt_disp_norm,
                         label_sizes_norm=label_sizes_norm,
                         im_size=im_size,
                         L_pck=L_pck,
                         image=img)

        return anno_dict

    def get_image(self, img_path):
        with Image.open(str(img_path)) as img:
            w, h = img.size
            img = self.trans(img)
            img = img.resize(self.img_resize, resample=Image.BICUBIC)   # self.img_resize: (256, 256)

        return img, np.array([w, h], dtype=float)

    def normalize_points(self, points, im_size):
        # normalize by image size
        w, h = im_size[0], im_size[1]
        points_norm = points.copy()
        points_norm[:, 0] /= w
        points_norm[:, 1] /= h

        return points_norm

    def find_nearest_anchor(self, label_point, anchors):
        distances = np.linalg.norm(anchors - label_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_anchor = anchors[nearest_index]
        return nearest_anchor
