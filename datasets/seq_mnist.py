import os
import numpy as np
import cv2
import torch
import torchvision
import math
import colorsys
from skimage.transform import resize
from copy import deepcopy
from utils.misc import get_RTmat
from utils.misc import freq_to_wave


class SequentialMNIST():
    # Rotate around z axis only.

    default_active_actions = [0, 1, 2]

    def __init__(
            self,
            root,
            train=True,
            transforms=torchvision.transforms.ToTensor(),
            T=3,
            max_angle_velocity_ratio=[-0.5, 0.5],
            max_angle_accl_ratio=[-0.0, 0.0],
            max_color_velocity_ratio=[-0.5, 0.5],
            max_color_accl_ratio=[-0.0, 0.0],
            max_pos=[-10, 10],
            max_trans_accl=[-0.0, 0.0],
            label=False,
            label_velo=False,
            label_accl=False,
            active_actions=None,
            max_T=9,
            only_use_digit4=False,
            backgrnd=False,
            shared_transition=False,
            color_off=False,
            rng=None
    ):
        self.T = T
        self.max_T = max_T
        self.rng = rng if rng is not None else np.random
        self.transforms = transforms
        self.data = torchvision.datasets.MNIST(root, train, download=True)
        self.angle_velocity_range = (-max_angle_velocity_ratio, max_angle_velocity_ratio) if isinstance(
            max_angle_velocity_ratio, (int, float)) else max_angle_velocity_ratio
        self.color_velocity_range = (-max_color_velocity_ratio, max_color_velocity_ratio) if isinstance(
            max_color_velocity_ratio, (int, float)) else max_color_velocity_ratio
        self.angle_accl_range = (-max_angle_accl_ratio, max_angle_accl_ratio) if isinstance(
            max_angle_accl_ratio, (int, float)) else max_angle_accl_ratio
        self.color_accl_range = (-max_color_accl_ratio, max_color_accl_ratio) if isinstance(
            max_color_accl_ratio, (int, float)) else max_color_accl_ratio
        self.color_off = color_off

        self.max_pos = max_pos
        self.max_trans_accl = max_trans_accl
        self.label = label
        self.label_velo = label_velo
        self.label_accl = label_accl
        self.active_actions = self.default_active_actions if active_actions is None else active_actions
        if backgrnd:
            print("""
                  =============
                  background ON
                  =============
                  """)
            fname = "MNIST/train_dat.pt" if train else "MNIST/test_dat.pt"
            self.backgrnd_data = torch.load(os.path.join(root, fname))

        if only_use_digit4:
            datas = []
            for pair in self.data:
                if pair[1] == 4:
                    datas.append(pair)
            self.data = datas
        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):
        self.angles_v = self.rng.uniform(math.pi * self.angle_velocity_range[0],
                                         math.pi * self.angle_velocity_range[1], size=1)
        self.angles_a = self.rng.uniform(math.pi * self.angle_accl_range[0],
                                         math.pi * self.angle_accl_range[1], size=1)
        self.color_v = 0.5 * self.rng.uniform(self.color_velocity_range[0],
                                              self.color_velocity_range[1], size=1)
        self.color_a = 0.5 * \
            self.rng.uniform(
                self.color_accl_range[0], self.color_accl_range[1], size=1)
        pos0 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        pos1 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        self.pos_v = (pos1-pos0)/(self.max_T - 1)
        self.pos_a = self.rng.uniform(
            self.max_trans_accl[0], self.max_trans_accl[1], size=[2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image = np.array(self.data[i][0], np.float32).reshape(28, 28)
        image = resize(image, [24, 24])
        image = cv2.copyMakeBorder(
            image, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        angles_0 = self.rng.uniform(0, 2 * math.pi, size=1)
        color_0 = self.rng.uniform(0, 1, size=1)
        pos0 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        pos1 = self.rng.uniform(self.max_pos[0], self.max_pos[1], size=[2])
        if self.shared_transition:
            (angles_v, angles_a) = (self.angles_v, self.angles_a)
            (color_v, color_a) = (self.color_v, self.color_a)
            (pos_v, pos_a) = (self.pos_v, self.pos_a)
        else:
            angles_v = self.rng.uniform(math.pi * self.angle_velocity_range[0],
                                        math.pi * self.angle_velocity_range[1], size=1)
            angles_a = self.rng.uniform(math.pi * self.angle_accl_range[0],
                                        math.pi * self.angle_accl_range[1], size=1)
            color_v = 0.5 * self.rng.uniform(self.color_velocity_range[0],
                                             self.color_velocity_range[1], size=1)
            color_a = 0.5 * \
                self.rng.uniform(
                    self.color_accl_range[0], self.color_accl_range[1], size=1)
            pos_v = (pos1-pos0)/(self.max_T - 1)
            pos_a = self.rng.uniform(
                self.max_trans_accl[0], self.max_trans_accl[1], size=[2])
        images = []
        for t in range(self.T):
            angles_t = (0.5 * angles_a * t**2 + angles_v * t +
                        angles_0) if 0 in self.active_actions else angles_0
            color_t = ((0.5 * color_a * t**2 + t * color_v + color_0) %
                       1) if 1 in self.active_actions else color_0
            pos_t = (0.5 * pos_a * t**2 + pos_v * t +
                     pos0) if 2 in self.active_actions else pos0
            mat = get_RTmat(0, 0, float(angles_t), 32, 32, pos_t[0], pos_t[1])
            _image = cv2.warpPerspective(image.copy(), mat, (32, 32))

            rgb = np.asarray(colorsys.hsv_to_rgb(
                color_t, 1, 1), dtype=np.float32)
            _image = np.concatenate(
                [_image[:, :, None]] * 3, axis=-1) * rgb[None, None]
            _image = _image / 255.

            if hasattr(self, 'backgrnd_data'):
                _imagemask = (np.sum(_image, axis=2, keepdims=True) < 3e-1)
                _image = torch.tensor(
                    _image) + self.backgrnd_data[i].permute([1, 2, 0]) * (_imagemask)
                _image = np.array(torch.clip(_image, max=1.))

            images.append(self.transforms(_image.astype(np.float32)))

        if self.label or self.label_velo:
            ret = [images]
            if self.label:
                ret += [self.data[i][1]]
            if self.label_velo:
                ret += [
                    freq_to_wave(angles_v.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_v).astype(np.float32)),
                    pos_v.astype(np.float32)
                ]
            if self.label_accl:
                ret += [
                    freq_to_wave(angles_a.astype(np.float32)),
                    freq_to_wave((2 * math.pi * color_a).astype(np.float32)),
                    pos_a.astype(np.float32)
                ]
            return ret
        else:
            return images
