import numpy as np
import torch
import torchvision
from collections import OrderedDict
import os
import numpy as np


_FACTORS_IN_ORDER = ['category', 'instance',
                     'lighting', 'elevation', 'azimuth']
_ELEV_V = [30, 35, 40, 45, 50, 55, 60, 65, 70]
_AZIM_V = np.arange(0, 350, 20)
assert len(_AZIM_V) == 18
_NUM_VALUES_PER_FACTOR = OrderedDict(
    {'category': 5, 'instance': 5, 'lighting': 6, 'elevation': 9, 'azimuth': 18})


def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
    indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


class SmallNORBDataset(object):

    default_active_actions = [3,4]

    def __init__(self, root,
                 train=True,
                 T=3,
                 label=False,
                 label_velo=False,
                 force_moving=False,
                 active_actions=None,
                 transforms=torchvision.transforms.ToTensor(),
                 shared_transition=False,
                 rng=None):
        assert T <= 6
        self.data = torch.load(os.path.join(
            root, 'smallNORB/train.pt' if train else 'smallNORB/test.pt'))
        self.label = label
        self.label_velo = label_velo
        print(self.data.shape)
        self.T = T
        self.transforms = transforms
        self.active_actions = self.default_active_actions if active_actions is None else active_actions
        self.force_moving = force_moving
        self.rng = rng if rng is not None else np.random
        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):
        self.vs = {}
        for kv in _NUM_VALUES_PER_FACTOR.items():
            key, value = kv[0], kv[1]
            self.vs[key] = self.gen_v(value)

    def __len__(self):
        return 5000 

    def gen_pos(self, max_n, v):
        _x = np.abs(v) * (self.T-1)
        if v < 0:
            return self.rng.randint(_x, max_n)
        else:
            return self.rng.randint(0, max_n-_x)
    
    def gen_v(self, max_n):
        v = self.rng.randint(1 if self.force_moving else 0, max_n//self.T + 1)
        if self.rng.uniform() > 0.5:
            v = -v
        return v

    def gen_factors(self):
        # initial state
        p_and_v_list = []
        sampled_indices = []
        for action_index, kv in enumerate(_NUM_VALUES_PER_FACTOR.items()):
            key, value = kv[0], kv[1]
            if key == 'category' or key == 'instance' or key == 'lighting':
                p_and_v_list.append([0, 0])
                index = self.rng.randint(0, _NUM_VALUES_PER_FACTOR[key])
                sampled_indices.append([index]*self.T)
            else:
                if not(action_index in self.active_actions):
                    v = 0
                else:
                    if self.shared_transition:
                        v = self.vs[key]
                    else:
                        v = self.gen_v(value)
                p = self.gen_pos(value, v)
                p_and_v_list.append((p, v))
                indices = [p + t * v for t in range(self.T)] 
                sampled_indices.append(indices)
        #print(p_and_v_list)
        return np.array(p_and_v_list, dtype=np.uint8), np.array(sampled_indices, dtype=np.uint8).T

    def __getitem__(self, i):
        p_and_v_list, sample_indices = self.gen_factors()
        imgs = []
        for t in range(self.T):
            ind = sample_indices[t]
            img = self.data[ind[0], ind[1], ind[2], ind[3], ind[4]]
            img = img/255.
            img = self.transforms(img[:, :, None])
            imgs.append(img)
        if self.T == 1:
            imgs = imgs[0]

        if self.label or self.label_velo:
            ret = [imgs]
            if self.label:
                ret += [sample_indices[0][0]]
            if self.label_velo:
                ret += [p_and_v_list[3][1][None], p_and_v_list[4][1][None]]
                
            return ret
        else:
            return imgs
