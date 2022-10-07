import numpy as np
import torch
import torchvision
from collections import OrderedDict
import os


_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = OrderedDict({'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15})


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


class ThreeDimShapesDataset(object):
    
    default_active_actions = [0,1,2,3,5]
    
    def __init__(self, root, train=True, T=3, label_velo=False, transforms=torchvision.transforms.ToTensor(),
                 active_actions=None, force_moving=False, shared_transition=False, rng=None):
        assert T <= 8
        self.images = torch.load(os.path.join(root, '3dshapes/images.pt')).astype(np.float32)
        self.label_velo = label_velo
        self.train = train
        self.T = T
        self.transforms = transforms
        self.active_actions = self.default_active_actions if active_actions is None else active_actions
        self.force_moving = force_moving
        self.rng = rng if rng is not None else np.random
        self.shared_transition = shared_transition
        if self.shared_transition:
            self.init_shared_transition_parameters()

    def init_shared_transition_parameters(self):
        vs = {}
        for kv in _NUM_VALUES_PER_FACTOR.items():
            key, value = kv[0], kv[1]
            vs[key] = self.gen_v(value)
        self.vs = vs
    
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
            if key == 'shape':
                p_and_v_list.append([0, 0])
                if self.train:
                    shape = self.rng.choice([0])
                else:
                    shape = self.rng.choice([1,2,3])
                sampled_indices.append([shape]*self.T)
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
        return np.array(p_and_v_list, dtype=np.uint8), np.array(sampled_indices, dtype=np.uint8).T
        
    
    def __getitem__(self, i):
        p_and_v_list, sample_indices = self.gen_factors()
        imgs = []
        for t in range(self.T):
            img = self.images[get_index(sample_indices[t])] / 255.
            img = self.transforms(img)
            imgs.append(img)
        if self.label_velo:
            return imgs, p_and_v_list[0][1][None], p_and_v_list[1][1][None], p_and_v_list[2][1][None], p_and_v_list[3][1][None], p_and_v_list[5][1][None]
        else:
            return imgs

