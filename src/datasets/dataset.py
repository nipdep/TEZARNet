import numpy as np
from torch.utils.data import Dataset
import torch
import random

# build mock dataloader
class PAMAP2Dataset(Dataset):
    def __init__(self, data, actions, attributes, action_feats, action_classes, seq_len=120):
        super(PAMAP2Dataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.actions = actions
        self.attributes = torch.from_numpy(attributes)
        self.action_feats = torch.from_numpy(action_feats)
        self.target_feat = torch.from_numpy(action_feats[action_classes, :])
        self.seq_len = seq_len
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind, ...]
        x_mask = np.array([0]) #self.padding_mask[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        y_feat = self.action_feats[target, ...]
        attr = self.attributes[target, ...]
        return x, y, y_feat, attr, x_mask

    def __len__(self):
        return self.data.shape[0]

class DaLiAcDataset(Dataset):
    def __init__(self, data, actions, attributes, action_feats, action_classes, seq_len=120):
        super(DaLiAcDataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.actions = actions
        self.attributes = torch.from_numpy(attributes)
        self.action_feats = torch.from_numpy(action_feats)
        self.target_feat = torch.from_numpy(action_feats[action_classes, :])
        self.seq_len = seq_len
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind, ...]
        x_mask = np.array([0]) #self.padding_mask[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        y_feat = self.action_feats[target, ...]
        attr = self.attributes[target, ...]
        return x, y, y_feat, attr, x_mask

    def __len__(self):
        return len(self.data)


class UTDDataset(Dataset):
    def __init__(self, data, actions, attributes, action_feats, action_classes, seq_len=120):
        super(UTDDataset, self).__init__()
        self.data = data
        self.actions = actions
        self.attributes = torch.from_numpy(attributes)
        self.action_feats = torch.from_numpy(action_feats)
        self.target_feat = torch.from_numpy(action_feats[action_classes, :])
        self.seq_len = seq_len
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind]
        x_len = x.shape[0]
        if x_len > self.seq_len:
            randIndex = random.sample(range(x_len), self.seq_len)
            randIndex.sort()
            x = x[randIndex, :]
        else:
            print(x_len)
        x_mask = np.array([0]) #self.padding_mask[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        y_feat = self.action_feats[target, ...]
        attr = self.attributes[target, ...]
        return torch.from_numpy(x), y, y_feat, attr, x_mask

    def __len__(self):
        return len(self.data)

class OPPDataset(Dataset):
    def __init__(self, data, actions, attributes, action_feats, action_classes, seq_len=120):
        super(OPPDataset, self).__init__()
        cols = list(range(4, 9))+list(range(16,18))+list(range(22,34))+list(range(37,133))
        self.data = torch.from_numpy(data)#[:, :, cols] # get only subject related IMU features
        self.actions = actions
        self.attributes = torch.from_numpy(attributes)
        self.action_feats = torch.from_numpy(action_feats)
        self.target_feat = torch.from_numpy(action_feats[action_classes, :])
        self.seq_len = seq_len
        # build action to id mapping dict
        self.n_action = len(self.actions)
        self.action2Id = dict(zip(action_classes, range(self.n_action)))

    def __getitem__(self, ind):
        x = self.data[ind, ...]
        x_mask = np.array([0]) #self.padding_mask[ind, ...]
        target = self.actions[ind]
        y = torch.from_numpy(np.array([self.action2Id[target]]))
        y_feat = self.action_feats[target, ...]
        attr = self.attributes[target, ...]
        return x, y, y_feat, attr, x_mask

    def __len__(self):
        return self.data.shape[0]