import os
import numpy as np
import torch
import h5py
import math
import random
from talkingface.data.dataset.dataset import Dataset

class starganDataset(Dataset):
    def __init__(self, config, datasplit):
        super().__init__(config, datasplit)
        self.config = config
        self.spk_list = sorted(os.listdir(config['dst2']))
        self.feat_dirs = [os.path.join(config['dst2'],spk) for spk in self.spk_list]
        self.filenames_all = [[os.path.join(d,t) for t in sorted(os.listdir(d))] for d in self.feat_dirs]
        self.n_domain = len(self.feat_dirs)
        
    def __len__(self):
        return min(len(f) for f in self.filenames_all)

    def __getitem__(self, idx):
        melspec_list = []
        for d in range(self.n_domain):
            with h5py.File(self.filenames_all[d][idx], "r") as f:
                melspec = f["melspec"][()]  # n_freq x n_time
            melspec_list.append(melspec)
        return melspec_list

def collate_fn(batch):
    #batch[b][s]: melspec (n_freq x n_frame)
    #b: batch size
    #s: speaker ID
    
    batchsize = len(batch)
    n_spk = len(batch[0])
    melspec_list = [[batch[b][s] for b in range(batchsize)] for s in range(n_spk)]
    #melspec_list[s][b]: melspec (n_freq x n_frame)
    #s: speaker ID
    #b: batch size

    n_freq = melspec_list[0][0].shape[0]

    X_list = []
    for s in range(n_spk):
        maxlen=0
        for b in range(batchsize):
            if maxlen<melspec_list[s][b].shape[1]:
                maxlen = melspec_list[s][b].shape[1]
        maxlen = math.ceil(maxlen/4)*4
    
        X = np.zeros((batchsize,n_freq,maxlen))
        for b in range(batchsize):
            melspec = melspec_list[s][b]
            melspec = np.tile(melspec,(1,math.ceil(maxlen/melspec.shape[1])))
            X[b,:,:] = melspec[:,0:maxlen]
        #X = torch.tensor(X)
        X_list.append(X)

    return X_list