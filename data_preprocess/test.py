import numpy as np

with open("/home/wang/px/vatex_fairseq/Data/bpe_data/train.vid-ench.vid") as f:
    for line in f:
        vid_pth  = line.strip() + ".npy"
        features = np.load(vid_pth)
        print(features.shape)