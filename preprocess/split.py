
import numpy as np
from PIL import Image
import pyxis as px
from rich.console import Console
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser(description="splits the csv files efficiently")
parser.add_argument('-c', "--csv", type=str, default='dataset.csv', help="csv file containing whole dataset which needs to be split")
parser.add_argument('-t', "--train", type=str, default='train.csv', help="path to save the train split of csv file")
parser.add_argument('-v', "--valid", type=str, default='valid.csv', help="path to save the valid split of csv file")
parser.add_argument('--split_valid', action='store_true')
# parser.add_argument('--split-train', action='store_true')
p = parser.parse_args()

# X = np.outer(np.arange(1, nb_samples + 1, dtype=np.uint8), np.arange(1, 4 + 1, dtype=np.uint8))
# y = np.arange(nb_samples, dtype=np.uint8)


with open("dataset.csv", "r") as f:
    data = f.readlines()

labels = np.array([int(l.strip().split(",")[-1]) for l in data[1:]]).reshape(-1,1)

unique = np.unique(labels)

ts_split = 0.05

ts_split = 1.0 - ts_split

idx = int(ts_split * len(unique))

# Console().print(f"{idx} / {len(unique)}")

if p.split_valid:
        
    x = np.where(labels>=idx)
    name = 'valid.csv'
else :
    x = np.where(labels<idx)
    name = 'train.csv'
    # print(x)


imgs = np.array([str(l.strip().split(",")[0]) for l in data[1:]]).reshape(-1,1)
identities = np.array([str(l.strip().split(",")[1]) for l in data[1:]])
# print(imgs[x])
Console().print(labels[x])

Console().print(imgs[x].shape)
Console().print(labels[x].shape)

to_write = np.concatenate([imgs[x].reshape(-1,1), labels[x].reshape(-1,1)], axis=1)
np.savetxt(f"{name}", to_write, delimiter=",", fmt='%s')
