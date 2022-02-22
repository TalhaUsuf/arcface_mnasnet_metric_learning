from hashlib import new
import numpy as np
from rich.console import Console
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

arg = ArgumentParser()
arg.add_argument('-f', '--file', default='valid.csv' , help='csv file which needs to be balanced')
p = arg.parse_args()


def main():
    with open(p.file, "r") as f:
        data = f.readlines()

    imgs = np.array([k.strip().split(',')[0] for k in data])
    labels = np.array([int(k.strip().split(',')[-1]) for k in data])
    
    # unique identities
    unique = np.unique(labels)
    Console().log(f"found {len(unique)} unique labels")
    # get 10 images for each of the 
    image = np.array(Image.open(imgs[-1]), dtype=np.float32)
    Console().log(f"image shape: {image.shape}")
    
    Console().log(f"SIZE IN MOMORY ===> {image.itemsize} bytes ---> {image.itemsize * 0.000001} MB", style="green on black")
    # select 10 images per identity
    PER_ID_IMGS = 10

    new_split = {}
    for id in tqdm(unique, colour="magenta", desc="ID:"): 
        idxs = np.where(id == labels)
        selected_labels = labels[idxs][:PER_ID_IMGS]
        selected_imgs = imgs[idxs][:PER_ID_IMGS]
        new_split.setdefault('image', []).extend(selected_imgs)
        new_split.setdefault('label', []).extend(selected_labels)
    
    pd.DataFrame.from_dict(new_split).to_csv('balanced.csv', index=False)
    
    
    
if __name__ == '__main__':
    main()