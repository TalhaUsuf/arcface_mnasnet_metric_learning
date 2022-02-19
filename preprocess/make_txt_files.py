
from pathlib import Path
from rich.console import Console
from absl import app, flags
from absl.flags import FLAGS
import pandas as pd
from tqdm import tqdm
from time import perf_counter
import os
from itertools import  repeat
from multiprocessing.dummy import Pool


flags.DEFINE_string('d', '/home/talha/metric_learning/glint360k_unpacked', 'Path to glint360K dataset')
flags.DEFINE_string('c', './dataset.csv', 'Path to save the csv file')

def main(argv):
    
    base_dir = Path(FLAGS.d)
    ids = os.listdir(str(base_dir)) # get names of all the folders
    id2label = {v:k for k,v in enumerate(ids)}
    Console().rule(title=f"found [color(magenta)]{len(ids)} identites", style="bold cyan")

    with open(f"{FLAGS.c}", "w") as f:
        # write the header file
        f.write("image,identity,label\n")
        # loop over identities
        for id in tqdm(ids, desc="FOLDER", colour="cyan", leave=True):
            imgs = [k for k in (base_dir / id).iterdir() if k.is_file()] # read only files
            for i in list(imgs):
                f.write(f"{str(i)},{i.parents[0].stem},{id2label[i.parents[0].stem]}\n")    
            # # loop over images in current folder
            # for img in tqdm(os.listdir(str(base_dir/id)), desc="IMG", colour="yellow", leave=False):
            #     # write the image path and label to the csv file
            #     f.write(f"{base_dir/id/img},{id},{id2label[id]}\n")

if __name__ == '__main__':
    app.run(main)