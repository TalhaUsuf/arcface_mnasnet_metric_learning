from pathlib import Path
from rich.console import Console
from argparse import ArgumentParser
from humanize import intword
from tqdm import trange
import pandas as pd
import os
import numpy as np


p = ArgumentParser(description="cleans the dataset csv files")
p.add_argument('-c', "--csv", type=str, default='valid.csv', help="csv file containing whole dataset which needs to be split")
p.add_argument('-t', "--clean", type=str, default='train_cleaned.csv', help="path to the cleaned csv file")
p = p.parse_args()



def main():
    with open(p.csv, "r") as f:
        data = f.readlines()
    data = [k.strip().split(",")[0] for k in data]
    Console().print(f"there are {intword(len(data))} lines in this csv")
    counts = []
    for k in trange(len(data)):
        reference = os.path.abspath(os.path.join(data[k], os.pardir))
        query = "glint360k_unpacked"
        # Console().print(f"{reference.count(query)}")
        counts.append(reference.count(query))

    counts = np.array(counts)
    indices = np.where(counts > 1)
    
    Console().print(intword(len(indices[0])), "paths need to be [green] filtered")
    data = np.array(data)[indices]
    pd.DataFrame({"wrong_image": data}).to_csv(p.clean, index=False)
    
if __name__ == '__main__':
    main()
