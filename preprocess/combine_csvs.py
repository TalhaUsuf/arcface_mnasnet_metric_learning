import pandas as pd
import os 
from pathlib import Path
from rich.console import Console
from tqdm import tqdm
from  argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--d", type=str, default="./tmp", help="Path to dir containing the identitiy csv files")
parser.add_argument("--c", type=str, default="./preprocess/dataset.csv", help="Path to save the dataset csv file")
p = parser.parse_args()

csv_files = [k.as_posix() for k in Path(p.d).iterdir() if k.is_file() and k.name.endswith(".csv")]

Console().rule(title=f"Found {len(csv_files)} csv files", style="bold green", characters="=")

CHUNK_SIZE = 50000
csv_file_list = csv_files
output_file = p.c
first_one = True
for csv_file_name in tqdm(csv_file_list, desc="CSV #", colour="cyan"):
    
    if not first_one: # if it is not the first csv file then skip the header row (row 0) of that file
        skip_row = [0]
    else:
        skip_row = []
        
    chunk_container = pd.read_csv(csv_file_name, chunksize=CHUNK_SIZE, skipinitialspace=True)
    for chunk in chunk_container:
            chunk.to_csv(output_file, mode="a", index=False)

    first_one = False