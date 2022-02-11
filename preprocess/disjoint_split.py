import pandas as pd
from rich.console import Console
from skmultilearn.model_selection import iterative_train_test_split
import seaborn as sns
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


p = ArgumentParser(description="Split a dataset into histogram balanced train / test split")
p.add_argument("-c", "--csv", type=str, help="csv file containing label, image and identity columns", required=True)
p.add_argument("-t", "--train", type=str, help="train csv file path to save", required=True)
p.add_argument("-v", "--valid", type=str, help="valid csv file path to save", required=True)
p.add_argument("-s", "--split", default=0.15, type=float, help="percentage of identities to use for test")

args = p.parse_args()


def main():
    
    assert args.split <=1.0 and args.split >= 0.0, "split must be between 0 and 1"
    df = pd.read_csv(args.csv, skipinitialspace=True)
    # test_size = int(args.split * df.shape[0])
    unique_id = df["identity"].unique()
    unique_lab = df["label"].unique()
    assert len(unique_lab) == len(unique_id), "labels and identities must have the same number of unique values"
    
    Console().rule(title=f'[color(50)]has [bold cyan]{len(unique_id)} [color(200)] identities', characters='-', style='bold magenta')

    identities_for_test = int(args.split * len(unique_id))
    indices_for_test = np.arange(identities_for_test)
    
    # shuffle the unqiue ids
    unique_id = np.random.permutation(unique_id)
    test_ids = unique_id[indices_for_test]    
    df_test = df[df["identity"].isin(test_ids)]
    df_train = df[~df["identity"].isin(test_ids)]
    df_test.to_csv(args.valid, index=False)
    df_train.to_csv(args.train, index=False)
    
    Console().rule(title=f'[bold cyan]Completed [bold green]saving csvs', characters='-', style='bold yellow')




if __name__ == "__main__":
    main()