import dask.dataframe as dd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-d', "--dir", type=str, default="./preprocess/dataset.csv", help="Path to dir containing the dataset.csv file")
parser.add_argument('-c', "--csv", type=str, default="./preprocess/cleaned_dataset.csv", help="Path to save the cleaned csv file")
p = parser.parse_args()

df = dd.read_csv(p.dir)
part = df.partitions[0]

df = df.loc[df["identity"]!='identity'].compute()
df.to_csv(p.csv, index=False)