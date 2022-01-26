from pathlib import Path
import pandas as pd
from rich.console import Console
from absl import app, flags
from absl.flags import FLAGS
from tqdm import tqdm
tqdm.pandas()


flags.DEFINE_string('csv', "arcface_mnasnet/glint360k_dataset.csv", 'csv file named arcface_mnasnet/glint360k_dataset.csv')
flags.DEFINE_string('train_df','arcface_mnasnet/train_df.csv',  help='train_df save path')
flags.DEFINE_string('test_df', 'arcface_mnasnet/test_df.csv', help='test_df save path')




def append_paths(row):
    # Console().print(row)
    return "/home/talha/metric_learning/" + row




def main(argv):
    df = pd.read_csv(FLAGS.csv, skipinitialspace=True)
    Console().print(df.head(6))
    df["image"] = df["image"].progress_apply(append_paths)
    df.to_csv("paths_appended_ds.csv", index=False)


if __name__ == '__main__':
    app.run(main)