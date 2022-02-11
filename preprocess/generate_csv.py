
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


flags.DEFINE_string('d', 'images', 'Path to vggface2 dataset')
flags.DEFINE_string('c', './dataset.csv', 'Path to save the csv file')


def write_dir_imgs(dir : str, mapping : dict):
    '''
    takes in a path to identity dir. and writes the csv file in tmp folder

    Parameters
    ----------
    dir : str
        identity dir path which contains images from the same identity
    mapping : dict
        map identity string to a specific unique index integer
    ''' 
        
    assert Path("tmp").exists() , "tmp folder does not exist"
    imgs = [k for k in Path(FLAGS.d , dir).iterdir() if k.is_file()]
    # Console().print(imgs)
    d = {}
    for k in tqdm(imgs, colour="green", leave=False):
        d.setdefault("image", []).append(str(k))
        d.setdefault("identity", []).append(str(k.parents[0].stem))
        d.setdefault("label", []).append(mapping[str(k.parents[0].stem)])
    # save to csv with specific identity name.
    pd.DataFrame.from_dict(d).to_csv(f"./tmp/{dir}.csv", index=False)
    
def main(argv):
    
    # get all identities strings
    identities = os.listdir(FLAGS.d)    
    # idn to label mapping
    idn2idx = {v:k for k,v in enumerate(identities)}
    t0 = perf_counter()
    with Pool(8) as p:
        
        p.starmap(write_dir_imgs, zip(identities, repeat(idn2idx)))
    t1 = perf_counter()
    Console().rule(title="DONE", style="bold green", characters="=")

    # ==========================================================================
    #                            Combine all csv files                                  
    # ==========================================================================
    
    csvs = [pd.read_csv(str(f), skipinitialspace=True) for f in Path("./tmp").iterdir() if f.is_file() and f.name.endswith(".csv")]
    Console().rule(title=f'[bold cyan]total [bold green]{len(csvs)} csv files found', characters='-', style='bold yellow')
    pd.concat(csvs).to_csv(FLAGS.c, index=False)
    Console().rule(title=f'[bold cyan]DONE [yellow] file saved at [bold green]{FLAGS.c}', characters='-', style='bold yellow')
    t2 = perf_counter()
    Console().print(f"\n[bold green]Writing individual csv files: [bold cyan]{t1-t0:.2f} [bold green]seconds\n[bold green]joining csv files: [bold cyan]{t2-t1:.2f} [bold green]seconds")
    
if __name__ == '__main__':
    Path("tmp").mkdir(parents=True, exist_ok=True)
    app.run(main)