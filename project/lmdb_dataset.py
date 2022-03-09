from torch.utils.data import Dataset, DataLoader
from rich.console import Console
import torch
import pandas as pd
import numpy as np
import cv2
from time import perf_counter
from pathlib import Path
from PIL import Image
from folder2lmdb import ImageFolderLMDB
from rich.console import Console
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torchvision.utils import make_grid, save_image


trf = transforms.Compose(
                                [
                                    transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
                                ]
                        )  

ds = ImageFolderLMDB(
                        db_path="/home/talha/metric_learning/glint360k_unpacked/val.lmdb",
                        transform=trf, 
                        target_transform=None
                    )

dl = DataLoader(ds, batch_size=200, shuffle=True, num_workers=1)

Path("sample_batches").mkdir(exist_ok=True)

times = []

ds = torch.utils.data.Subset(ds, range(0, len(ds), 5000))
Console().log(f"total dataset [yellow]{len(ds)}", style="green on black")


for k in range(10):
        batch = next(iter(dl))    
        Console().log(f"batch-{k}    label-[yellow]{batch[1].tolist()}", style="red on green")
#     t0 = perf_counter()
#     batch = next(iter(dl))

#     t1 = perf_counter()
#     max = torch.max(batch[0])
#     min = torch.min(batch[0])
#     Console().log(f"max pixel --> {max}")
#     Console().log(f"min pixel --> {min}")
#     save_image(batch[0], fp=f"sample_batches/batch_{k}_grid.jpg", nrow=15, padding=1, pad_value=0)
#     t = t1 - t0
#     times.append(t)

# times = np.mean(times)
# Console().rule(title=f"Dataloading time ---> {times} seconds", style="red on black", characters="-")
