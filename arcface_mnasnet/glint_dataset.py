from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
from pytorch_lightning import LightningDataModule, seed_everything
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import os
from rich.console import Console
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from rich.table import Table 
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


tqdm.pandas()
seed_everything(10)



class glint360k_dataset(Dataset):
    '''
    load a single image and label from glint360K dataset in the same fashion as MNIST dataset

    '''    
    def __init__(self, df, label_encoder, transforms):
        self.df = df
        self.label_encoder = label_encoder
        self.transforms = transforms
        self.df["label"] = self.label_encoder.transform(self.df["identity"])
    def __getitem__(self, idx):
        '''
        returns a single image, label , each as tensor

        Parameters
        ----------
        idx : int
            index of the sample pair to be returned

        Returns
        -------
        img : torch.Tensor
            torch.float32 tensor of shape [C, H, W]
        label : torch.Tensor
            torch.int64 tensor of shape [1,]
        '''
        current_row = self.df.iloc[idx, :]        
        img = cv2.imread(str(current_row["image"]))
        assert img.size > 0, "image not read correctly"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            img = self.transforms(img).type(torch.float32) # [C, H, W]
        else:
            # if transforms is None ---> convert image to tensor
            img = transforms.ToTensorv2(img).type(torch.float32) # [C, H, W]
        label = current_row["identity"]
        label = torch.tensor(current_row["label"], dtype=torch.long)

        return img, label

    def __len__(self):
        return self.df.shape[0]

class glint360k_DM(LightningDataModule):
    def __init__(self, train_trf, val_trf, dataset_dir):
        super(glint360k_DM, self).__init__()
        self.train_trf = train_trf
        self.val_trf = val_trf
        self.dataset_dir = dataset_dir
        self.csv = "glint360k_dataset.csv"
        self.le = LabelEncoder()
        Console().rule(title=f"[red]reading from data-dir ----> [cyan]{self.dataset_dir}", style="bold green")
    def prepare_data(self):
            pass
        # idn = [j for j in os.listdir(self.dataset_dir)]
        # for_csv = {}
        # for identity in tqdm(idn, desc="IDENTITY", colour="magenta", leave=True):
        #     path = list(Path(self.dataset_dir, identity).iterdir())
        #     for image in tqdm(path, desc="IMAGE", leave=False, colour="green"):
        #         for_csv.setdefault("image", []).append(image.as_posix())
        #         for_csv.setdefault("identity", []).append(identity)
        # df = pd.DataFrame.from_dict(for_csv)
        # df.to_csv(self.csv, index=False)
        
    def setup(self, stage=None):
        # pathlib use here is very slow
        # _all_identities = [k for k in Path(self.dataset_dir).iterdir() if k.is_dir()]
        df = pd.read_csv(self.csv, skipinitialspace=True)
        df = df.sample(frac=1).reset_index(drop=True) # shuffle the dataset

        labels = df["identity"].tolist()
        self.le.fit(labels)
        df["label"] = df["identity"]

        self.train, self.test = train_test_split(df, test_size=0.05)
        tab = Table(title="[green]GLINT360K DATASET SPLITS", title_style="bold green", style="cyan on black", show_lines=True)
        tab.add_column("SPLIT", style="yellow")
        tab.add_column("No. of Images", style="magenta")
        
        
        tab.add_row("train", f"{len(self.train)}")
        tab.add_row("test", f"{len(self.test)}")
        Console().print(tab)
        
        
        if stage == "validate" or stage == None:
            self.testdataset = glint360k_dataset(self.test, self.le, self.val_trf)
    def val_dataloader(self):
        return DataLoader(self.testdataset, batch_size=10, shuffle=True, num_workers=8)
        # ==========================================================================
        #                             split the data_into_train and val                                  
        # ==========================================================================
        # self.train_set = glint360k_dataset(self.dataset_dir, self.train_trf)
        # self.val_set = glint360k_dataset(self.dataset_dir, self.val_trf)


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)



dm = glint360k_DM(train_trf = None, val_trf = transform, dataset_dir="glint360k_unpacked")
# dm.prepare_data()
dm.setup()
dl = dm.val_dataloader()
batch = iter(dl).next()
print(batch[0].shape, batch[0].dtype)
print(batch[1], batch[1].dtype)

joblib.dump(batch[0], "batch_X.pkl")
joblib.dump(batch[1], "batch_Y.pkl")