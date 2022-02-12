from torch.utils.data import Dataset
from rich.console import Console
import torch
import pandas as pd
import cv2
from PIL import Image

class identities_ds(Dataset):
    def __init__(self, csv:str, transform = None):
        '''
        takes a csv file and makes a pytorch dataset class

        Parameters
        ----------
        csv : str
            csv must have 3 columns image, identity and label 
        transform : [type], 
            transformations to apply on the each image, must be of torchvision type
        '''        
        self.csv = pd.read_csv(csv, skipinitialspace=True, skip_blank_lines=True)
        # bad practice to load into memory wholly but this info. is needed outside the class (by arcface loss classification layer) 
        self.targets = self.csv.iloc[:,-1].values
        self.trf = transform
        
        assert self.trf is not None, "transform is None, there should be atleast resize and normalize transform"
        
        assert "image" in self.csv.columns, "csv must have image column"
        assert "identity" in self.csv.columns, "csv must have identity column"
        assert "label" in self.csv.columns, "csv must have label column"
        
        assert self.csv.columns[0] == "image", "image column must be first"
        Console().rule(title=f"csv passed column exist checks  [color(yellow)] ....", characters="=", style="bold cyan")
        
    def __len__(self):
        return len(self.csv)
        
    def __getitem__(self,idx):
        
        img = Image.open(self.csv.iloc[idx,0]).convert('RGB')
        if img is not None:
            target = self.csv.iloc[idx,-1] # label       
            if self.trf:
                img = self.trf(img) # [C, H, W]
                
                return torch.tensor(img).float(), torch.tensor(target).long()
                