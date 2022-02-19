### Deep learning project seed
Use this seed to start new deep learning / ML projects.

- Built in setup.py
- Built in requirements
- Examples with MNIST
- Badges
- Bibtex

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT  
 
---

<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/deep-learning-project-template

# install project   
cd deep-learning-project-template 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd project

# run module (example: mnist as your main contribution)   
python lit_classifier_main.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   

# Pre-processing


## Make a csv file efficiently 💡


```bash
python preprocess/make_txt_files.py --d  /home/talha/metric_learning/glint360k_unpacked --c dataset.csv
```


## Split efficiently


```bash
python preprocess/split.py --split_valid  --csv dataset.csv
```



## Dataset Dir.  📃

|Dataset|Dir. |
|:---|:---|
|  Glint360k processed dataset | `/home/talha/metric_learning/glint360k_unpacked`  |
|vggFace2 dataset|`/home/talha/metric_learning/train`|
  

## Image operations :hammer: 🗡️ 🔧

 - Face detector
 - face alignmment
 - resize to `224x224`
  
> code is on Hasnain PC :tv:

## Make a csv

Here `train` has `vggface2` datset :green_book:  ⚠️ 

```bash
python preprocess/generate_csv.py --c project/dataset.csv --d /home/talha/metric_learning/train
```

flag options: 🇵🇰
```
flags:

generate_csv.py:
  --c: Path to save the csv file
    (default: './dataset.csv')
  --d: Path to vggface2 dataset
    (default: 'images')
```

## Combine the csv files

> ⚠️ in case of `glint360k` dataset, `generate_csv.py` was able to generate individual csv files ☑️ but stuck ⁉️ on combining them. In that case run below script:

fast 🚥  way to combine the individual csv files

```
python preprocess/combine_csvs.py
```
It will create combined csv file `dataset.csv` inside the `preprocess` dir.


FLAGs are defined below
```
optional arguments:
  -h, --help  show this help message and exit
  --d D       Path to dir containing the individual identitiy csv files
  --c C       Path to save the dataset csv file
```

**More faster :zap: implementation USING DASK**

```
python preprocess/combine_csvs_dask.py
```

## Clean the csv files

After avove operation there will be rows with `image, identity, label` in between the csvs, Remove those rows using this.

```
python preprocess/clean_rows.py --dir ./preprocess/dataset.csv  --csv ./preprocess/cleaned_dataset.csv
```

FLAGS 🇵🇰
```
optional arguments:
  -h, --help         show this help message and exit
  -d DIR, --dir DIR  Path to dir containing the dataset.csv file
  -c CSV, --csv CSV  Path to save the cleaned csv file
```

## Split the csv

Split the csv 🗒️ into **disjoint** ⚔️ datset csv files. This means train.csv will have different identities than valid.csv

```bash
python preprocess/disjoint_split.py  -c project/dataset.csv -t project/train.csv -v project/valid.csv -s 0.20
```

Flags 🇵🇰

```
optional arguments:
  -h, --help            show this help message and exit
  -c CSV, --csv CSV     csv file containing label, image and identity columns
  -t TRAIN, --train TRAIN
                        train csv file path to save
  -v VALID, --valid VALID
                        valid csv file path to save
  -s SPLIT, --split SPLIT
                        percentage of identities to use for test
```


