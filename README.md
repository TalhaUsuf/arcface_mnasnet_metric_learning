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


## Make a csv file efficiently ????


```bash
python preprocess/make_txt_files.py --d  /home/talha/metric_learning/glint360k_unpacked --c dataset.csv
```


## Split efficiently


```bash
python preprocess/split.py --split_valid  --csv dataset.csv
```

## sample 10 images per identity

To make the test split fit into memory, 10 images per identity have been sampled

```bash
python preprocess/balance.py 
```

manually remove the header from the resultant csv file. `balanced.csv` file ?????? will be created. 

## Dataset Dir.  ????

|Dataset|Dir. |
|:---|:---|
|  Glint360k processed dataset | `/home/talha/metric_learning/glint360k_unpacked`  |
|vggFace2 dataset|`/home/talha/metric_learning/train`|
  

## Image operations :hammer: ??????? ????

 - Face detector
 - face alignmment
 - resize to `224x224`
  
> code is on Hasnain PC :tv:

## Make a csv

Here `train` has `vggface2` datset :green_book:  ?????? 

```bash
python preprocess/generate_csv.py --c project/dataset.csv --d /home/talha/metric_learning/train
```

flag options: ????????
```
flags:

generate_csv.py:
  --c: Path to save the csv file
    (default: './dataset.csv')
  --d: Path to vggface2 dataset
    (default: 'images')
```

## Combine the csv files

> ?????? in case of `glint360k` dataset, `generate_csv.py` was able to generate individual csv files ?????? but stuck ?????? on combining them. In that case run below script:

fast ????  way to combine the individual csv files

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

FLAGS ????????
```
optional arguments:
  -h, --help         show this help message and exit
  -d DIR, --dir DIR  Path to dir containing the dataset.csv file
  -c CSV, --csv CSV  Path to save the cleaned csv file
```

## Split the csv

Split the csv ??????? into **disjoint** ?????? datset csv files. This means train.csv will have different identities than valid.csv

```bash
python preprocess/disjoint_split.py  -c project/dataset.csv -t project/train.csv -v project/valid.csv -s 0.20
```

Flags ????????

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






# Train


```bash
python project/train.py --embed_sz 256 --batch_size 250 --lr_trunk 0.00001 --lr_embedder 0.001 --lr_arcface 0.0001 --image_size 224  --gpus 2 --strategy ddp --log_every_n_steps 5 --warmup_epochs 3 --check_val_every_n_epoch 1 --precision 16 --amp_backend native --train_ds /home/talha/Downloads/arcface_mnasnet_metric_learning/train.csv  --valid_ds /home/talha/Downloads/arcface_mnasnet_metric_learning/valid.csv   --replace_sampler_ddp False --num_sanity_val_steps 0 --T_0 1 --T_mult 2
```


## Resume training

```bash
python project/train.py --embed_sz 256 --batch_size 250 --lr_trunk 0.00001 --lr_embedder 0.001 --lr_arcface 0.0001 --image_size 224 --gpus 2 --strategy ddp --log_every_n_steps 5 --warmup_epochs 3 --check_val_every_n_epoch 1 --precision 16 --amp_backend native --train_ds /home/talha/Downloads/arcface_mnasnet_metric_learning/train.csv --valid_ds /home/talha/Downloads/arcface_mnasnet_metric_learning/balanced.csv  --replace_sampler_ddp False --num_sanity_val_steps 1 --T_0 1 --T_mult 2 --lfw_dir lfw --ver_batch 16  --pairs_file pairs.txt  --resume_from_checkpoint '/home/talha/Downloads/arcface_mnasnet_metric_learning/arcface_mnasnet_metric_learning-project/2i4p43ar/checkpoints/epoch=3-step=103999-last.ckpt'
```

# Validation

1. Following script will use the mxnet way to perform evaluation, it however is reporting the same accuracy on all three datasets (which is strange ????)


    Use `project/eval_mxnet.py` script ???????.

    ???? **Checkpoint path** has been set inside the script  

    ```bash
    python project/eval_mxnet.py 
    ```
2. Following script will use user provided pairs to perform evaluation. It uses the [pairs.txt](pairs.txt) file to get pairs.

    Pairs are needed in the following format (the trailing `1` or `0` tells if this is a positive pair ??????  or a negative pair ??????)
    ```
    Abel_Pacheco/Abel_Pacheco_0001.jpg Abel_Pacheco/Abel_Pacheco_0004.jpg 1
    Akhmed_Zakayev/Akhmed_Zakayev_0001.jpg Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg 1
    Akhmed_Zakayev/Akhmed_Zakayev_0002.jpg Akhmed_Zakayev/Akhmed_Zakayev_0003.jpg 1
    .
    .
    .
    Abdel_Madi_Shabneh/Abdel_Madi_Shabneh_0001.jpg Mikhail_Gorbachev/Mikhail_Gorbachev_0001.jpg 0
    Abdul_Rahman/Abdul_Rahman_0001.jpg Portia_de_Rossi/Portia_de_Rossi_0001.jpg 0
    Abel_Pacheco/Abel_Pacheco_0001.jpg Jong_Thae_Hwa/Jong_Thae_Hwa_0002.jpg 0
    Abel_Pacheco/Abel_Pacheco_0002.jpg Jean-Francois_Lemounier/Jean-Francois_Lemounier_0001.jpg 0
    ```


# Conversion to tflite

## Openvino docker[onnx ---> IR onnx representation]

Convert to `onnx` using the default way of exporting *pytorch model to onnx*. It works ?????? without error on current **conda environment**.

### Install openvino docker image

```bash
 docker pull openvino/ubuntu18_dev
```
### Running the image

`ONNX` file is located at : 

```/home/talha/Downloads/arcface_mnasnet_metric_learning```

Dir. structure inside `arcface_mnasnet_metric_learning` is shown below:

```bash
.
????????? arcface_mnasnet_metric_learning-project
????????? balanced.csv
????????? data
????????? dataset.csv
????????? events.out.tfevents.1645796091.Talha-System-PC.14468.0
????????? example_logs
????????? example_tensorboard
????????? IR_model
????????? lfw
????????? lfw_funneled
????????? LICENSE
????????? mnasnet_embedder.onnx
????????? pairs.txt
????????? preprocess
????????? project
????????? README.md
????????? requirements.txt
????????? setup.cfg
????????? setup.py
????????? tests
????????? tmp
????????? train.csv
????????? valid.csv
????????? wandb
????????? weights

```


> ????????? Absolute path must be given to mount this dir. volume inside docker


```bash
docker run -it --device /dev/dri:/dev/dri \ 
            --device-cgroup-rule='c 189:* rmw' \
            -v /home/talha/Downloads/arcface_mnasnet_metric_learning:/dev/bus/usb \
            --rm openvino/ubuntu18_dev:latest
```

Now the contents inside dir. `arcface_mnasnet_metric_learning` are shown in `/dev/bus/usb`

### Convert onnx to IR ??????

Now the path to model is given as `/dev/bus/usb`/`*.onnx` because the dir. has been mounted inside docker image


Model IR reporesentation will be saved in `IR_model` which was present in ```/home/talha/Downloads/arcface_mnasnet_metric_learning```.


```bash
python3 deployment_tools/model_optimizer/mo.py  \
          --input_model /dev/bus/usb/mnasnet_embedder.onnx \
          --model_name mnasnet_arcfrace_224_rgb_3_channel_fp32 \
          --output_dir /dev/bus/usb/IR_model/  \
          --input "input[1 3 224 224]"
```

Now IR files are located at ```/home/talha/Downloads/arcface_mnasnet_metric_learning/IR_model``` ????


## openvino to tflite

### install and run docker image

To pull docker image:

```bash
 docker pull ghcr.io/pinto0309/openvino2tensorflow:latest
```


To run docker image, `cd` to the root folder `/home/talha/Downloads/arcface_mnasnet_metric_learning`:

```bash
xhost +local: && \
  docker run -it --rm \
  -v `pwd`:/home/user/workdir \
  -v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
  --device /dev/video0:/dev/video0:mwr \
  --net=host \
  -e LIBVA_DRIVER_NAME=iHD \
  -e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
  -e DISPLAY=$DISPLAY \
  --privileged \
  ghcr.io/pinto0309/openvino2tensorflow:latest
```



### conversion openvino IR --> tf saved model

dir. structure:

```
.
????????? arcface_mnasnet_metric_learning-project
????????? balanced.csv
????????? data
.
.
.
.
.
????????? SAVED_MODEL_TF
???    ????????? saved_model.pb (copy)
???        ????????? assets
???        ????????? model_dynamic_range_quant.tflite
???        ????????? model_float16_quant.tflite
???        ????????? model_float32.tflite
???        ????????? model_full_integer_quant_edgetpu.log
???        ????????? model_full_integer_quant_edgetpu.tflite
???        ????????? model_full_integer_quant.tflite
???        ????????? model_weight_quant.tflite
???        ????????? saved_model.pb
???        ????????? variables
???
.
.
.
????????? weights
```

following will convert the openvino IR model file into tf saved model 

?????? this will change `[N, C, H, W]` to `[N, H, W, C]` **without** adding *transpose , pad, transpose* before every layer and is the only way to 
convert torch model to tflite model so far. 

```bash
openvino2tensorflow --model_path IR_model/mnasnet_arcfrace_224_rgb_3_channel_fp32.xml  \
                    --model_output_path "SAVED_MODEL_TF/saved_model.pb (copy)/" \
                    --output_saved_model
```

### conversion tf saved model --> tflite


?????? `SAVED_MODEL_TF/saved_model.pb (copy)/` is a dir. as shown above.
 This dir. already contains the *tf saved model* from above step. 

Following will take the tf saved model and IR file to convert to float32 tflite model.
```bash
openvino2tensorflow --model_path IR_model/mnasnet_arcfrace_224_rgb_3_channel_fp32.xml  \
                    --model_output_path "SAVED_MODEL_TF/saved_model.pb (copy)/" \
                    --output_no_quant_float32_tflite
```

???? more flags of conversion can be found here: ???? [openvino2tf](https://github.com/PINTO0309/openvino2tensorflow#5-usage)

flags starting with --output_* decide the output format.