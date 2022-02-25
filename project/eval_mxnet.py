"""
Following code has been adopted from following link and has been used for evaluation:
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/test.py#L18
it takes datasets in form of bin files and performs evaluation on them, this file can be used as a stand-alone file for evaluation
on the said datasets
"""
from rich.console import Console
import verification
import timm
import logging
import os
from typing import List
import torch
from torchvision.transforms import transforms
import json
import torch.nn as nn
from pytorch_metric_learning import losses
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter


# def get_lfw_list(pair_list):
#     with open(pair_list, 'r') as fd:
#         pairs = fd.readlines()
#     data_list = []
#     for pair in pairs:
#         splits = pair.split()

#         if splits[0] not in data_list:
#             data_list.append(splits[0])

#         if splits[1] not in data_list:
#             data_list.append(splits[1])
#     return data_list


# identity_list = get_lfw_list('pairs.txt')
# Console().print(len(identity_list))


class mnasnet_embedder(pl.LightningModule):
    def __init__(self, 
                    train_ds : str = "project/train.csv", 
                    valid_ds : str = "project/valid.csv", 
                    embed_sz : int = 256, 
                    batch_size : int = 200, 
                    image_size : int = 224,
                    samples_per_iter : int = 20,
                    lr_trunk : float = 0.00001,
                    lr_embedder : float = 0.0001, 
                    lr_arcface : float = 0.0001,
                    warmup_epochs : int = 2,
                    T_0 : float = 1,
                    T_mult : float = 2,
                    **kwargs):

            super(mnasnet_embedder, self).__init__()

            self.save_hyperparameters()
            self.image_size = image_size
            self.embed_sz = embed_sz
            self.train_ds = train_ds
            self.valid_ds = valid_ds
            self.T_0 = T_0
            self.T_mult = T_mult
            self.batch_size = batch_size
            self.samples_per_iter = samples_per_iter
            self.lr_trunk = lr_trunk
            self.lr_embedder = lr_embedder
            self.lr_arcface = lr_arcface
            self.warmup_epochs = warmup_epochs
            self.train_transform = transforms.Compose(
                                                            [
                                                                transforms.Resize(size=(self.image_size, self.image_size)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
                                                            ]
                                                    )       

            
            self.val_transform = transforms.Compose(
                                                            [
                                                                transforms.Resize(size=(self.image_size, self.image_size)),
                                                                transforms.ToTensor(),
                                                                transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
                                                            ]
                                                    )
            
            
            
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
            #            defining the [trunk ----> embedder] model
            #                       Trunk : Mnasnet                                                
            #                       Embedder : MLP
            #                       Classifier : ArcFace                                                 
            # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
            
            # ==========================================================================
            #                             trunk                                  
            # ==========================================================================
            self.trunk = timm.create_model('mnasnet_100', pretrained=True)
            # put backbone in train mode
            self.trunk.train()
            
            in_features = self.trunk.classifier.in_features
            
            # replace the classifier layer with identity layer
            self.trunk.classifier = nn.Identity()
            # make trunk wholly trainable        
            for p in self.trunk.parameters():
                p.requires_grad = True
                
            self.trunk_output_size = in_features
            # ==========================================================================
            #                             embedder                                  
            # ==========================================================================
            # define embedder
            self.embedder = self.get_embedder(self.trunk_output_size, self.embed_sz)
            
            # ==========================================================================
            #                             classification head                                  
            # ==========================================================================
            self.arcface_loss_layer = losses.ArcFaceLoss(163552, self.embed_sz, margin=28.6, scale=64)

    class MLP(nn.Module):
        # layer_sizes[0] is the dimension of the input
        # layer_sizes[-1] is the dimension of the output
        def __init__(self, layer_sizes, final_relu=False):
            super().__init__()
            layer_list = []
            layer_sizes = [int(x) for x in layer_sizes]
            num_layers = len(layer_sizes) - 1
            final_relu_layer = num_layers if final_relu else num_layers - 1
            for i in range(len(layer_sizes) - 1):
                input_size = layer_sizes[i]
                curr_size = layer_sizes[i + 1]
                if i < final_relu_layer:
                    layer_list.append(nn.ReLU(inplace=False))
                layer_list.append(nn.Linear(input_size, curr_size))
            self.net = nn.Sequential(*layer_list)
            self.last_linear = self.net[-1]

        def forward(self, x):
            return self.net(x)

    def get_embedder(self, trunk_output_size : int, embed_sz : int = 64):

        embedder = mnasnet_embedder.MLP([trunk_output_size, embed_sz])
        return embedder
            
    def forward(self, x):
        # given a batch `x` ---> generate the embedding vector
        x = self.trunk(x)  # Mnasnet 512 , feature extractor dims.
        x = self.embedder(x) # MLP get the embedding using features as input
        return x
    
    

class CallBackVerification(object):
    
    def __init__(self, val_targets, rec_prefix, summary_writer=None, image_size=(112, 112)):
        # self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        # if self.rank is 0:
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

        self.summary_writer = summary_writer

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            # logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            Console().print(f'[{self.ver_name_list[i]}][{global_step}]XNorm: {xnorm}')
            # logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            Console().print(f'[{self.ver_name_list[i]}][{global_step}]Accuracy-Flip: {acc2}+-{std2}')

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            # logging.info(
            #     '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
            Console().print(f'[{self.ver_name_list[i]}][{global_step}]Accuracy-Highest: {self.highest_acc_list[i]}')
            results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            Console().log(f"reading ---> {path}")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        # if self.rank is 0 and num_update > 0:
        backbone.eval()
        self.ver_test(backbone, num_update)
        backbone.train()
            
            
            
sw = SummaryWriter(log_dir='./')
# cback = CallBackVerification(['lfw', 'agedb_30', 'cfp_ff'], rec_prefix="/home/talha/metric_learning/faces_emore/", summary_writer=sw, image_size=(112, 112))
cback = CallBackVerification(['lfw'], rec_prefix="/home/talha/metric_learning/faces_emore/", summary_writer=sw, image_size=(112, 112))
model = mnasnet_embedder(train_ds = None, valid_ds = None, embed_sz = 256, batch_size = 250, image_size = 224)
params = torch.load("/home/talha/Downloads/arcface_mnasnet_metric_learning/weights/epoch=5-step=150999-last.ckpt")
Console().log(f"following keys are present in the checkpoint ....")
model.load_state_dict(params["state_dict"])
Console().log(f"model loaded successfully")
cback.__call__(0, model)