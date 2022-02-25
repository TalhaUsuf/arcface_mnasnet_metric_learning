# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function
import os
import cv2
# from models import *
from rich.console import Console
import torch.nn as nn
import torch
import pytorch_lightning as pl
from torchvision.transforms import transforms
import timm
import torch
from pytorch_metric_learning import losses
import numpy as np
import time
import matplotlib.pyplot as plt
# from config import Config
# from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


#  self.val_transform = transforms.Compose(
#                                                             [
#                                                                 transforms.Resize(size=(self.image_size, self.image_size)),
#                                                                 transforms.ToTensor(),
#                                                                 transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
#                                                             ]
#                                                     )

def load_image(img_path):
    image = cv2.imread(img_path) # [H, W, C]
    if image is None:
        return None
    # scale the pixel values between [0.0, 1.0]
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32FC3)
    # cvt to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # pre-process image
    image -= np.array([[[0.4850, 0.4560, 0.4060]]])
    image /= np.array([[[0.2290, 0.2240, 0.2250]]]) 
    # image = np.hstack((image, np.fliplr(image)))
    # plt.imshow(image)
    # plt.show()
    image = image.transpose((2, 0, 1)) # [C, H, W]
    image = np.stack((image, np.fliplr(image))) # 2 x [H, W, C]
    # image = image[:, np.newaxis, :, :]
    # image = image.astype(np.float32, copy=False)
    
    return image # [2, C, H, W]


def get_featurs(model, test_list, device, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(img_path)
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(device)
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]  # [N, 256]
            fe_2 = output[1::2] # [N, 256]
            feature = np.hstack((fe_1, fe_2)) # [N, 512]
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


# def load_model(model, model_path):
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(model_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size, device):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, device, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc



# class mnasnet_embedder(pl.LightningModule):
#     def __init__(self, 
#                     train_ds : str = "project/train.csv", 
#                     valid_ds : str = "project/valid.csv", 
#                     embed_sz : int = 256, 
#                     batch_size : int = 200, 
#                     image_size : int = 224,
#                     samples_per_iter : int = 20,
#                     lr_trunk : float = 0.00001,
#                     lr_embedder : float = 0.0001, 
#                     lr_arcface : float = 0.0001,
#                     warmup_epochs : int = 2,
#                     T_0 : float = 1,
#                     T_mult : float = 2,
#                     **kwargs):

#             super(mnasnet_embedder, self).__init__()

#             self.save_hyperparameters()
#             self.image_size = image_size
#             self.embed_sz = embed_sz
#             self.train_ds = train_ds
#             self.valid_ds = valid_ds
#             self.T_0 = T_0
#             self.T_mult = T_mult
#             self.batch_size = batch_size
#             self.samples_per_iter = samples_per_iter
#             self.lr_trunk = lr_trunk
#             self.lr_embedder = lr_embedder
#             self.lr_arcface = lr_arcface
#             self.warmup_epochs = warmup_epochs
#             self.train_transform = transforms.Compose(
#                                                             [
#                                                                 transforms.Resize(size=(self.image_size, self.image_size)),
#                                                                 transforms.ToTensor(),
#                                                                 transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
#                                                             ]
#                                                     )       

            
#             self.val_transform = transforms.Compose(
#                                                             [
#                                                                 transforms.Resize(size=(self.image_size, self.image_size)),
#                                                                 transforms.ToTensor(),
#                                                                 transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
#                                                             ]
#                                                     )
            
            
            
#             # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
#             #            defining the [trunk ----> embedder] model
#             #                       Trunk : Mnasnet                                                
#             #                       Embedder : MLP
#             #                       Classifier : ArcFace                                                 
#             # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
            
#             # ==========================================================================
#             #                             trunk                                  
#             # ==========================================================================
#             self.trunk = timm.create_model('mnasnet_100', pretrained=True)
#             # put backbone in train mode
#             self.trunk.train()
            
#             in_features = self.trunk.classifier.in_features
            
#             # replace the classifier layer with identity layer
#             self.trunk.classifier = nn.Identity()
#             # make trunk wholly trainable        
#             for p in self.trunk.parameters():
#                 p.requires_grad = True
                
#             self.trunk_output_size = in_features
#             # ==========================================================================
#             #                             embedder                                  
#             # ==========================================================================
#             # define embedder
#             self.embedder = self.get_embedder(self.trunk_output_size, self.embed_sz)
            
#             # ==========================================================================
#             #                             classification head                                  
#             # ==========================================================================
#             self.arcface_loss_layer = losses.ArcFaceLoss(163552, self.embed_sz, margin=28.6, scale=64)

#     class MLP(nn.Module):
#         # layer_sizes[0] is the dimension of the input
#         # layer_sizes[-1] is the dimension of the output
#         def __init__(self, layer_sizes, final_relu=False):
#             super().__init__()
#             layer_list = []
#             layer_sizes = [int(x) for x in layer_sizes]
#             num_layers = len(layer_sizes) - 1
#             final_relu_layer = num_layers if final_relu else num_layers - 1
#             for i in range(len(layer_sizes) - 1):
#                 input_size = layer_sizes[i]
#                 curr_size = layer_sizes[i + 1]
#                 if i < final_relu_layer:
#                     layer_list.append(nn.ReLU(inplace=False))
#                 layer_list.append(nn.Linear(input_size, curr_size))
#             self.net = nn.Sequential(*layer_list)
#             self.last_linear = self.net[-1]

#         def forward(self, x):
#             return self.net(x)

#     def get_embedder(self, trunk_output_size : int, embed_sz : int = 64):

#         embedder = mnasnet_embedder.MLP([trunk_output_size, embed_sz])
#         return embedder
            
#     def forward(self, x):
#         # given a batch `x` ---> generate the embedding vector
#         x = self.trunk(x)  # Mnasnet 512 , feature extractor dims.
#         x = self.embedder(x) # MLP get the embedding using features as input
#         return x
    
    


# if __name__ == '__main__':



#     identity_list = get_lfw_list("pairs.txt")
#     img_paths = [os.path.join("lfw", each) for each in identity_list]

#     # model.eval()
#     model = mnasnet_embedder(train_ds = None, valid_ds = None, embed_sz = 256, batch_size = 250, image_size = 224)
#     params = torch.load("/home/talha/Downloads/arcface_mnasnet_metric_learning/weights/last_1_epoch_last_step.ckpt")
#     Console().log(f"following keys are present in the checkpoint ....")
#     model.load_state_dict(params["state_dict"])
#     model.eval()
#     model.to(torch.device("cuda:1"))    
    
#     acc = lfw_test(model, img_paths, identity_list, "pairs.txt", 32)
#     Console().rule(title=f"Accuracy is {acc:3f}", style="green on black", characters="=")


