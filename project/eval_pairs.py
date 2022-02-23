# Following code has been adopted from:
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/test.py#L18
from rich.console import Console
import verification
import logging
import os
from typing import List
import torch
from torch.utils.tensorboard import SummaryWriter


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


identity_list = get_lfw_list('pairs.txt')
Console().print(len(identity_list))




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
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))

            self.summary_writer: SummaryWriter
            self.summary_writer.add_scalar(tag=self.ver_name_list[i], scalar_value=acc2, global_step=global_step, )

            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                '[%s][%d]Accuracy-Highest: %1.5f' % (self.ver_name_list[i], global_step, self.highest_acc_list[i]))
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
cback = CallBackVerification(['lfw', 'agedb_30', 'cfp_ff'], rec_prefix="/home/talha/metric_learning/faces_emore/", summary_writer=sw, image_size=(112, 112))
