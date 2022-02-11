

from gc import callbacks
import logging
from xml.etree.ElementTree import Comment
import json
import matplotlib.pyplot as plt
import numpy as np
import record_keeper
import torch
import torch.nn as nn
import torchvision
import numpy as np
from rich.console import Console
import umap
from uuid import uuid4
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms
import umap
import pytorch_metric_learning
from pytorch_metric_learning import samplers, miners
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
import wandb
from pytorch_lightning.loggers import WandbLogger



sns.set_style('darkgrid')
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)




class mnist_embedder(pl.LightningModule):
    def __init__(self, 
                root_dataset, 
                embed_sz : int = 128, 
                batch_size : int = 128, 
                lr : float = 0.0001, 
                image_size : int = 64,
                min_lr : float = 1e-8,
                t_max : int = 50,
                samples_per_iter : int = 20,
                **kwargs):

        super(mnist_embedder, self).__init__()

        self.save_hyperparameters()
        self.min_lr = min_lr
        self.t_max = t_max
        self.image_sz = image_size
        self.embed_sz = embed_sz
        self.root = root_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.samples_per_iter = samples_per_iter
        
        self.train_transform = transforms.Compose(
                                                        [
                                                            transforms.Resize(self.image_sz),
                                                            transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=self.image_sz),
                                                            transforms.RandomHorizontalFlip(0.5),
                                                            transforms.ToTensor(), # [C, H, W]
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ]
                                                )       
        
        self.val_transform = transforms.Compose(
                                                        [
                                                            transforms.Resize(self.image_sz),
                                                            transforms.ToTensor(), # [C, H, W]
                                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ]
                                                )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
        #            defining the [trunk ----> embedder] model
        #                       Trunk : resnet                                                
        #                       Embedder : MLP                                                 
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                                                
        self.trunk = torchvision.models.resnet18(pretrained=True)
        for p in self.trunk.parameters():
            p.requires_grad = True
        # replacing the end-classifier with a Identity layer
        self.trunk_output_size = self.trunk.fc.in_features
        self.trunk.fc = common_functions.Identity() # clf replaced by identity layer

        self.embedder = self.get_embedder(self.trunk_output_size, self.embed_sz)

        # self.model = torch.nn.Sequential([
        #                                         self.trunk,
        #                                         self.embedder
        #                                     ])
        self.params_list = list(self.trunk.parameters()) + list(self.embedder.parameters())
        # Console().print(list(self.trunk.parameters()))
        # Set the mining function
        self.record_keeper, _, _ = logging_presets.get_record_keeper(
                                                                    "example_logs", "example_tensorboard"
                                                                )
        self.hooks = logging_presets.get_hook_container(self.record_keeper)
        self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
       
        self.tester = testers.GlobalEmbeddingSpaceTester(
                                                            end_of_testing_hook=self.hooks.end_of_testing_hook,
                                                            visualizer=umap.UMAP(),
                                                            visualizer_hook=self.visualizer_hook,
                                                            dataloader_num_workers=4,
                                                            batch_size=512,
                                                            # data_device=self.device,
                                                            # accuracy_calculator=AccuracyCalculator(avg_of_avgs=True, k="max_bin_count", device=self.device),
                                                            accuracy_calculator=AccuracyCalculator(avg_of_avgs=True, k="max_bin_count"),
                                                        )
    
    def visualizer_hook(self, umapper, umap_embeddings, labels, split_name, keyname, *args):
        logging.info(
            "UMAP plot for the {} split and label set {}".format(split_name, keyname)
        )
        label_set = np.unique(labels)
        num_classes = len(label_set)
        fig = plt.figure(figsize=(20, 15))
        plt.gca().set_prop_cycle(
            cycler(
                "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
            )
        )
        for i in range(num_classes):
            idx = labels == label_set[i]
            plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
        plt.title(f"epoch_{self.current_epoch}")
        # plt.savefig(f"embeddings_{nm}_epoch_{self.current_epoch}.jpg")
        # wandb.log({f"embeddings" : wandb.Image("embeddings_{nm}_epoch_{self.current_epoch}.jpg")})
        wandb.log({"embeddings" : plt,
                   "captions" : wandb.Html(f"<h1>epoch_{self.current_epoch}</h1>"),
                   "epoch" : self.current_epoch,})
        plt.close(fig)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This will be used to create train and val sets that are class-disjoint
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    class ClassDisjointCIFAR100(torch.utils.data.Dataset):
        def __init__(self, original_train, original_val, train, transform):
            super().__init__()
            # if train flag is true then it will allow only indices below 50
            # if train flag is flag then it will allow only indices above 50
            rule = (lambda x: x < 50) if train else (lambda x: x >= 50)
            train_filtered_idx = [i for i, x in enumerate(original_train.targets) if rule(x)]   
            val_filtered_idx = [i for i, x in enumerate(original_val.targets) if rule(x)]

            # combine the training data from both original train as well as original val. dataset
            self.data = np.concatenate(
                                        [
                                            original_train.data[train_filtered_idx],
                                            original_val.data[val_filtered_idx],
                                        ],
                                        axis=0,
                                    )
            self.targets = np.concatenate(
                                            [
                                                np.array(original_train.targets)[train_filtered_idx],
                                                np.array(original_val.targets)[val_filtered_idx],
                                            ],
                                            axis=0,
                                        )
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            img, target = self.data[index], self.targets[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #        classifier model defined
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

        embedder = mnist_embedder.MLP([trunk_output_size, embed_sz])
        return embedder
        

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                            Lightning Data Module
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def prepare_data(self):
        # Called only on single-GPU

        # don't assign the states, only process dataset

        # download the datasets
        datasets.CIFAR100(
                            root=f"{self.root}/CIFAR100_Dataset", train=True, transform=None, download=True
                            )
        datasets.CIFAR100(
                            root=f"{self.root}/CIFAR100_Dataset", train=False, transform=None, download=True
                            )
        
    def setup(self, stage=None):
        # Called on each GPU

        # define the states

        # DOWNLOAD THE ORIGINAL DATASETS
        self.original_train = datasets.CIFAR100(
                                                    root=f"{self.root}/CIFAR100_Dataset", train=True, transform=None, download=True
                                                )
        self.original_val = datasets.CIFAR100(
                                                    root=f"{self.root}/CIFAR100_Dataset", train=False, transform=None, download=True
                                                )
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #           Making the dis-joint datasets
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Class disjoint training and validation set
        self.train_dataset = mnist_embedder.ClassDisjointCIFAR100(
                                                        self.original_train, self.original_val, train=True, transform=self.train_transform
                                                    )
        self.val_dataset = mnist_embedder.ClassDisjointCIFAR100(
                                                        self.original_train, self.original_val, train=False, transform=self.val_transform
                                                 )
        self.dataset_dict = {"val": self.val_dataset}
        self.classes_in_training = np.unique(self.train_dataset.targets)

        assert set(self.train_dataset.targets).isdisjoint(set(self.val_dataset.targets)), "labels are NOT dis-joint"

        # sampler needs the no. of training dataset examples
        self.sampler = samplers.MPerClassSampler(
                                                    self.train_dataset.targets, m=self.samples_per_iter, length_before_new_iter=len(self.train_dataset)
                                                )
        # loss function layer needs the no. of classes
        # see : https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/examples/notebooks/TrainWithClassifier.ipynb
        self.arcface_loss_layer = losses.ArcFaceLoss(len(self.classes_in_training), self.embed_sz, margin=28.6, scale=64)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=6, pin_memory=True, prefetch_factor=24, persistent_workers=True, sampler=self.sampler)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=12)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                pl-lightning realted functions
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, x):
        # given a batch `x` ---> generate the embedding vector
        x = self.trunk(x)  # 512 , feature extractor dims.
        x = self.embedder(x) # get the embedding using features as input
        # x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # generate the embedding
        images, labels = batch
        embedding = self(images)

        # get the hard examples        
        miner_output = self.miner(embedding, labels) # in your training for-loop
        loss = self.arcface_loss_layer(embedding, labels, miner_output)
        # loss = self.arcface_loss_layer(embedding, labels)
        self.log("train_loss", loss, sync_dist=True)
        return {"loss" : loss}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # below line is necessary or it will perform testing on each mini-batch of the validation set
        if batch_idx == 0:
            # only perfom testing once per validation epoch
            all_accs = self.tester.test(dataset_dict=self.dataset_dict, 
                             epoch=self.current_epoch,
                             trunk_model=self,
                             embedder_model=None,
                             )
            for k,v in all_accs["val"].items():
                
                # self.log("val_acc", all_accs["val"]["precision_at_1_level0"], sync_dist=True)
                self.log(f"{k}", all_accs["val"][f"{k}"], sync_dist=True)
            
            return {"val_acc": all_accs["val"]["precision_at_1_level0"]}
        else:
            pass
        
    def configure_optimizers(self):
        
        self.optimizer_model = torch.optim.Adam(self.params_list, lr=self.lr)
        self.scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_model, T_max=self.t_max, eta_min=self.min_lr)

        self.optimizer_arcface = torch.optim.Adam(self.arcface_loss_layer.parameters(), lr=self.lr)
        self.scheduler_arcface = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_arcface, T_max=self.t_max, eta_min=self.min_lr)
        
        return [self.optimizer_model, self.optimizer_arcface], [ {"scheduler":self.scheduler_model, "name":"model_lr"}, {"scheduler" : self.scheduler_arcface, "name" : "arcface_lr"} ]



    @staticmethod
    def add_model_specific_args(parent_parser):

        # root_dataset, 
        #         embed_sz : int = 128, 
        #         batch_size : int = 128, 
        #         lr : float = 0.0001, 
        #         img_sz : int = 64):


        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--root_dataset", type=str, default="./", help="dir.at where to download the datasets")
        parser.add_argument("--embed_sz", type=int, default=128, help="embedding size")
        parser.add_argument("--batch_size", type=int, default=128, help="batch size for dataloader")
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for the optimizer")
        parser.add_argument("--image_size", type=int, default=64, help="resolution at which to feed images to model")
        parser.add_argument("--min_lr", type=float, default=1e-8, help="minimum lr which the optimizer should retain")
        parser.add_argument("--t_max", type=int, default=50, help="at which epoch `time-period / 2 value` should happen")
        parser.add_argument("--samples_per_iter", type=int, default=20, help="The number of samples per class to fetch at every iteration. \n SEE: https://kevinmusgrave.github.io/pytorch-metric-learning/samplers/#mperclasssampler")
        
        return parser


def cli_main(args=None):
   
    wandb.init() # needed for wandb.watch
    wandb.login()

    pl.seed_everything(1234)

    parser = ArgumentParser()
    # parser.add_argument("--dataset", default="mnist", type=str, help="mnist, cifar10, stl10, imagenet")
    # script_args, _ = parser.parse_known_args(args)

    # if script_args.dataset == "mnist":
    #     dm_cls = MNISTDataModule
    # elif script_args.dataset == "cifar10":
    #     dm_cls = CIFAR10DataModule
    # elif script_args.dataset == "stl10":
    #     dm_cls = STL10DataModule
    # elif script_args.dataset == "imagenet":
    #     dm_cls = ImagenetDataModule

    
    parser = pl.Trainer.add_argparse_args(parser)
    parser = mnist_embedder.add_model_specific_args(parser)
    args = parser.parse_args(args)

    model = mnist_embedder(**vars(args))
    # model.prepare_data()
    # model.setup()
    # print(len(model.train_dataset))

    wandb_logger = WandbLogger(project='Arcface-FaceRecognition-glint', 
                           config=vars(args),
                           group='face-recognition', 
                           job_type='train')
    wandb.watch(
            model, criterion=None, log="all",
            log_graph=True, log_freq=20
        )
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[pl.callbacks.LearningRateMonitor()], weights_summary='top', progress_bar_refresh_rate=20)
    trainer.fit(model)
    # return dm, model, trainer

    wandb.finish()
    

if __name__ == "__main__":
    # dm, model, trainer = cli_main()
    cli_main()




# # Set the loss function
# loss = losses.TripletMarginLoss(margin=0.1)

# # Set the mining function
# miner = miners.MultiSimilarityMiner(epsilon=0.1)

# # Set the dataloader sampler
# sampler = samplers.MPerClassSampler(
#     train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)
# )

# # Set other training parameters
# batch_size = 32
# num_epochs = 4

# # Package the above stuff into dictionaries.
# models = {"trunk": trunk, "embedder": embedder} # trunk ---> resnet , embedder ---> MLP model

# # making separate optimizers

# optimizers = {
#     "trunk_optimizer": trunk_optimizer,
#     "embedder_optimizer": embedder_optimizer,
# }
# loss_funcs = {"metric_loss": loss}
# mining_funcs = {"tuple_miner": miner}

# # Remove logs if you want to train with new parameters
# !rm -rf example_logs/ example_saved_models/ example_tensorboard/

# """## Create the training and testing hooks"""

# record_keeper, _, _ = logging_presets.get_record_keeper(
#     "example_logs", "example_tensorboard"
# )
# hooks = logging_presets.get_hook_container(record_keeper)
# dataset_dict = {"val": val_dataset}
# model_folder = "example_saved_models"


# def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
#     logging.info(
#         "UMAP plot for the {} split and label set {}".format(split_name, keyname)
#     )
#     label_set = np.unique(labels)
#     num_classes = len(label_set)
#     fig = plt.figure(figsize=(20, 15))
#     plt.gca().set_prop_cycle(
#         cycler(
#             "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
#         )
#     )
#     for i in range(num_classes):
#         idx = labels == label_set[i]
#         plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
#     plt.show()


# # Create the tester
# tester = testers.GlobalEmbeddingSpaceTester(
#     end_of_testing_hook=hooks.end_of_testing_hook,
#     visualizer=umap.UMAP(),
#     visualizer_hook=visualizer_hook,
#     dataloader_num_workers=2,
#     accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
# )

# end_of_epoch_hook = hooks.end_of_epoch_hook(
#     tester, dataset_dict, model_folder, test_interval=1, patience=1
# )

# """## Create the trainer"""

# trainer = trainers.MetricLossOnly(
#     models,
#     optimizers,
#     batch_size,
#     loss_funcs,
#     mining_funcs,
#     train_dataset,
#     sampler=sampler,
#     dataloader_num_workers=2,
#     end_of_iteration_hook=hooks.end_of_iteration_hook,
#     end_of_epoch_hook=end_of_epoch_hook,
# )

# """## Start Tensorboard
# (Turn off adblock and other shields)
# """

# # Commented out IPython magic to ensure Python compatibility.
# # %load_ext tensorboard
# # %tensorboard --logdir example_tensorboard

# """## Train the model"""

# trainer.train(num_epochs=num_epochs)

