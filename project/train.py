from project.dataset_cls import identities_ds
import timm
from gc import callbacks
import logging
from xml.etree.ElementTree import Comment
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
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




class mnasnet_embedder(pl.LightningModule):
    def __init__(self, 
                train : str = "project/train.csv", 
                valid : str = "project/valid.csv", 
                embed_sz : int = 128, 
                batch_size : int = 128, 
                lr : float = 0.0001, 
                image_size : int = 64,
                min_lr : float = 1e-8,
                t_max : int = 50,
                samples_per_iter : int = 20,
                **kwargs):

        super(mnasnet_embedder, self).__init__()

        self.save_hyperparameters()
        self.min_lr = min_lr
        self.t_max = t_max
        self.image_sz = image_size
        self.embed_sz = embed_sz
        self.train = train
        self.valid = valid
        self.batch_size = batch_size
        self.lr = lr
        self.samples_per_iter = samples_per_iter
        
        self.train_transform = transforms.Compose(
                                                        [
                                                            transforms.Resize(size=self.img_sz),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
                                                        ]
                                                )       

        
        self.val_transform = transforms.Compose(
                                                        [
                                                            transforms.Resize(size=self.img_sz),
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
        
        #DEFINED LATER IN SETUP FUNCTION
        
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
  
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #        embedder model defined
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

        embedder = mnasnet_embedder.MLP([trunk_output_size, embed_sz])
        return embedder
        

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                            Lightning Data Module
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def prepare_data(self):
        # Called only on single-GPU

        # don't assign the states, only process dataset
        pass
        
    def setup(self, stage=None):
        # Called on each GPU

        # define the states i.e. self.abc

        # DOWNLOAD THE ORIGINAL DATASETS
       
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #           Making the dis-joint datasets
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Class disjoint training and validation set
        self.train_dataset = identities_ds(self.train, self.train_transform)
                                                    
        self.val_dataset = identities_ds(self.valid, self.val_transform)
        
        # needed by embedding-tester                                                        
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
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=6)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                pl-lightning realted functions
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, x):
        # given a batch `x` ---> generate the embedding vector
        x = self.trunk(x)  # Mnasnet 512 , feature extractor dims.
        x = self.embedder(x) # MLP get the embedding using features as input
        return x
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # generate the embedding
        images, labels = batch
        embedding = self(images) # everything in forward  will be executed

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
        
        optimizer_trunk = torch.optim.Adam(self.trunk_model.parameters(), lr=self.lr_trunk)
        lr_scheduler_trunk = LinearWarmupCosineAnnealingLR(optimizer_trunk, self.warmup_epochs, self.trainer.max_epochs,
                                                     warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)
        
        optimizer_embedder = torch.optim.Adam(self.embedder_model.parameters(), lr=self.lr_embedder)
        lr_scheduler_embedder = LinearWarmupCosineAnnealingLR(optimizer_embedder, self.warmup_epochs, self.trainer.max_epochs,
                                                     warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)
        
        optimizer_arcface_clf = torch.optim.Adam(self.arcface_loss_layer.parameters(), lr=self.lr_arcface)
        lr_scheduler_arcface = LinearWarmupCosineAnnealingLR(optimizer_arcface_clf, self.warmup_epochs, self.trainer.max_epochs,
                                                     warmup_start_lr=0.0, eta_min=0.0, last_epoch=- 1)
        

        # self.optimizer_arcface = torch.optim.Adam(self.arcface_loss_layer.parameters(), lr=self.lr)
        # self.scheduler_arcface = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_arcface, T_max=self.t_max, eta_min=self.min_lr)
        
        return [optimizer_trunk, 
                optimizer_embedder, 
                optimizer_arcface_clf], [ {"scheduler":lr_scheduler_trunk, "name":"trunk_lr"},
                                         {"scheduler":lr_scheduler_embedder, "name":"embedder_lr"}, 
                                         {"scheduler" : lr_scheduler_arcface, "name" : "arcface_lr"} ]



    @staticmethod
    def add_model_specific_args(parent_parser):


        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train", type=str, default="project/train.csv", help="path to the train.csv file")
        parser.add_argument("--valid", type=str, default="project/valid.csv", help="path to the valid.csv file")
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

