from __future__ import absolute_import

import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

from torch.utils.data import DataLoader, random_split
import os
from pytorch_metric_learning import losses, samplers
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset
import optuna
from infilling import Inpainter

import pytorch_lightning as pl
from argparse import ArgumentParser

train_path = '/lab/vislab/DATA/CUB/images/justin-train/'
valid_path = '/lab/vislab/DATA/CUB/images/justin-test/'

# num_sanity_val_steps
# val_check_interval

class Basic(LightningModule):
    def __init__(self, hparams, trial, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.trial = trial

        self.epoch = 0
        self.learning_rate = self.hparams.lr
        self.model = models.resnet50(pretrained=True)
        self.loss = losses.TripletMarginLoss(margin=0.1, triplets_per_anchor="all", normalize_embeddings=True)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        return self.model(x)

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        self.trainset = datasets.ImageFolder(train_path, transform)
        self.validset = datasets.ImageFolder(valid_path, transform)

    def train_dataloader(self):
        train_sampler = samplers.MPerClassSampler(self.trainset.targets, 8, len(self.trainset))
        return DataLoader(self.trainset, batch_size=64, sampler=train_sampler, num_workers=4)

    def val_dataloader(self):
        valid_sampler = samplers.MPerClassSampler(self.validset.targets, 8, len(self.validset))
        return DataLoader(self.validset, batch_size=64, sampler=valid_sampler, num_workers=4)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # location = [(a,b) for a in range(255) for b in range(255)]
        # small_mask = Image.new('L', (4, 4), 0)
        # masks = []
        # for _ in range(32):
        #     base = Image.new('L',(256,256),255)
        #     r = random.choice(location)
        #     base.paste(small_mask, r)
        #     location.pop(location.index(r))
        #     base = np.ascontiguousarray(np.expand_dims(base, 0)).astype(np.uint8)
        #     masks.append(base)
        # masks = np.array(masks)
        # masks = torch.from_numpy(masks)

        # pretrained_model_path = '/lab/vislab/DATA/just/infilling/model/model_places2.pth'
        # inpainter = Inpainter(pretrained_model_path, 256, 32)
        # inpainted_img_batch = inpainter.inpaint(inputs, masks)


        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        train_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
        return {
            'log': {'train_loss': train_loss},
            'progress_bar': {'train_loss': train_loss}
        }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs, labels)
        # calculate that stepped accuracy
        val_acc = self.computeAccuracy(outputs, labels)
        val_acc = torch.tensor(val_acc, dtype=torch.float32)
        # needs to be a tensor

        return {
            'val_loss': loss, 
            'val_acc': val_acc
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        # stack the accuracies and average PER EPOCH
        val_acc = torch.stack([x['val_acc'] for x in validation_step_outputs]).mean()
        # print(validation_step_outputs['val_acc'])
        print(val_acc)
        print(val_loss)

        self.trial.report(val_loss, self.epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        self.epoch += 1
        return {
            'log': {
                'val_loss': val_loss,
                'val_acc': val_acc
                },
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc': val_acc
            }
        }
    
    def computeAccuracy(self, outputs, labels):
        incorrect = correct = 0
        for idx, emb in enumerate(outputs):
            pairwise = torch.nn.PairwiseDistance(p=2)
            dist = pairwise(emb, outputs)
            closest = torch.topk(dist, 2, largest=False).indices[1]
            if labels[idx] == labels[closest]:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.1)
        return parser