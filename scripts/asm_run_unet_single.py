import os
import sys
import random
import shutil
import multiprocessing as mp
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchvision.transforms import Resize, InterpolationMode, ToPILImage
from torchvision.transforms.functional import pad
from torchmetrics import AUROC
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import geopandas as gpd
import rasterio
import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

from asm_datamodules import *
from asm_models import *

# PARAMETERS
model = "unet_dropout"
split = False # generate new splits if True; use saved splits if False
project = "ASM_stat288" # project name in WandB
run_name = "unet_impute_trainonly_square" # run name in WandB
n_epoch = 100
backbone = "resnet18"
pretrained_weights = True
in_channels = 4
encoder_depth = 5
bands = ["R", "G", "B", "NIR"]
augment = False
dropout_rate = 0.1
dropout_2d = False
first_only = False
class_weights = [0.25,0.75]
alpha = None
gamma = None
lr = 1e-5
image_loss_type = None
image_weight = None
weight_decay = 0.1
loss = "ce_imputed"
impute_eval = False
patience = 10
early_stop_patience = 10
batch_size = 64

# device configuration
device, num_devices = ("cuda", torch.cuda.device_count()) if torch.cuda.is_available() else ("cpu", len(os.sched_getaffinity(0)))
workers = len(os.sched_getaffinity(0))
print(f"Running on {num_devices} {device}(s) with {workers} cpus")
torch.set_float32_matmul_precision("medium")

# ---PREPARE DATA---
if split:
    # unique file path for this split
    out_path = "../data/split_regular"

    # generate split and save as pickle file
    split_asm_data(stratify_col="country", 
            save = True,
           out_path=out_path)

# datamodule configuration
num_workers = workers # set to maximum number of available CPUs
root = "/n/holyscratch01/tambe_lab/kayan/karena/" # root for data files
split_path = "../data/split_regular" # one of the randomly generated splits

# create and set up datamodule
datamodule = ASMDataModule(batch_size=batch_size, 
                           num_workers=num_workers, 
                           split=split, 
                           root=root,
                           bands=bands,
                           transforms=min_max_transform,
                           augment=augment,
                           split_path=split_path)

# set up dataloaders
datamodule.setup("fit")
train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader() 
datamodule.setup("test")
test_dataloader = datamodule.test_dataloader()

# callback to calculate val AUC at end of each epoch
# uses a simple global averaging function to transform seg output -> image pred
class AUCCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        val_dataloader = trainer.val_dataloaders
        
        pl_module.eval()
        preds = []
        targets = []
        with torch.inference_mode():
            for samples in val_dataloader:
                inputs = samples["image"].to(device)
                masks = samples["mask"].to(device)
                
                outputs = pl_module(inputs) # get model output
                outputs = torch.softmax(outputs, dim=1) # softmax over class dimension
                
                img_preds = torch.mean(outputs, dim=(-1,-2)) # average over x and y dimensions
                img_targets = (torch.sum(masks, dim=(-1,-2)) > 0) # will be >0 if contains a mine, 0 otherwise
                preds.append(img_preds[:,1]) # append probability of mine class
                targets.append(img_targets) # append true labels
                
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        auc_task = AUROC(task="binary")
        auc_score = auc_task(preds,targets)
        
        wandb.log({"val_AUC": auc_score.item()}, step=trainer.global_step)

# callback for WandB logging
class WandBCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # log train loss to WandB
        train_loss = trainer.callback_metrics.get("train_loss_epoch")
        if train_loss is not None:
            wandb.log({"train_loss": train_loss.item()}, step=trainer.global_step)
            
    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation loss to WandB
        val_loss = trainer.callback_metrics.get("val_loss_epoch")
        if val_loss is not None:
            wandb.log({"val_loss": val_loss.item()}, step=trainer.global_step)
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
 
        # outputs corresponds to our model predictions
        # log n sample image predictions from every other batch
        if batch_idx%4 == 0:
            n = 1
            imgs = batch["image"]
            masks = batch["mask"].to(torch.float64)
            outputs = outputs.to(torch.float64)
            captions = ["Image", "Ground truth", "Prediction"]
            for i in range(n):
                if len(imgs[i]>3): img = ToPILImage()(imgs[i][:-1]) # remove NIR channel for plotting purposes
                mask = ToPILImage()(masks[i])
                output = ToPILImage()(outputs[i])
                wandb_logger.log_image(key=f"Val {batch_idx}-{i}", images=[img, mask, output], caption=captions)

# set up model
task = CustomSemanticSegmentationTask(
    model=model,
    backbone=backbone,
    weights=pretrained_weights, # whether or not to use ImageNet weights
    loss=loss,
    impute_eval=impute_eval,
    class_weights = torch.Tensor(class_weights) if class_weights is not None else None,
    in_channels=in_channels,
    encoder_depth=encoder_depth,
    num_classes=2,
    lr=lr,
    weight_decay=weight_decay,
    dropout_rate=dropout_rate,
    dropout_2d=dropout_2d,
    first_only=first_only,
    image_weight=image_weight,
    image_loss_type=image_loss_type,
    alpha=alpha,
    gamma=gamma,
    patience=patience,
    freeze_backbone=False,
    freeze_decoder=False
)

wandb_logger = WandbLogger(project=project, name=run_name, log_model=True, save_code=True)
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
early_stop_callback = EarlyStopping(
           monitor='val_loss',
           min_delta=0.00,
           patience=early_stop_patience,
           verbose=False,
           mode='min'
        )
lr_callback = LearningRateMonitor(logging_interval="epoch")

trainer = Trainer(
        accelerator=device,
        devices=num_devices,
        max_epochs=n_epoch,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[AUCCallback(), WandBCallback(), checkpoint_callback, early_stop_callback, lr_callback],
        logger=wandb_logger
    )

trainer.fit(model=task, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
# set up datamodule for testing
datamodule.setup("test")
test_dataloader = datamodule.test_dataloader()

trainer.test(model=task, dataloaders=test_dataloader)

# test evaluation
task.eval()

y_probs = []
y_true = []

with torch.inference_mode():
    for idx,samples in enumerate(test_dataloader):
        # Move input data to the device
        inputs = samples['image']
        labels = samples['mask']

        # Forward pass
        outputs = task(inputs)
        outputs = torch.softmax(outputs, dim=1)

        for output, label in zip(outputs, labels):
            preds = output[1].cpu().numpy()
            label = label.numpy()
            y_probs += list(preds.flatten())
            y_true += list(label.flatten())

prec, rec, _ = precision_recall_curve(y_true, y_probs)
fig,ax = plt.subplots()
ax.plot(prec, rec)
ax.set_title("Test Precision-Recall Curve")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
wandb.log({"ROC": wandb.Image(fig)})

ap = average_precision_score(y_true, y_probs)
wandb.log({"Test Average Precision": ap})
print(f"Average precision is: {ap}")

pixelwise_predictions = {}

with torch.inference_mode():
    for idx,samples in enumerate(test_dataloader):
        unique_ids = samples['id']
        # Move input data to the device
        inputs = samples['image']

        # Forward pass
        outputs = task(inputs)
        outputs = torch.softmax(outputs, dim=1)
        #outputs = outputs.argmax(dim=1).squeeze()
        
        for unique_id,output in zip(unique_ids, outputs):
            pixelwise_predictions[unique_id] = output[1].cpu().numpy()
            
def pixelwise_to_class(pixelwise_preds):
    class_proba = {}
    for (unique_id,preds) in pixelwise_preds.items():
        # average probability across pixels
        class_proba[unique_id] = np.mean(preds)
    return class_proba

class_proba = pixelwise_to_class(pixelwise_predictions)
path="/n/home07/kayan/asm/data/filtered_labels.geojson"
label_df = gpd.read_file(path)

true_labels = [label_df[label_df["unique_id"]==x]["label"].values[0] for x in class_proba.keys()]
class_proba = list(class_proba.values())
fig,ax = plt.subplots()
fpr, tpr, _ = roc_curve(true_labels, class_proba)
ax.plot(fpr, tpr)
ax.set_title("Test ROC")
ax.set_xlabel("False Positive")
ax.set_ylabel("True Positive")
wandb.log({"ROC": wandb.Image(fig)})

auc = roc_auc_score(true_labels, class_proba)
wandb.log({"Test AUC": auc})
print(f"AUC is: {auc}")

wandb.finish()
