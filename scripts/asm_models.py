import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchgeo.datasets import NonGeoDataset, stack_samples, unbind_samples
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.trainers import PixelwiseRegressionTask, SemanticSegmentationTask
from torchgeo.models import RCF, FCN
from torchvision.transforms.functional import pad
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import AUROC, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, BinaryAccuracy, BinaryJaccardIndex, BinaryAveragePrecision
import lightning as L
import geopandas as gpd
import rasterio
import numpy as np
from asm_losses import *
        
class UnetWithDropout(smp.Unet):
    '''Unet with dropout layers added after ReLU activation'''
    def __init__(
        self,
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=4,
        classes=2,
        dropout_rate=0.3,
        dropout_2d=False,
        first_only=False,
    ):
        super().__init__(encoder_name=encoder_name, 
                         encoder_weights=encoder_weights,
                         in_channels=in_channels,
                         classes=classes)
        
        # add dropout layers
        for module_name, module in self.encoder.named_children():
            if isinstance(module, nn.Sequential) and not first_only: 
                for block_name, block in module.named_children():
                    for layer_name, layer in block.named_children():
                        if layer_name == 'relu':
                            # Insert dropout after the activation layer in each block
                            if dropout_2d:
                                setattr(block, layer_name, nn.Sequential(
                                    layer,
                                    nn.Dropout2d(dropout_rate)
                                ))
                            else:
                                setattr(block, layer_name, nn.Sequential(
                                    layer,
                                    nn.Dropout(dropout_rate)
                                ))
            else:
                if module_name == 'relu':
                    # Add dropout after the first activation layer in the encoder
                    if dropout_2d:
                        setattr(self.encoder, module_name, nn.Sequential(
                            module,
                            nn.Dropout2d(dropout_rate)
                        ))
                    else:
                        setattr(self.encoder, module_name, nn.Sequential(
                            module,
                            nn.Dropout(dropout_rate)
                        ))

class CustomSemanticSegmentationTask(SemanticSegmentationTask):
    def __init__(
        self,
        model,
        backbone,
        weights = None,
        in_channels = 4,
        num_classes = 2,
        encoder_depth = 5,
        num_filters = 64,
        dropout_rate = 0.3,
        dropout_2d = True,
        first_only = False,
        loss = "focal",
        impute_eval = False,
        class_weights = None,
        alpha = None,
        gamma = 2.0,
        focal_normalized = False,
        image_weight = 0.5, 
        image_loss_type = "ce",
        image_ce_weights = None,
        lr = 1e-3,
        patience = 10,
        weight_decay = 1e-2,
        freeze_backbone = False,
        freeze_decoder = False,
    ):
        super().__init__(model, backbone, weights, in_channels, num_classes, loss=loss,
                         class_weights=class_weights, lr=lr, freeze_backbone=freeze_backbone,       
                         freeze_decoder=freeze_decoder)

        # add weight decay parameter
        self.hparams["weight_decay"] = weight_decay
        self.hparams["alpha"] = alpha
        self.hparams["gamma"] = gamma
        self.hparams["focal_normalized"] = focal_normalized
        self.hparams["image_weight"] = image_weight
        self.hparams["image_loss_type"] = image_loss_type
        self.hparams["image_ce_weights"] = image_ce_weights
        self.hparams["num_filters"] = num_filters
        self.hparams["dropout_rate"] = dropout_rate
        self.hparams["dropout_2d"] = dropout_2d
        self.hparams["first_only"] = first_only
        self.hparams["encoder_depth"] = encoder_depth
        self.hparams["impute_eval"] = impute_eval
        
    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce" or loss == "ce_imputed":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams["class_weights"]
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=self.hparams["focal_normalized"], 
                alpha=self.hparams["alpha"], gamma=self.hparams["gamma"]
            )
        elif loss == "joint":
            self.criterion = JointLoss(image_weight=self.hparams["image_weight"], ignore_index=ignore_index, 
                normalized=self.hparams["focal_normalized"], alpha=self.hparams["alpha"], gamma=self.hparams["gamma"], 
                image_loss_type=self.hparams["image_loss_type"], image_ce_weights=self.hparams["image_ce_weights"])
        elif loss == "confidence_weighted":
            self.criterion = ConfidenceLoss(ignore_index=ignore_index, class_weights=self.hparams["class_weights"])
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )
        
    def configure_metrics(self) -> None:
        num_classes = self.hparams["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        metrics = MetricCollection(
            {
                "binary_accuracy": BinaryAccuracy(
                    ignore_index=ignore_index,
                    multidim_average="global",
                ),
                "binary_jaccard_0-5": BinaryJaccardIndex(
                    threshold=0.5, ignore_index=ignore_index
                ),
                "binary_jaccard_0-3": BinaryJaccardIndex(
                    threshold=0.3, ignore_index=ignore_index
                ),
                "binary_jaccard_0-1": BinaryJaccardIndex(
                    threshold=0.1, ignore_index=ignore_index
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        
    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]
        num_filters: int = self.hparams["num_filters"]
        dropout_rate: int = self.hparams["dropout_rate"]
        dropout_2d: bool = self.hparams["dropout_2d"]
        first_only: bool = self.hparams["first_only"]
        encoder_depth: int = self.hparams["encoder_depth"]
        decoder_channels = (256, 128, 64, 32, 16)[-encoder_depth:]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                encoder_depth=encoder_depth,
                decoder_channels=decoder_channels
            )
        elif model == "unet_dropout":
            self.model = UnetWithDropout(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
                dropout_rate=dropout_rate,
                dropout_2d = dropout_2d,
                first_only = first_only
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        elif model == "rcf":
            self.model = RCFRegression(input_features=in_channels, num_classes=num_classes)
            
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model in ["unet", "unet_dropout", "deeplabv3+"]:
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "unet_dropout", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "unet_dropout", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
                
    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams["patience"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
    
    def training_step(
        self, batch, batch_idx, dataloader_idx = 0):
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]
        confidence_scores = batch["confidence"]
        y_hat = self(x)
        if self.hparams["loss"] == "confidence_weighted":
            loss = self.criterion(y_hat, y, confidence_scores)
        elif self.hparams["loss"] == "ce_imputed":
            # probability of switching label is inversely proportional to confidence
            prob_switch = 1 - torch.square(confidence_scores/5)
            # generate random numbers to simulate probabilistic draw
            rand_nums = torch.rand(len(prob_switch)).cuda()
            # switch mine to no mine with desired probability (switch occurs if rand_num < prob_switch)
            y_imputed = y*(rand_nums > prob_switch).unsqueeze(1).unsqueeze(2)
            loss = self.criterion(y_hat, y_imputed)
        else:
            loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        y_hat_mine = torch.softmax(y_hat,axis=1)[:,1,:,:] # predicted probability of mine class
        self.train_metrics(y_hat_mine, y)
        self.log_dict(self.train_metrics)
        return loss
                
    def validation_step(
        self, batch, batch_idx, dataloader_idx=0):
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
            
        Returns:
            The predicted mask.
        """
        x = batch["image"]
        y = batch["mask"]
        confidence_scores = batch["confidence"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        if self.hparams["loss"] == "confidence_weighted":
            loss = self.criterion(y_hat, y, confidence_scores=None)
        elif self.hparams["impute_eval"]:
            # probability of switching label is inversely proportional to confidence
            prob_switch = 1 - confidence_scores/5
            # generate random numbers to simulate probabilistic draw
            rand_nums = torch.rand(len(prob_switch)).cuda()
            # switch mine to no mine with desired probability (switch occurs if rand_num < prob_switch)
            y_imputed = y*(rand_nums > prob_switch).unsqueeze(1).unsqueeze(2)
            loss = self.criterion(y_hat, y_imputed)
        else:
            loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        y_hat_mine = torch.softmax(y_hat,axis=1)[:,1,:,:] # predicted probability of mine class
        if self.hparams["impute_eval"]:
            self.val_metrics(y_hat_mine, y_imputed)
        else:
            self.val_metrics(y_hat_mine, y)
        self.log_dict(self.val_metrics)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass
        return torch.softmax(y_hat,dim=1)[:,1] # return output for logging purposes
    
    def test_step(self, batch, batch_id, dataloader_idx = 0):
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        confidence_scores = batch["confidence"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        if self.hparams["loss"]=="confidence_weighted":
            loss = self.criterion(y_hat, y, confidence_scores=None)
        elif self.hparams["impute_eval"]:
            # probability of switching label is inversely proportional to confidence
            prob_switch = 1 - confidence_scores/5
            # generate random numbers to simulate probabilistic draw
            rand_nums = torch.rand(len(prob_switch)).cuda()
            # switch mine to no mine with desired probability (switch occurs if rand_num < prob_switch)
            y_imputed = y*(rand_nums > prob_switch).unsqueeze(1).unsqueeze(2)
            loss = self.criterion(y_hat, y_imputed)
        else:
            loss = self.criterion(y_hat, y)
        
        #if batch_id==0:
        #    print(f"Input data type: {x.dtype}")
        #    print(f"Example input: {x[0]}")
        #    print(f"Sum of each channel: {x[0].sum(axis=-1).sum(axis=-1)}")
        #    print(f"Example output: {y_hat_hard[0]}")
        
        self.log("test_loss", loss)
        y_hat_mine = torch.softmax(y_hat,axis=1)[:,1,:,:] # predicted probability of mine class
        if self.hparams["impute_eval"]:
            self.test_metrics(y_hat_mine, y_imputed)
        else:
            self.test_metrics(y_hat_mine, y)
        self.log_dict(self.test_metrics)
