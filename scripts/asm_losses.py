import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from functools import partial

class ConfidenceLoss(_Loss):
    '''Custom loss function that attaches weight lambda to each sample based on its label confidence'''
    def __init__(self, ignore_index=None, class_weights=[0.5,0.5]):
        super().__init__()
        ignore_value = -1000 if ignore_index is None else ignore_index
        # class-weighted cross-entropy loss as base loss function; reduction="none" to keep batch dimension
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_value, 
                                              weight=torch.Tensor(class_weights), 
                                              reduction="none")
        
    def forward(self, y_pred, y_true, confidence_scores):
        ce_loss = self.ce_loss_fn(y_pred, y_true)
        if confidence_scores is not None:
            # remove torch.square() to test linear weighting
            scaled_loss = (ce_loss * torch.square((confidence_scores/5)).unsqueeze(1).unsqueeze(2)).mean()
        else:
            scaled_loss = ce_loss.mean()
        return scaled_loss
