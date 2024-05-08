import pickle
import matplotlib.pyplot as plt
import kornia.augmentation as K
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchgeo.datasets import NonGeoDataset
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms import AugmentationSequential
from torchvision.transforms import Resize, InterpolationMode, RandomCrop
from torchvision.transforms.functional import pad
import geopandas as gpd
import rasterio
import numpy as np
from asm_train_test_split import *
from asm_models import *

def min_max_transform(sample, target_size=(256,256), crop_size=None):
    '''Perform min-max normalization on each channel of image, and resize image + mask to be 256x256'''
    img = sample["image"].permute(1, 2, 0) # moves spectral channels to final dimension
    mask = sample["mask"]
    
    # min-max scaling by channel
    img = img.numpy()
    img_norm = (img - np.min(img,axis=(0,1))) / (np.max(img,axis=(0,1)) - np.min(img,axis=(0,1))) 
    img_norm = torch.tensor(img_norm).permute(2, 0, 1) # re-permute spectral channels to first dimension
    
    # resize data to be 256x256
    img_norm = Resize((256,256),antialias=True)(torch.unsqueeze(img_norm,0))
    mask = Resize((256,256),interpolation=InterpolationMode.NEAREST)(torch.unsqueeze(mask, 0))
    
    if crop_size is not None:
        crop_transform = RandomCrop(crop_size)
        img_norm = crop_transform(img_norm)
        mask = crop_transform(mask)
    
    sample["image"] = torch.squeeze(img_norm)
    sample["mask"] = torch.squeeze(mask)
    return sample

class ASMDataset(NonGeoDataset):
    splits = ["train", "val", "test"]
    all_bands = ["B", "G", "R", "NIR", "Mask"]
    rgb_bands = ["R", "G", "B"]
    
    def __init__(
        self,
        root = "/n/holyscratch01/tambe_lab/kayan/karena/",
        transforms = None,
        split = "train",
        bands = ["R", "G", "B", "NIR"],
        split_path = "/n/home07/kayan/asm/data/train_test_split",
        label_path = "../data/filtered_labels.geojson",
        **kwargs
    ) -> None:
        """Initialize a new ASMDataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            split: one of "train," "val", or "test"
            bands: the subset of bands to load
            split_path: path to file containing unique identifiers for train/test/val split, generated with scripts/train_test_split.py
            label_path: path to geojson file containing label information
            **kwargs: Additional keyword arguments passed on to transform function
        """  
        self.root = root
        self.transforms = transforms
        self.transform_args = kwargs
        assert split in ["train", "val", "test"]
        self.bands = bands
        self.band_indices = [self.all_bands.index(b) + 1 for b in bands if b in self.all_bands] # add 1 since rasterio starts index at 1, not 0
        
        # get unique identifiers of desired split
        with open(split_path,'rb') as handle:
            split_data = pickle.load(handle)
        self.ids = split_data[split]
        
        # convert unique identifiers to file names
        self.image_filenames = [f"{self.root}images/{unique_id}.tif" for unique_id in self.ids]
        self.mask_filenames = [f"{self.root}rasters/{unique_id}.tif" for unique_id in self.ids]
        
        self.label_df = gpd.read_file(label_path)
        
    def __len__(self):
        """Return the number of chips in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.image_filenames)
        
    def __getitem__(self, index: int):
        """Return item at an index within the dataset.

        Args:
            index: index to return

        Returns:
            a dict containing image, mask, transform, crs, and metadata at index.
        """
        img_fn = self.image_filenames[index]
        mask_fn = self.mask_filenames[index]
        
        img = rasterio.open(img_fn).read(self.band_indices)
        img = torch.from_numpy(np.array(img, dtype=np.float32))
        
        mask = rasterio.open(mask_fn).read(1)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        confidence = self.label_df[self.label_df["unique_id"]==self.ids[index]]["confidence"].values[0]
        
        sample = {"image": img, "mask": mask, "id": self.ids[index], "confidence": confidence}

        if self.transforms is not None:
            sample = self.transforms(sample, **self.transform_args)
            
        return sample
    
    def plot(self, sample):
        # Find the correct band index order
        rgb_indices = []
        for band in self.rgb_bands:
            rgb_indices.append(self.bands.index(band))

        # Reorder and rescale the image
        image = sample["image"][rgb_indices].permute(1, 2, 0)
        image = image.numpy()
        # min-max scaling
        #image_norm = (image - np.min(image,axis=(0,1))) / (np.max(image,axis=(0,1)) - np.min(image,axis=(0,1))) 
        
        # Reorder mask
        mask = sample["mask"]

        # Plot the image
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(image)
        axs[1].imshow(mask,cmap="gray")
        axs[0].axis("off")
        axs[0].set_title("Image")
        axs[1].axis("off")
        axs[1].set_title("Mask")

        return fig
    
class ASMDataModule(NonGeoDataModule):
    def __init__(
        self, 
        batch_size: int = 8, 
        num_workers: int = 1,
        split: bool = False,
        split_n: int = None,
        save: bool = False,
        mines_only: bool = False,
        augment: bool = False,
        **kwargs
    ) -> None:
        """Initialize a new ASMModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            split: Whether or not to perform a new train-test-val split of data.
            split_n: Number of tiles to include in train-test-val split
            mines_only: restrict data to only images that have mines in them
            augment: Whether or not to perform image augmentation
            **kwargs: Additional keyword arguments passed to ASMDataset.
        """
        if augment: 
            self.augmentations = AugmentationSequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90,p=0.5),
                data_keys=["image", "mask"],
            )
        super().__init__(ASMDataset, batch_size, num_workers, **kwargs)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                split = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                split = "val"
            elif self.trainer.testing:
                split = "test"
            elif self.trainer.predicting:
                split = "predict"

            aug = self.augmentations
            if aug is not None and split == "train": 
                batch = aug(batch)

        return batch