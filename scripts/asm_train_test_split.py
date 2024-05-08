import os
from pathlib import Path
import pickle
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_confident_train(
    path="../data/filtered_labels.geojson", 
    data_path="/n/holyscratch01/tambe_lab/kayan/karena/images/",
    stratify_col="country", 
    save = True,
    out_path = "../data/train_test_split",
    confidence_threshold = 5,
    maintain_mine_prop = True,
    random_state = None
):
    """Split data into train/test/val sets. Train set consists only of samples with confidence 5, sampled to maintain the ratio of positive and negative samples. Val and test set sizes are taken from the remaining samples (confidence < 5) in a 0.16/0.2 ratio, also sampled to maintain the ratio of positive and negative samples.
    """  
    
    label_df = gpd.read_file(path)
        
    # take out any files that are not present in the image directory
    dir_ids = [Path(file_name).stem for file_name in os.listdir(data_path)]
    label_df = label_df[label_df["unique_id"].isin(dir_ids)]
    
    # restrict to only use samples meeting confidence threshold
    confident_label_df = label_df[label_df["confidence"] >= confidence_threshold]
    unconfident_label_df = label_df[label_df["confidence"] < confidence_threshold]

    if maintain_mine_prop:
        # retain original proportion of mines in confident dataset
        confident_mine_df = confident_label_df[confident_label_df["label"] == 1]
        confident_nomine_df = confident_label_df[confident_label_df["label"] == 0]

        num_mines_confident = len(confident_mine_df)
        scale_factor = len(label_df[label_df["label"] == 0])/len(label_df[label_df["label"] == 1])
        confident_nomine_df = confident_nomine_df.sample(int(num_mines_confident*scale_factor))

        confident_label_df = pd.concat([confident_mine_df, confident_nomine_df])
            
    # the high confident samples become our train dataset
    train = confident_label_df
    
    if maintain_mine_prop:
        # retain original proportion of mines in unconfident dataset
        unconfident_mine_df = unconfident_label_df[unconfident_label_df["label"] == 1]
        unconfident_nomine_df = unconfident_label_df[unconfident_label_df["label"] == 0]

        num_nomines_unconfident = len(unconfident_nomine_df)
        scale_factor = len(label_df[label_df["label"] == 1])/len(label_df[label_df["label"] == 0])
        unconfident_mine_df = unconfident_mine_df.sample(int(num_nomines_unconfident*scale_factor))

        unconfident_label_df = pd.concat([unconfident_mine_df, unconfident_nomine_df])
    
    # split into val/test
    val, test = train_test_split(unconfident_label_df, 
                stratify=unconfident_label_df[stratify_col] if stratify_col is not None else None,
                test_size=5/9, # with a 0.16-0.2 val-test ratio
                random_state=random_state
            )
                                  
    # get unique identifiers for each split
    train_ids = train["unique_id"].values
    val_ids = val["unique_id"].values
    test_ids = test["unique_id"].values
    
    split_ids = {"train": train_ids, "val": val_ids, "test":test_ids}
    if save:
        # save as pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(split_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path

def split_data_confident_balanced_split(
    path="../data/filtered_labels.geojson", 
    data_path="/n/holyscratch01/tambe_lab/kayan/karena/images/",
    stratify_col="country", 
    save = True,
    out_path = "../data/train_test_split",
    confidence_threshold = 5,
    maintain_mine_prop = True,
    random_state = None
):
    """Split data into train/test/val sets. Train set consists only of samples with confidence 5, sampled to maintain the ratio of positive and negative samples. The train/val/test set sizes are chosen to maintain a 0.64/0.16/0.2 ratio, and val/test sets are sampled to maintain the ratio of confident to unconfident samples.
    """  
    
    label_df = gpd.read_file(path)

    # take out any files that are not present in the image directory
    dir_ids = [Path(file_name).stem for file_name in os.listdir(data_path)]
    label_df = label_df[label_df["unique_id"].isin(dir_ids)]

    # restrict to only use samples meeting confidence threshold
    confident_label_df = label_df[label_df["confidence"] >= confidence_threshold]
    unconfident_label_df = label_df[label_df["confidence"] < confidence_threshold]

    if maintain_mine_prop:
        # retain original proportion of mines in confident dataset
        confident_mine_df = confident_label_df[confident_label_df["label"] == 1]
        confident_nomine_df = confident_label_df[confident_label_df["label"] == 0]

        num_mines_confident = len(confident_mine_df)
        scale_factor = len(label_df[label_df["label"] == 0])/len(label_df[label_df["label"] == 1])
        confident_nomine_df = confident_nomine_df.sample(int(num_mines_confident*scale_factor))

        confident_label_df = pd.concat([confident_mine_df, confident_nomine_df])

        # retain original proportion of mines in unconfident dataset
        unconfident_mine_df = unconfident_label_df[unconfident_label_df["label"] == 1]
        unconfident_nomine_df = unconfident_label_df[unconfident_label_df["label"] == 0]

        num_nomines_unconfident = len(unconfident_nomine_df)
        scale_factor = len(label_df[label_df["label"] == 1])/len(label_df[label_df["label"] == 0])
        unconfident_mine_df = unconfident_mine_df.sample(int(num_nomines_unconfident*scale_factor))

        unconfident_label_df = pd.concat([unconfident_mine_df, unconfident_nomine_df])

    # split confident dataset into train/val/test
    train_confident, test_confident = train_test_split(confident_label_df, 
            stratify=confident_label_df[stratify_col] if stratify_col is not None else None,
            test_size=0.2,
            random_state=random_state
        )
    train_confident, val_confident = train_test_split(train_confident, 
                stratify=train_confident[stratify_col] if stratify_col is not None else None,
                test_size=0.2,
                random_state=random_state
            )

    # subsample unconfident labels to maintain train-val-test split sizes
    scale_factor = len(label_df[label_df["confidence"] < confidence_threshold])/len(label_df[label_df["confidence"] >= confidence_threshold])
    unconfident_label_df = unconfident_label_df.sample(int((len(test_confident)+len(val_confident))*scale_factor))
    # split unconfident dataset into train/val/test
    val_unconfident, test_unconfident = train_test_split(unconfident_label_df, 
                stratify=unconfident_label_df[stratify_col] if stratify_col is not None else None,
                test_size=5/9,
                random_state=random_state
            )

    train = train_confident
    val = pd.concat([val_confident, val_unconfident])
    test = pd.concat([test_confident, test_unconfident])
                                  
    # get unique identifiers for each split
    train_ids = train["unique_id"].values
    val_ids = val["unique_id"].values
    test_ids = test["unique_id"].values
    
    split_ids = {"train": train_ids, "val": val_ids, "test":test_ids}
    if save:
        # save as pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(split_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path

def split_asm_data(
    path="/n/home07/kayan/asm/data/filtered_labels.geojson", 
    data_path="/n/holyscratch01/tambe_lab/kayan/karena/images/",
    stratify_col="country", 
    save = True,
    out_path = "/n/home07/kayan/asm/data/train_test_split",
    random_state = None
):
    """Split data into train/test/val sets.

    Args:
        path: str, optional
            path to geojson file used to load dataframe. Default is '/n/home07/kayan/data/filtered_labels.geojson'
        data_path: str, optional
            path to directory with image files, used to cross-reference unique_ids in geojson file. Default is '/n/holyscratch01/tambe_lab/kayan/karena/images/'
        stratify_col: str, optional
            the name of the column used to stratify the data. Default is 'country'.
        save: bool, optional
            whether or not to save the split in a pickle file. Default is True.
        out_path: str, optional
            path used to save file if save is True. Default is '/n/home07/kayan/data/train_test_split'
        n: int, optional
            restrict split to first n items in dataframe. Default is None.
        mines_only: bool, optional
            restrict data to only images that have mines in them. Default is False.
        random-state: int, optional
            random seed used to generate train-test split. Default is None.
    """  
    
    label_df = gpd.read_file(path)
        
    # take out any files that are not present in the image directory
    dir_ids = [Path(file_name).stem for file_name in os.listdir(data_path)]
    label_df = label_df[label_df["unique_id"].isin(dir_ids)]
    
    # split into train/val and test
    train, test = train_test_split(label_df, 
                stratify=label_df[stratify_col] if stratify_col is not None else None,
                test_size=0.2,
                random_state=random_state
            )
    # split further into train and val
    train, val = train_test_split(train,
                stratify=train[stratify_col] if stratify_col is not None else None,
                test_size=0.2,
                random_state=random_state)
                                  
    # get unique identifiers for each split
    train_ids = train["unique_id"].values
    val_ids = val["unique_id"].values
    test_ids = test["unique_id"].values
    
    split_ids = {"train": train_ids, "val": val_ids, "test":test_ids}
    if save:
        # save as pickle file
        with open(out_path, 'wb') as handle:
            pickle.dump(split_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path