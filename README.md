# STAT288 Final Project
This repository contains the code used to perform analysis for a STAT288 final project entitled "Deep Learning Detection of Artisanal Mines in Satellite Imagery: Multi-Phase Challenges." Running the code depends on data files that are not provided due to their use in a larger ongoing research project; therefore this serves as a reference only. The scripts used to train models were run on Harvard University's FAS Research Computing cluster via SLURM job submissions. Results were logged using Weights and Biases, with the project located [here](https://wandb.ai/asm_detect/ASM_stat288?nw=nwuserkayan).

**Repository contents**
- notebooks
    - `00_exploratory_analysis.ipynb`: used to get counts for the number of samples per confidence rating and to plot representative examples
    - `01_split_by_confidence.ipynb`: used to generate train/test/val splits for method 1 experiments (training with high-confidence data only)
- scripts
    - `asm_datamodules.py`: custom Dataset and DataModule class
    - `asm_losses.py`: custom loss class that implements uncertainty-informed weighting
    - `asm_models.py`: custom classes for UNet with dropout and associated Lightning Trainer; contains detail about model architecture and training
    - `asm_run_unet_single.py`: script used to run training and evaluation of a new UNet model, with desired parameters specified at the top of the file
    - `asm_train_test_split.py`: used to generate a new train-test-val split using a number of possible protocols, with option to save split as pickle file
    - `run_experiment.sbatch`: batch script used to submit job to SLURM scheduler on the FASRC cluster
