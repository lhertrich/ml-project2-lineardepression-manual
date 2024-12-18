# Road Segmentation with Machine Learning
This is the repository for the project 2 of the course CS433 - Machine Learning at EPFL in 2024/25 for the group lineardepression. It was created manually because GitHub classroom is not working properly.

This repository contains all files used to create the results mentioned in the project report. The goal of the project is to do binary semantic segmenation of roads from satellite images. Therefore different models were tested and a combination of DeepLabV3+ and SegFormer was found to provide the best result with an AICrowd *F1*-score of **0.914**. Please follow the instructions below to use this project and have a look at the results section to recreate our results.

# Project Setup Instructions

To use this project follow the setup instructions provided. You can use a `conda` environment (recommended) or a standard Python virtual environment.

---

## Installation Instructions

1. Install Anaconda if not already installed.

2. Create a new conda environment:
   ```bash
   conda create -n project_env python=3.11
   ```
3. Activate the conda environment:
    ```bash
   conda activate project_env
   ```

4. Install pip inside the environment:
    ```bash
   conda install pip
   ```
5. Install pytorch on your machine:
    Visit https://pytorch.org and follow the instructions to install pytorch on your system. Take care to install the CUDA version if you want to use GPU support locally.

6. Install project dependencies:
    After installing pytorch successfully install the project dependencies with `pip` by running the command:
    ```bash
   pip install -r requirements.txt
   ```

## Project Data

The directory structure of our data for this project is the following:

```
├── data <- Project data files 
│ 
│   ├── external_data <- External datasets
│   │   ├── chicago <- Contains the Chicago external dataset 
│   │
│   ├── test_set_images <- Test set images for evaluation 
│   │
│   ├── training <- Training dataset directory 
│   │   ├── augmented <- Augmented training data 
│   │   │   ├── images <- Augmented training images 
│   │   │   ├── masks <- Masks corresponding to augmented images 
│   │   │
│   │   ├── chicago_data_augmented <- Augmented Chicago training data 
│   │   │   ├── images <- Augmented images for Chicago dataset 
│   │   │   ├── masks <- Masks corresponding to augmented images 
│   │   │
│   │   ├── groundtruth <- Ground truth masks for training 
│   │   │
│   │   ├── groundtruth_binarize <- Binarized ground truth masks 
│   │   │
│   │   ├── images <- Original training images 
```
It is crucial to maintain this structure so all files within this project work properly. You can obtain this structure by following these steps:
1. Download the original dataset from: https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files
2. Unpack the dataset and move the folders inside the project data folder
3. We use an external dataset which can be downloaded from: https://zenodo.org/record/1154821/files/chicago.zip
4. Unpack the dataset, create a subfolder `external_data` and move the unpacked `chicago` folder in it
5. Augment the original data by running:
    ```bash 
    python -m data_preparation.data_augmentation
    ``` 
    from the root directory

6. Augment the external chicago data by running:
    ```bash 
   python -m data_preparation.preprocess_external_data
    ``` 
    from the root directory

You now should have the same data structure as shown in the picture.

# Results
To recreate our results, you can simply run the `run.py` file in the root directory by running:
```bash
python run.py
```
from the root directory.
This file trains two models with the external chicago dataset and combines them to create a `final_submission.csv` file on the root level. We recommend using CUDA to create the submission. The runtime of this file with support of an A100 GPU on Google Colab was around 25 minutes. 

To recreate our results when testing different models, you can run:
```bash
python -m models.model_name.model_name_train
```
Except for `combined_model` which was only used for submission and `logistic_regression` which was only used once to create a simple baseline. The produced results, like train and validation loss plot and sample predictions, are then stored in the `trained_models` folder.
