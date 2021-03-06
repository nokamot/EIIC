# EIIC
This is a repository of codes for Extended Invariant Information Clustering is Effective for Leave-One-Site-Out Cross-Validation in Resting State Functional Connectivity Modelling

## Requirements
We used docker container of pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime for analysis.
Besides it, scikit-learn, pandas and openpyxl are required.

## How to run

### 1. Download data
Download preprocessed ROI time series of ABIDE dataset and phenotypic data from [Download page of ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/download.html).

### 2. Set config file
1. Make 2 directories for saving intermediate files (input data and labels) and final output (trained models).
2. Edit 4 relative path items of param_set.py.

    1. source_dir: Directory of intermediate files
    2. output_dir: Directory of result files
    3. preparation_params['label_file_path']: Phenotypic data
    4. preparation_params['path_structure']: Each ROI time series files

### 3. Train models and output results
1. Pull PyTorch docker image.
```bash
sudo docker pull pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
```
2. Start docker container mounting a directory including these codes and necessary files we mentioned in section 2 (Set config file).
```bash
sudo docker run -it --rm --gpus device=0 -v /Path/to/codes/in/host:/Path/to/codes/in/container pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
```

3. Install required packages by pip and run in the container.
```bash
pip install scikit-learn pandas openpyxl
cd /Path/to/codes/in/container
python run.py
```
