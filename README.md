# EIIC
This is a repository of codes for An Extended Invariant Information Clustering is Effective for the Leave-One-Site-Out Cross-validation in Resting State Functional Connectivity Modelling

## Requirements
We used docker container of pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime for analysis.
Besides it, scikit-learn, pandas and openpyxl are required.

## How to run

### 1. Download data
Download preprocessed ROI timeseries of ABIDE dataset and phenotypic data from [Download page of ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/download.html).

### 2. Set config file
Make 2 directories for save intermidiate files (input data and labels) and final output (trained models), and edit 4 relative path items of param_set.py.

1. source_dir: Directory of intermidiate files
2. output_dir: Directory of result files
3. preparation_params['label_file_path']: Phenotypic data
4. preparation_params['path_structure']: Each ROI timeseries files

### 3. Train models and output results
1. Start docker container mounting a directory including these codes and necessary files we mentiond in section 2 (Set config file).
sudo docker run -it --rm --gpus device=0 -v /Path/to/codes/in/host:Path/to/codes/in/container pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
cd /ws
2. Install required packages by pip and run.
pip install scikit-learn pandas openpyxl
python run.py

