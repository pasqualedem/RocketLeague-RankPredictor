# Rocket League Rank Predictor

CRISP-DM case study for Big Data course in Artificial Intelligence MSc.

Prediction of the rank of a player given its statistic from the match


See [docs](./docs/main.pdf) for futher details.

## Requirements

 Python 3.9+

## Installation

Create a virtual enviroment

    python -m venv venv
    
Install requirements

    pip install -r requirements.txt
    
## Usage

### Retrieve replays

Configure `PARAMETERS` variable in retrieve.py and then:

    python src/retrieve.py
    
### Preprocessing

Launch jupyter lab

    jupyter-lab
    
Follow data_exploration.ipynb and data_preprocess.ipynb

### Modeling

Launch grid searches using:
        
    python src/train.py

Parameters are set in `parameters.yaml`.
It is already filled with parameters used in the case study.

Change from `dataset_normal_preprocessed.csv` to `dataset_factored.csv` to run experiments with factored dataset

### View results

Use the interactive view of mlflow:

    mlflow ui

