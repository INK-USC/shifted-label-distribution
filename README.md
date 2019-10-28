[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


<h1 align="center">Shifted Label Distribution Matters</h1>
<h5 align="center">Looking Beyond Label Noise in Distantly Supervised Relation Extraction</h5>

TL;DL We identify __shifted label distribution__, an important yet long-overlooked issue in DSRE; introduce a simple yet effective __bias adjustment__ to adapt a trained model along such shift explicitly; release a RE codebase. 

## Table of Contents

- [Introduction](#introduction)
- [Example](#example)
- [Training Recipes](#training-recipes)
    - [Environment Setup](#environment-setup)
    - [Download and Pre-processing](#download-and-pre-processing)
    - [Commands for training](#running-instructions)
        - Feature-based models
            - [ReHession](ReHession/README.md)
            - [CoType](CoType/README.md)
            - [Logistic Regression](LogisticRegression/README.md)
        - Neural models
            - [Bi-GRU / Bi-LSTM / PCNN / CNN / PositionAware-LSTM](Neural/README.md)
            - [Bi-GRU+ATT / PCNN+ATT](NeuralATT/README.md)
- [Reference](#reference)

## Introduction

We observed __diminishing phenomenon__ in model performance (i.e., neural models outperform feature-based models by a lot on human-annoated dataset, but the gap diminishes on DS datasets.)
In addtion, we found two __heuristic threshold__ techniques to be effective on DS datasets.
These observation leads to __shifted label distribution__, an important yet long-overlooked issue in DSRE.
We further introduced __bias adjustment__ to adapt a trained model along such shift explicitly.

<p align="center"><img width="70%" src="overview.png"/></p>

## Example

__Repository Highlight__: 
We release code and provide detailed instructions for all models used in the paper, 
including feature-based models (ReHession, CoType and Logistic Regression) 
and neural models (Bi-GRU, Bi-LSTM, PCNN, CNN, PositionAware LSTM, Bi-GRU+ATT, PCNN+ATT). 
Below is an example for ...

## Training Recipes

### Environment Setup
We set up our environment in Anaconda3 (version: 5.2.0, build: py36_3) with the following commands.
```
conda create --name shifted
conda activate shifted
conda install pytorch=0.3.1
source deactivate

conda create --name shifted-neural
conda activate shifted-neural
conda install cudnn=7.1.2 pytorch=0.4.0 tqdm
source deactivate
```

### Download and Pre-processing

Please check data download and pre-processing instructions in each data directory in `./data`. Also, check [this](data/neural/vocab/README.md) to download our processed word embeddings and word2id file.


### Running Instructions

Click the model name to view the running instructions for each model.

#### Feature-based Models

Run `conda activate shifted` first to activate the environment for feature-based models.

1. [ReHession](ReHession/README.md)
2. [CoType](CoType/README.md)
3. [Logistic Regression](LogisticRegression/README.md)

#### Neural Models

Run `conda activate shifted-neural` first to activate the environment for neural models.

1. [Bi-GRU / Bi-LSTM / PCNN / CNN / PositionAware-LSTM](Neural/README.md)
2. [Bi-GRU+ATT / PCNN+ATT](NeuralATT/README.md)


## Reference
Please cite the following paper if you find the paper and the code to be useful : -)
```
@inproceedings{ye2019shifted,
 title={Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction},
 author={Ye, Qinyuan and Liu, Liyuan and Zhang, Maosen and Ren, Xiang},
 booktitle={Proc. of EMNLP-IJCNLP},
 year={2019}
}
```
