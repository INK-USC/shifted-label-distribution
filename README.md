Code for EMNLP 2019 paper "Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction" [[Link]](https://arxiv.org/abs/1904.09331)

_Todo: Briefly introduce our findings here._

### Content

- [Environment Setup](#environment-setup)
- [Download and Pre-processing](#download-and-pre-processing)
- [Running Instructions](#running-instructions)

### Environment Setup
We set up our environment in Anaconda (version: 5.2.0, build: py36_3) with the following commands.
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

Click on the model name to see the instructions on how to run each model.

#### Feature-based Models

Run `conda activate shifted` first to activate the environment for feature-based models.

1. [ReHession](ReHession/README.md)
2. [CoType](CoType/README.md)
3. [Logistic Regression](LogisticRegression/README.md)

#### Neural Models

Run `conda activate shifted-neural` first to activate the environment for neural models.

1. [Bi-GRU / Bi-LSTM / PCNN / CNN](Neural/README.md)
2. [Bi-GRU + ATT / PCNN + ATT](NeuralATT/README.md)