Code for EMNLP 2019 paper "Looking Beyond Label Noise: Shifted Label Distribution Matters in Distantly Supervised Relation Extraction" [[Link]](https://arxiv.org/abs/1904.09331)

### Environment Setup

```
conda create --name shifted
conda activate shifted
conda install pytorch=0.3.1
deactivate shifted

conda create --name shifted-neural
conda activate shifted-neural
conda install cudnn=7.1.2 pytorch=0.4.0 tqdm
deactivate shifted-neural
```

### Pre-processing

```
conda activate shifted
sh ./brown_clustering.sh KBP
sh ./feature_generation.sh KBP
python3 DataProcessor/gen_data_neural.py --in_dir ./data/intermediate/KBP/rm --out_dir ./data/neural/KBP
```
Replace 'KBP' with 'NYT' to generate features for NYT

Also, download our pruned word embeddings and word2id file, and place in `./data/neural/vocab`

### Feature-based Models

For feature-based models, run `conda activate shifted` first to activate the environment.

#### 1. ReHession

KBP (hyper-params are using the default settings)
```
python ReHession/run.py --seed 1
python ReHession/eva.py --seed 1
```


NYT
```
python ReHession/run.py --dataset NYT --info NYT-default --input_dropout 0.5 --output_dropout 0.0 --seed 1
```

Note:

By default, run.py trains the default model. Set "--bias fix" to use "Fix Bias" as said in the paper.
By default, eva.py evaluates the performance (1) without threshold, (2) with max threshold, (3) with entropy threshold. Set "--bias set" to enable "Set Bias" during evaluation. If you train a model with "--bias fix", you should pass the same flag to eva.py.

#### 2. CoType

KBP

```
CoType/retype-rm -data KBP -mode m -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -iters 2000 -lr 0.001
python2 CoType/Evaluation/emb_dev_n_test.py extract KBP retypeRm cosine 0.0
```
NYT
```
CoType/retype-rm -data NYT -mode m -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -iters 1000 -lr 0.01
python2 CoType/Evaluation/emb_dev_n_test.py extract NYT retypeRm cosine 0.0
```

#### 3. Logistic Regression

First, move to the model directory with `cd LogisticRegression`

KBP (data_dir is using default)
```
python2 train.py
python2 test.py
```

NYT
```
python2 train.py --save_filename result_nyt.pkl --data_dir ../data/intermediate/NYT/rm
python2 test.py --save_filename result_nyt.pkl
```

### Neural Models

First activate the environment with `source activate shifted-neural`

#### 1. Bi-GRU / Bi-LSTM / PCNN / CNN

KBP (data_dir is using default)
```
python Neural/train.py --repeat 1
python Neural/eva.py --repeat 1
```

You may specify the dataset, save directory, hyperparams (dropout, lr, lr_decay, etc.) by passing arguments. Try `python Neural/train.py -h` to check the usage of each argument.
