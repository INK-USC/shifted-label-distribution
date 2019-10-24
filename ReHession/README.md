### Arguments
You can select dataset, set hyperparameters, choose the way to handle bias term by passing arguments. For simplicity, we're only listing some important arguments here. Check the usage of all available arguments with `python ReHession/run.py -h` and `python ReHession/eva.py -h`

```
run.py
--dataset DATASET     name of the dataset, (KBP|NYT|TACRED).
--bias BIAS           ways to handle bias term, (default|fix).
--info INFO           description, also used as filename to save model.
```

```
eva.py
--dataset DATASET     name of the dataset, (KBP|NYT|TACRED).
--bias BIAS           ways to handle bias term, (default|fix|set)
--info INFO           description, also used as filename to load model.
--thres_ratio THRES_RATIO
                      proportion of data to tune thres.
--bias_ratio BIAS_RATIO
                      proportion of data to estimate bias.
```

By default, `eva.py` evaluates the performance (1) without threshold, (2) with max threshold, (3) with entropy threshold. Set `--bias set` to enable "Set Bias" during evaluation. If you train a model with `--bias fix`, you should pass the same flag to eva.py.


### Example Usage
KBP (Using default args)
```
python ReHession/run.py --seed 1
python ReHession/eva.py --seed 1
```


NYT
```
python ReHession/run.py --dataset NYT --info NYT-default --input_dropout 0.5 --output_dropout 0.0 --seed 1
python ReHession/eva.py --info NYT-default
```


TACRED
```
python ReHession/run.py --dataset TACRED --info TACRED-default --input_dropout 0.2 --output_dropout 0.1 --seed 2
python ReHession/eva.py --info TACRED-default
```