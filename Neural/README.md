### Arguments

You can select dataset, set hyperparameters, choose the way to handle bias term by passing arguments. For simplicity, we're only listing some important arguments here. Check the usage of all available arguments with `python Neural/train.py -h` and `python Neural/test.py -h`

``` 
train.py
--data_dir DATA_DIR   specify dataset with directory.
--model MODEL         model name, (cnn|pcnn|bgru|lstm).
--fix_bias            Train model with fix bias (not fixed by default).
--repeat REPEAT       train the model for multiple times.
--info INFO           description, also used as filename to save model.
```
```
test.py
--info INFO           description, also used as filename to save model.
--repeat REPEAT       test the model for multiple trains.
--thres_ratio THRES_RATIO
                      proportion of data to tune thres.
--bias_ratio BIAS_RATIO
                      proportion of data to estimate bias.
--cvnum CVNUM         # samples to tune thres or estimate bias
--fix_bias            test model with fix bias (not fixed by default).
```

### Example Usage

KBP (Using default args)
```
python Neural/train.py --repeat 1
python Neural/eva.py --repeat 1
```

