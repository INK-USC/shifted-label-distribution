Please download [NYT](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing) and unzip the file here. Then execute the following pre-processsing script.

```
sh ./brown_clustering.sh NYT
sh ./feature_generation.sh NYT
python DataProcessor/gen_data_neural.py --in_dir ./data/intermediate/NYT/rm --out_dir ./data/neural/NYT
python DataProcessor/gen_bag_level_data.py --in_dir ./data/neural/NYT --out_dir ./data/neural_att/NYT
```