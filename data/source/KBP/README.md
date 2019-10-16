Please download [KBP](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RjFLUkVQTm93WVU?usp=sharing) and unzip the file here. Then execute the following pre-processsing script.

```
sh ./brown_clustering.sh KBP
sh ./feature_generation.sh KBP
python DataProcessor/gen_data_neural.py --in_dir ./data/intermediate/KBP/rm --out_dir ./data/neural/KBP
python DataProcessor/gen_bag_level_data.py --in_dir ./data/neural/KBP --out_dir ./data/neural_att/KBP
```