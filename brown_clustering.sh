#!/usr/bin/env bash

Data=$1
echo $Data

### Split Original Train Data into 90% Train and 10% Dev
python2 DataProcessor/dev_set_partition.py --dataset $Data --ratio 0.1

### Generate brown file (clusters raw.txt into 300 clusters)
cd data/source/$Data
python3 generateBClusterInput.py
cd ../../..
cd DataProcessor/brown-cluster/
make
./wcluster --text ../../data/source/$Data/bc_input.txt --c 300 --output_dir ../../data/source/$Data/brown-out
cd ../../
mv data/source/$Data/brown-out/paths data/source/$Data/brown
