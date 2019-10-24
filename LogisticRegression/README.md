### Example Usage

First, move to the model directory with `cd LogisticRegression`

KBP (Using default args)
```
python2 train.py
python2 test.py
```

NYT
```
python2 train.py --save_filename result_nyt.pkl --data_dir ../data/intermediate/NYT/rm
python2 test.py --save_filename result_nyt.pkl --data_dir ../data/intermediate/NYT/rm
```

TACRED
```
python2 train.py --save_filename result_tacred.pkl --data_dir ../data/intermediate/TACRED/rm
python2 test.py --save_filename result_tacred.pkl --data_dir ../data/intermediate/TACRED/rm
```