### Example Usage

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

TACRED
```
CoType/retype-rm -data TACRED -mode m -size 50 -negative 3 -threads 3 -alpha 0.0001 -samples 1 -iters 1000 -lr 0.01
python2 CoType/Evaluation/emb_dev_n_test.py extract TACRED retypeRm cosine 0.0
```