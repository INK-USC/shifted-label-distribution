Please download [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24) from LDC and place the three __json__ files here.
```
./train.json
./dev.json
./test.json
```

You may return to project root and execute the following to generate transform the data format to feature-based models (saved to `{ROOT}/data/source/TACRED`).
```
python DataProcessor/gen_tacred.py 
```