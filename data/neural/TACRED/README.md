Please download [TACRED](https://catalog.ldc.upenn.edu/LDC2018T24) from LDC and place the three __json__ files here.
```
./train.json
./dev.json
./test.json
```

After download, you may return to project root and execute the following to generate transform the data format to feature-based models (saved to `{ROOT}/data/source/TACRED`).
```
python DataProcessor/gen_tacred.py
```

For TACRED, we use provided NER tags as feature input, so please modify the following code snippet in `./DataProcessor/Feature/other_features.py` (~line 41).
```python
class EMTypeFeature(AbstractFeature):
    def apply(self, sentence, mention, features):
        for em in sentence.entityMentions:
            if em.start == mention.em1Start and em.end == mention.em1End:
                # features.append('EM1_TYPE_%s' % sentence.ner[em.start]) # comment this line
                features.append('EM1_TYPE_%s' % em.labels) # uncomment this line
            if em.start == mention.em2Start and em.end == mention.em2End:
                # features.append('EM2_TYPE_%s' % sentence.ner[em.start]) # comment this line
                features.append('EM2_TYPE_%s' % em.labels) # uncomment this line
```

Then run feature extraction with the following script, which is similar to the pre-processing of KBP and NYT
```
sh ./brown_clustering.sh TACRED
sh ./feature_generation.sh TACRED
python DataProcessor/gen_bag_level_data.py --in_dir ./data/neural/TACRED --out_dir ./data/neural_att/TACRED
```