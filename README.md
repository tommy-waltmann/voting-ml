# voting-ml
Using machine learning to analyze voting decisions.

git clone git@github.com:tommy-waltmann/voting-ml.git

git fetch --recurse-submodules

```
cd ./voting_ml
python3 example_*.py
```

main.py - to run over specific feature selection method
main_common_fts.py - to run over all manually selected features and to run over features common in three selection methods
main_combined_fts.py - to run over features which are union of first 7 features from all three selection methods