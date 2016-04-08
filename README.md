# orthoclean

## Purpose

## Methodology

## Installation/Required software

1. python
1. scikit-learn
1. aliscore
1. mafft

## Tutorial

1. Training a filtering model

```bash
# Get a dataset from OrthoDB.
# This can be done via the OrthoDB website, or you can use wget if you know how to query their APIs
wget "http://orthodb.org/fasta?query=&level=6656&species=6656&universal=1&singlecopy=0.9"

# Run the model training script
# This script will take care of everything for you after you have a dataset from OrthoDB, includeing:
#   1. Parsing the sequences into their OrthoDB Groups
#   2. Generate false-positive homology clusters from the true-positive homology clusters
#   3. Align the clusters using MAFFT
#   4. Featurize the clusters
#   5. Train a filtering model
bin/train_model.py
```

1. Filtering using a trained model

## Citing this package

## Acknowledgements
