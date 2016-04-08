# orthoclean

## Purpose

This software package is designed as an all-inclusive source for taking putative orthology clusters and filtering them.

## Methodology

Our methodology is based on a machine-learning approach published in BMC Bioinformatics 

## Required software

1. python 2
1. scikit-learn
1. [Aliscore](https://www.zfmk.de/en/research/research-centres-and-groups/aliscore)
1. [MAFFT](http://mafft.cbrc.jp/alignment/software/)
1. [PAML](http://abacus.gene.ucl.ac.uk/software/paml.html)
1. [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

Note: all necessary software packages (aside from the python modules) are included.
The python modules can be installed via pip and the included requirements.txt.
You can use your own installation of each of these software packages, but we suggest using the included packages.
Follow these steps to install all software.

```bash
# Install python dependencies (with root permissions)
pip install -r requirements.txt

# Install python dependenceis (without root permissions)
pip install --user -r requirements.txt

# Install Aliscore

# Install MAFFT

# Install PAML

# Install Seq-Gen
```

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

2. Filtering using a trained model

## Citing this package

Please use the following to cite us:

```tex
@article{fujimoto2016detecting,
  title={Detecting false positive sequence homology: a machine learning approach},
  author={Fujimoto, M Stanley and Suvorov, Anton and Jensen, Nicholas O and Clement, Mark J and Bybee, Seth M},
  journal={BMC bioinformatics},
  volume={17},
  number={1},
  pages={1},
  year={2016},
  publisher={BioMed Central}
}
```

## Acknowledgements

The authors would like to thank:

1. BYU Computational Sciences Laboratory
1. Christophe Giraud-Carrier
