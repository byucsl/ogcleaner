# orthoclean

## Purpose

This software package is designed as an all-inclusive source for taking putative orthology clusters and filtering them.

## Methodology

Our methodology is based outlined in [**Detecting false positive sequence homology: a machine learning approach**](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0955-3) published in BMC Bioinformatics on 24 February 2016.

## Required software

1. python 2
1. scikit-learn
1. [Aliscore](https://www.zfmk.de/en/research/research-centres-and-groups/aliscore)
   1. This program requires perl
1. [MAFFT](http://mafft.cbrc.jp/alignment/software/)
1. [PAML](http://abacus.gene.ucl.ac.uk/software/paml.html)
1. [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

Note: all necessary software packages (aside from the python modules) are included.
The python modules can be installed via pip and the included requirements.txt.
You can use your own installation of each of these software packages, but we suggest using the included packages.
Follow these steps to install all software.

### Compiling MAFFT
We include a modified version of MAFFT that is altered for installation without root permissions.
No other modifications were made to it, feel free to use your own MAFFT installation if you already have it by using the ```--aligner_path``` option.
We suggest using the included MAFFT package.

### Compiling PAML
For this application, we require the PAML evolverRandomTree package.
This is not built in the default PAML software package.
The version of PAML that is included in this software package contains the modifications as outlined in the [PAML documentation](http://www.molecularevolution.org/molevolfiles/paml/pamlDOC.pdf) necessary to compile the evolverRandomTree binary.
It also contains modifications that allow the evolverRandomTree program to save output to a user-specified destination.
It is suggested that you use the included PAML distribution in this package unless you are able to make the necessary modifications to your PAML installation.

## Installation

```bash
# Python dependenecies
## With root permissions
pip install -r requirements.txt

## Without root permissions
pip install --user -r requirements.txt

# Install Aliscore
make aliscore

# Install MAFFT
make mafft

# Install PAML
make paml

# Install Seq-Gen
make seq-gen
```

## Tutorial

### Training a filtering model

```bash
# Get a dataset from OrthoDB.
# This can be done via the OrthoDB website, or you can use wget if you want to query their APIs directly
wget "http://orthodb.org/fasta?query=&level=6656&species=6656&universal=1&singlecopy=0.9"

# Run the model training script on the included test dataset (a very small subset of OrthoDB data)
# This script will take care of everything for you after you have a dataset from OrthoDB, includeing:
#   1. Parsing the sequences into their OrthoDB Groups
#   2. Generate false-positive homology clusters from the true-positive homology clusters
#   3. Align the clusters using MAFFT
#   4. Featurize the clusters
#   5. Train a filtering model
python bin/train_model.py --orthodb_fasta data/small.fasta
```
This script will train a model for you and save the model to disk to be used in the following script.
It also generates lots of intermediary files that can be removed if you do not wish to keep them.
Use the ```make rm_int``` command to remove all intermediary files but still retain the trained models.
Note that this command only removes the default folders, if you specify your own folders during runtime they must be manually deleted.

### Filtering using a trained model

```bash
# This will use the trained model in created in the previous step.
run this commmand
```

You now have a filtered set of orthology clusters!

### Notes on running the program:

The train_model.py script is all-inclusive and will do everything for you.
There are flags provided to skip steps in the pipeline.
These flags are listed in the order that they are evaluated in the pipeline.
If you skip a step all previous steps will be skipped as well.
Each flag requires you to pass in a path to the directory containing the expected output from all previous steps.
The skip flags are:

```
  --skip_segregate SKIP_SEGREGATE
                        Skip segregating the fasta file from OrthoDB into
                        separate fasta files for each group. Provide the path
                        to the directory that contains all the segregated
                        ortho groups in fasta format.
  --skip_align_orthodb SKIP_ALIGN_ORTHODB
                        Skip alignment process for each OrthoDB orthology
                        group. Provide the path to the directory with the
                        OrthBD alignments in fasta format.
  --skip_generate_nh SKIP_GENERATE_NH
                        Skip the generation process of false-positive homology
                        clusters. Provide the path to the directory with all
                        false-positive homology clusters in fasta format.
  --skip_align_nh SKIP_ALIGN_NH
                        Skip the alignment process for each false-positive
                        homoloy clusters. Provide the path to the directory
                        with all fasle-positive homology cluster alignments in
                        fasta format.
```

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
1. Quinn Snell
