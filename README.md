# OGCleaner (Orthology Group Cleaner)

## Purpose

This software package is designed as an all-inclusive source for taking putative orthology clusters and filtering them.

## Methodology

Our methodology is based outlined in [**Detecting false positive sequence homology: a machine learning approach**](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0955-3) published in BMC Bioinformatics on 24 February 2016.

This application was published in [**The OGCleaner: filtering false-positive homology clusters**](http://bioinformatics.oxfordjournals.org/content/early/2016/09/07/bioinformatics.btw571.abstract) published in Bioinformatics on 10 September 2016.

## Required software

1. python 2
1. [scikit-learn](https://github.com/scikit-learn/scikit-learn)
    1. This currently requires the development branch (0.18.dev0) of scikit-learn for the neural network. You can install the developer branch by following the instructions [here](https://github.com/scikit-learn/scikit-learn).
1. [Aliscore](https://www.zfmk.de/en/research/research-centres-and-groups/aliscore)
   1. This program requires perl
1. [pandas](http://pandas.pydata.org/)
1. [matplotlib](http://matplotlib.org/)
1. [MAFFT](http://mafft.cbrc.jp/alignment/software/)
1. [PAML](http://abacus.gene.ucl.ac.uk/software/paml.html)
1. [Seq-Gen](http://tree.bio.ed.ac.uk/software/seqgen/)

Note: all necessary software packages are included except:

1. python packages
1. scikit-learn dev branch

The python modules can be installed via pip and the included requirements.txt and from [here](https://github.com/scikit-learn/scikit-learn) for scikit-learn.
We strongly suggest using a [virtualenv](https://virtualenv.pypa.io/en/stable/) as a way to set up an isolated python module environment.
Follow these steps to install all other software.

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

# Install Aliscore
make aliscore

# Install MAFFT
make mafft

# Install PAML
make paml

# Install Seq-Gen
make seq-gen

# Install scikit-learn developer branch
git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
python setup.py install
```

**If you don't have root permissions on your system**, you can run ```pip install --user -r requirements.txt``` to install python dependencies and ```python setup.py install --user``` in the scikit-learn directory.

## Tutorial

If you don't want to train your own models and want to skip to filtering an existing set of clusters using a pre-trained model **Using your own data** section.

This program has two modes that are specified as positional arguments when running.
These two modes are: ```train``` and ```classify```.
You can specify them by doing:

```bash
# To train a model
python bin/ogcleaner.py train <additional arguments>

# To classify clusters
python bin/ogcleaner.py classify <additional arguments>
```

See the below sections for walkthroughs.


### Training a filtering model

To train a model you must use the ```train``` positional argument.

```bash
# Get a dataset from OrthoDB.
# This can be done via the OrthoDB website, or you can use wget if you want to query their APIs directly as shownn below
# this file is written to disk with the name 'universal.singlecopy0.9.fasta' as seen in the wget options
wget -O arthro.universal0.8.single0.8.fasta "http://orthodb.org/fasta?query=&level=6656&species=6656&universal=0.8&singlecopy=0.8&limit=100000"

# Run the model training script on the included test dataset (a very small subset of OrthoDB data)
# This script will take care of everything for you after you have a dataset from OrthoDB, includeing:
#   1. Parsing the sequences into their OrthoDB Groups
#   2. Generate false-positive homology clusters from the true-positive homology clusters
#   3. Align the clusters using MAFFT
#   4. Featurize the clusters
#   5. Train a filtering model
python bin/ogcleaner.py train --orthodb_fasta arthro.universal0.8.single0.8.fasta --threads 10 --clean
```

Using the ```--threads NUM_FLAGS``` flag will multithread the program and make it go faster.

This script will train a model for you and save the model to disk to be used in the following script.
The model is saved to disk to default locations but can be set by the ```--trained_model_dir``` and ```--save_prefix``` arguments.
It also generates lots of intermediary files that can be removed if you do not wish to keep them.
Use the ```make clean``` command to remove all intermediary files but still retain the trained models.
You can also use the ```--clean``` to remove the log files as you go.
It is highly suggested that you run with the ```--clean``` option.
Note that this command only removes the default folders, if you specify your own folders during runtime they must be manually deleted.
If you need to rebuild any of the packaged software (Aliscore, MAFFT, etc.) you can run ```make deepclean``` and it will delete all compiled versions of the program.

You can use your own orthology group data to train the model as well.
Simply supply a FASTA file that contains all orthology groups.
Each entry in the FASTA requires an ID that is unique to its respective orthology group.
Use the ```--og_field_pos``` to provie the 0-based index of the orthology group IDs during the training phase.

### Filtering using a trained model

```bash
# This will use the trained model in created in the previous step.
# It will filter the orthodb fasta files, all clusters should come back as H (homology clusters).
python bin/ogcleaner.py classify --fasta_dir train_orthodb_groups_fasta/ --model trained_model/filter --threads 10
```

You now have a filtered set of orthology clusters!

#### Using Your Own Data
To filter your own data, you must have one FASTA file that corresponds to each putative orthology cluster.
The FASTA file must have all protein sequences that belong to that cluster.
This can be generated by using ```scripts/gen_clusters_from_good_proteins.py``` to generate these FASTA files from OrthoMCL files.
Running the following will generate the FASTA cluster files and then run the classification step on them.

```bash
# Generate the FASTA cluster files from OrthoMCL files
python scripts/gen_clusters_from_good_proteins.py --groups groups.txt --proteins goodProteins.fasta --out fasta_cluster/

# Now run the filtering on these clusters
python bin/ogcleaner.py classify --fasta_dir fasta_cluster/ --model trained_model/filter --threads 10
```


### Output files

The results of the filtering will be output into a file called ```results.txt```.
To use these results in conjunction with OrthoMCL, use ```scripts/filter_orthomcl_output.py``` to filter your OrthoMCL ```groups.txt``` file.

#### Using Pre-Trained Models

A pre-trained model is provided in the repository.
Use the prefix ```pretrained_models/pretrained_filter``` while using OGCleaner in classify mode.

### Testing your training dataset

We have also included the ability to reproduce the plots in our paper and for you to be able to validate the effectiveness of your trained models.
This includes doing bootstrap analysis for each model with all features and for each individual feature using a neural network.
To run these tests and generate the plots, use the ```--test``` flag when

```bash
python bin/ogcleaner.py train --orthodb_fasta data/arthro.universal.singlecopy0.9.fasta --test
```

### Notes on running the program:

The ogcleaner.py script is all-inclusive and will do everything for you.
You may save time doing some of the following.

```
  --featurize_only      Only featurize the data, no testing or model training.
  --featurized_data FEATURIZED_DATA
                        Skip all steps and use the pickled, featurized data.
  --test_only           Only perform validation of models and features, do not
                        train final models. If --featurized_data is not set,
                        it will featurize your data and a OrthoDB fasta is
                        required.
```

## Citing this package

Please use the following to cite us:

[**The OGCleaner: filtering false-positive homology clusters**](http://bioinformatics.oxfordjournals.org/content/early/2016/09/07/bioinformatics.btw571.abstract)

## Acknowledgements

The authors would like to thank:

1. BYU Computational Sciences Laboratory
1. Christophe Giraud-Carrier
1. Quinn Snell
