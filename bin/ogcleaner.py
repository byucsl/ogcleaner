'''
Copyright (C) 2016 Masaki Stanley Fujimoto

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import sys, argparse, re, time, os, pickle
import numpy as np
from subprocess32 import call, Popen, PIPE, TimeoutExpired
from pathlib import Path
from multiprocessing import Pool, Lock
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier

from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import math, warnings

from sklearn.utils import shuffle as skshuffle

from random import shuffle


available_models = "svm,neural_network,random_forest,naive_bayes,logistic_regression"
default_models = "neural_network"
available_features_str = "aliscore,length,num_seqs,num_gaps,num_amino_acids,range,amino_acid_charged,amino_acid_uncharged,amino_acid_special,amino_acid_hydrophobic"
available_features = available_features_str.split( ',' )


# our own meta-classifier
class Metaclassifier:
    '''
    a meta-classifier that uses several other base classifiers to produce its own classifications
    '''

    def __init__( self, model_names = default_models.split( ',' ) ):
        '''
        model_names = a python list of model names, the model names should correspond to the used models
        '''
        self.model_names = model_names
        for x in model_names:
            model_class = model_to_class[ x ]
            self.models = [ model_class( **( model_params[ model_class ] ) ) for x in model_names ]
        self.skip_models = [ 0 ] * len( self.models )
        self.models.append( MLPClassifier() )

    def generate_base_classifier_predictions( self, x, sx ):
        predictions = []
        # create predictions for all classifier except for the last one, the meta-classifier
        for idx, model in enumerate( self.models[ : -1 ] ):
            # if the model was untrainable during the bootstrapping process, we mark everything
            # as a homology cluster
            if self.skip_models[ idx ] == 1:
                predictions.append( np.array( [ 'H' ] * len( x ) ) )
            else:
                if isinstance( model, MultinomialNB ):
                    predictions.append( model.predict( x ) )
                else:
                    predictions.append( model.predict( sx ) )

        predictions = pd.DataFrame( predictions ).replace( { 'H' : 1, "NH" : 0 } ).transpose()
        predictions.columns = self.model_names
        return predictions

    def train_base_classifiers( self, x, sx, y ):
        for idx, model in enumerate( self.models[ : -1 ] ):
            try:
                if isinstance( model, MultinomialNB ):
                    # don't scale for NB
                    model.fit( x, y )
                else:
                    # scale it for all others
                    model.fit( sx, y )
            except ValueError:
                # could not train the model
                # should only happen during bootstrap analysis when both H and NH are not present in
                # the training data, not too concerned about this
                # mark a model as untrained
                self.skip_models[ idx ] = 1
            #except ConvergenceWarning:
            #    # the model didn't converge
            #    pass

    def fit( self, x, sx, y ):
        self.train_base_classifiers( x, sx, y )
        predictions = self.generate_base_classifier_predictions( x, sx )
        try:
            self.models[ -1 ].fit( predictions, y )
        except:
            pass
        #except ConvergenceWarning:
        #    # the model didn't converge
        #    pass

    def predict( self, x, sx ):
        predictions = self.generate_base_classifier_predictions( x, sx )
        return self.models[ -1 ].predict( predictions )

# global variables
version = "0.1a"
p = Path( __file__ )
base_path = p.resolve().parents[ 1 ]
max_rand_int = 100000
default_mafft_path = str( base_path ) + "/lib/mafft_bin/mafft"
default_mafft_opts = "--anysymbol"
default_paml_path = str( base_path ) + "/lib/paml_bin/evolverRandomTree"
default_seqgen_path = str( base_path ) + "/lib/seq-gen_bin/seq-gen"
default_aliscore_path = str( base_path ) + "/lib/Aliscore_v.2.0/Aliscore.02.2.pl"
default_seqgen_opts = "-mWAG -k1 -n1"

# number of replicates for bootstrap analysis when testing the models
# TODO: change this to 100
replicates = 30

# amino acid properties
# courtesy of Nick Jensen, thanks Nick!
AAcodes = {
        'W' : 4,
        'I' : 4,
        'E' : 1,
        'S' : 2,
        'D' : 1,
        'P' : 3,
        'Y' : 4,
        'F' : 4,
        'U' : 3,
        'V' : 4,
        'K' : 1,
        'M' : 4,
        'G' : 3,
        'N' : 2,
        'H' : 1,
        'R' : 1,
        'L' : 4,
        'Q' : 2,
        'T' : 2,
        'C' : 3,
        'A' : 4
        }
code_to_class = {
        3 : 'AAspecial',
        2 : 'AAuncharged_polar',
        1 : 'AAcharged',
        4 : 'AAhydrophobic'
        }
aa_special = [ 'P', 'U', 'G', 'C' ]
aa_uncharged_polar = [ 'S', 'N', 'Q', 'T' ]
aa_charged = [ 'E', 'D', 'K', 'H', 'R' ]
aa_hydrophobic = [ 'W', 'I', 'Y', 'F', 'V', 'M', 'L', 'A' ]

bootstrap_percentages_str = "1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"
bootstrap_percentages = map( float, bootstrap_percentages_str.split( ',' ) )

model_to_class = {
        "svm" : SVC,
        "neural_network" : MLPClassifier,
        "random_forest" : RandomForestClassifier,
        "naive_bayes" : MultinomialNB,
        "logistic_regression" : LogisticRegression,
        "metaclassifier" : Metaclassifier
        }

model_params = {
        SVC : {
            "kernel" : "poly",
            "cache_size" : 4000
            },
        RandomForestClassifier : {},
        MultinomialNB : {},
        LogisticRegression : {},
        Metaclassifier : {
            "model_names" : [ "svm", "neural_network", "naive_bayes", "logistic_regression" ]
            },
        MLPClassifier : {}
        }

def share_models( model_to_class_, model_params_ ):
    global model_to_class, model_params
    model_to_class = model_to_class_
    model_params = model_params_


def init_child(lock_):
    global lock
    lock = lock_


def init_child_train_models( lock_, x_, y_ ):
    global lock, x, y
    lock = lock_
    x = x_
    y = y_


def init_child_featurize( lock_, working_dir_, label_, aliscore_path_, skip_aliscore_, aliscore_timeout_, skip_prev_aliscore_ ):
    global lock, working_dir, label, aliscore_path, skip_aliscore, aliscore_timeout, skip_prev_aliscore
    lock = lock_
    working_dir = working_dir_
    label = label_
    aliscore_path = aliscore_path_
    skip_aliscore = skip_aliscore_
    aliscore_timeout = aliscore_timeout_
    skip_prev_aliscore = skip_prev_aliscore_


def init_child_cluster_seqs( lock_, nh_groups_dir_ ):
    global lock, nh_groups_dir
    lock = lock_
    nh_groups_dir = nh_groups_dir_


def init_child_tree_builder( lock_, config_file_paths_, paml_path_, logs_dir_, paml_trees_dir_ ):
    global lock, config_file_paths, paml_path, logs_dir, paml_trees_dir
    lock = lock_
    config_file_paths = config_file_paths_
    paml_path = paml_path_
    logs_dir = logs_dir_
    paml_trees_dir = paml_trees_dir_


def init_child_cluster_gen( lock_, trees_, nh_groups_dir_, seqgen_path_, seqgen_opts_, logs_dir_, evolved_seqs_dir_ ):
    global lock, trees, nh_groups_dir, seqgen_path, seqgen_opts, logs_dir, evolved_seqs_dir
    lock = lock_
    trees = trees_
    nh_groups_dir = nh_groups_dir_
    seqgen_path = seqgen_path_
    seqgen_opts = seqgen_opts_
    logs_dir = logs_dir_
    evolved_seqs_dir = evolved_seqs_dir_


def remove_ambiguous_amino_acids( seq ):
    return seq.replace( 'U', '' ).replace( 'B', '' ).replace( 'X', '' ).replace( 'Z', '' )


def generate_paml_config( seed, v ):
    config_text = ( "0        * 0: paml format (mc.paml); 1:paup format (mc.nex)\n"
    "{}   * random number seed (odd number)\n\n"
    "{} 20 10 * <# seqs>  <# nucleotide sites>  <# replicates>\n\n"
    "0.1 0.2 0.3 1.5   * birth rate, death rate, sampling fraction, and mutation rate (tree height)\n\n"
    "0          * model: 0:JC69, 1:K80, 2:F81, 3:F84, 4:HKY85, 5:T92, 6:TN93, 7:REV\n\n"
    "1 * kappa or rate parameters in model\n\n"
    "0.5  4     * <alpha>  <#categories for discrete gamma>\n\n"
    "0.1 0.2 0.3 0.4    * base frequencies\n\n"
    "T   C   A   G" )

    return config_text.format( seed, int( v ) )


def segregate_orthodb_groups( fasta_file_path, groups_dir, og_field_pos ):

    errw( "\tSegregating the sequences into their orthogroups..." )
    # read in the fasta file
    # separate into groups as you're reading
    #orthogroups = dict()
    regex = re.compile( "orthodb\d+_OG=(\S*)" )

    group_names = []

    with open( fasta_file_path ) as fh:
        cur_group = ""
        cur_seq_header = ""
        cur_seq_seq = ""

        cur_group_seqs = []

        for line in fh:
            if line[ 0 ] == '>':
                if og_field_pos != -1:
                    t_group = line.split()[ og_field_pos ]
                else:
                    m = regex.search( line )
                    t_group = m.group( 1 )

                if cur_seq_header != "":
                    cur_group_seqs.append( cur_seq_header + "\n" + cur_seq_seq )

                if t_group != cur_group:
                    if cur_group != "":
                        group_names.append( cur_group )
                        with open( groups_dir + "/" + cur_group, 'w' ) as out:
                            for item in cur_group_seqs:
                                out.write( item )
                                out.write( "\n" )
                    cur_group_seqs = []
                    cur_group = t_group
                
                cur_seq_header = line.strip()
                cur_seq_seq = ""
            else:
                # TODO: possible remove the need for removing ambiguous amino acids because
                #       mafft allows for any character matching with the option --anysymbol
                cur_seq_seq += remove_ambiguous_amino_acids( line.strip() )
        cur_group_seqs.append( cur_seq_header + "\n" + cur_seq_seq )
        with open( groups_dir + "/" + cur_group, 'w' ) as out:
            group_names.append( cur_group )
            for item in cur_group_seqs:
                out.write( item )
                out.write( "\n" )

        errw( " Done!\n" )

        return group_names


def align_cluster( aligner_path, aligner_opts, cluster_path, cluster_name, aligned_dir, logs_dir ):
    align_out_file_path = aligned_dir + "/" + cluster_name
    align_stderr_file_path = logs_dir + "/" + cluster_name + ".mafft_alignment.stderr.out"
    
    with open( align_out_file_path, 'w' ) as alignment_fh, open( align_stderr_file_path, 'w' ) as stderr_fh:
        status = call( [ aligner_path, aligner_opts, cluster_path ], stdout = alignment_fh, stderr = stderr_fh )
        
        with lock:
            errw( "\t\tAligning " + cluster_path + "..." )
            if status != 0:
                errw( "Alignment error!!\n" )
            else:
                errw( " Done!\n" )


def align_worker( item ):
    aligner_path = item[ 0 ]
    aligner_opts = item[ 1 ]
    clusters_out_path = item[ 2 ]
    cluster_name = item[ 3 ]
    aligned_dir = item[ 4 ]
    logs_dir = item[ 5 ]
    align_cluster( aligner_path, aligner_opts, clusters_out_path, cluster_name, aligned_dir, logs_dir )


def align_clusters( aligner_path, aligner_opts, clusters_dir, cluster_names, aligned_dir, threads, logs_dir ):
    errw( "\tAligning clusters...\n" )

    tasks = []
    for cluster_name in cluster_names:
        tasks.append( ( aligner_path, aligner_opts, clusters_dir + "/" + cluster_name, cluster_name, aligned_dir, logs_dir ) )

    # randomize the tasks because clusters with lower IDs take longer to align due to how OrthoMCL assigns IDs
    shuffle( tasks )


    lock = Lock()
    pool = Pool( threads, initializer = init_child, initargs = ( lock, ) )
    pool.map( align_worker, tasks )

    pool.close()
    pool.join()
    errw( "\tDone aligning clusters!\n" )


def generate_paml_configs( outputs_dir ):
    errw( "\t\tGenerating PAML config files..." )
    # these are fixed hyperparameters
    # TODO: make these user-settable
    s = np.random.normal( loc = 50, scale = 15, size = 1000 )
    file_paths = []

    for i, v in enumerate( s ):
        paml_config_text = generate_paml_config( str( np.random.randint( max_rand_int ) ), v ) 
        file_path = outputs_dir + "/tree" + str( i )
        with open( file_path, 'w' ) as fh:
            fh.write( paml_config_text )
        file_paths.append( ( str( i ), file_path ) )
    
    errw( " Done!\n" )
    return file_paths


def generate_paml_tree( item ):
    id = item[ 0 ]
    config_path = item[ 1 ]
    output_path = paml_trees_dir + "/" + str( id )
    log_out = logs_dir + "/" + str( id ) + ".paml"
    with open( log_out, 'w' ) as log_fh:
        status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh, stderr = log_fh )
        tries = 0
        # It looks like PAML's exit codes aren't correct... we'll ignore them for now
        #while status != 0 or tries < 2:
            #with lock:
            #    errw( "\t\t\tTree " + config_path + " generation failed... trying again\n" )
            #status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh )
            #tries += 1
        status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh, stderr = log_fh )

        with lock:
            if status != 0:
                errw( "\t\t\tTree " + config_path + " generation..." )
                errw( "failed!\n" )
            #else:
            #    errw( "Success!\n" )


def generate_paml_trees( paml_trees_dir, config_file_paths, threads, paml_path, logs_dir ):
    errw( "\t\tGenerating PAML tree construction task list..." )
    # generate a list of tasks
    tasks = config_file_paths
    errw( " Done!\n" )

    errw( "\t\tConstructing trees...\n" )

    # send the list of tasks to a pool of threads to do work
    lock = Lock()
    pool = Pool(
            threads,
            initializer = init_child_tree_builder,
            initargs = (
                lock,
                config_file_paths,
                paml_path,
                logs_dir,
                paml_trees_dir,
                )
            )
    pool.map( generate_paml_tree, tasks )
    pool.close()
    pool.join()

    errw( "\t\tDone constructing trees!\n" )

    return [ ( x[ 0 ], paml_trees_dir + "/" + str( x[ 0 ] ) ) for x in tasks if os.path.isfile( paml_trees_dir + "/" + str( x[ 0 ] ) ) ]


def merge_all_trees( tree_file_paths ):

    all_trees = []
    for tree_id, tree_file_path in tree_file_paths:
        with open( tree_file_path, 'r' ) as fh:
            for line in fh:
                if line[ 0 ] == '(':
                    all_trees.append( line.strip() )
    return all_trees


def evolve_seq( id, invariable_sites, output_dir, header, seq, tree ):
    seq_id = header[ 1 : ].split()[ 0 ].split( ':' )[ 1 ]
    seqgen_input = "1\t" + str( len( seq ) ) + "\n" + seq_id + "\t" + seq + "\n1\n" + tree
    id = id.split()[ 0 ]
    output_path = output_dir + "/" + id + "/" + seq_id + ".evolved"

    with open( output_path, 'w' ) as fh, open( logs_dir + "/" + id + "." + seq_id + ".seqgen_out", 'w' ) as stderr_out:
        p = Popen( [ default_seqgen_path ] + seqgen_opts + [ "-i" + str( invariable_sites ) ], stdout = fh, stdin = PIPE, stderr = stderr_out )
        #fh.write( "\n" )
        p.communicate( input = seqgen_input )
        p.wait()


def generate_evolved_sequence( item ):
#def generate_evolved_sequence( ( id, cluster_path, invariable_sites ) ):
    id = item[ 0 ]
    cluster_path = item[ 1 ]
    invariable_sites = item[ 2 ]
    new_cluster = []

    dir_check( evolved_seqs_dir + "/" + id )

    headers = []
    seqs = []

    with open( cluster_path, 'r' ) as fh:
        header = ""
        seq = ""
        for line in fh:
            if line[ 0 ] == ">":
                if header != "":
                    # evolve individual sequence
                    #evolved_seq = evolve_seq( id, nh_groups_dir, header, seq, tree )
                    #new_cluster.append( ( header, evolved_seq ) )

                    headers.append( header )
                    seqs.append( seq )

                # reset the sequence
                header = line.strip()
                seq = ""
            else:
                seq += line.strip()
        #evolved_seq = evolve_seq( id, nh_gruop_dir, header, seq, tree )
        #new_cluster.append( ( header, evolved_seq ) )

        headers.append( header )
        seqs.append( seq )

    tree_indices = np.random.randint( len( trees ), size = len( seqs ) )
    for idx, full_seq in enumerate( zip( headers, seqs ) ):
        header = full_seq[ 0 ]
        seq = full_seq[ 1 ]
        evolve_seq( id, invariable_sites, evolved_seqs_dir, header, seq, trees[ tree_indices[ idx ] ] )

    with lock:
        errw( "\t\t\tEvolved cluster " + id + " with " + str( invariable_sites * 100 ) + "% invariable sites\n" )


def generate_evolved_sequences( nh_groups_dir, all_trees, homology_cluster_paths, threads, seqgen_path, seqgen_opts, logs_dir, evolved_seqs_dir ):
    errw( "\t\tGenerating evolved sequences...\n" )

    # prep tasks
    #tasks = [ x for x in homology_cluster_paths ]
    invariable_sites = [ 0.0 ] * ( len( homology_cluster_paths ) / 3 )
    invariable_sites += [ 0.25 ] * ( len( homology_cluster_paths ) / 3 )
    invariable_sites += [ 0.5 ] * ( len( homology_cluster_paths ) - len( invariable_sites ) )

    shuffle( invariable_sites )

    assert len( invariable_sites ) == len( homology_cluster_paths )

    ids = [ x[ 0 ] for x in homology_cluster_paths ]
    file_paths = [ x[ 1 ] for x in homology_cluster_paths ]
    tasks = zip( ids, file_paths, invariable_sites )

    # distribute tasks
    lock = Lock()
    pool = Pool(
            threads,
            initializer = init_child_cluster_gen,
            initargs = (
                lock,
                all_trees,
                nh_groups_dir,
                seqgen_path,
                seqgen_opts,
                logs_dir,
                evolved_seqs_dir
                )
            )
    pool.map( generate_evolved_sequence, tasks )
    pool.close()
    pool.join()

    errw( "\t\tDone generating evolved sequences!\n" )

    return [ ( x[ 0 ], evolved_seqs_dir + "/" + x[ 0 ] ) for x in homology_cluster_paths ]


def create_evolved_cluster( item ):
    group_id = item[ 0 ]
    dir_path = item[ 1 ]
    cluster_seqs = []

    for dirname, dirnames, filenames in os.walk( dir_path ):
        for filename in filenames:
            full_path = os.path.join( dirname, filename )

            if os.stat( full_path ).st_size == 0:
                continue

            with open( full_path ) as fh:
                spec_seqs = []
                species_name = filename[ : filename.rfind( '.' ) ]
                fh.next()
                for line in fh:
                    spec_seqs.append( line.strip().split()[ 1 ] )
                if len( spec_seqs ) == 0:
                    continue

                rand_seq = spec_seqs[ np.random.randint( len( spec_seqs ) - 1 ) ]
                cluster_seqs.append( ( species_name, rand_seq ) )

    with open( nh_groups_dir + "/" + group_id, 'w' ) as fh:
        for header, seq in cluster_seqs:
            fh.write( ">" + header + "\n" + seq + "\n" )


def create_evolved_clusters( evolved_cluster_dir_paths, threads, nh_groups_dir ):
    errw( "\t\tForming clusters of evolved sequences...\n" )
    lock = Lock()
    pool = Pool(
            threads,
            initializer = init_child_cluster_seqs,
            initargs = (
                lock,
                nh_groups_dir
                )
            )
    pool.map( create_evolved_cluster, evolved_cluster_dir_paths )
    pool.close()
    pool.join()
    
    errw( "\t\tDone forming clusters!\n" )
    
    return [ x[ 0 ] for x in evolved_cluster_dir_paths ]


def generate_nh_clusters( orthodb_group_paths, evolved_seqs_dir, nh_groups_dir, paml_configs_dir, paml_trees_dir, threads, paml_path, logs_dir, seqgen_path, seqgen_opts ):
    errw( "\tGenerating false-positive homology clusters...\n" )
    # generate paml config files
    config_file_paths = generate_paml_configs( paml_configs_dir )

    # generate paml trees
    tree_file_paths = generate_paml_trees( paml_trees_dir, config_file_paths, threads, paml_path, logs_dir )

    # concatenate all trees into a single file
    all_trees = merge_all_trees( tree_file_paths )

    # evolve sequences
    evolved_cluster_dir_paths = generate_evolved_sequences(
            nh_groups_dir,
            all_trees,
            orthodb_group_paths,
            threads,
            seqgen_path,
            seqgen_opts,
            logs_dir,
            evolved_seqs_dir )

    # form clusters
    evolved_cluster_paths = create_evolved_clusters(
            evolved_cluster_dir_paths,
            threads,
            nh_groups_dir
            )

    errw( "\tDone generating false-positive homology clusters!\n" )

    return evolved_cluster_paths


def featurize_cluster( item ):
    idx = item[ 0 ]
    group = item[ 1 ]
    path = item[ 2 ]

    # compute all features unless it's aliscore
    length = 0
    num_seqs = 0
    num_gaps = 0
    num_aa = 0
    range = 0
    clust_aa_charged = []
    clust_aa_uncharged = []
    clust_aa_special = []
    clust_aa_hydrophobic = []

    min_seq_len = float( 'infinity' )
    max_seq_len = 0

    with open( path, 'r' ) as fh:
        seq_aa_charged = 0
        seq_aa_uncharged = 0
        seq_aa_special = 0
        seq_aa_hydrophobic = 0
        seen_seq = False

        # while counting up amino acids also prep cluster for aliscore
        # headers for sequences must be formatted for aliscore to
        # not throw an error
        with open( working_dir + "/" + group, 'w' ) as ofh:
            for line in fh:
                if line[ 0 ] == '>':
                    num_seqs += 1

                    fixed_header = line.split()[ 0 ].replace( ':', '' ).replace( '|', '' ).replace( '(', '' ).replace( ')', '' ).replace( ';', '' ).replace( "--", '' ).replace( ',', '' ).replace( '*', '' )
                    ofh.write( fixed_header + "\n" )

                    if seen_seq:
                        clust_aa_charged.append( seq_aa_charged )
                        clust_aa_uncharged.append( seq_aa_uncharged )
                        clust_aa_special.append( seq_aa_special )
                        clust_aa_hydrophobic.append( seq_aa_hydrophobic )
                    seq_aa_charged = 0
                    seq_aa_uncharged = 0
                    seq_aa_special = 0
                    seq_aa_hydrophobic = 0
                    seen_seq = True
                    length = 0
                else:
                    ofh.write( line )
                    length += len( line.strip() )

                    c_gaps = line.count( '-' )
                    seq_len = len( line.strip() ) - c_gaps

                    num_gaps += c_gaps
                    num_aa += seq_len

                    if seq_len > max_seq_len:
                        max_seq_len = seq_len
                    if seq_len < min_seq_len:
                        min_seq_len = seq_len

                    # aa_special = [ 'P', 'U', 'G', 'C' ]
                    # aa_uncharged_polar = [ 'S', 'N', 'Q', 'T' ]
                    # aa_charged = [ 'E', 'D', 'K', 'H', 'R' ]
                    # aa_hydrophobic = [ 'W', 'I', 'Y', 'F', 'V', 'M', 'L', 'A' ]

                    # count amino acid types
                    ## charged
                    seq_aa_charged += sum( line.count( x ) for x in aa_charged )

                    ## uncharged
                    seq_aa_uncharged += sum( line.count( x ) for x in aa_uncharged_polar )

                    ## special
                    seq_aa_special += sum( line.count( x ) for x in aa_special )

                    ## hydrophobic
                    seq_aa_hydrophobic += sum( line.count( x ) for x in aa_hydrophobic )
            clust_aa_charged.append( seq_aa_charged )
            clust_aa_uncharged.append( seq_aa_uncharged )
            clust_aa_special.append( seq_aa_special )
            clust_aa_hydrophobic.append( seq_aa_hydrophobic )

    # calculate all actual values for the cluster
    my_range = max_seq_len - min_seq_len

    # sanity checking, each of these should have something in them.
    assert len( clust_aa_charged ) > 0
    assert len( clust_aa_uncharged ) > 0
    assert len( clust_aa_special ) > 0
    assert len( clust_aa_hydrophobic ) > 0

    clust_aa_charged = np.std( clust_aa_charged, ddof = 1 )
    clust_aa_uncharged = np.std( clust_aa_uncharged, ddof = 1 )
    clust_aa_special = np.std( clust_aa_special, ddof = 1 )
    clust_aa_hydrophobic = np.std( clust_aa_hydrophobic, ddof = 1 )

    # compute aliscore
    # perl Aliscore.02.2.pl -i 80.groupID8701.aln
    aliscore = -1

    aliscore_outfile_path = working_dir + "/" + group + "_List_random.txt"
    if skip_aliscore:
        #errw( "Skipping aliscore and attempting to read from disk..."
        try:
            aliscore = get_aliscore_from_file( aliscore_outfile_path )
        except:
            with lock:
                errw( "\t\t\tAliscore file does not exist...\n" )
    else:
        if skip_prev_aliscore:
            try:
                aliscore = get_aliscore_from_file( aliscore_outfile_path )
            except:
                aliscore = run_aliscore( working_dir, group, aliscore_outfile_path )
                with lock:
                    errw( "\t\t\tAliscore file does not exist, recomputing " + group + " ... Complete!\n" )
        else:
            aliscore = run_aliscore( working_dir, group, aliscore_outfile_path )

    if aliscore == -1:
        aliscore = length

    # store the featurized instance
    #data_dest[ idx ] = [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ]
    #data_dest.put( [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ] )

    if label != '?':
        group = group + "_" + label

    my_data = [ group, aliscore, length, num_seqs, num_gaps, num_aa, my_range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ]

    with lock:
        #data_dest.append( [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ] )
        errw( "\t\t\tFeaturizing of " + group + " Complete\n" )
        #print group, my_data

    return my_data


def get_aliscore_from_file( aliscore_file_path ):
    aliscore = 0
    with open( aliscore_file_path, 'r' ) as fh:
        for line in fh:
            line = line.strip()
            if line == '':
                continue
            aliscore += len( line.split() )
    return aliscore


def run_aliscore( working_dir, group, aliscore_outfile_path ):
    aliscore_pm_path = os.path.dirname( aliscore_path )
    with open( working_dir + "/" + group + ".aliscore.err", 'w' ) as err_fh, open( working_dir + "/" + group + ".aliscore.out", 'w' ) as out_fh:
        try:
            status = call( [ "perl", "-I", aliscore_pm_path, aliscore_path, "-i", working_dir + "/" + group ], stdout = out_fh, stderr = err_fh, timeout = aliscore_timeout )
            if status != 0:
                with lock:
                    errw( "\t\t\tAliscore failed :( ! Continuing...\n" )

            # grab the aliscore from the output
            if status == 0:
                return get_aliscore_from_file( aliscore_outfile_path )
            else:
                #print "ERROR " + str( status )
                # we don't know why it failed so we'll give it an aliscore of 0
                # TODO: check why it failed and handle different errors
                #       only two possible reasons for aliscore to fail:
                #           1. not enough taxa left!
                #           2. taxon names of tree and sequence files do not match!\nprocess terminated!
                return 0
        except TimeoutExpired:
            with lock:
                errw( "\t\t\tAliscore timed out :( ! Continuing...\n" )
            return -1


def featurize_clusters( cluster_dir, working_dir, cluster_ids, threads, label, aliscore_path, skip_aliscore, aliscore_timeout, skip_prev_aliscore ):
    errw( "\t\tFeaturizing " + label + " clusters...\n" )
    tasks = [ ( idx, x, cluster_dir + "/" + x ) for idx, x in enumerate( cluster_ids ) ]

    # randomize the tasks to spread the workload
    shuffle( tasks )

    lock = Lock()
    pool = Pool(
            threads,
            initializer = init_child_featurize,
            initargs = (
                lock,
                working_dir,
                label,
                aliscore_path,
                skip_aliscore,
                aliscore_timeout,
                skip_prev_aliscore,
                )
            )
    featurized_clusters = pool.map( featurize_cluster, tasks )
    pool.close()
    pool.join()

    errw( "\t\tDone featurizing clusters!\n" )

    return featurized_clusters


def save_featurized_dataset( dest_path, readable_data, pickle_dest_path, pd_data ):
    errw( "\t\tSaving featurized data set to " + dest_path + "..." )
    with open( dest_path, 'w' ) as fh:
        for item in readable_data:
            fh.write( ", ".join( map( str, item ) ) + "\n" )

    errw( " Done!\n" )

    errw( "\t\tPickling featurized data set to " + pickle_dest_path + "..." )
    with open( pickle_dest_path, "wb" ) as fh:
        pickle.dump( pd_data, fh )
    errw( " Done!\n" )

def train_model( model ):
    model_class = model_to_class[ model ]
    cur_model = model_class( **( model_params[ model_class ] ) )
    if isinstance( cur_model, Metaclassifier ):
        cur_model.fit( x, sx, y )
    else:
        cur_model.fit( x, y )

    with lock:
        errw( "\t\tTraining " + model + "... Done!\n" )

    return cur_model


def format_data_for_training( data ):
    # prep the data
    ## turn the data into a pandas dataframe with appropriate labels
    columns = [ 'group' ]
    columns.extend( available_features_str.split( ',' ) )
    columns.append( "class" )
    data = pd.DataFrame( data, columns = columns )
    data.set_index( 'group', drop = False, inplace = True )

    return data


def create_trained_model( model, features, data, threads ):
    errw( "\tTraining model\n" )
    # prep the model
    model = model.split( ',' )

    # prep features
    # needed for data prep
    features = features.split( ',' )

    # generate the scaled data
    sdata = data.copy()

    ss = StandardScaler()
    ss.fit( sdata[ features ] )
    sdata[ features ] = ss.transform( sdata[ features ] )

    ## separate into features and labels
    x = data[ features ]
    y = data[ "class" ]

    ## split into test and train
    ## TODO: parameterize test size
    ## TODO: parameterize random state for testing reproducibility
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2 )

    # get the scaled x values after the split
    sx_train = sdata.loc[ x_train.index ][ features ]
    sx_test = sdata.loc[ x_test.index ][ features ]

    errw( "\t\tTrain size: " + str( len( x_train ) ) + " instances\n" )
    errw( "\t\tTest size: " + str( len( x_test ) ) + " instances\n" )


    if len( model ) > 1:
        model_class = model_to_class[ "metaclassifier" ]
    else:
        model_class = model_to_class[ model[ 0 ] ]
    model = model_class( **model_params[ model_class ] )

    if isinstance( model_class, Metaclassifier ):
        model.fit( x_train, sx_train, y_train )
        test_predictions = model.predict( x_test, sx_test )
    elif isinstance( model_class, MultinomialNB ):
        model.fit( x_train, y_train )
        test_predictions = model.predict( x_test )
    else:
        model.fit( sx_train, y_train )
        test_predictions = model.predict( sx_test )

    # output accuracy
    accuracy = accuracy_score( y_test, test_predictions )
    errw( "\t\tTest set accuracy: " + str( accuracy ) + "\n" )

    errw( "\tDone training model!\n" )

    return model, ss


def train_individual_model( model, x_train, sx_train, y_train ):
    if isinstance( model, Meteclassifier ):
        model.fit( x_train, sx_train, y_train )
    elif isinstnace( model, MultinomialNB ):
        model.fit( x_train, y_train )
    else:
        model.fit( sx_train, y_train )


def get_correct_preds( model, x_test, sx_test ):
    if isinstance( model, Meteclassifier ):
        preds = model.predict( x_test, sx_test )
    elif isinstnace( model, MultinomialNB ):
        preds = model.predict( x_test )
    else:
        preds = model.predict( sx_test )
    return preds
   

def bootstrap( model, model_params = {}, data = None, sdata = None, features = available_features, verbosity = 0 ):

    acc_train = []
    acc_test = []
    for perc in bootstrap_percentages:
        # resample data
        num_instances = int( math.ceil( len( data ) * ( perc / 100 ) ) )

        acc_perc_train = []
        acc_perc_test = []

        for rep in range( replicates ):
            train, test, strain, stest = train_test_split( data, sdata, test_size = 0.2 )
            x_test = test[ features ]
            sx_test = stest[ features ]
            
            test_labels = test[ "class" ]
            
            sub_train = train.sample( num_instances, replace = True )
            sub_strain = strain.loc[ sub_train.index ]

            x_train = sub_train[ features ]
            sx_train = sub_strain[ features ]

            train_labels = sub_train[ "class" ]

            mod = model( **model_params )
            try:
                if isinstance( mod, Metaclassifier ):
                    mod.fit( x_train, sx_train, train_labels )
                    
                    preds = mod.predict( x_train, sx_train )
                    acc_perc_train.append( accuracy_score( preds, train_labels ) )
                    
                    preds = mod.predict( x_test, sx_test )
                    acc_perc_test.append( accuracy_score( preds, test_labels ) )
                elif isinstance( mod, MultinomialNB ):
                    mod.fit( x_train, train_labels )
                    
                    preds = mod.predict( x_train )
                    acc_perc_train.append( accuracy_score( preds, train_labels ) )
                    
                    preds = mod.predict( x_test )
                    acc_perc_test.append( accuracy_score( preds, test_labels ) )
                else:
                    mod.fit( sx_train, train_labels )
            
                    preds = mod.predict( sx_train )
                    acc_perc_train.append( accuracy_score( preds, train_labels ) )
                    
                    preds = mod.predict( sx_test )
                    acc_perc_test.append( accuracy_score( preds, test_labels ) )
            except ValueError as e:
                pass

            # train dataset
            

            # test dataset
        
        if verbosity > 0:
            errw( "perc:\t" +  str( perc ) + "\t" + str( np.mean( acc_perc_train ) ) + "\t" + str( np.mean( acc_perc_test ) ) + "\t" + str( np.std( acc_perc_train ) ) + str( np.std( acc_perc_test ) ) + "\n" )
        acc_train.append( acc_perc_train )
        acc_test.append( acc_perc_test )
    return acc_train, acc_test


def gen_plot( model, acc_train, acc_test, y_lim_min = 0.5, y_lim_max = 1.0 ):
    acc_train_ = np.asarray( acc_train )
    acc_test_ = np.asarray( acc_test )

    avgs_train = acc_train_.mean( axis = 1 )
    errs_train = acc_train_.std( axis = 1 )

    avgs_test = acc_test_.mean( axis = 1 )
    errs_test = acc_test_.std( axis = 1 )

    avgs = pd.DataFrame( [ avgs_train, avgs_test ] ).transpose()
    avgs.columns = [ "Train", "Test" ]
    avgs[ "perc" ] = pd.Series( map( int, bootstrap_percentages_str.split( ',' ) ), index = avgs.index )
    avgs.set_index( "perc", inplace = True )

    errs = pd.DataFrame( [ errs_train, errs_test ] ).transpose()
    errs.columns = [ "Train", "Test" ]
    errs[ "perc" ] = pd.Series( map( int, bootstrap_percentages_str.split( ',' ) ), index = errs.index )
    errs.set_index( "perc", inplace = True )

    fig, ax = plt.subplots()
    fig.set_size_inches( 12, 8, forward = True )
    plt.title( model + " accuracy" )
    plt.ylim( y_lim_min, y_lim_max )
    plt.xlabel( "% of total training set" )
    plt.ylabel( "% accuracy" )
    avgs.plot.line( ax = ax, color = [ 'b', 'r' ] )
    plt.fill_between( avgs.index, avgs[ "Train" ] - errs[ "Train" ], avgs[ "Train" ] + errs[ "Train" ], facecolor = 'blue', alpha = 0.2 )
    plt.fill_between( avgs.index, avgs[ "Test" ] - errs[ "Test" ], avgs[ "Test" ] + errs[ "Test" ], facecolor = 'red', alpha = 0.2 )
    plt.xticks( avgs.index.values, map( str, avgs.index.values ), fontsize = 8 )
    leg = plt.legend( fontsize = 8 )


# These are tests that will help verify results of the models
def run_validation( test_dir, data, threads ):
    shuffled_data = data.copy()
    shuffled_data = skshuffle( shuffled_data )
    scaled_shuffled_data = shuffled_data.copy()
    scaled_shuffled_data[ available_features ] = scale( shuffled_data[ available_features ] )
    sdata = scaled_shuffled_data.copy()
    for model_name, model_class in model_to_class.iteritems():
        errw( "\t\tValidating " + model_name + "..." )
        acc_train, acc_test = bootstrap(
                model_class,
                model_params = model_params[ model_class ],
                data = shuffled_data,
                sdata = scaled_shuffled_data
                )
        gen_plot( model_name, acc_train, acc_test, y_lim_min = 0.7 )
        plt.savefig( test_dir + "/model_validation.bootstrap_plot." + model_name + ".png" )
        errw( " Done!\n" )

    for feature in available_features:
        model_for_feature_validation = "neural_network"
        errw( "\t\tValidating " + feature + " with " + model_for_feature_validation + "..." )
        model_class = model_to_class[ model_for_feature_validation ]
        feat_acc_train, feat_acc_test = bootstrap(
                model_class,
                model_params = model_params[ model_class ],
                data = shuffled_data,
                sdata = scaled_shuffled_data,
                features = [ feature ],
                verbosity = 0
                )
        gen_plot( model_for_feature_validation + ": " + feature, feat_acc_train, feat_acc_test, y_lim_min = 0.5 )
        plt.savefig( test_dir + "/feature_validation.bootstrap_plot." + model_for_feature_validation + "." + feature + ".png" )
        errw( " Done!\n" )


def load_featurized_data( file_path ):
    return pickle.load( open( file_path, "rb" ) )


def save_model( save_dir, save_prefix, model, features, trained_model, scaler ):
    errw( "\tSaving model..." )
    
    with open( save_dir + "/" + save_prefix + ".trained_model", "wb" ) as fh:
        pickle.dump( trained_model, fh )

    with open( save_dir + "/" + save_prefix + ".features", 'w' ) as fh:
        fh.write( features )

    with open( save_dir + "/" + save_prefix + ".specified_model", 'w' ) as fh:
        fh.write( model )

    with open( save_dir + "/" + save_prefix + ".scaler", "wb" ) as fh:
        pickle.dump( scaler, fh )

    errw( " Done!\n" )


def errw( text ):
    sys.stderr.write( text )


# return true if it needs to compute, return false if directory already exists and data should be used
def dir_check( dir_path ):
    # check if the output directory exists
    if not os.path.exists( dir_path ):
        status = call( [ "mkdir", dir_path ] )

        if status == 0:
            errw( "Created directory " + dir_path + "\n" )
        else:
            sys.exit( "ERROR! Could not create the directory " + dir_path + ". Aborting!" )

        return True
    return False


def clean_dir( dir ):
    call( [ "rm", "-rf", dir + "/*" ] )


def generate_trained_model( args ):
    errw( "Aligner: " + args.aligner_path + "\n" )
    errw( "Aligner args: " + args.aligner_options + "\n" )

    # verify parameters
    ## verify specified model
    check_model = args.model.split( ',' )
    for model in check_model:
        if model not in model_to_class.keys():
            sys.exit( "ERROR! User specified invalid model: " + model )

    ## verify specified features
    check_feats = args.features.split( ',' )
    feats_avail = available_features_str.split( ',' )
    for feat in check_feats:
        if feat not in feats_avail:
            sys.exit( "ERROR! User specified invalid feature: " + feat )

    if args.test_only:
        args.test = True
    # end verify parameters
    
    if not args.featurized_data:

        if not args.orthodb_fasta:
            sys.exit( "ERROR! User did not provide featurized dataset or OrthoDB fasta file! Abort!" )

        ## check if the output directory exists
        dir_check( args.orthodb_groups_dir )
        dir_check( args.nh_groups_dir )
        dir_check( args.paml_configs_dir )
        dir_check( args.logs_dir )
        dir_check( args.aligned_homology_dir )
        dir_check( args.aligned_nh_dir )
        dir_check( args.paml_trees_dir )
        dir_check( args.evolved_seqs_dir )
        dir_check( args.featurized_clusters_dir )
        dir_check( args.aliscore_homology_dir )
        dir_check( args.aliscore_nh_dir )    

        if args.seed != -1:
            errw( "Setting random number seed to: " + str( args.seed ) + "\n" )
            np.random.seed( args.seed )

        errw( "Beginning...\n" )

        ortho_groups = segregate_orthodb_groups( args.orthodb_fasta, args.orthodb_groups_dir, args.og_field_pos )
        errw( "Number of groups: " + str( len( ortho_groups ) ) + "\n" )

        if args.clean:
            clean_dir( args.logs_dir )

        # align the orthodb clusters
        align_clusters(
                args.aligner_path,
                args.aligner_options,
                args.orthodb_groups_dir,
                ortho_groups,
                args.aligned_homology_dir,
                args.threads,
                args.logs_dir
                )

        if args.clean:
            clean_dir( args.logs_dir )

        # process the seqgen_opts
        args.seqgen_opts = args.seqgen_opts.split()

        # non-homology cluster generation
        nh_group_paths = [ ( x, args.orthodb_groups_dir + "/" + x ) for x in ortho_groups ]
        nh_groups = generate_nh_clusters(
                nh_group_paths,
                args.evolved_seqs_dir,
                args.nh_groups_dir,
                args.paml_configs_dir,
                args.paml_trees_dir,
                args.threads,
                args.paml_path,
                args.logs_dir,
                args.seqgen_path,
                args.seqgen_opts
                )

        if args.clean:
            clean_dir( args.logs_dir )

        # align the non-homology clusters
        align_clusters(
                args.aligner_path,
                args.aligner_options,
                args.nh_groups_dir,
                nh_groups,
                args.aligned_nh_dir,
                args.threads,
                args.logs_dir
                )

        if args.clean:
            clean_dir( args.logs_dir )

        # featurize datasets
        ## featurize orthodb groups
        h_featurized = featurize_clusters(
                args.aligned_homology_dir,
                args.aliscore_homology_dir,
                ortho_groups,
                args.threads,
                'H',
                args.aliscore_path,
                False,
                args.aliscore_timeout,
                False,
                )

        if args.clean:
            clean_dir( args.logs_dir )

        ## featurize nh groups
        nh_featurized = featurize_clusters(
                args.aligned_nh_dir,
                args.aliscore_nh_dir,
                nh_groups, args.threads,
                "NH",
                args.aliscore_path,
                False,
                args.aliscore_timeout,
                False,
                )

        if args.clean:
            clean_dir( args.logs_dir )

        ## format data with correct column headers
        data = format_data_for_training( h_featurized + nh_featurized  )

        if args.clean:
            clean_dir( args.logs_dir )

        ## concatenate the featurized tables into a single file and write to disk
        save_featurized_dataset(
                args.featurized_clusters_dir + "/featurized_data.txt",
                h_featurized + nh_featurized,
                args.featurized_clusters_dir + "/featurized_data.pickle",
                data
                )

        if args.clean:
            clean_dir( args.logs_dir )

        if args.featurize_only:
            errw( "Featurizing dataset successful, quitting because --featurize_only set.\n" )
            sys.exit()
    else:
        data = load_featurized_data( args.featurized_data )

    # run tests
    if args.test:
        dir_check( args.test_dir )
        run_validation( args.test_dir, data, args.threads )

        if args.test_only:
            errw( "Model and feature validation successful, quitting because --test_only set.\n" )
            sys.exit()

    # train model
    model, scaler = create_trained_model( args.model, args.features, data, args.threads )

    # save model
    dir_check( args.trained_model_dir )
    save_model( args.trained_model_dir, args.save_prefix, args.model, args.features, model, scaler )


def parse_cluster_paths( file_path ):
    errw( "\tParsing cluster paths..." )
    cluster_paths = []
    with open( file_path ) as fh:
        for line in fh:
            cluster_paths.append( line.strip() )
    wrre( " Done!\n" )

    return cluster_paths


def load_model( model_prefix ):
    errw( "\tLoading model..." )
    with open( model_prefix + ".trained_model", "rb" ) as fh:
        model = pickle.load( fh )
    with open( model_prefix + ".scaler", "rb" ) as fh:
        scaler = pickle.load( fh )
    errw( " Done!\n" )

    return model, scaler


def read_cluster_names( dir_path ):
    names = []
    for item in os.listdir( dir_path ):
        if os.path.isfile( dir_path + "/" + item ):
            # parse out name and append to list
            names.append( item )
    return names


def classify_clusters( args ):
    errw( "Classifying clusters!\n" )

    # verify parameters
    ## check if all directories required exist
    dir_check( args.logs_dir )
    dir_check( args.featurized_clusters_dir )
    dir_check( args.aliscore_dir )

    # end verify parameters

    # load model
    model, scaler = load_model( args.model_prefix )

    # get cluster names
    cluster_names = read_cluster_names( args.fasta_dir )

    # check if aligned dir exsists
    dir_check( args.aligned_dir )

    if args.skip_align:
        errw( "\t\tSkipping the alignment step...\n" )
    else:
        align_clusters(
                args.aligner_path,
                args.aligner_options,
                args.fasta_dir,
                cluster_names,
                args.aligned_dir,
                args.threads,
                args.logs_dir
                )

    clean_dir( args.logs_dir )

    # featurize clusters
    featurized = featurize_clusters(
            args.aligned_dir,
            args.aliscore_dir,
            cluster_names,
            args.threads,
            '?',
            args.aliscore_path,
            args.skip_aliscore,
            args.aliscore_timeout,
            args.skip_prev_aliscore,
            )
    data = format_data_for_training( featurized )
    #data[ 'names' ] = pd.Series( cluster_names, index = data.index )

    # classify
    x = data[ available_features ]
    sx = scaler.transform( data[ available_features ] )



    if isinstance( model, Metaclassifier ):
        preds = model.predict( x, sx )
    elif isinstance( model, MultinomialNB ):
        preds = model.predict( x )
    else:
        preds = model.predict( sx )

    if args.clean:
        clean_dir( args.logs_dir )

    with open( args.results_file, 'w' ) as fh:
        for name, pred in zip( data[ 'group' ], preds ):
            fh.write( name + "\t" + pred + "\n" )

    errw( "Done!\n" )


def main( args ):
    errw( "Command: " + " ".join( sys.argv ) + "\n" )
    errw( "OrthoClean model training module version " + version + "\n" )

    if args.which == "train":
        generate_trained_model( args )
    elif args.which == "classify":
        classify_clusters( args )

    errw( "Finished!\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Train models to clean putative orthology clusters. Methodology published in Detecting false positive sequence homology: a machine learning approach, BMC Bioinformatics (24 February 2016, http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0955-3). Please contact M. Stanley Fujimoto at sfujimoto@gmail.com for any questions.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
            )

    # sub parsers
    sp = parser.add_subparsers()
    sp_train = sp.add_parser( "train",
            help = "Train filtering models."
            )
    sp_train.set_defaults( which = "train" )

    sp_classify = sp.add_parser( "classify",
            help = "Classify clusters. Cluster classifications are written to standard out, redirect this output to a file to store results."
            )
    sp_classify.set_defaults( which = "classify" )


    # sub parser classify options
    classify_group_in = sp_classify.add_argument_group( "Input", "Input files to run the program." )
    classify_group_in.add_argument( "--fasta_dir",
            type = str,
            required = True,
            help = "A directory containing all your clusters with each sluter in a separate fasta file."
            )
    classify_group_in.add_argument( "--model_prefix",
            type = str,
            required = True,
            help = "Trained model prefix for classification. Prepend any directories to saved models."
            )

    classify_group_opts = sp_classify.add_argument_group( "Options", "Options for running the program." )
    classify_group_opts.add_argument( "--threads",
            type = int,
            default = 1,
            help = "Number of threads to use during alignment (Default 1)."
            )
    classify_group_opts.add_argument( "--aliscore_path",
            type = str,
            default = default_aliscore_path,
            help = "Path to the aliscore binary."
            )
    classify_group_opts.add_argument( "--aligner_path",
            type = str,
            default = default_mafft_path,
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here. NOTE: this program is only designed to work with MAFFT, other aligners will take modification."
            )
    classify_group_opts.add_argument( "--clean",
            default = True,
            action = "store_false",
            help = "Set this flag to delete temporary files as you go."
            )
    classify_group_opts.add_argument( "--aligner_options",
            type = str,
            default = default_mafft_opts,
            help = "Options for your aligner."
            )
    classify_group_opts.add_argument( "--results_file",
            type = str,
            default = "results.txt",
            help = "Cluster classification output file path."
            )
    classify_group_opts.add_argument( "--aliscore_timeout",
            type = int,
            default = 120,
            help = "Time (in seconds) to wait for aliscore to finish."
            )


    classify_group_dir = sp_classify.add_argument_group( "Directories", "Directories where output will be stored." )
    classify_group_dir.add_argument( "--aligned_dir",
            type = str,
            default = "classify_aligned_dir",
            help = "Directory to store cluster alignments."
            )
    classify_group_dir.add_argument( "--logs_dir",
            type = str,
            default = "logs",
            help = "Directory to store progress output from programs."
            )
    classify_group_dir.add_argument( "--featurized_clusters_dir",
            type = str,
            default = "classify_featurized_clusters",
            help = "Directory to store the matrix of featurized clusters."
            )
    classify_group_dir.add_argument( "--aliscore_dir",
            type = str,
            default = "classify_aliscores",
            help = "Directory to store the aliscore for clusters."
            )

    classify_group_skip = sp_classify.add_argument_group( "Skipping options", "Steps that you can skip if they've already been completed." )
    classify_group_skip.add_argument( "--skip_align",
            default = False,
            action = "store_true",
            help = "Skip the alignment process. You should use the same aligner and parameters during classification that you used during training."
            )
    classify_group_skip.add_argument( "--skip_aliscore",
            default = False,
            action = "store_true",
            help = "Skip the aliscore process because you've already computed the aliscore previously."
            )
    classify_group_skip.add_argument( "--skip_prev_aliscore",
            default = False,
            action = "store_true",
            help = "Skip clusters that have already had their aliscore computed. This will attempt to compute the aliscore only for clusters that were not successful. Use the --aliscore_timeout option to give more time to the aliscore algorithm."
            )
    # TODO: implement this option
    classify_group_skip.add_argument( "--skip_featurize",
            default = False,
            action = "store_true",
            help = "Skip the featurizing of clusters and read in from file."
            )


    # sub parser train options
    group_in = sp_train.add_argument_group( "Input", "Input files to run the program." )
    group_in.add_argument( "--orthodb_fasta",
            type = str,
            help = "Fasta file downloaded from orthodb."
            )

    train_group_dir = sp_train.add_argument_group( "Directories", "Directories where output will be stored." )
    train_group_dir.add_argument( "--orthodb_groups_dir",
            type = str,
            default = "train_orthodb_groups_fasta",
            help = "Directory to store OrthoDB sequence clusters in fasta format."
            )
    train_group_dir.add_argument( "--nh_groups_dir",
            type = str,
            default = "train_nh_groups_fasta",
            help = "Directory to store the non-homology sequence clusters in fasta format."
            )
    train_group_dir.add_argument( "--paml_configs_dir",
            type = str,
            default = "train_paml_configs",
            help = "Directory to store the PAML configuration files."
            )
    train_group_dir.add_argument( "--logs_dir",
            type = str,
            default = "logs",
            help = "Directory to store progress output from programs."
            )
    train_group_dir.add_argument( "--aligned_homology_dir",
            type = str,
            default = "train_cluster_alignments_homology",
            help = "Directory to store all OrthoDB homology alignments."
            )
    train_group_dir.add_argument( "--aligned_nh_dir",
            type = str,
            default = "train_cluster_alignments_nh",
            help = "Directory to store all false-positive homology alignments."
            )
    train_group_dir.add_argument( "--paml_trees_dir",
            type = str,
            default = "train_paml_trees",
            help = "Directory to store trees generated from PAML."
            )
    train_group_dir.add_argument( "--evolved_seqs_dir",
            type = str,
            default = "train_evolved_seqs",
            help = "Directory to store evolved sequences."
            )
    train_group_dir.add_argument( "--featurized_clusters_dir",
            type = str,
            default = "train_featurized_clusters",
            help = "Directory to store the matrix of featurized clusters. 1 file will be placed in this directory: a concatenation of the OrthoDB clusters and the non-homology clusters."
            )
    train_group_dir.add_argument( "--aliscore_homology_dir",
            type = str,
            default = "train_aliscores_homology",
            help = "Directory to store the aliscore for all homology clusters."
            )
    train_group_dir.add_argument( "--aliscore_nh_dir",
            type = str,
            default = "train_aliscores_nh",
            help = "Directory to store the aliscore for all non-homology clusters."
            )
    train_group_dir.add_argument( "--trained_model_dir",
            type = str,
            default = "trained_model",
            help = "Directory to save the trained model for classification."
            )

    train_group_aligner = sp_train.add_argument_group( "Aligner options", "Options to use for aligning your sequence clusters." )
    train_group_aligner.add_argument( "--aligner_path",
            type = str,
            default = default_mafft_path,
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here. NOTE: this program is only designed to work with MAFFT, other aligners will take modification."
            )
    train_group_aligner.add_argument( "--aligner_options",
            type = str,
            default = default_mafft_opts,
            help = "Options for your aligner."
            )

    train_group_paml = sp_train.add_argument_group( "PAML options" )
    train_group_paml.add_argument( "--paml_path",
            type = str,
            default = default_paml_path,
            help = "Path to the PAML evolverRandomTree binary."
            )

    train_group_seqgen = sp_train.add_argument_group( "Seq-Gen options", "Options for Seq-Gen, used for evolving sequences." )
    train_group_seqgen.add_argument( "--seqgen_path",
            type = str,
            default = default_seqgen_path,
            help = "Path to the Seq-Gen binary."
            )
    train_group_seqgen.add_argument( "--seqgen_opts",
            type = str,
            default = default_seqgen_opts,
            help = "Options for running Seq-Gen."
            )

    train_group_aliscore = sp_train.add_argument_group( "Aliscore options" )
    train_group_aliscore.add_argument( "--aliscore_path",
            type = str,
            default = default_aliscore_path,
            help = "Path to the aliscore binary."
            )
    train_group_aliscore.add_argument( "--aliscore_timeout",
            type = int,
            default = 120,
            help = "Time (in seconds) to wait for aliscore to finish."
            )


    train_group_model = sp_train.add_argument_group( "Model training", "Models and features available for training" )
    train_group_model.add_argument( "--model",
            type = str,
            default = default_models,
            help = "A comma separated list of models to use. Available models include: " + ", ".join( available_models.split( ',' ) ) + ". If more than one model is selected, a meta-classifier that utilizes stacking is used that combined all specified models."
            )
    train_group_model.add_argument( "--features",
            type = str,
            default = available_features_str,
            help  = "A comma separted list of features to use when training models. Available features include: " + ", ".join( available_features_str.split( ',' ) ) + "."
            )

    train_group_misc = sp_train.add_argument_group( "Misc.", "Extra options you can set when you're running the program." )
    train_group_misc.add_argument( "--threads",
            type = int,
            default = 1,
            help = "Number of threads to use during alignment (Default 1)."
            )
    train_group_misc.add_argument( "--seed",
            type = int,
            default = -1,
            help = "Seed to use for generating trees in PAML and for sampling sequences for non-homology cluster generation."
            )
    train_group_misc.add_argument( "--og_field_pos",
            type = int,
            default = -1,
            help = "0-based index of orthology group ID when using custom data (not from OrthoDB). If using data from OrthoDB, this argument should be ignored."
            )

    train_group_skip = sp_train.add_argument_group( "Skip parts of the pipeline", "If you've already completed parts of the pipeline you can skip steps with these flags. Note: each flag requires you specify an output directory for the step you're skipping. Steps are listed in order, if you skip a step, all steps before it are skipped as well." )
    '''train_group_skip.add_argument( "--skip_segregate",
            type = str,
            help = "Skip segregating the fasta file from OrthoDB into separate fasta files for each group. Provide the path to the directory that contains all the segregated ortho groups in fasta format."
            )
    train_group_skip.add_argument( "--skip_align_orthodb",
            type = str,
            help = "Skip alignment process for each OrthoDB orthology group. Provide the path to the directory with the OrthBD alignments in fasta format."
            )
    train_group_skip.add_argument( "--skip_generate_nh",
            type = str,
            help = "Skip the generation process of false-positive homology clusters. Provide the path to the directory with all false-positive homology clusters in fasta format."
            )
    train_group_skip.add_argument( "--skip_align_nh",
            type = str,
            help = "Skip the alignment process for each false-positive homoloy clusters. Provide the path to the directory with all fasle-positive homology cluster alignments in fasta format."
            )'''
    train_group_skip.add_argument( "--featurize_only",
            default = False,
            action = "store_true",
            help = "Only featurize the data, no testing or model training."
            )
    train_group_skip.add_argument( "--featurized_data",
            type = str,
            help = "Skip all steps and use the pickled, featurized data."
            )
    train_group_skip.add_argument( "--test_only",
            default = False,
            action = "store_true",
            help = "Only perform validation of models and features, do not train final models. If --featurized_data is not set, it will featurize your data and a OrthoDB fasta is required. This flag will turn on --test."
            )

    train_group_save = sp_train.add_argument_group( "Saving", "Options for saving your trained model." )
    train_group_save.add_argument( "--save_prefix",
            type = str,
            default = "filter",
            help = "Save prefix for your trained model"
            )

    train_group_test = sp_train.add_argument_group( "Testing", "Test your machine learning models." )
    train_group_test.add_argument( "--test",
            default = False,
            action = "store_true",
            help = "Do boostrapping analysis, tests, etc. to see how well your trained models are performing."
            )
    train_group_test.add_argument( "--test_dir",
            type = str,
            default = "tests",
            help = "Directory to store output of running tests."
            )
    
    train_group_clean = sp_train.add_argument_group( "Cleaning", "There are many intermediary files that are generated while running this program, set this flag to clean as you go." )
    train_group_clean.add_argument( "--clean",
            default = True,
            action = "store_false",
            help = "Set this flag to delete temporary files as you go."
            )
    args = parser.parse_args()

    main( args )

