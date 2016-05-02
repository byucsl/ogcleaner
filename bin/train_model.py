'''
This script is designed to train the models necessary for 
'''

import sys, argparse, re, time, os, pickle
import numpy as np
from subprocess import call, Popen, PIPE
from pathlib import Path
from multiprocessing import Pool, Lock
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math, warnings
from random import shuffle


available_models = "svm,neural_network,random_forest,naive_bayes,logistic_regression"
available_features = "aliscore,length,num_seqs,num_gaps,num_amino_acids,range,amino_acid_charged,amino_acid_uncharged,amino_acid_special,amino_acid_hydrophobic"


# our own meta-classifier
class Metaclassifier:
    '''
    a meta-classifier that uses several other base classifiers to produce its own classifications
    '''

    def __init__( self, model_names = available_models.split( ',' ) ):
        '''
        model_names = a python list of model names, the model names should correspond to the used models
        '''
        self.model_names = model_names
        self.models = [ model_to_class[ x ]() for x in model_names ]
        self.skip_models = [ 0 ] * len( self.models )
        self.models.append( MLPClassifier() )

    def generate_base_classifier_predictions( self, x ):
        predictions = []
        # create predictions for all classifier except for the last one, the meta-classifier
        for idx, model in enumerate( self.models[ : -1 ] ):
            # if the model was untrainable during the bootstrapping process, we mark everything
            # as a homology cluster
            if self.skip_models[ idx ] == 1:
                predictions.append( np.array( [ 'H' ] * len( x ) ) )
            else:
                predictions.append( model.predict( x ) )

        predictions = pd.DataFrame( predictions ).replace( { 'H' : 1, "NH" : 0 } ).transpose()
        predictions.columns = self.model_names
        return predictions

    def train_base_classifiers( self, x, y ):
        for idx, model in enumerate( self.models[ : -1 ] ):
            try:
                model.fit( x, y )
            except ValueError:
                # could not train the model
                # should only happen during bootstrap analysis when both H and NH are not present in
                # the training data, not too concerned about this
                # mark a model as untrained
                self.skip_models[ idx ] = 1
            #except ConvergenceWarning:
            #    # the model didn't converge
            #    pass

    def fit( self, x, y ):
        self.train_base_classifiers( x, y )
        predictions = self.generate_base_classifier_predictions( x )
        try:
            self.models[ -1 ].fit( predictions, y )
        except:
            pass
        #except ConvergenceWarning:
        #    # the model didn't converge
        #    pass

    def predict( self, x ):
        predictions = self.generate_base_classifier_predictions( x )
        return self.models[ -1 ].predict( predictions )


# global variables
version = "0.1a"
p = Path( __file__ )
base_path = p.resolve().parents[ 1 ]
max_rand_int = 100000
default_mafft_path = str( base_path ) + "/lib/mafft_bin/mafft"
default_paml_path = str( base_path ) + "/lib/paml_bin/evolverRandomTree"
default_seqgen_path = str( base_path ) + "/lib/seq-gen_bin/seq-gen"
default_aliscore_path = str( base_path ) + "/lib/Aliscore_v.2.0/Aliscore.02.2.pl"
default_seqgen_opts = "-mWAG -k1 -n1"

# number of replicates for bootstrap analysis when testing the models
# TODO: change this to 100
num_replicates = 100

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

default_bootstrap_percentages = "1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"

model_to_class = {
        "svm" : SVC,
        "neural_network" : MLPClassifier,
        "random_forest" : RandomForestClassifier,
        "naive_bayes" : MultinomialNB,
        "logistic_regression" : LogisticRegression,
        "metaclassifier" : Metaclassifier
        }


def share_model_to_class( model_to_class_ ):
    global model_to_class
    model_to_class = model_to_class_


def init_child(lock_):
    global lock
    lock = lock_


def init_child_train_models( lock_, x_, y_ ):
    global lock, x, y
    lock = lock_
    x = x_
    y = y_


def init_child_featurize( lock_, working_dir_, label_, aliscore_path_ ):
    global lock, working_dir, label, aliscore_path
    lock = lock_
    working_dir = working_dir_
    label = label_
    aliscore_path = aliscore_path_


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


def segregate_orthodb_groups( fasta_file_path, groups_dir ):

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
                m = regex.search( line )
                t_group = m.group( 1 )
                #print t_group
                if cur_seq_header != "":
                    cur_group_seqs.append( cur_seq_header + "\n" + cur_seq_seq )

                if t_group != cur_group:

                    
                    if cur_group != "":
                        group_names.append( cur_group )
                        #print cur_group + "\t" + str( len( cur_group_seqs ) )
                        # print group to file
                        with open( groups_dir + "/" + cur_group, 'w' ) as out:
                            for item in cur_group_seqs:
                                out.write( item )
                                out.write( "\n" )
                    cur_group_seqs = []
                    cur_group = t_group
                
                cur_seq_header = line.strip()
                cur_seq_seq = ""
            else:
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
        status = call( [ aligner_path, cluster_path ], stdout = alignment_fh, stderr = stderr_fh )
        
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
    #print seqgen_input + "\n"
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

    #with lock:
    #    print group_id, dir_path

    for dirname, dirnames, filenames in os.walk( dir_path ):
        for filename in filenames:
            full_path = os.path.join( dirname, filename )
            print "\t", filename

            if os.stat( full_path ).st_size == 0:
                continue

            with open( full_path ) as fh:
                spec_seqs = []
                species_name = filename[ : filename.rfind( '.' ) ]
                fh.next()
                for line in fh:
                    spec_seqs.append( line.strip().split()[ 1 ] )
                if len( spec_seqs ) == 0:
                    print "\t\tnothing in file"
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

                    fixed_header = line.split()[ 0 ].replace( ':', '' ).replace( '(', '' ).replace( ')', '' ).replace( ';', '' ).replace( '|', '' ).replace( "--", '' ).replace( ',', '' ).replace( '*', '' )
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
    range = max_seq_len - min_seq_len

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
    aliscore_pm_path = os.path.dirname( aliscore_path )

    with open( working_dir + "/" + group + ".aliscore.err", 'w' ) as err_fh, open( working_dir + "/" + group + ".aliscore.out", 'w' ) as out_fh:
        status = call( [ "perl", "-I", aliscore_pm_path, aliscore_path, "-i", working_dir + "/" + group ], stdout = out_fh, stderr = err_fh )

    if status != 0:
        with lock:
            errw( "\t\t\tAliscore failed :( ! Continuing...\n" )

    # grab the aliscore from the output

    aliscore = 0
    if status == 0:
        with open( working_dir + "/" + group + "_List_random.txt", 'r' ) as fh:
            for line in fh:
                line = line.strip()
                if line == '':
                    continue
                aliscore += len( line.split() )

    # store the featurized instance
    #data_dest[ idx ] = [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ]
    #data_dest.put( [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ] )

    with lock:
        #data_dest.append( [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ] )
        errw( "\t\t\tFeaturizing of " + group + " Complete\n" )

    return [ aliscore, length, num_seqs, num_gaps, num_aa, range, clust_aa_charged, clust_aa_uncharged, clust_aa_special, clust_aa_hydrophobic, label ]


def featurize_clusters( cluster_dir, working_dir, cluster_ids, threads, label, aliscore_path ):
    errw( "\t\tFeaturizing " + label + " clusters...\n" )
    tasks = [ ( idx, x, cluster_dir + "/" + x ) for idx, x in enumerate( cluster_ids ) ]
    #featurized_clusters = [ 0 for x in cluster_ids ]
    #featurized_clusters = Queue
   
    lock = Lock()
    pool = Pool(
            threads,
            initializer = init_child_featurize,
            initargs = (
                lock,
                working_dir,
                label,
                aliscore_path,
                )
            )
    featurized_clusters = pool.map( featurize_cluster, tasks )
    pool.close()
    pool.join()

    #for featurized_cluster in featurized_clusters:
    #    print featurized_cluster

    errw( "\t\tDone featurizing clusters!\n" )

    return featurized_clusters


def save_featurized_dataset( dest_path, readable_data, pickle_dest_path, pd_data ):
    errw( "\t\tSaving featurized data set to " + dest_path + "..." )
    #print data
    with open( dest_path, 'w' ) as fh:
        for item in readable_data:
            #print item
            fh.write( ", ".join( map( str, item ) ) + "\n" )

    errw( " Done!\n" )

    errw( "\t\tPickling featurized data set to " + pickle_dest_path + "..." )
    with open( pickle_dest_path, "wb" ) as fh:
        pickle.dump( pd_data, fh )
    errw( " Done!\n" )

def train_model( model ):
    cur_model = model_to_class[ model ]()
    cur_model.fit( x, y )

    with lock:
        errw( "\t\tTraining " + model + "... Done!\n" )

    return cur_model


def format_data_for_training( data ):
    # prep the data
    ## turn the data into a pandas dataframe with appropriate labels
    columns = available_features.split( ',' )
    columns.append( "class" )
    data = pd.DataFrame( data, columns = columns )

    return data


def create_trained_models( models, features, data, threads ):
    errw( "\tTraining models\n" )
    # prep the models
    models = models.split( ',' )

    # prep features
    # needed for data prep
    features = features.split( ',' )

    ## separate into features and labels
    x = data[ features ]
    y = data[ "class" ]

    ## split into test and train
    ## TODO: parameterize test size
    ## TODO: parameterize random state for testing reproducibility
    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2 )

    errw( "\t\tTrain size: " + str( len( x_train ) ) + " instances\n" )
    errw( "\t\tTest size: " + str( len( x_test ) ) + " instances\n" )

    trained_models = []
    if len( models ) > 1:
        model = Metaclassifier( models )
    else:
        model = models[ 0 ]()
    
    model.fit( x_train, y_train )

    predictions_test = model.predict( x_test )

    # output accuracy
    accuracy = accuracy_score( y_test, predictions_test )
    errw( "\t\tTest set accuracy: " + str( accuracy ) + "\n" )

    errw( "\tDone training models!\n" )

    return model


def train_return_acc( item ):
    x_train = item[ 0 ]
    y_train = item[ 1 ]
    x_test = item[ 2 ]
    y_test = item[ 3 ]
    model = model_to_class[ item[ 4 ] ]()

    with warnings.catch_warnings():
        warnings.simplefilter( "ignore" )
        try:
            model.fit( x_train, y_train )
            preds = model.predict( x_test )
            acc = accuracy_score( y_test, preds )

            return acc
        except ValueError:
            return 0.
        except Warning:
            return 0.
    return 0.


# These are tests that will help verify results of the models
def run_validation( test_dir, data, threads ):
    errw( "\tRunning validation...\n" )

    boot_percs = map( int, default_bootstrap_percentages.split( ',' ) )

    # set up features
    features = available_features.split( ',' )

    # first, split the data into train and test
    # TODO: parameterize the test size
    train, test = train_test_split( data, test_size = 0.2 )

    # set up test data
    x_test = test[ features ]
    y_test = test[ "class" ]
    
    # pool for multithreading

    ## check inidivudal model performance with bootstrap analysis
    ## this is run for 100 replicates
    errw( "\t\tValidate models...\n" )
    all_percs_train = dict()
    all_percs_test = dict()
    for model_name, model_class in model_to_class.iteritems():
        errw( "\t\t\tValidating " + model_name + "..." )
        model_acc_train = []
        model_acc_test = []
        #model = model_class()

        for boot_perc in boot_percs:
            boot_perc_acc = []
            num_instances = int( math.ceil( ( float( boot_perc ) / 100 ) * len( train ) ) )

            #print "number of instances: " + str( num_instances )
            tasks_test = []
            tasks_train = []
            for i in range( num_replicates ):
                # get the sub-sampled data
                sub_train = train.sample( n = num_instances, replace = True )
                x_train = sub_train[ features ]
                y_train = sub_train[ "class" ]

                tasks_train.append( ( x_train, y_train, x_train, y_train, model_name ) )
                tasks_test.append( ( x_train, y_train, x_test, y_test, model_name ) )

            pool = Pool( threads, initializer = share_model_to_class, initargs = ( model_to_class, ) )
            boot_perc_acc = pool.map( train_return_acc, tasks_test )
            pool.close()
            pool.join()
            model_acc_test.append( boot_perc_acc )

            pool = Pool( threads, initializer = share_model_to_class, initargs = ( model_to_class, ) )
            boot_perc_acc = pool.map( train_return_acc, tasks_train )
            pool.close()
            pool.join()
            model_acc_train.append( boot_perc_acc )

        errw( " Done!\n" )
        formatted_percs = pd.DataFrame( model_acc_test ).transpose()
        formatted_percs.columns = boot_percs
        all_percs_test[ model_name ] = formatted_percs

        formatted_percs = pd.DataFrame( model_acc_train ).transpose()
        formatted_percs.columns = boot_percs
        all_percs_train[ model_name ] = formatted_percs
    errw( "\t\tCompleted model validation\n" )

    errw( "\t\tWriting accuracy plots to disk...\n" )
    for model, percs in all_percs_train.iteritems():
        errw( "\t\t\tGenerating plot for " + model + "..." )

        avgs_train = percs.mean()
        errs_train = percs.std()

        avgs_test = all_percs_test[ model ].mean()
        errs_test = all_percs_test[ model ].std()

        avgs = pd.DataFrame( [ avgs_train, avgs_test ] ).transpose()
        avgs.columns = [ "Train", "Test" ]

        errs = pd.DataFrame( [ errs_train, errs_test ] ).transpose()
        errs.columns = [ "Train", "Test" ]

        fig, ax = plt.subplots()
        #leg = plt.legend( fontsize = 8 )
        #leg = plt.gca().get_legend()
        #ltext  = leg.get_texts()
        #plt.setp( ltext, fontsize = 8 )
        fig.set_size_inches( 12, 8, forward = True )
        plt.title( model + " accuracy" )
        plt.ylim( 0.5, 1.0 )
        plt.xlabel( "% of total training set" )
        plt.ylabel( "% accuracy" )
        avgs.plot.line( ax = ax, color = [ 'b', 'r' ] )
        plt.fill_between( avgs.index.values, avgs[ "Train" ] - errs[ "Train" ], avgs[ "Train" ] + errs[ "Train" ], facecolor = 'blue', alpha = 0.2 )
        plt.fill_between( avgs.index.values, avgs[ "Test" ] - errs[ "Test" ], avgs[ "Test" ] + errs[ "Test" ], facecolor = 'red', alpha = 0.2 )
        plt.xticks( avgs.index.values, map( str, avgs.index.values ), fontsize = 8 )
        leg = plt.legend( fontsize = 8 )
        plt.savefig( test_dir + "/model_validation.bootstrap_plot." + model + ".png" )

        errw( " Done!\n" )

    errw( "\t\tCompleted writing data to disk!\n" )

    errw( "\t\tWriting model accuracies to disk...\n" )
    # create plots for each model
    for dataset, all_percs in [ ( "train", all_percs_train ), ( "test", all_percs_test ) ]:
        for model, percs in all_percs.iteritems():
            errw( "\t\t\tWriting " + model + " " + dataset + " dataset accuracy values to disk..." )

            with open( test_dir + "/model_validation." + dataset + ".bootstrap_values." + model + ".csv", 'w' ) as fh:
                fh.write( percs.to_csv() )

            errw( " Done!\n" )

    errw( "\t\tCompleted writing data to disk!\n" )

    errw( "\t\tValidate features...\n" )
    ## check individual feature performance using the metaclassifier
    ## this is run for 100 replicates
    all_feature_percs = dict()
    for feature in features:
        errw( "\t\t\tValidating " + feature + "..." )
        feature_acc = []
        for boot_perc in boot_percs:
            boot_perc_acc = []
            num_instances = int( math.ceil( ( float( boot_perc ) / 100 ) * len( train ) ) )

            tasks = []
            for i in range( num_replicates ):
                sub_train = train.sample( n = num_instances, replace = True )
                x_train = sub_train[ feature ].reshape(-1, 1)
                y_train = sub_train[ "class" ]
                tasks.append( ( x_train, y_train, x_test[ feature ].reshape( -1, 1 ), y_test, "metaclassifier" ) )
                
                #model = model_to_class[ "metaclassifier" ]()
                #model.fit( x_train, y_train )

                #preds = model.predict( x_test[ feature ].reshape(-1, 1) )
                #acc = accuracy_score( y_test, preds )
                #boot_perc_acc.append( acc )
            pool = Pool( threads, initializer = share_model_to_class, initargs = ( model_to_class, ) )
            boot_perc_acc = pool.map( train_return_acc, tasks )
            pool.close()
            pool.join()
            feature_acc.append( boot_perc_acc )
        formatted_percs = pd.DataFrame( feature_acc ).transpose()
        formatted_percs.columns = boot_percs
        all_feature_percs[ feature ] = formatted_percs
        errw( " Done!\n" )

    pool.close()
    errw( "\t\tWriting accuracies to disk...\n" )
    # create plots for each model

    model = "metaclassifier"
    all_means = []
    all_errs = []
    col_names = []
    for feature, percs in all_feature_percs.iteritems():
        errw( "\t\t\tWriting " + feature + " accuracy values to disk..." )

        with open( test_dir + "/feature_validation.bootstrap_values." + model + "." + feature + ".csv", 'w' ) as fh:
            fh.write( percs.to_csv() )

        col_names.append( feature )
        avgs = percs.mean()
        errs = percs.std()

        all_means.append( avgs )
        all_errs.append( errs )

        errw( " Done!\n" )

    errw( "\t\tGenerating features plot..." )

    all_means = pd.DataFrame( all_means ).transpose()
    all_means.columns = col_names
    colors = [ "blue", "red", "yellow", "orange", "green", "black", "cyan", "gray", "purple", "pink" ]

    fig, ax = plt.subplots()
    #leg = plt.gca().get_legend()
    #ltext  = leg.get_texts()
    #plt.setp( ltext, fontsize = 8 )
    fig.set_size_inches( 12, 8, forward = True )
    plt.title( model + " test set accuracy per feature" )
    plt.ylim( 0.0, 1.0 )
    plt.xlabel( "% of total training set" )
    plt.ylabel( "% accuracy" )
    all_means.plot.line( ax = ax, color = colors )

    for idx, feature in enumerate( col_names ):
        plt.fill_between( all_means.index.values, all_means[ feature ] - errs, all_means[ feature ] + errs, facecolor = colors[ idx ], alpha = 0.1 )

    plt.xticks( all_means.index.values, map( str, all_means.index.values ), fontsize = 8 )
    leg = plt.legend( fontsize = 8 )
    plt.savefig( test_dir + "/feature_validation.bootstrap_plot." + model + ".png" )
    errw( " Done!\n" )

    errw( "\t\tCompleted writing data and plots to disk!\n" )

    errw( "\tCompleted validation!\n" )


def load_featurized_data( file_path ):
    return pickle.load( open( file_path, "rb" ) )

def save_models( save_prefix, models, features, trained_models ):
    errw( "\tSaving models..." )
    
    with open( save_prefix + ".trained_models", "wb" ) as fh:
        pickle.dump( trained_models, fh )

    with open( save_prefix + ".features", 'w' ) as fh:
        fh.write( features )

    with open( save_prefix + ".models", 'w' ) as fh:
        fh.write( models )

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


def train_models( args ):
    errw( "Aligner: " + args.aligner_path + "\n" )
    errw( "Aligner args: " + args.aligner_options + "\n" )

    # verify parameters
    ## verify specified models
    check_models = args.models.split( ',' )
    for model in check_models:
        if model not in model_to_class.keys():
            sys.exit( "ERROR! User specified invalid model: " + model )

    ## verify specified features
    check_feats = args.features.split( ',' )
    feats_avail = available_features.split( ',' )
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

        ortho_groups = segregate_orthodb_groups( args.orthodb_fasta, args.orthodb_groups_dir )
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

    # train models
    if args.test:
        dir_check( args.test_dir )
        run_validation( args.test_dir, data, args.threads )

        if args.test_only:
            errw( "Model and feature validation successful, quitting because --test_only set.\n" )
            sys.exit()

    models = create_trained_models( args.models, args.features, data, args.threads )

    # save models
    save_models( args.save_prefix, args.models, args.features, models )


def parse_cluster_paths( file_path ):
    errw( "\tParsing cluster paths..." )
    cluster_paths = []
    with open( file_path ) as fh:
        for line in fh:
            cluster_paths.append( line.strip() )
    wrre( " Done!\n" )

    return cluster_paths


def load_models( models_prefix ):
    errw( "\tLoading models..." )
    with open( models_prefix + ".trained_models", "rb" ) as fh:
        models = pickle.load( fh )
    errw( " Done!\n" )

    return models

# TODO: finish this
def classify_clusters( args ):
    errw( "Classifying clusters!\n" )

    # verify parameters
    ## check if all directories required exist
    dir_check( args.logs_dir )
    dir_check( args.featurized_clusters_dir )
    dir_check( args.aliscore_dir )

    # end verify parameters

    # get cluster paths
    cluster_paths = parse_cluster_paths( args.fasta_list )

    # if need to align, align clusters
    if not args.aligned:
        # check if aligned dir exsists
        dir_check( args.aligned_dir )

        # align the clusters and save to a folder

        # reset cluster_paths to the clusters in the folder

    # featurize clusters

    # load models
    models = load_models( args.models_prefix )

    # classify


def main( args ):
    errw( "OrthoClean model training module version " + version + "\n" )

    if args.which == "train":
        train_models( args )
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
    classify_group_in.add_argument( "--fasta_list",
            type = str,
            required = True,
            help = "A list containing all the paths to your fasta files. One file per line."
            )
    classify_group_in.add_argument( "--models",
            type = str,
            required = True,
            help = "Trained models used for classification."
            )

    classify_group_opts = sp_classify.add_argument_group( "Options", "Options for running the program." )
    classify_group_opts.add_argument( "--aligned",
            default = False,
            action = "store_true",
            help = "If your fasta file is already aligned, alignment will skip."
            )

    classify_group_models = sp_classify.add_argument_group( "Model Options", "Options for loading trained models." )
    classify_group_models.add_argument( "--models_prefix",
            type = str,
            required = True,
            help = "Prefix of the saved models."
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

    train_group_aligner = sp_train.add_argument_group( "Aligner options", "Options to use for aligning your sequence clusters." )
    train_group_aligner.add_argument( "--aligner_path",
            type = str,
            default = default_mafft_path,
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here. NOTE: this program is only designed to work with MAFFT, other aligners will take modification."
            )
    train_group_aligner.add_argument( "--aligner_options",
            type = str,
            default = "",
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

    train_group_model = sp_train.add_argument_group( "Model training", "Models and features available for training" )
    train_group_model.add_argument( "--models",
            type = str,
            default = available_models,
            help = "A comma separated list of models to use. Available models include: " + ", ".join( available_models.split( ',' ) ) + ". If more than one model is selected, a meta-classifier is used that combined all specified models."
            )
    train_group_model.add_argument( "--features",
            type = str,
            default = available_features,
            help  = "A comma separted list of features to use when training models. Available features include: " + ", ".join( available_features.split( ',' ) ) + "."
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

    train_group_save = sp_train.add_argument_group( "Saving", "Options for saving your trained models." )
    train_group_save.add_argument( "--save_prefix",
            type = str,
            default = "trained_model",
            help = "Save prefix for your trained models"
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
            default = False,
            action = "store_true",
            help = "Set this flag to delete temporary files as you go."
            )
    args = parser.parse_args()

    main( args )

