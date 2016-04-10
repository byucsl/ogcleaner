import sys, argparse, re, time, os
import numpy as np
from subprocess import call, Popen, PIPE
from pathlib import Path
from multiprocessing import Pool, Lock


# global variables
version = "0.1a"
p = Path( __file__ )
base_path = p.resolve().parents[ 1 ]
max_rand_int = 100000
default_mafft_path = str( base_path ) + "/lib/mafft_bin/mafft"
default_paml_path = str( base_path ) + "/lib/paml_bin/evolverRandomTree"
default_seqgen_path = str( base_path ) + "/lib/seq-gen_bin/seq-gen"
default_seqgen_opts = "-mWAG -k1 -i0 -n1"

# TODO: enumerate all possible models and features
available_models = "svm,neural_network,random_forest,naive_bayes,logistic_regression"
available_features = "aliscore,length,num_seqs,num_gaps,num_amino_acids,range,amino_acid_charged,amino_acid_uncharged,amino_acid_special,amino_acid_hydrophobic"


def init_child(lock_):
    global lock
    lock = lock_


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

        errw( "Done!\n" )

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
                errw( "Done!\n" )

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
    
    errw( "Done!\n" )
    return file_paths


def generate_paml_tree( item ):
    id = item[ 0 ]
    config_path = item[ 1 ]
    output_path = paml_trees_dir + "/" + str( id )
    log_out = logs_dir + "/" + str( id ) + ".paml"
    with open( log_out, 'w' ) as log_fh:
        status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh )
        tries = 0
        # It looks like PAML's exit codes aren't correct... we'll ignore them for now
        #while status != 0 or tries < 2:
            #with lock:
            #    errw( "\t\t\tTree " + config_path + " generation failed... trying again\n" )
            #status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh )
            #tries += 1
        status = call( [ paml_path, '5', config_path, output_path ], stdout = log_fh )

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
    errw( "Done!\n" )

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


def evolve_seq( id, output_dir, header, seq, tree ):
    seq_id = header[ 1 : ].split()[ 0 ].split( ':' )[ 1 ]
    seqgen_input = "1\t" + str( len( seq ) ) + "\n" + seq_id + "\t" + seq + "\n1\n" + tree
    #print seqgen_input + "\n"
    id = id.split()[ 0 ]
    output_path = output_dir + "/" + id + "/" + seq_id + ".evolved"

    with open( output_path, 'w' ) as fh, open( logs_dir + "/" + id + "." + seq_id + ".seqgen_out", 'w' ) as stderr_out:
        p = Popen( [ default_seqgen_path ] + seqgen_opts, stdout = fh, stdin = PIPE, stderr = stderr_out )
        #fh.write( "\n" )
        p.communicate( input = seqgen_input )
        p.wait()


def generate_evolved_sequence( item ):
    id = item[ 0 ]
    cluster_path = item[ 1 ]
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
        evolve_seq( id, evolved_seqs_dir, header, seq, trees[ tree_indices[ idx ] ] )

    with lock:
        errw( "\t\t\tEvolved cluster " + id + "\n" )

def generate_evolved_sequences( nh_groups_dir, all_trees, homology_cluster_paths, threads, seqgen_path, seqgen_opts, logs_dir, evolved_seqs_dir ):
    errw( "\t\tGenerating evolved sequences...\n" )

    # prep tasks
    #tasks = [ x for x in homology_cluster_paths ]
    tasks = homology_cluster_paths
    #print homology_cluster_paths

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
            with open( full_path ) as fh:
                spec_seqs = []
                species_name = filename[ : filename.rfind( '.' ) ]
                fh.next()
                for line in fh:
                    spec_seqs.append( line.strip().split()[ 1 ] )
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


def featurized_cluster( item ):
    pass


def featurize_clusters( cluster_paths ):
    pass


def errw( text ):
    sys.stderr.write( text )

def dir_check( dir_path ):
    # check if the output directory exists
    if not os.path.exists( dir_path ):
        status = call( [ "mkdir", dir_path ] )

        if status == 0:
            errw( "Created directory " + dir_path + "\n" )
        else:
            sys.exit( "ERROR! Could not create the directory " + dir_path + ". Aborting!" )


def main( args ):
    errw( "OrthoClean model training module version " + version + "\n" )
    errw( "Aligner: " + args.aligner_path + "\n" )
    errw( "Aligner args: " + args.aligner_options + "\n" )

    if args.seed != -1:
        errw( "Setting random number seed to: " + str( args.seed ) + "\n" )
        np.random.seed( args.seed )

    errw( "Beginning...\n" )

    # param checking
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

    ortho_groups = segregate_orthodb_groups( args.orthodb_fasta, args.orthodb_groups_dir )

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

    # featurize datasets

    # train models

    # perform model tests

    errw( "Finished!\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Train models to clean putative orthology clusters. Methodology published in Detecting false positive sequence homology: a machine learning approach, BMC Bioinformatics (24 February 2016, http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0955-3). Please contact M. Stanley Fujimoto at sfujimoto@gmail.com for any questions.",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument_group( "Input", "Input files to run the program." )
    parser.add_argument( "--orthodb_fasta",
            type = str,
            required = True,
            help = "Fasta file downloaded from orthodb."
            )
    group_dir = parser.add_argument_group( "Directories", "Directories where output will be stored." )
    group_dir.add_argument( "--orthodb_groups_dir",
            type = str,
            default = "orthodb_groups_fasta",
            help = "Directory to store OrthoDB sequence clusters in fasta format."
            )
    group_dir.add_argument( "--nh_groups_dir",
            type = str,
            default = "nh_groups_fasta",
            help = "Directory to store the non-homology sequence clusters in fasta format."
            )
    group_dir.add_argument( "--paml_configs_dir",
            type = str,
            default = "paml_configs",
            help = "Directory to store the PAML configuration files."
            )
    group_dir.add_argument( "--logs_dir",
            type = str,
            default = "logs",
            help = "Directory to store progress output from programs."
            )
    group_dir.add_argument( "--aligned_homology_dir",
            type = str,
            default = "cluster_alignments_homology",
            help = "Directory to store all OrthoDB homology alignments."
            )
    group_dir.add_argument( "--aligned_nh_dir",
            type = str,
            default = "cluster_alignments_nh",
            help = "Directory to store all false-positive homology alignments."
            )
    group_dir.add_argument( "--paml_trees_dir",
            type = str,
            default = "paml_trees",
            help = "Directory to store trees generated from PAML."
            )
    group_dir.add_argument( "--evolved_seqs_dir",
            type = str,
            default = "evolved_seqs",
            help = "Directory to store evolved sequences."
            )
    group_dir.add_argument( "--featurized_clusters_dir",
            type = str,
            default = "featurized_clusters",
            help = "Directory to store the matrix of featurized clusters. 2 files will be placed in this directory: 1 for the OrthoDB clusters and 1 for the non-homology clusters."
            )
    group_aligner = parser.add_argument_group( "Aligner options", "Options to use for aligning your sequence clusters." )
    group_aligner.add_argument( "--aligner_path",
            type = str,
            default = default_mafft_path,
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here. NOTE: this program is only designed to work with MAFFT, other aligners will take modification."
            )
    group_aligner.add_argument( "--aligner_options",
            type = str,
            default = "",
            help = "Options for your aligner."
            )
    group_paml = parser.add_argument_group( "PAML options" )
    group_paml.add_argument( "--paml_path",
            type = str,
            default = default_paml_path,
            help = "Path to the PAML evolverRandomTree binary."
            )
    group_seqgen = parser.add_argument_group( "Seq-Gen options", "Options for Seq-Gen, used for evolving sequences." )
    group_seqgen.add_argument( "--seqgen_path",
            type = str,
            default = default_seqgen_path,
            help = "Path to the Seq-Gen binary."
            )
    group_seqgen.add_argument( "--seqgen_opts",
            type = str,
            default = default_seqgen_opts,
            help = "Options for running Seq-Gen."
            )
    group_model = parser.add_argument_group( "Model training", "Models and features available for training" )
    group_model.add_argument( "--models",
            type = str,
            default = available_models,
            help = "A comma separated list of models to use. Available models include: " + ", ".join( available_models.split( ',' ) ) + ". If more than one model is selected, a meta-classifier is used that combined all specified models."
            )
    group_model.add_argument( "--features",
            type = str,
            default = available_features,
            help  = "A comma separted list of features to use when training models. Available features include: " + ", ".join( available_features.split( ',' ) ) + "."
            )
    group_misc = parser.add_argument_group( "Misc.", "Extra options you can set when you're running the program." )
    group_misc.add_argument( "--threads",
            type = int,
            default = 1,
            help = "Number of threads to use during alignment (Default 1)."
            )
    group_misc.add_argument( "--seed",
            type = int,
            default = -1,
            help = "Seed to use for generating trees in PAML and for sampling sequences for non-homology cluster generation."
            )
    group_skip = parser.add_argument_group( "Skip parts of the pipeline", "If you've already completed parts of the pipeline you can skip steps with these flags. Note: each flag requires you specify an output directory for the step you're skipping. Steps are listed in order, if you skip a step, all steps before it are skipped as well." )
    group_skip.add_argument( "--skip_segregate",
            type = str,
            help = "Skip segregating the fasta file from OrthoDB into separate fasta files for each group. Provide the path to the directory that contains all the segregated ortho groups in fasta format."
            )
    group_skip.add_argument( "--skip_align_orthodb",
            type = str,
            help = "Skip alignment process for each OrthoDB orthology group. Provide the path to the directory with the OrthBD alignments in fasta format."
            )
    group_skip.add_argument( "--skip_generate_nh",
            type = str,
            help = "Skip the generation process of false-positive homology clusters. Provide the path to the directory with all false-positive homology clusters in fasta format."
            )
    group_skip.add_argument( "--skip_align_nh",
            type = str,
            help = "Skip the alignment process for each false-positive homoloy clusters. Provide the path to the directory with all fasle-positive homology cluster alignments in fasta format."
            )
    group_test = parser.add_argument_group( "Testing", "Test your machine learning models." )
    group_test.add_argument( "--test",
            default = False,
            action = "store_true",
            help = "Do boostrapping analysis, tests, etc. to see how well your trained models are performing."
            )

    args = parser.parse_args()

    main( args )

