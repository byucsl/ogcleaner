import sys, argparse, re, time, os
from subprocess import call
from pathlib import Path
from multiprocessing import Pool, Lock, Queue


# global variables
version = "0.1a"
p = Path( __file__ )
default_mafft_path = str( p.resolve().parents[ 1 ] ) + "/lib/mafft_bin/mafft"
q = Queue()

# TODO: enumerate all possible models and features
available_models = "svm"
available_features = "length"


def init_child(lock_):
    global lock
    lock = lock_


def remove_ambiguous_amino_acids( seq ):
    return seq.replace( 'U', '' ).replace( 'B', '' ).replace( 'X', '' ).replace( 'Z', '' )


def segregate_orthodb_groups( fasta_file_path, group_dir ):

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
                        with open( group_dir + "/" + cur_group + ".fasta", 'w' ) as out:
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
        with open( group_dir + "/" + cur_group + ".fasta", 'w' ) as out:
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
        tasks.append( ( aligner_path, aligner_opts, clusters_dir + "/" + cluster_name + ".fasta", cluster_name, aligned_dir, logs_dir ) )

    lock = Lock()
    pool = Pool( threads, initializer = init_child, initargs = ( lock, ) )
    pool.map( align_worker, tasks )

    pool.close()
    pool.join()
    errw( "\tDone aligning clusters!" )


def errw( text ):
    sys.stderr.write( text )

def dir_check( dir_path ):
    # check if the output directory exists
    if not os.path.exists( dir_path ):
        status = call( [ "mkdir", dir_path ] )

        if status == 0:
            errw( "Created orthogroup fasta folder " + dir_path + "\n" )
        else:
            sys.exit( "ERROR! Could not create the directory " + dir_path + ". Aborting!" )


def main( args ):
    errw( "OrthoClean model training module version " + version + "\n" )
    errw( "Aligner: " + args.aligner_path + "\n" )
    errw( "Aligner args: " + args.aligner_options + "\n" )
    errw( "Beginning...\n" )

    # param checking
    ## check if the output directory exists
    dir_check( args.orthodb_group_dir )
    dir_check( args.aligned_dir )
    dir_check( args.logs_dir )

    ortho_groups = segregate_orthodb_groups( args.orthodb_fasta, args.orthodb_group_dir )

    # align the ortho clusters
    align_clusters( args.aligner_path, args.aligner_options, args.orthodb_group_dir, ortho_groups, args.aligned_dir, args.threads, args.logs_dir )

    # non-homology cluster generation
    ## sample sequences from homology clusters

    ## evolve sequences

    # align the non-homology clusters
    align_clusters( args.nh_group_dir, nh_groups )

    # featurize datasets

    # train models

    errw( "Finished!\n" )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Train models to clean putative orthology clusters. Methodology published in Detecting false positive sequence homology: a machine learning approach, BMC Bioinformatics (24 February 2016, http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-0955-3). Please contact M. Stanley Fujimoto at sfujimoto@gmail.com for any questions.",
            )
    parser.add_argument( "--models",
            type = str,
            help = "A comma separated list of models to use. Available models include: " + available_models + "."
            )
    parser.add_argument( "--features",
            type = str,
            help  = "A comma separted list of features to use when training models. Available features include: " + available_features + "."
            )
    parser.add_argument( "--orthodb_fasta",
            type = str,
            required = True,
            help = "Fasta file downloaded from orthodb."
            )
    parser.add_argument( "--orthodb_group_dir",
            type = str,
            default = "orthodb_groups_fasta",
            help = "Directory to store OrthoDB sequence clusters in fasta format."
            )
    parser.add_argument( "--nh_group_dir",
            type = str,
            default = "nh_groups_fasta",
            help = "Directory to store the non-homology sequence clusters in fasta format."
            )
    parser.add_argument( "--aligner_path",
            type = str,
            default = default_mafft_path,
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here. NOTE: this program is only designed to work with MAFFT, other aligners will take modification."
            )
    parser.add_argument( "--aligner_options",
            type = str,
            default = "",
            help = "Options for your aligner."
            )
    parser.add_argument( "--aligned_dir",
            type = str,
            default = "cluster_alignments",
            help = "Directory to store all alignments."
            )
    parser.add_argument( "--threads",
            type = int,
            default = 1,
            help = "Number of threads to use during alignment (Default 1 )."
            )
    parser.add_argument( "--logs_dir",
            type = str,
            default = "logs_dir",
            help = "Directory to store progress output from programs."
            )

    args = parser.parse_args()

    main( args )

