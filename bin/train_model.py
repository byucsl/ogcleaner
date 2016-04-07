import sys, argparse, re, time, os
from subprocess import call
from pathlib import Path


# global variables
version = "0.1a"
p = Path( __file__ )
default_mafft_path = str( p.resolve() ) + "/mafft"

# TODO: enumerate all possible models and features
available_models = "svm"
available_features = "length"

def segregate_orthodb_groups( fasta_file_path, group_dir ):
    # check if the output directory exists
    if not os.path.exists( group_dir ):
        status = call( [ "mkdir", group_dir ] )

        if status == 0:
            errw( "Created orthogroup fasta folder " + group_dir + "\n" )
        else:
            sys.exit( "ERROR! Could not create the orthogroup fasta folder. Aborting!" )

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
                cur_seq_seq += line.strip()
        cur_group_seqs.append( cur_seq_header + "\n" + cur_seq_seq )
        with open( group_dir + "/" + cur_group + ".fasta", 'w' ) as out:
            group_names.append( cur_group )
            for item in cur_group_seqs:
                out.write( item )
                out.write( "\n" )

        errw( "Done!\n" )

        return group_names


def align_cluster( aligner_path, aligner_opts, cluster_path ):
    errw( "\t\tAligning " + cluster_path + "..." )
    status = call( [ aligner_path, aligner_opts, 
    if status != 0:
        errw( "Alignment error!!\n" )
    else:
        errw( "Done!\n" )


def align_clusters( aligner_path, aligner_opts, clusters_dir, cluster_names  ):
    errw( "\tAligning clusters...\n" )
    for cluster_name in cluster_names:
        align_cluster( aligner_path, aligner_opts, clusters_dir + "/" + cluster_name + ".fasta" )
    errw( "Done!\n" )


def errw( text ):
    sys.stderr.write( text )


def main( args ):
    errw( "OrthoClean model training module version " + version + "\n" )
    ortho_groups = segregate_orthodb_groups( args.orthodb_fasta, args.orthodb_group_dir )

    # align the ortho clusters
    align_clusters( args.aligner_path, args.aligner_options, args.orthodb_group_dir, ortho_groups )

    # non-homology cluster generation
    ## sample sequences from homology clusters

    ## evolve sequences

    # align the non-homology clusters
    align_clusters( args.nh_group_dir, nh_groups )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "Train models to clean putative orthology clusters."
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
            help = "Default aligner is MAFFT and is set up during install. If you already have MAFFT or another aligner installed, provide the path here."
            )
    parser.add_argument( "--aligner_options",
            type = str,
            default = "",
            help = "Options for your aligner."
            )

    args = parser.parse_args()

    main( args )

