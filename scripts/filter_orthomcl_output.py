import sys, argparse


def main( args ):

    to_keep = []
    # read in the cluster labels, keep a list of the clusters to keep
    with open( args.cluster_labels, 'r' ) as fh:
        for line in fh:
            line = line.strip().split()
            if line[ 1 ] == 'H':
                to_keep.append( line[ 0 ] )

    with open( args.groups, 'r' ) as fh:
        for line in fh:
            group_id = line.strip().split()[ 0 ][ : -1 ]
            
            if group_id in to_keep:
                print line.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'This script takes the cluster classification results output from the OGCleaner and filters the OrthoMCL groups.txt file.'
            )
    parser.add_argument( '--groups',
            type = str,
            required = True,
            help = 'The path to the groups.txt output file from OrthoMCL.'
            )
    parser.add_argument( '--cluster_labels',
            type = str,
            required = True,
            help = 'The path to the OGCleaner results.txt cluster classification file.'
            )

    args = parser.parse_args()
    main( args )
