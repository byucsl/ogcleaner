import argparse


def main( args ):
    groups = dict()
    groups_seq = dict()
    with open( args.groups, 'r' ) as fh:
        for line in fh:
            line = line.split( ':' )
            group_name = line[ 0 ]
            for ind in line[ 1 ].split():
                groups[ ind ] = group_name
            groups_seq[ group_name ] = []
    print "Parsed", str( len( groups ) ), "groups"

    with open( args.proteins, 'r' ) as fh:
        header = ""
        seq = ""
        for line in fh:
            if line[ 0 ] == '>':
                # header
                if len( header ) > 0:
                    try:
                        groups_seq[ groups[ header ] ].append( ( header, seq ) )
                    except:
                        pass
                header = line[ 1 : ].strip()
                seq = ""
            else:
                seq += line
        try:
            groups_seq[ groups[ header ] ].append( ( header, seq ) )
        except:
            pass

    print "Sequences separated into groups!"

    print "Writing groups to files"
    for group, seqs in groups_seq.iteritems():
        with open( args.out_dir + "/" + group, 'w' ) as fh:
            for head, seq in seqs:
                fh.write( '>' + head + "\n" + seq )
    print "\tDone!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description = "This program is designed to take the output of OrthoMCL (groups.txt) and the goodProteins.fasta file. A separate FASTA file will be created for each group identified by OrthoMCL and will be populated with the appropriate protein sequences taken from goodProteins.fasta as identified by the groups.txt file."
            )
    parser.add_argument( "--groups",
            required = True,
            type = str,
            help = "The groups.txt output file from OrthoMCL"
            )
    parser.add_argument( "--proteins",
            required = True,
            type = str,
            help = "The goodProteins.fasta file used as input to OrthoMCL"
            )
    parser.add_argument( "--out_dir",
            type = str,
            help = "The directory to store the output files",
            default = ""
            )
    args = parser.parse_args()
    main( args )
