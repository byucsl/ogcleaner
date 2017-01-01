import sys, argparse


def main( args ):
    names = {}

    with open( args.names, 'r' ) as fh:
        for line in fh:
            line = line.split()
            names[ line[ 0 ] ] = line[ 1 ]

    with open( args.groups, 'r' ) as fh:
        for line in fh:
            line = line.split()
            reformatted_entry = [ line[ 0 ] ]
            for entry in line[ 1 : ]:
                entry = entry.split( '|' )
                entry[ 0 ] = names[ entry[ 0 ] ]
                reformatted_entry.append( '|'.join( entry ) )
            print " ".join( reformatted_entry )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description = 'Replace the species IDs that are used by OrthoMCL during processing with the original species names.'
            )
    parser.add_argument( '--groups',
            type = str,
            required = True,
            help = 'groups.txt file.'
            )
    parser.add_argument( '--names',
            type = str,
            required = True,
            help = 'tab delimitted file that has the OrthoMCL ID followed by the species names.'
            )

    args = parser.parse_args()
    main( args )
