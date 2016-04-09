all: mafft paml aliscore seq-gen
	@echo "Compiled all software dependencies!"

mafft: lib/mafft_bin/mafft

lib/mafft_bin/mafft:
	if [ -f lib/mafft_bin ]; \
		then \
			rm lib/mafft/bin; \
		fi;

	@echo "Compiling MAFFT"
	cd lib; \
		tar zxvf mafft-7.273-without-extensions-src.tgz; \
		cd mafft-7.273-without-extensions/core; \
		$(MAKE) install; \
		cd ../..; \
		ln -s mafft-7.273-without-extensions/core/bin mafft_bin
	@echo "Done!"

paml: lib/paml_bin/evolverRandomTree
	
lib/paml_bin/evolverRandomTree:
	if[ -f lib/paml_bin ]; \
		then \
			rm lib/paml_bin; \
		fi;

	@echo "Compiling PAML"
	cd lib; \
		tar zxvf paml4.9a.tgz; \
		cd paml4.9a/src; \
		cc -lm -o evolverRandomTree -O2 evolver.c tools.c; \
		mv evolverRandomTree ../bin; \
		cd ../..; \
		ln -s paml4.9a/bin paml_bin
	@echo "Done!"

aliscore:
	@echo "Compling Aliscore"
	@echo "Done!"

seq-gen: lib/seq-gen_bin/seq-gen
	
lib/seq-gen_bin/seq-gen:
	if[ -f lib/seq-gen_bin ]; \
		then \
			rm lib/seq-gen_bin; \
		fi;

	@echo "Compling Seq-Gen"
	cd lib; \
		tar zxvf Seq-Gen.v1.3.3.tgz; \
		cd Seq-Gen.v1.3.3; \
		mkdir bin; \
		cd source; \
		$(MAKE); \
		mv seq-gen ../bin; \
		cd ../..; \
		ln -s Seq-Gen.v1.3.3/bin seq-gen_bin
	@echo "Done!"

clean_run:
	rm -rf cluster_alignments_homology cluster_alignments_nh evolved_seqs logs nh_groups_fasta orthodb_groups_fasta paml_configs paml_trees siterates.txt SeedUsed mc.paml evolver.out
