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
	if [ -f lib/paml_bin ]; \
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

aliscore: lib/aliscore_bin/Aliscore.02.2.pl

lib/aliscore_bin/Aliscore.02.2.pl:
	@echo "Compling Aliscore"
	cd lib; \
		unzip Aliscore_v.2.0.zip; \
		ln -s Aliscore_v.2.0 aliscore_bin
	@echo "Done!"

seq-gen: lib/seq-gen_bin/seq-gen
	
lib/seq-gen_bin/seq-gen:
	if [ -f lib/seq-gen_bin ]; \
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

rm_int:
	rm -rf train_cluster_alignments_homology train_cluster_alignments_nh train_evolved_seqs train_nh_groups_fasta train_orthodb_groups_fasta train_paml_configs train_featurized_clusters train_evolved_seqs train_aliscores_nh train_aliscores_homology train_paml_trees
	rm -rf classify_aligned_dir classify_featurized_clusters classify_aliscores
	rm -rf logs

clean: rm_int
	rm -rf model

deepclean: clean
	rm -rf lib/Aliscore_v.2.0
	rm lib/aliscore_bin
	rm -rf lib/mafft-7.273-without-extensions
	rm lib/mafft_bin
	rm -rf lib/paml4.9a
	rm lib/paml_bin
	rm -rf lib/Seq-Gen.v1.3.3
	rm lib/seq-gen_bin
