all: mafft paml aliscore seq-gen
	@echo "Compiled all software dependencies!"

mafft:
	@echo "Compiling MAFFT"
	@echo "Done!"

paml: lib/paml_bin/evolverRandomTree
	
lib/paml_bin/evolverRandomTree:
	rm lib/paml_bin
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

seq-gen:
	@echo "Compling Seq-Gen"
	@echo "Done!"
