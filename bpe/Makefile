learn_vocab: learn_vocab.cpp corpus.o shapecounting.o ;
	g++ -g -fopenmp corpus.o shapecounting.o learn_vocab.cpp -o learn_vocab

shapecounting.o: shapecounting.cpp shapecounting.hpp corpus.hpp ;
	g++ -g -c -fopenmp shapecounting.cpp -o shapecounting.o

corpus.o: corpus.cpp corpus.hpp ;
	g++ -g -c corpus.cpp -o corpus.o
