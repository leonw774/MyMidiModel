all: apply_vocab learn_vocab ;

apply_vocab: apply_vocab.cpp corpus.o shapescore.o ;
	g++ -g -fopenmp corpus.o shapescore.o apply_vocab.cpp -o apply_vocab

learn_vocab: learn_vocab.cpp corpus.o shapescore.o ;
	g++ -g -fopenmp corpus.o shapescore.o learn_vocab.cpp -o learn_vocab

shapescore.o: shapescore.cpp shapescore.hpp corpus.hpp ;
	g++ -g -c -fopenmp shapescore.cpp -o shapescore.o

corpus.o: corpus.cpp corpus.hpp ;
	g++ -g -c corpus.cpp -o corpus.o

clean: 
	rm *.o learn_vocab apply_vocab
