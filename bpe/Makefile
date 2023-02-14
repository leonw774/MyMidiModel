all: apply_vocab learn_vocab ;

apply_vocab: apply_vocab.cpp classes.o functions.o ;
	g++ -g -fopenmp classes.o functions.o apply_vocab.cpp -o apply_vocab

learn_vocab: learn_vocab.cpp classes.o functions.o ;
	g++ -g -fopenmp classes.o functions.o learn_vocab.cpp -o learn_vocab

functions.o: functions.cpp functions.hpp classes.hpp ;
	g++ -g -c -fopenmp functions.cpp -o functions.o

classes.o: classes.cpp classes.hpp ;
	g++ -g -c classes.cpp -o classes.o

clean: 
	rm *.o learn_vocab apply_vocab
