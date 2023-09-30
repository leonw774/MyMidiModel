CFLAGS = -O2

all: apply_vocab learn_vocab;

debug: CFLAGS = -pg -g -Og
debug: apply_vocab learn_vocab;

apply_vocab: apply_vocab.cpp classes.o functions.o;
	g++ $(CFLAGS) -fopenmp classes.o functions.o apply_vocab.cpp -o apply_vocab

learn_vocab: learn_vocab.cpp classes.o functions.o;
	g++ $(CFLAGS) -fopenmp classes.o functions.o learn_vocab.cpp -o learn_vocab

functions.o: functions.cpp functions.hpp classes.hpp;
	g++ $(CFLAGS) -c -fopenmp functions.cpp -o functions.o

classes.o: classes.cpp classes.hpp;
	g++ $(CFLAGS) -c classes.cpp -o classes.o

clean: 
	rm functions.o classes.o learn_vocab apply_vocab
