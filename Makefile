all: logr

logr: temp/logr.o temp/vect.o
	g++ -std=c++11 temp/logr.o temp/vect.o
	mv a.out logr

temp/logr.o: src/main.cpp include/vect.h
	g++ -std=c++11 -c src/main.cpp
	mv main.o temp/logr.o

temp/vect.o: src/vect.cpp include/vect.h
	g++ -std=c++11 -c src/vect.cpp
	mv vect.o temp/vect.o

clean:
	rm temp/*.o
	rm *~
	rm logr
