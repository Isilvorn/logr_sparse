all: logr

logr: temp/logr.o temp/vect.o temp/datamodule.o
	g++ -std=c++11 temp/logr.o temp/vect.o temp/datamodule.o
	mv a.out logr

temp/logr.o: src/main.cpp include/vect.h
	g++ -std=c++11 -c src/main.cpp
	mv main.o temp/logr.o

temp/datamodule.o: src/datamodule.cpp include/datamodule.h
	g++ -std=c++11 -c src/datamodule.cpp
	mv datamodule.o temp/datamodule.o

temp/vect.o: src/vect.cpp include/vect.h
	g++ -std=c++11 -c src/vect.cpp
	mv vect.o temp/vect.o

clean:
	rm -f temp/*.o
	rm -f *~
	rm -f logr
