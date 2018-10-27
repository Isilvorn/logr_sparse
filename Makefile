all: logr

logr: logr.o
	g++ -std=c++11 -g logr.o
	mv a.out logr

logr.o: main.cpp
	g++ -std=c++11 -c main.cpp
	mv main.o logr.o
