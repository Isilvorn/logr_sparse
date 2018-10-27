all: procdata

procdata: procdata.o
	g++ -std=c++11 -g procdata.o
	mv a.out procdata

procdata.o: procdata.cpp
	g++ -std=c++11 -c procdata.cpp

clean:
	rm *.o
	rm *~
	rm procdata
