
all: main

intmasseval.o: intmasseval.cc intmasseval.h
	g++ -c intmasseval.cc -O3 -ffast-math -std=c++17 -g

floatmasseval.o: floatmasseval.cc floatmasseval.h
	g++ -c floatmasseval.cc -O3 -ffast-math -std=c++17 -g
main.o: main.cc
	g++ -c main.cc -O3 -ffast-math -std=c++17 -g

main: main.o intmasseval.o floatmasseval.o
	g++ main.o intmasseval.o floatmasseval.o -o main
