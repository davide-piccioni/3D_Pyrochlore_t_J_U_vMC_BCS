CC = g++
#CC = icpc
	CFLAGS = -Wall -O3 --std=c++17 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/Eigen -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3
all: SR_lattice.exe
SR_lattice.exe : main_SR_lattice.o walker.o random.o distances.h 
	mpicxx walker.o random.o main_SR_lattice.o -o SR_lattice.exe
main_SR_lattice.o : main_SR_lattice.cpp distances.h 
	mpicxx -c main_SR_lattice.cpp -o main_SR_lattice.o $(CFLAGS)
walker.o : walker.cpp walker.h distances.h 
	$(CC) -c walker.cpp -o walker.o $(CFLAGS)
random.o : random.cpp random.h
	$(CC) -c random.cpp -o random.o $(CFLAGS)
clean :
	rm *.o SR_lattice.exe
