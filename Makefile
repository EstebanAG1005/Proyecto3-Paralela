all: pgm.o hough hough_constante

hough: houghBase.cu pgm.o
	@nvcc -arch=sm_75 houghBase.cu pgm.o -o hough

hough_constante: hough_Constante.cu pgm.o
	@nvcc -arch=sm_75 hough_Constante.cu pgm.o -o hough_constante

pgm.o: common/pgm.cpp
	@g++ -c common/pgm.cpp -o ./pgm.o
