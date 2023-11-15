all: pgm.o hough hough_constante hough_Compartida

hough: houghBase.cu pgm.o
	@nvcc -arch=sm_75 houghBase.cu pgm.o -o hough

hough_constante: hough_Constante.cu pgm.o
	@nvcc -arch=sm_75 hough_Constante.cu pgm.o -o hough_constante

hough_Compartida: hough_Compartida.cu pgm.o
	@nvcc -arch=sm_75 hough_Compartida.cu pgm.o -o hough_Compartida

pgm.o: common/pgm.cpp
	@g++ -c common/pgm.cpp -o ./pgm.o
