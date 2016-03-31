all:
	g++ `pkg-config --cflags opencv` -o video video.cpp `pkg-config --libs opencv`