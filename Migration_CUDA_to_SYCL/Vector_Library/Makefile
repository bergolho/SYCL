CC=nvcc
CFLAGS=-I.
DEPS = vector_library.h

%.o: %.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

main: main.o vector_library.o 
	$(CC) -o main main.o vector_library.o