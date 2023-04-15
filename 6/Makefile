# for gs_hybrid.cpp
TARGET = gs_hybrid

CXXFLAGS = -fopenmp

.PHONY: all
all: $(TARGET)

.PHONY: run
run:
	mpirun -np 2 ./$(TARGET) OMP_NUM_THREADS=2

$(TARGET): $(TARGET).cpp Makefile
	mpic++ $(CXXFLAGS) $< -o $(TARGET)