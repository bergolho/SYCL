# Compiler
CXX = icpx
#CXX = nvcc

# Compilation flags
CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -Wall -Wno-deprecated-declarations -Wno-unused-variable
#CXXFLAGS = -O3 -Wno-deprecated-declarations

# Executable name
TARGET = matrix_add_sycl_v2
#TARGET = matrix_add_cuda

# Source file
SRC = matrix_add_v2.cpp
#SRC = matrix_add.cu

# Default rule: Build the executable
all: $(TARGET)

# Rule to compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET)

