# Compiler
CXX = nvcc

# Compilation flags
CXXFLAGS = -O2

# Executable name
TARGET = main

# Source file
SRC = main.cu

# Default rule: Build the executable
all: $(TARGET)

# Rule to compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET)

clear_results:
	rm outputs/sv_history.txt
