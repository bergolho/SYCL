# Compiler
CXX = icpx

# Compilation flags
CXXFLAGS = -fsycl -O2 -Wall -Wno-deprecated-declarations -Wno-unused-variable

# Executable name
#TARGET = main_using_memcpy_sycl
#TARGET = main_using_buffer_sycl
TARGET = main_using_buffer_sycl_v2

# Source file
#SRC = main_using_memcpy.cpp
#SRC = main_using_buffers.cpp
SRC = main_using_buffers_v2.cpp

# Default rule: Build the executable
all: $(TARGET)

# Rule to compile the program
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET)

clear_results:
	rm outputs/sv_sycl_history.txt
