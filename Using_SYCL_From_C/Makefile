# Compiler
CXX = g++  # Use clang++ or another compiler if not using SYCL

# Compiler flags
CXXFLAGS = -O2

# Source files
SRCS = main.cpp sycl_library/vector_sycl.cpp c_library/vector_c.c
OBJS = $(SRCS:.cpp=.o)

# Output executable
TARGET = my_program

# Default rule
all: $(TARGET)

# Link the object files
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
