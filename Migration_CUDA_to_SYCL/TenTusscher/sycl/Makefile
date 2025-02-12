# Compiler
CXX = icpx

# Check if USE_GPU is set to 'y'
ifeq ($(USE_GPU), y)
	CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O2 -w
else
	CXXFLAGS = -fsycl -O2 -w
endif

# Executable name
TARGET = main_dp

# Source file
SRC = main_dp.cpp

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
