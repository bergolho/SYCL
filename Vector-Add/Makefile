ICPX = icpx
ICPX_FLAGS = -fsycl

all: vector_add

vector_add: vector_add.cpp
	$(ICPX) $(ICPX_FLAGS) -o $@ $<

clear:
	rm -r vector_add
