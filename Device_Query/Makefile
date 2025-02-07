ICPX = icpx
ICPX_FLAGS = -fsycl

all: device_query

device_query: device_query.cpp
	$(ICPX) $(ICPX_FLAGS) -o $@ $<

clear:
	rm -r device_query
