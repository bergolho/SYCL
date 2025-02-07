ICPX = icpx
ICPX_FLAGS = -fsycl

all: select_device

select_device: select_device.cpp
	$(ICPX) $(ICPX_FLAGS) -o $@ $<

clear:
	rm -r select_device
