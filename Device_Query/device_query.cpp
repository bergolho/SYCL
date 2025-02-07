#include <CL/sycl.hpp>
#include <iostream>

int main () {
	for (const cl::sycl::platform& platform :
			cl::sycl::platform::get_platforms()) {
		std::cout << "========================================" << std::endl;
		std::cout << "Platform : " << platform.get_info< cl::sycl::info::platform::name >() << std::endl;
	        
		for (const cl::sycl::device& device : platform.get_devices()) {
			std::cout << " Device : " << device.get_info< cl::sycl::info::device::name >() << std::endl;
		}	
	}
	return 0;
}
