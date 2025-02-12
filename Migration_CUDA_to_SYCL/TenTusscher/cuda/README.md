## Convert CUDA to SYCL

To convert a cellular model written with CUDA code to SYCL have all the necessary files inside the folder and execute the following command:

```sh
$ c2s main.cu --gen-helper-function --out-root sycl_code
```

This will generate SYCL code using the SYCLomatic and DPCT tools. The generated files will be stored inside the folder '_sycl_code_'.

Here are a few notes about the conversion process:

0) The _'main_dp.cpp'_ is the SYCL converted file.

1) The math functions inside the _'ten_tusscher_3_RS_common.inc'_ will be replaced by their associated SYCL ones, e.g. _sycl::pow_, _sycl::exp_, etc. Looks like the original are not compatible with all devices.

2) Remember to copy the new _'ten_tusscher_3_RS_common.inc'_ file together with the _'main_dp.cpp'_ one.
 
