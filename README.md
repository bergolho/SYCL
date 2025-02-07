# SYCL
Repository with simple programs to learn SYCL.

## Install Intel oneAPI for Fedora 40

- Go to this link and download the Intel oneAPI Base Toolkit for Linux Offline Installer.
- https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=offline

- Create a folder called _"OneAPI-Install"_ in your _/home_ directory and move the file you downloaded to here

```sh
$ cd; mkdir OneAPI-Install
$ mv Downloads/intel-oneapi-base-toolkit-2025.0.1.46_offline.sh OneAPI-Install
```

- Execute the setup using the GUI and follow all the instructions

```sh
$ cd OneAPI-Install
$ sudo sh ./intel-oneapi-base-toolkit-2025.0.1.46_offline.sh
```

## Install the Codeplay NVIDIA compatibility plugin

- This guide contains information on using DPC++ to run SYCL™ applications on NVIDIA® GPUs via the DPC++ CUDA® plugin.

- Go to this website: https://developer.codeplay.com/products/oneapi/nvidia/download/

- Download a version according to your GPU device and CUDA version

- Execute the shell script and install everything

```sh
$ sudo sh oneapi-for-nvidia-gpus-2025.0.0-linux.sh
``` 

- Load the enviroment variables

```sh
$ . /opt/intel/oneapi/setvars.sh --include-intel-llvm
```

- Check if SYCL can identify your GPU now with the command:

```sh
$ sycl-ls
[opencl:cpu][opencl:0] Intel(R) OpenCL, Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz OpenCL 3.0 (Build 0) [2024.18.12.0.05_160000]
[cuda:gpu][cuda:0] NVIDIA CUDA BACKEND, NVIDIA GeForce GTX 1060 6GB 6.1 [CUDA 12.7]
```


 


