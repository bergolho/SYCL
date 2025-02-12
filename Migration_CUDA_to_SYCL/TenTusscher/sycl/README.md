## Compile

To enable the SYCL program to run using the GPU, build using the following command.

```sh
$ make USE_GPU=y
```
*IMPORTANT!* You have to pass a different flag to enable compatibility with NVIDIA GPUs. This flag is already in the Makefile.

For CPU:

```sh
$ make USE_GPU=n
```

## Run

To run, just execute the following command

```sh
$ ./main_dp
```
