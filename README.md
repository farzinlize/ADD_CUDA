
# Add Cuda Project
Add cuda is a learning purpose project based on a simple task: Adding all element of an array in a single integer (**Summation**)

# Kernel
The kernel used in this project is based on a **Optimizing Parallel Reduction in CUDA** lecture by *Mark Harris* introduced at GPGPU course at Shiraz University by *Dr. Farshad khunjush*. The **reduction** lecture contain several kernels that tying to achieve best speedup compare to the very first and simple add kernel at the begging of lecture.

## Reduction
As it discussed in a lecture, reduction is a common and important data parallel primitive problem when after an iteration of working on data, number of active threads or potentially parallel tasks decreases.
The kernel written in this project work on an input array. Each block uses an integer array as shared memory in size of number of threads in block so that each thread works on one element of this shared memory. All kernel versions used in this project provided at discussed **Optimizing Parallel Reduction in CUDA** lecture.
In result, kernel provide the program shorter array of elements that contains summation of parts of input array. Summation of the result array in one integer is the final result. 

**Dose using a different kernel on result array affect the total speedup?**
It is very likely that invoking more kernels to do more things in parallel can effect the total speedup. Even all input data for new kernels is already presented at device but actually **removing final sequential summation has no effect** on the total speedup due to personal experiments. Bigger input array leads to bigger output and bigger output array leads to more sequential works at the end but even with 400MB input array (that make 20MB output array!), removing sequential part leads to only **3%** speedup (**1.0338x**). Consider that 3% speedup gained only because of removing sequential parts that corrupts final result. If we want to implement a parallel way to get the summation of output array in a single integer, we will get less than 3% speedup that not worth anything.

# Compile Project 
This project contains several ways to compile in order to report and measure several kind of speedup or any good information that helps the writer to achieve a better understanding about cuda codes and possible speedups.
Project has a **`make.bat`** file containing a simple *CMD script* to run a `nvcc` command for compile like this:

    nvcc -o add_cuda.exe %source%  %__defines%

The `source` variable contains all source files name included in project and the `__defines` variable contain defined values for compilation in order to compile in several ways.
possible values as argument passed to the script is shown below:

Value | Description 
--------- | --------- 
inarray | choose between an in array operation or out array *(described below)* 
overlap | choose between parallel data transfer using several streams and sequential data transfer run with no streams (using only `streams0`) 
debug | Active debug parts of code in purpose of debugging 
test | Active a separated `main` in purpose of testing functions individually 

> **`make.bat`** also has a `clean` option to remove compiled libraries and
> executable programs

**Why is there an overlap option?** 
We can run project with any number of streams we want as a argument passed to main program so we can run the overlap version with only one stream therefore it seems there is no need of overlap option that chose between using one stream or several. But it must be considered that running the overlap version even with only one stream dose not provide us a way to measure pure execute time of kernel because of asynchronous data transfer and kernel launches. At the end the purpose of overlap option considered as a way to calculate pure kernel execution time.

**In array operation against out array**
in array operation use input array itself to save the result instead of using a different array (out array). It use smaller amount of global memory on device but not that much. the size of output array depends on number of all used thread blocks in all kernels (each stream)

    int block_size =  1024;
    int stream_size = size / stream_count;
    int block_count = (stream_size/block_size)/2;
    int output_size = block_count * stream_count;

We can see output array size is `(size / block_size)/2` and its much smaller than the input size (`size`). choosing between in array or out array operation only saves a little memory and gave us no speedup.

# Project Structure

 - **`kernels.cu` with `kernels.cuh` header**

Kernels in this project held in separated file with cuda suffix (`.cu`). two in array kernel and out array kernel written in a single function using preprocessor conditions. Header of this file contain kernel prototypes respond to *inarray* option and `typedef struct` of argument using by kernel. The purpose of using a single struct as kernel argument instead of passing the arrays is **dealing** with struct in **C** and learning how to **initial, transfer data and pass** structs to kernels and it has no need in this project as an important feature.

 - **`app.cu` with no header**

Main program that set kernels variables and get arguments from user written in this file. This module contain several mains corresponding to each compile option like `overlap` option. each main function get needed arguments from user and call the special function that only compile if needed to. Input of the program is listed below:

 1. **factor**: the factor in scale of MB that specify size of the input array. for example factor=20 means 20 MB array (`size=1024 * 1024 * factor;`) 
 2. **stream_count**: number of stream (only ask with overlap option in compile)

Each function (`overlap_transfer_kernel` and `one_add_kernel`) has the same structure described below:

 - Define and set variables
 - Define and set kernel variables (like `dim3 grid_dim` and `block_dim` and `streams`)
 - Initial data on host and device
 - Sequential run
 - Parallel run	
 - Printing result and validation
 - Free allocated memory on host and device

There is some informing line that print out some information about compiled options like `overlap` and `inarray` option

> Note that the `app.cu` module suffix is also .cu because of it contains launching kernels command like: `kernel_function<<<kernel_config_variables>>>(kernel_arguments)`

 - **`helper_function.c` with `helper_function.h` header and `fuzzu_timing.c` with `fuzzy_timing.h` header**

These two module contain some helper functions like initialing random arrays or zero arrays on host with **`malloc`** function or **`cudaMallocHost`** function to provide a fixed framed allocation in memory. All functions in this two modules except `gettimeofday` function provided by [*Amir Hossein Sojoodi*](https://github.com/amirsojoodi) at repository of [GPGPU-2018Fall](https://github.com/amirsojoodi/GPGPU-2018Fall) hosted on github. The `gettimeofday` function provided manually from internet in order to using other functions at `fuzzy_timing.c` module in the way it used in a Linux based system.

# Results
After running the program without parallel data transferring with streams, speedup increase with size of input array but the total speedup limited (**1.9x** maximum) because of data transfer time. After running the program with several streams with `overlap` option total speedup increase but still limited (**5.4x** maximum) in term of data transfers. Kernel time wasn't issue because each part of kernel in stream ended before the next data transfer completed. So the **total speedup** always **limited** to **how much data transfer takes time**.  There is a table to show speedups due to number of streams:

Number of Stream | Total Speedup *(avrage 4 run)*
--------- | ----------
1 | 3.5153
2 | 4.324125
4 | 4.736175
8 | 5.2361
16 | 5.405225 

The picture below shows how data transfer limits the speedup even if kernel optimized gratefully.
  
![enter image description here](https://lh3.googleusercontent.com/fUjqjBH-aeqoBE2bouhbKSf2KxlbwhafCItJAMOzPMtoAeuI6b5aRNcLrJftMBauDx-ABIjlzg37 "Visual Profiler - Parallel run of 16 stream")

The total speedup used for only one purpose program that only want to launch one kernel and get speedup from it. But what if the data actually be **present at device**? In this case we must consider another speedup variable: **Execute Speedup**

> Execute speedup present the speedup of only kernel execution time
> against sequential time.

Seven version of kernel presented in the discussed lecture about reduction that each of them gain more speedup step by step. The table below present the execute speedup over implemented kernels correspond to kernels in the lecture on 40MB data:

Kernel Number | Total Speedup | Execute Speedup | Step Speedup *Over execution*
--------- | --------- | --------- | ---------
Kernel 4 (first add during global load) | 1.9454 | 8.6061 |
Kernel 5 (unroll last warp) | 2.1319 | 13.9464 | 1.62
Kernel 6 (completely unrolled) | 2.1725 | 15.75 | 1.12

**Why completely unrolled kernel gain less speedup?**
completely unrolling reduces plenty amount of works in kernel but it only gain 1.12 speedup factor that can be ignored but in the discussed lecture this kernel promises speedup around 1.8. The most logical reason can be **fixed number of threads in block** (1024) that can make too much overhead making all this thread per blocks. *Currently* I'm working on this reason about this lack of speedup by running experiments with different block sizes and even looking for more details and reasons around this issue.


> This Report Written by Farzin Mohammdi with [StackEdit](https://stackedit.io/).

