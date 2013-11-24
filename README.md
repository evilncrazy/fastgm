fastgm
======
A CUDA library for computations involving grid-shaped conditional random fields (CRFs). For usage, see the example in the example directory.

Features
========
- Works with CRFs that only have unary and pairwise potentials.
- Potential functions are functors that can be evaluated on the GPU, rather than just being a simple matrix of potential values.
- Templated on the potential functions, so you can implement your potential functions with functors.
- Most functions read and write to device memory only, so there's no host-device memory transfer overheads.
- Inference and decoding of CRFs on the GPU.
- Simple parameter estimation algorithms to compute maximum likelihood.

Compiling
=========
The library itself cannot be compiled into a standalone object file because it uses templates. So you will need some wrapper code that instantiates the necessary objects and compile that with the library to create an object file. The object file can then be used to link with applications that use libraries like OpenCV (as far as I know, you can't compile .cu files that include OpenCV headers).