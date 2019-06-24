# Programming GPU across the Language Boundaries

This article tries to illustrate the fact that NVRTC can be a powerful tool for 
building GPU libraries that can be reused in an arbitary language that supports C/C++ interop.
This has been the motivation of my "NVRTC Based GPU libraries" series of projects.

## GPU library reuse in a non-C/C++ language

Computationally intensive libraries are mostly written in C/C++ or another compiled language. 
Interpreted languages usually can also benefit from these libraries through library reusing. 

This pattern also applies to some of the GPU libraries. As long as all functionalities are exposed
through host APIs, these GPU libraries have no difference with other C/C++ libraries.

However, there are exceptions, known as "template libraries". These libraries are basically source-code
that are not compiled util they are used. In GPU programming, templates are especially interesting and
indispensible because of the inefficiency of dynamic polymorphism in GPU programming models. One famous 
example of this kind of libraries is Thrust. Because these libraries are provided in uncompiled 
form, there is no way to reuse them in a language other than the one these libraries are programmed in.

## Run-time compilation of GPU device code




