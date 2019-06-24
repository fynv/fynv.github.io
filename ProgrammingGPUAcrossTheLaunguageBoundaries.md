# Programming GPU across the Language Boundaries

This article tries to illustrate the fact that NVRTC can be a powerful tool for 
building GPU libraries that can be reused in an arbitary language that supports C/C++ interop.
This has been the motivation of my "NVRTC Based GPU libraries" series of projects.

## GPU library reuse in a non-C/C++ language

Computationally intensive libraries are mostly written in C/C++ or another compiled language. 
Interpreted languages usually can also benefit from these libraries through library reusing. 

This pattern also applies to some of the GPU libraries. As long as all functionalities are exposed
through concrete host APIs, these GPU libraries have no difference with other C/C++ libraries.

However, there are exceptions, known as "template libraries". These libraries are basically source-code
that are not compiled util they are used. In GPU programming, templates are especially interesting and
indispensible because of the inefficiency of dynamic polymorphism in GPU programming models. One famous 
example of this kind of libraries is Thrust. Because these libraries are provided in uncompiled 
form, there is no way to reuse them in a language other than the one these libraries are programmed in.

```cpp
template<typename ForwardIterator , typename T >
void thrust::replace	(	ForwardIterator 	first,
							ForwardIterator 	last,
							const T & 	old_value,
							const T & 	new_value 
```

This is just a simple example of a function provided by Thrust. All parameters are templated. "T" can be
anything that has a definition, and "ForwardIterator" should "T_*" or something compatible. Functions 
like this can be very powerful and useful, but they are only available to C++.

## Run-time compilation of GPU device code

It is not difficult to imagine that most problem that can be solved using template programming can also be
solved by automated string modifications of the source-code. The "instantiation" of a template is no more
than replacing the template parameters with the concrete types and values. 

Therefore, when a compiler is available at run-time, we are able to complete some tasks that could have 
required templating by manipulating the source-code using run-time type information.




