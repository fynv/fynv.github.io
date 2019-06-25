# Programming GPU across the Language Boundaries

This article tries to illustrate the fact that NVRTC can be a powerful tool for 
building GPU libraries that can be reused from an arbitary language that supports C/C++ interop.
This has been the motivation of my "NVRTC Based GPU libraries" series of projects.

## Importance and limitation of templates in GPU libraries

Computationally intensive libraries are mostly written in C/C++ or another compiled language. 
Interpreted languages usually can also benefit from these libraries through library reusing. 

This pattern also applies to some of the GPU libraries. As long as all functionalities are exposed
through concrete host APIs, these GPU libraries have no difference with other C/C++ libraries.

However, there are exceptions, known as "template libraries". These libraries are basically source-code
that are not compiled util they are used. In GPU programming, templates are especially interesting and
indispensible because of the inefficiency of dynamic polymorphism in GPU programming models. One famous 
example of this kind of libraries is Thrust. Because these libraries are provided in uncompiled 
form, there is no way to reuse them from a language other than the one these libraries are programmed in.

```cpp
template<typename ForwardIterator , typename T >
void thrust::replace(ForwardIterator first,
                     ForwardIterator last,
                     const T & old_value,
                     const T & new_value)
```

This is just a simple example of a function provided by Thrust. All parameters are templated. "T" can be
anything that has a definition, and "ForwardIterator" should "T_*" or something compatible. Functions 
like this can be very powerful and useful, but they are only available to C++.

## Run-time compilation of GPU device code

Run-time compilation is not a new thing to GPU programmers. In graphics programming, we use it to compile 
shaders to adapt to different running environment. In OpenCL, we also use it as the default way to compile
device code. In CUDA programming, however, we didn't have run-time compilation until CUDA 7.x, the new 
module of the CUDA SDK is called NVRTC. 

Now let's see how run-time compilation can be used to build libraries as poweful as the template libraries
while they are still reusable from another language. 

### As powerful as templates

First, it is not difficult to imagine that most problem that can be solved using template programming can 
also be solved by automated string modifications of the source-code. The "instantiation" of a template is 
no more than replacing the template parameters with the concrete types and values. 

Second, we don't need to use templates for the host code in most cases. Because dynamic polymorphism is 
well supported in host code. It is just efficiency issue, and here we are trying to use GPU for the 
computational intensive part.

Therefore, all we need to do is to make the "instantiation" process of device code happen at run-time,
so we can make it work together with the dynamic host code. 2 pieces are required to make this happen,
one is automated string modifications, the other is run-time compilation of device code.

Here is the ThrustRTC function corresponding to thrust::replace:

```cpp
bool TRTC_Replace(DVVectorLike& vec, 
                  const DeviceViewable& old_value, 
                  const DeviceViewable& new_value);
```

DVVectorLike and DeviceViewable are both host classes. They are abstract classes and their sub-classes
can contain data of different types, and the type information is recorded by the members of the run-time
objects.

There are 2 interface functions that each Device Viewable Object much implement:

```cpp
class DeviceViewable
{
public:
    ...
    virtual std::string name_view_cls() const = 0;
    virtual ViewBuf view() const = 0;
};
```

The function *name_view_cls()* returns how this object is represented in GPU device code, 
which must be something that the GPU compiler recognizes. The string will be involved in 
the string modification processes when the object is used.

The function *view()* returns a byte buffer containing the data that can be copied to 
device into a variable of the *name_view_cls()* type. 

In the library we can have some built-in code (as string) like:

```cpp
template<T_Vec, T_Value>
extern "C" __global__ 
void replace(T_Vec view_vec, 
             T_Value old_value,
             T_Value new_value)
{
    uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid>=view_vec.size()) return;
    if (view_vec[idx] == (decltype(view_vec)::value_t)old_value) 
        view_vec[idx] = (decltype(view_vec)::value_t)new_value;
}

```

At runtime, when we try to launch the kernel using some concrete parameters, it can be
easily "instantiated" as something like:

```cpp
template<class _T>
struct VectorView
{
    typedef _T value_t;
    typedef _T& ref_t;

    value_t* _data;
    size_t _size;

    __device__ size_t size() const
    {
        return _size;
    }

    __device__ ref_t operator [](size_t idx)
    {
        return _data[idx];
    }
};

extern "C" __global__ 
void replace(VectorView<float> view_vec, 
             float old_value,
             float new_value)
{
    uint32_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid>=view_vec.size()) return;
    if (view_vec[idx] == (decltype(view_vec)::value_t)old_value) 
        view_vec[idx] = (decltype(view_vec)::value_t)new_value;
}

```

You can see that here we don't need to change the function body and only need to change
the function header.

### Why it is portable

First, the compiler itself is nothing special as a C library, except that it is only available 
as a shared library, don't know why NVIDIA doesn't provide a static version. It is used together
with the CUDA driver API, which is also nothing special.

Second, the library APIs are also nothing special. All it exposes are compiled C/C++ APIs. 

These compiled APIs can be called from any programming language that has a C-interop capability.
Classes like "DeviceViewable" can be wrapped as a classes in the target languages. This requires
the target language supports OO programming though, which most of them do nowadays.

Therefore, you can find very similar functions in the C++/Python/C#/JAVA versions of ThrustRTC:

```cpp
// C++
bool TRTC_Replace(DVVectorLike& vec, 
                  const DeviceViewable& old_value, 
                  const DeviceViewable& new_value);
```

```python
# Python
def Replace(vec, old_value, new_value):
    ...
```

```cs
// C#
using ThrustRTCLR;

namespace ThrustRTCSharp
{
    public partial class TRTC
    {
        ...
        public static bool Replace(DVVectorLike vec, 
                                   DeviceViewable old_value, 
                                   DeviceViewable new_value)    
        ...
    }
}
```

```java
// JAVA
package JThrustRTC;

public class TRTC 
{
    ...
    public static boolean Replace(DVVectorLike vec, 
                                  DeviceViewable old_value, 
                                  DeviceViewable new_value)
    ...
}
```

## An alternative way to build CUDA libraries?

It is not surprising that most existing CUDA libraries are based on CUDA runtime. 
An important reason is that it is indeed very handy. Mixing both host code and device code 
in a single source-file in the "same" language is attractive, and makes people believe 
that knowing C++ programming is sufficient for them to program GPU. 
CUDA runtime + static compilation + templates is also the officially recommended paradigm
of CUDA programming. Before there is NVRTC, there seems to be little choice. Even after NVRTC 
has been there, people still seldomly realize how the game can be changed.

Here, we see that NVRTC + dynamic instantiation can be a serious alternative paradigm for 
CUDA programming in general. It is as powerful as templates, and it is portable, which is
important for libraries. In addition, it reduces the library compilation time and generates
very slim binary. These comes at a cost that an application can be quite slow when it is 
executed for the first time (the compilation time is moved to here!).

The fact that CUDA is so widely used these days makes it a broad area to explore: 

How does the alternative paradigm behave in each of the application fields of CUDA?

At the time this article is written. I have only finished ThrustRTC, and it is the first 
time we can compare it with Thrust.

The next fields I'm going to explore includes:

* Matrix operations
* Neural networks
* Visualization

Don't be surprised if you see some libraries named BlasRTC, DNNRTC or VisRTC.


