# Web Browser + Game Engine: combining 2 monsters

In Metaverse development, it is becoming more and more crucial to have both modern web and modern graphics technologies working collaboratively in the same application.

However, in reality, we rarely have both at the same time. While web browsers are at the center of the modern web, game engines/3D rendering engines are at the center of modern realtime graphics. Projects of both kinds are typically huge ones, maintained by different groups of programmers. Since that both web browsers and 3D engines have a "rendering" part and require their own UI areas, when you run 2 of them side-by-side, they would appear as 2 clearly separated component. When you chose to use one of them, your system will likely to be weak in some aspects. For web browsers, there is WebGL for 3D graphics. WebGL2 is a low-level graphics interface vaguely corresponding to GLES 3.0's function set, with some limiters applied. Threre are "3D engines" like Three.js built on top of this interface, but they are significantly less powerful and less efficient than modern game engines. For game engines, most of them don't have built-in web support, so you will likely have to choose some additional wheels for network communication. Some of the modern web features are specfic to web browsers. Native environments have long been left behind by browsers when it comes to network functionalities.

This article tries to discuss why the 2 monsters should be and can be combined togther, and what are the current difficulties. By sharing my experience, I also want to persuade more people to join me to try to make it happen.

## To combine the two - Why?

Web browsers and game engines share some similarities. 

* Both of them are rendering engines
* Both of them are mainly written in native code
* Both of them can run platform neutral user scripts

Web browsers and game engines are both rendering engines. They are both trying to visualize some underlying data model. For browsers, the model is DOM. For game engines, the model can be a scene graph or some more complicated form. The rendering functionalities are all implementated on top of some low-level graphics APIs. There can be some kind of abstraction layers, but the calls will go all the way down to one of the native graphics APIs, like Direct3D, OpenGL, OpenGL ES, and Vulkan. While the web browsers focus on 2D rendering of text/image/video contents, game engines stresses more on 3D contents.

Scripting engines are commonly seen in both kinds of engines. For web browsers, the scripts mostly come from remote servers. The scripts define the content of a web pages as well as client logics. For game engines, the scripts are written by game developers to define game contents and logics. For a networked game, it also makes perfect sense for the scripts to come from a remote server. 

So basically, web browsers and game engines can be seen as rendering engines sharing some similar engineering structures, with different focuses of design. We should be able to combine the two just by improving the "weak" part of one engine, using the "strong" part of design from another engine.

## Why building a 3D rendering engine in a scripting language is bad

During the recent years of web browser development, the community choose to do a relatively strightforward exposure of the low-level 3D APIs to the scripting layer and let script programmers to design their 3D rendering engine using the scripting language. Both WebGL and WebGPU are following the similar philosophy. This can work in many cases, but also have significant limitations.

1. JavaScript is bad for maths, especially geometry maths. The lack of operator overloading and vectorization tools makes it appears redundant even doing some simple operations.

2. The engines have to be open-sourced. Also they are required to be downloaded from remote server so they cannot be very big.

3. High frequent API calls can be inefficient. There can be some improvement in WebGPU, only in this aspect.

4. Graphics capabilities are deliberately weakened. An important thing the WebGL wrapper does is to try to limit the GPU capabilities. The concern is mostly security and stability related. The user scripts, especially remote scripts are more likely to be buggy or even malicious than compiled, fixed functions. Therefore JS engines will likely to remain weak in the future while the architecture remains unchanged.

Therefore, having a built-in 3D engine is more desirable than building one on top of the scripting engine. It can be more powerful and more efficient without compromising the security.

## Some preliminary attempts and current problems

Based on the above thinking, I did 2 research projects. 

The 1st one is [Three.V8](https://github.com/fynv/Three.V8). Three.V8 is a simple 3D rendering engine directly based on V8 scripting engine and native OpenGL 4.3. The source code is super small, so that I can use it as a reference then play around in different environment setups.

The 2nd one is [ChromiumHacks](https://github.com/fynv/ChromiumHacks). This one actually tries to port the Three.V8 code into the huge Chromium code trunk.

The projects prove that the idea is technically feasible to do, given enough programming work force. However there are problems which have already shown up. The main issue is about the OpenGL ES wrapping layers, among which, [ANGLE](https://github.com/google/angle), and the layer above it, the code under the "src/gpu" folder. To use the existing context created together with a DOM Canvas, it is natural to build the engine against the GLES2Interface APIs, which are provided by the "src/gpu" module. However these APIs are designed to be directly used by WebGL2, therefore all the limiters are already applied. Far from the real native APIs, GLES2Interface is not even ANGLE. It took me a lot of hacking effort to upgrade the context to GLES 3.1, but there are still features missing, like SRGB framebuffers. 

It might be possible to create a separate rendering context which is closer to the native APIs. However, that seems to involve some huge coding effort which I can't do currently.

There are also some inherent issues about building a 3D rendering engine inside a web browser. One of them is that 3D rendering engines are extremely diversified. There are too many possibilities, and maybe non of them can become a standard and get a W3C approvement. Every engine company can make their own version of Chromium, and the only use-case would be inside their own proprietary products.


## Possibility to do it the other way round

As a graphics programmer, I'm able to write my own 3D rendering engine from scratch. Now that I've learnt how to wrap the interfaces using V8, I can build things like [Three.V8](https://github.com/fynv/Three.V8) which is already a little bit like a "browser". It seems possible for me to add more and more wheels, including network APIs into the engine then finally achieve the goal to "combine a game engine with a web browser". However, the amount of work seems to be even more huge.



