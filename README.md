# Who is fynv

Fei Yang, former GPU software engineer of NVIDIA. Now working in my friend's start-up.

# My Blogs

## Cascaded Residual Encoding for HDR Lightmap Compression

... During the development of the rendering engine [Three.V8](https://github.com/fynv/three.v8), we also hit this problem. Because of that the system is mobile & web targeting, transfer efficiency is among our top concerns. After testing several ideas, we found that a JPEG based solution performs surprisingly well.

[More](https://fynv.github.io/Cascaded-Residual-Encoding-for-HDR-Lightmap-Compression/)

## Mixed-resolution Grid of Light Probes

This article introduces an experimental feature of the open-source project [Three.V8]( https://github.com/fynv/Three.V8), namely Mixed-resolution Grid of Light Probes (or LODProbeGrid which is used for coding). 

[More](https://fynv.github.io/MixedResolutionGridOfLightProbes/)

##  Web Browser + Game Engine: Combining 2 Monsters

In Metaverse development, it is becoming more and more crucial to have both modern web and modern graphics technologies working collaboratively in the same application.

My opinion is that it is best we can combine a web browser with a game engine at native level. 

This article tries to discuss why the 2 monsters should be and can be combined togther, and what are the current difficulties. By sharing my experience, I also want to persuade more people to join me to try to make it happen.

[More](https://fynv.github.io/WebBrowserPlusGameEngine.html)

## Programming GPU across the Language Boundaries

This article tries to illustrate the fact that NVRTC + dynamic-instantiation can be a powerful
CUDA programming paradigm for building GPU libraries that can be reused from an arbitary 
language that supports C/C++ interop. 

This has been the motivation of my "NVRTC Based GPU libraries" series of projects.

[More](https://fynv.github.io/ProgrammingGPUAcrossTheLaunguageBoundaries.html)

## Ray Tracing - What GPU Solves and Doesn't Solve

As I quited NVIDIA recently, now I have more freedom of talking about not just the positive sides of GPUs.

[More](https://fynv.github.io/RayTracingWhatGPUSolvesAndDoesntSolve.html)

# My Projects

## Three.V8

Embedabble 3D rendering engine using JavaScript as user script.

Project Page:
[https://github.com/fynv/Three.V8](https://github.com/fynv/Three.V8)

## LiveKit - Video Toolkit for Python

Project Page:
[https://github.com/fynv/livekit](https://github.com/fynv/livekit)

## VkInline

A Python interface to access the computation/rasterization/ray-tracing functionalities of Vulkan. Can be used for GPU computing and off-screen rendering.

Project Page:
[https://github.com/fynv/vkinline](https://github.com/fynv/vkinline)

## FeiRays - Vulkan based ray-tracing

Project Page:
[https://github.com/fynv/FeiRays](https://github.com/fynv/FeiRays)


## NVRTC Based GPU libraries

### ThrustRTC - Thrust-like standard library

Project Page:
[https://github.com/fynv/ThrustRTC](https://github.com/fynv/ThrustRTC)

Documentations:
[https://fynv.github.io/ThrustRTC/](https://fynv.github.io/ThrustRTC/)

### CURandRTC - Random number generator

Project Page:
[https://github.com/fynv/CURandRTC](https://github.com/fynv/CURandRTC)

### RTRTC - Ray-tracing

Project Page:
[https://github.com/fynv/RTRTC](https://github.com/fynv/RTRTC)

## Music & Singing Synthesizer

### ScoreDraft

ScoreDraft is a simple music/singing synthesizer that provides a Python based score authoring interface.

Project Page:
[https://github.com/fynv/ScoreDraft](https://github.com/fynv/ScoreDraft)

Documentations:
[https://fynv.github.io/ScoreDraft/](https://fynv.github.io/ScoreDraft/)

### ScoreDraftEditor

YAML editor UI for ScoreDraft.

Project Page:
[https://github.com/fynv/ScoreDraftEditor](https://github.com/fynv/ScoreDraftEditor)

Documentations:
[https://fynv.github.io/ScoreDraftEditor/](https://fynv.github.io/ScoreDraftEditor/)


### Score Demo Pages

[月半小夜曲](https://fynv.github.io/scores/yueban.html)


